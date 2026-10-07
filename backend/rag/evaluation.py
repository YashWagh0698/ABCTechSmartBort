"""Step 9 - Retrieval evaluation.

"Bad retrieval -> bad context -> bad answer." (project documentation)
Before any LLM is added we MEASURE whether the retriever finds the right chunks.

HOW A CHUNK COUNTS AS "RELEVANT"
    Each answerable question lists its EVIDENCE: the exact text that answers it.
    A retrieved chunk is relevant if it contains that text. We label by evidence
    (not by section name) because in our document most facts appear twice, in the
    policy section AND in the FAQ. Both are valid sources, and labelling by
    evidence stays correct even if chunk sizes or chunk ids change.

    Evidence items are FACTS. A fact is a string, or a list of alternative
    phrasings (any one counts):   [["last working day of each month",
                                    "last working day of every month"]]
    A multi-section question has several facts and needs all of them.

THE METRICS (K = how many top results we look at)
    Recall@K     facts found in the top K chunks / facts needed.
                 One fact: it is simply "was the right chunk in the top K?"
    Precision@K  relevant chunks in the top K / K.
                 NOTE: with only 1-2 relevant chunks in the whole index this can never
                 reach 1.0 for big K. It measures noise, so read it with that in mind.
    MRR          mean of 1/rank of the first relevant chunk (1st -> 1, 2nd -> 0.5, 3rd -> 0.33...)

UNANSWERABLE QUESTIONS cannot be scored with these metrics (there is nothing to find).
For them we record how HIGH the best similarity score is. If a wrong question scores as
high as a right one, no score threshold can separate them. That is exactly the data
Step 12 needs to choose a refusal threshold.
"""
import json
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from backend.rag.embeddings import EmbeddingService
from backend.rag.ingestion import ingest_document
from backend.rag.retriever import Retriever
from backend.rag.vector_store import VectorStore

QUESTION_TYPES = ("direct", "paraphrase", "multi_section", "unsupported", "near_miss")
ANSWERABLE_TYPES = ("direct", "paraphrase", "multi_section")
UNANSWERABLE_TYPES = ("unsupported", "near_miss")
TYPE_LABELS = {
    "direct": "Direct", "paraphrase": "Paraphrased", "multi_section": "Multi-section",
    "unsupported": "Unsupported", "near_miss": "Similar but incorrect",
}
DEFAULT_KS = (1, 3, 5)
MAX_RELEVANT_PER_QUESTION = 4          # more than this means the evidence text is too generic


class EvaluationDataError(Exception):
    """The question file is missing or malformed."""


# =============================================================================
# The dataset
# =============================================================================
@dataclass
class EvalQuestion:
    id: str
    type: str
    question: str
    answer_available: bool
    expected_sections: List[str] = field(default_factory=list)
    facts: List[List[str]] = field(default_factory=list)        # each fact = list of alternative phrasings
    absent_terms: List[str] = field(default_factory=list)
    related_section: Optional[str] = None


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _as_fact(item, where: str) -> List[str]:
    phrases = [item] if isinstance(item, str) else item
    if (not isinstance(phrases, list) or not phrases
            or not all(isinstance(p, str) and p.strip() for p in phrases)):
        raise EvaluationDataError(f"{where}: an evidence item must be a text or a non-empty list of texts")
    return [p.strip() for p in phrases]


def load_questions(path) -> List[EvalQuestion]:
    """Read and strictly validate evaluation/questions.json."""
    path = Path(path)
    if not path.is_file():
        raise EvaluationDataError(f"Question file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationDataError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise EvaluationDataError("The question file must contain a non-empty list")

    questions, seen_ids = [], set()
    for index, item in enumerate(raw, 1):
        where = f"question #{index}"
        if not isinstance(item, dict):
            raise EvaluationDataError(f"{where}: must be an object")
        qid = item.get("id")
        if not isinstance(qid, str) or not qid.strip():
            raise EvaluationDataError(f"{where}: missing 'id'")
        where = f"question {qid}"
        if qid in seen_ids:
            raise EvaluationDataError(f"{where}: duplicate id")
        seen_ids.add(qid)

        qtype = item.get("type")
        if qtype not in QUESTION_TYPES:
            raise EvaluationDataError(f"{where}: 'type' must be one of {', '.join(QUESTION_TYPES)}")
        text = item.get("question")
        if not isinstance(text, str) or not text.strip():
            raise EvaluationDataError(f"{where}: missing 'question'")
        available = item.get("answer_available")
        if not isinstance(available, bool):
            raise EvaluationDataError(f"{where}: 'answer_available' must be true or false")
        if available != (qtype in ANSWERABLE_TYPES):
            raise EvaluationDataError(f"{where}: type '{qtype}' does not match answer_available={available}")

        question = EvalQuestion(id=qid, type=qtype, question=text.strip(), answer_available=available)
        if available:
            evidence = item.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise EvaluationDataError(f"{where}: an answerable question needs a non-empty 'evidence' list")
            question.facts = [_as_fact(e, where) for e in evidence]
            sections = item.get("expected_section")
            sections = [sections] if isinstance(sections, str) else sections
            if (not isinstance(sections, list) or not sections
                    or not all(isinstance(s, str) and s.strip() for s in sections)):
                raise EvaluationDataError(f"{where}: 'expected_section' must be a text or a list of texts")
            question.expected_sections = sections
            if qtype == "multi_section" and len(question.facts) < 2:
                raise EvaluationDataError(f"{where}: a multi_section question needs at least 2 evidence facts")
        else:
            terms = item.get("absent_terms")
            if (not isinstance(terms, list) or not terms
                    or not all(isinstance(t, str) and t.strip() for t in terms)):
                raise EvaluationDataError(f"{where}: an unanswerable question needs a non-empty 'absent_terms' list")
            question.absent_terms = terms
            question.related_section = item.get("related_section")
        questions.append(question)
    return questions


def check_against_chunks(questions: Sequence[EvalQuestion], chunks) -> List[str]:
    """Verify the dataset against the real chunks. Returns problems (empty list = dataset is sound)."""
    problems: List[str] = []
    texts = [(c.section, _norm(c.text)) for c in chunks]
    seen_questions = set()
    for q in questions:
        key = _norm(q.question)
        if key in seen_questions:
            problems.append(f"{q.id}: the same question text appears twice")
        seen_questions.add(key)

        if q.answer_available:
            for fact in q.facts:
                matching = [section for section, text in texts if any(_norm(p) in text for p in fact)]
                if not matching:
                    problems.append(f"{q.id}: the evidence {fact[0]!r} is not in any chunk")
                elif not set(matching) & set(q.expected_sections):
                    problems.append(f"{q.id}: evidence {fact[0]!r} is in {sorted(set(matching))}, "
                                    f"not in the expected section(s) {q.expected_sections}")
            relevant = sum(1 for _, text in texts if is_relevant(text, q.facts))
            if relevant > MAX_RELEVANT_PER_QUESTION:
                problems.append(f"{q.id}: {relevant} chunks count as relevant; the evidence is too generic")
        else:
            for term in q.absent_terms:
                if any(_norm(term) in text for _, text in texts):
                    problems.append(f"{q.id}: '{term}' IS in the document, so this question is answerable")
    return problems


# =============================================================================
# Metrics (pure functions: easy to test by hand)
# =============================================================================
def is_relevant(text: str, facts: Sequence[Sequence[str]]) -> bool:
    """Does this chunk contain the evidence for ANY of the facts?"""
    normalised = _norm(text)
    return any(_norm(p) in normalised for fact in facts for p in fact)


def facts_covered(texts: Sequence[str], facts: Sequence[Sequence[str]]) -> int:
    """How many facts appear in at least one of the texts."""
    normalised = [_norm(t) for t in texts]
    return sum(1 for fact in facts if any(_norm(p) in t for t in normalised for p in fact))


def recall_at_k(texts: Sequence[str], facts, k: int) -> float:
    return facts_covered(texts[:k], facts) / len(facts)


def precision_at_k(texts: Sequence[str], facts, k: int) -> float:
    return sum(1 for t in texts[:k] if is_relevant(t, facts)) / k


def reciprocal_rank(texts: Sequence[str], facts) -> float:
    for rank, text in enumerate(texts, 1):
        if is_relevant(text, facts):
            return 1.0 / rank
    return 0.0


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


# =============================================================================
# Running the evaluation
# =============================================================================
@dataclass
class RetrievedItem:
    rank: int
    chunk_id: str
    heading: str
    similarity: float
    relevant: Optional[bool]          # None for unanswerable questions


@dataclass
class QuestionResult:
    question: EvalQuestion
    retrieved: List[RetrievedItem]
    recall: Dict[int, Optional[float]]
    precision: Dict[int, Optional[float]]
    reciprocal_rank: Optional[float]
    first_relevant_rank: Optional[int]

    @property
    def top1_similarity(self) -> Optional[float]:
        return self.retrieved[0].similarity if self.retrieved else None


@dataclass
class EvaluationRun:
    results: List[QuestionResult]
    ks: Sequence[int]
    max_k: int
    meta: dict = field(default_factory=dict)

    # ---- grouping --------------------------------------------------------
    def of_type(self, *types) -> List[QuestionResult]:
        return [r for r in self.results if r.question.type in types]

    def metrics(self, results: Sequence[QuestionResult]) -> dict:
        """Mean Recall@K / Precision@K and MRR over the given (answerable) results."""
        return {
            "count": len(results),
            "recall": {k: _mean([r.recall[k] for r in results]) for k in self.ks},
            "precision": {k: _mean([r.precision[k] for r in results]) for k in self.ks},
            "mrr": _mean([r.reciprocal_rank for r in results]),
        }

    def overall(self) -> dict:
        return self.metrics(self.of_type(*ANSWERABLE_TYPES))

    def by_type(self) -> Dict[str, dict]:
        return {t: self.metrics(self.of_type(t)) for t in ANSWERABLE_TYPES if self.of_type(t)}

    def failures(self, k: Optional[int] = None) -> List[QuestionResult]:
        """Answerable questions whose evidence was NOT fully found in the top K."""
        k = k or max(self.ks)
        return [r for r in self.of_type(*ANSWERABLE_TYPES) if (r.recall[k] or 0) < 1.0]

    # ---- scores for the refusal threshold (Step 12) ---------------------------
    def top1_scores(self, *types) -> List[float]:
        return [r.top1_similarity for r in self.of_type(*types) if r.top1_similarity is not None]

    def score_summary(self) -> Dict[str, Optional[dict]]:
        groups = {"answerable": ANSWERABLE_TYPES, "unsupported": ("unsupported",), "near_miss": ("near_miss",)}
        out: Dict[str, Optional[dict]] = {}
        for name, types in groups.items():
            scores = self.top1_scores(*types)
            out[name] = ({"n": len(scores), "min": min(scores), "median": statistics.median(scores),
                          "max": max(scores)} if scores else None)
        return out

    def threshold_sweep(self, thresholds: Optional[Sequence[float]] = None) -> List[dict]:
        """For each candidate threshold: how many answerable questions would still be answered,
        and how many unanswerable ones would be correctly refused."""
        thresholds = thresholds or [round(0.10 + 0.05 * i, 2) for i in range(13)]       # 0.10 ... 0.70
        answerable = self.top1_scores(*ANSWERABLE_TYPES)
        unsupported = self.top1_scores("unsupported")
        near = self.top1_scores("near_miss")
        rows = []
        for t in thresholds:
            rows.append({
                "threshold": t,
                "answerable_kept": _mean([1.0 if s >= t else 0.0 for s in answerable]),
                "unsupported_refused": _mean([1.0 if s < t else 0.0 for s in unsupported]),
                "near_miss_refused": _mean([1.0 if s < t else 0.0 for s in near]),
            })
        return rows

    def separation(self) -> dict:
        """Is there ANY threshold that keeps every answerable question and refuses every unanswerable one?"""
        a, u, n = (self.top1_scores(*ANSWERABLE_TYPES), self.top1_scores("unsupported"),
                   self.top1_scores("near_miss"))
        info = {
            "answerable_min": min(a) if a else None,
            "unsupported_max": max(u) if u else None,
            "near_miss_max": max(n) if n else None,
        }
        info["clean_gap_unsupported"] = bool(a and u and min(a) > max(u))
        info["clean_gap_near_miss"] = bool(a and n and min(a) > max(n))
        return info

    # ---- output --------------------------------------------------------------------
    def to_dict(self) -> dict:
        def r4(x):
            return None if x is None else round(x, 4)
        overall = self.overall()
        return {
            "meta": self.meta,
            "ks": list(self.ks),
            "overall": {"count": overall["count"],
                        "recall": {str(k): r4(v) for k, v in overall["recall"].items()},
                        "precision": {str(k): r4(v) for k, v in overall["precision"].items()},
                        "mrr": r4(overall["mrr"])},
            "score_summary": self.score_summary(),
            "separation": self.separation(),
            "threshold_sweep": self.threshold_sweep(),
            "questions": [{
                "id": r.question.id, "type": r.question.type, "question": r.question.question,
                "answer_available": r.question.answer_available,
                "recall": {str(k): r4(v) for k, v in r.recall.items()},
                "precision": {str(k): r4(v) for k, v in r.precision.items()},
                "reciprocal_rank": r4(r.reciprocal_rank),
                "first_relevant_rank": r.first_relevant_rank,
                "retrieved": [{"rank": i.rank, "chunk_id": i.chunk_id, "heading": i.heading,
                               "similarity": r4(i.similarity), "relevant": i.relevant} for i in r.retrieved],
            } for r in self.results],
        }


def evaluate(retriever: Retriever, questions: Sequence[EvalQuestion],
             ks: Sequence[int] = DEFAULT_KS, meta: Optional[dict] = None) -> EvaluationRun:
    """Run every question through the retriever and score the results."""
    ks = tuple(sorted(set(ks)))
    if not ks or ks[0] < 1:
        raise ValueError("ks must be positive whole numbers")
    max_k = max(max(ks), 10)
    results: List[QuestionResult] = []
    for q in questions:
        response = retriever.retrieve(q.question, top_k=max_k)
        texts = [hit.text for hit in response.results]
        retrieved = [
            RetrievedItem(rank, hit.chunk_id, hit.heading, hit.similarity,
                          is_relevant(hit.text, q.facts) if q.answer_available else None)
            for rank, hit in enumerate(response.results, 1)
        ]
        if q.answer_available:
            first = next((i.rank for i in retrieved if i.relevant), None)
            results.append(QuestionResult(
                q, retrieved,
                recall={k: recall_at_k(texts, q.facts, k) for k in ks},
                precision={k: precision_at_k(texts, q.facts, k) for k in ks},
                reciprocal_rank=reciprocal_rank(texts, q.facts), first_relevant_rank=first))
        else:
            results.append(QuestionResult(q, retrieved, {k: None for k in ks}, {k: None for k in ks}, None, None))
    return EvaluationRun(results=results, ks=ks, max_k=max_k, meta=meta or {})


# =============================================================================
# Experiments with a temporary index (does not touch your real one)
# =============================================================================
def build_temporary_index(embedder: EmbeddingService, document_path, max_words: int, overlap_words: int):
    """Build a throw-away index so settings can be compared without changing the real index.

    Returns (retriever, temp_dir). Call temp_dir.cleanup() when finished.
    """
    temp_dir = tempfile.TemporaryDirectory(prefix="abc_eval_", ignore_cleanup_errors=True)
    store = VectorStore(Path(temp_dir.name) / "chroma", "eval_tmp")
    ingest_document(document_path, embedder, store, max_words=max_words, overlap_words=overlap_words)
    return Retriever(embedder, store), temp_dir


# =============================================================================
# The report
# =============================================================================
def _fmt(x: Optional[float], digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def render_markdown(run: EvaluationRun) -> str:
    ks = list(run.ks)
    meta = run.meta
    lines: List[str] = ["# Retrieval Evaluation Report", ""]

    lines += ["## Setup", "", "| Item | Value |", "|---|---|"]
    for key, label in (("generated_at", "Generated"), ("embedding_model", "Embedding model"),
                       ("document_version", "Document version"), ("chunk_max_words", "Chunk size limit (words)"),
                       ("chunk_overlap_words", "Chunk overlap (words)"), ("chunks_in_index", "Chunks in index"),
                       ("index_mode", "Index used")):
        if key in meta:
            lines.append(f"| {label} | {meta[key]} |")
    answerable = run.of_type(*ANSWERABLE_TYPES)
    lines += [f"| Questions | {len(run.results)} ({len(answerable)} answerable, "
              f"{len(run.results) - len(answerable)} not answerable) |", ""]
    if "embedding_model" in meta and "hashing" in str(meta["embedding_model"]):
        lines += ["> **Warning:** these numbers come from the offline `hashing` stand-in, which only matches shared "
                  "words. They are NOT the quality of your real system. Run the evaluation with the real model.", ""]

    lines += ["## Results for answerable questions", "",
              "| Question type | N | " + " | ".join(f"Recall@{k}" for k in ks) + " | "
              + " | ".join(f"Precision@{k}" for k in ks) + " | MRR |",
              "|---|---|" + "---|" * (2 * len(ks) + 1)]
    rows = [(TYPE_LABELS[t], m) for t, m in run.by_type().items()] + [("**All answerable**", run.overall())]
    for label, m in rows:
        lines.append(f"| {label} | {m['count']} | " + " | ".join(_fmt(m["recall"][k]) for k in ks) + " | "
                     + " | ".join(_fmt(m["precision"][k]) for k in ks) + f" | {_fmt(m['mrr'])} |")
    lines += ["", "How to read this: **Recall@K** is the share of needed facts found in the top K chunks. "
              "**MRR** rewards putting the right chunk first (1.0 = always first). **Precision@K** is low by "
              "design here: each fact is in only 1-2 chunks, so most of the K results cannot be relevant.", ""]

    k_fail = max(ks)
    failures = run.failures(k_fail)
    lines += [f"## Answerable questions NOT fully found in the top {k_fail}", ""]
    if not failures:
        lines += ["None. Every answerable question had all its evidence in the top "
                  f"{k_fail} chunks.", ""]
    else:
        for r in failures:
            q = r.question
            lines += [f"**{q.id}** ({TYPE_LABELS[q.type]}): {q.question}", "",
                      f"- needed evidence: " + "; ".join(f"`{f[0]}`" for f in q.facts),
                      f"- recall@{k_fail}: {_fmt(r.recall[k_fail])}"
                      + (f", first relevant chunk at rank {r.first_relevant_rank}" if r.first_relevant_rank
                         else f", no relevant chunk in the top {run.max_k}")]
            for i in r.retrieved[:3]:
                lines.append(f"- retrieved #{i.rank}: {i.similarity:+.3f} {i.heading}")
            lines.append("")

    not_first = [r for r in answerable if r.first_relevant_rank and r.first_relevant_rank > 1]
    lines += ["## Found, but not ranked first", ""]
    if not not_first:
        lines += ["None.", ""]
    else:
        lines += ["| Question | Rank of first relevant chunk |", "|---|---|"]
        lines += [f"| {r.question.id}: {r.question.question} | {r.first_relevant_rank} |" for r in not_first]
        lines.append("")

    lines += ["## Questions with no answer in the document", "",
              "The retriever always returns the closest chunks, even when none of them can answer. "
              "The similarity of the best chunk is what a refusal rule can use.", "",
              "| Id | Type | Question | Best similarity | Closest chunk |", "|---|---|---|---|---|"]
    for r in run.of_type(*UNANSWERABLE_TYPES):
        top = r.retrieved[0] if r.retrieved else None
        lines.append(f"| {r.question.id} | {TYPE_LABELS[r.question.type]} | {r.question.question} | "
                     f"{_fmt(top.similarity if top else None, 3)} | {top.heading if top else '-'} |")
    lines.append("")

    summary = run.score_summary()
    lines += ["## Best-similarity scores by question group", "",
              "| Group | N | Lowest | Median | Highest |", "|---|---|---|---|---|"]
    for name, label in (("answerable", "Answerable"), ("unsupported", "Unsupported"),
                        ("near_miss", "Similar but incorrect")):
        s = summary[name]
        lines.append(f"| {label} | {s['n']} | {_fmt(s['min'], 3)} | {_fmt(s['median'], 3)} | {_fmt(s['max'], 3)} |"
                     if s else f"| {label} | 0 | - | - | - |")
    sep = run.separation()
    lines += [""]
    if sep["answerable_min"] is not None and sep["unsupported_max"] is not None:
        if sep["clean_gap_unsupported"]:
            lines.append(f"- **Unsupported questions:** the highest unsupported score ({sep['unsupported_max']:.3f}) is "
                         f"below the lowest answerable score ({sep['answerable_min']:.3f}), so a threshold between "
                         "them separates these two groups on this data.")
        else:
            lines.append(f"- **Unsupported questions:** the groups OVERLAP (highest unsupported "
                         f"{sep['unsupported_max']:.3f} vs lowest answerable {sep['answerable_min']:.3f}). "
                         "No single threshold refuses every unsupported question and keeps every answerable one.")
    if sep["answerable_min"] is not None and sep["near_miss_max"] is not None:
        if sep["clean_gap_near_miss"]:
            lines.append(f"- **Similar-but-incorrect questions:** also separable (highest {sep['near_miss_max']:.3f}).")
        else:
            lines.append(f"- **Similar-but-incorrect questions:** these OVERLAP with answerable ones (highest "
                         f"{sep['near_miss_max']:.3f}). They are about a real topic, so the retriever finds a related "
                         "chunk. A score threshold cannot catch them; the grounded prompt and answer check "
                         "(Steps 11-12) must.")

    lines += ["", "## Threshold sweep", "",
              "If the refusal rule were `best similarity >= threshold`:", "",
              "| Threshold | Answerable questions still answered | Unsupported refused | Similar-but-incorrect refused |",
              "|---|---|---|---|"]
    for row in run.threshold_sweep():
        lines.append(f"| {row['threshold']:.2f} | {_fmt(row['answerable_kept'])} | "
                     f"{_fmt(row['unsupported_refused'])} | {_fmt(row['near_miss_refused'])} |")
    lines += ["", "A good threshold keeps the first column high and the other two high at the same time. "
              "Choose it in Step 12 from these numbers, using the REAL embedding model.", ""]
    return "\n".join(lines)
