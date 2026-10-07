"""Step 9 tests: retrieval evaluation.

A) Metrics, checked against numbers worked out BY HAND.
B) The question file: schema rules, and verification against the real chunks.
C) evaluate() with a fake retriever whose results are fixed, so every expected
   number below can be recomputed on paper.
D) The report, and a full run on the real document with the offline embedder.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.models.search import RetrievalResponse, SearchResult
from backend.rag.evaluation import (
    EvalQuestion, EvaluationDataError, build_temporary_index, check_against_chunks, evaluate,
    facts_covered, is_relevant, load_questions, precision_at_k, recall_at_k, reciprocal_rank,
    render_markdown,
)
from backend.rag.ingestion import ingest_document
from backend.rag.retriever import Retriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"


# =============================================================================
# A) Metrics by hand
# =============================================================================
# Ranks:      1         2                 3     4              5
TEXTS = ["nothing", "has ALPHA inside", "zzz", "alpha again", "more"]
FACT_ALPHA = [["alpha"]]


def test_relevance_ignores_case_and_whitespace():
    assert is_relevant("The   ALPHA\n beta", [["alpha beta"]])
    assert not is_relevant("gamma", FACT_ALPHA)


def test_alternative_phrasings_count_as_one_fact():
    fact = [["foo", "bar"]]
    assert is_relevant("only bar here", fact) and is_relevant("only foo here", fact)
    assert facts_covered(["foo and bar"], fact) == 1               # still ONE fact


def test_recall_for_one_fact_is_found_or_not_found():
    assert [recall_at_k(TEXTS, FACT_ALPHA, k) for k in (1, 2, 5)] == [0.0, 1.0, 1.0]


def test_recall_for_several_facts_is_a_fraction():
    facts = [["alpha"], ["beta"]]
    texts = ["alpha", "x", "beta"]
    assert [recall_at_k(texts, facts, k) for k in (1, 2, 3)] == [0.5, 0.5, 1.0]


def test_precision_divides_by_k_not_by_the_number_found():
    assert precision_at_k(TEXTS, FACT_ALPHA, 1) == 0.0
    assert precision_at_k(TEXTS, FACT_ALPHA, 2) == 0.5             # ranks 2 -> 1 of 2
    assert precision_at_k(TEXTS, FACT_ALPHA, 5) == 0.4             # ranks 2 and 4 -> 2 of 5
    assert precision_at_k(["alpha"], FACT_ALPHA, 10) == 0.1        # only 1 result exists, K is still 10


def test_reciprocal_rank_is_one_over_the_rank_of_the_first_hit():
    assert reciprocal_rank(TEXTS, FACT_ALPHA) == 0.5               # first hit at rank 2
    assert reciprocal_rank(["alpha"], FACT_ALPHA) == 1.0
    assert reciprocal_rank(["x", "y", "alpha"], FACT_ALPHA) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], FACT_ALPHA) == 0.0


def test_nothing_found_scores_zero_everywhere():
    assert recall_at_k(["x", "y"], FACT_ALPHA, 5) == 0.0
    assert precision_at_k(["x", "y"], FACT_ALPHA, 5) == 0.0


# =============================================================================
# B) The question file
# =============================================================================
VALID = [
    {"id": "D1", "type": "direct", "question": "Q?", "answer_available": True,
     "expected_section": "S", "evidence": ["alpha"]},
    {"id": "U1", "type": "unsupported", "question": "Z?", "answer_available": False,
     "expected_section": None, "absent_terms": ["zzz"]},
]


def write_questions(tmp_path, data) -> Path:
    path = tmp_path / "q.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def mutated(index, **changes):
    data = json.loads(json.dumps(VALID))
    for key, value in changes.items():
        if value is KeyError:
            data[index].pop(key, None)
        else:
            data[index][key] = value
    return data


def test_valid_file_loads(tmp_path):
    qs = load_questions(write_questions(tmp_path, VALID))
    assert [q.id for q in qs] == ["D1", "U1"]
    assert qs[0].facts == [["alpha"]] and qs[0].expected_sections == ["S"]
    assert qs[1].absent_terms == ["zzz"] and not qs[1].answer_available


def test_evidence_may_be_text_or_list_of_alternatives(tmp_path):
    data = mutated(0, evidence=["one", ["two", "deux"]])
    assert load_questions(write_questions(tmp_path, data))[0].facts == [["one"], ["two", "deux"]]


@pytest.mark.parametrize("data, message", [
    (mutated(0, id=""), "missing 'id'"),
    (mutated(0, type="weird"), "'type' must be one of"),
    (mutated(0, question="  "), "missing 'question'"),
    (mutated(0, answer_available="yes"), "true or false"),
    (mutated(0, answer_available=False), "does not match"),
    (mutated(1, answer_available=True), "does not match"),
    (mutated(0, evidence=KeyError), "needs a non-empty 'evidence'"),
    (mutated(0, evidence=[]), "needs a non-empty 'evidence'"),
    (mutated(0, evidence=[[]]), "evidence item"),
    (mutated(0, evidence=[42]), "evidence item"),
    (mutated(0, evidence=[["ok", ""]]), "evidence item"),
    (mutated(0, expected_section=KeyError), "expected_section"),
    (mutated(0, expected_section=[]), "expected_section"),
    (mutated(0, type="multi_section"), "at least 2 evidence"),
    (mutated(1, absent_terms=KeyError), "absent_terms"),
    (mutated(1, absent_terms=[]), "absent_terms"),
    (mutated(1, absent_terms=[3]), "absent_terms"),
    ([VALID[0], {**VALID[0]}], "duplicate id"),
    (["not an object"], "must be an object"),
    ([], "non-empty list"),
    ({"a": 1}, "non-empty list"),
])
def test_malformed_question_files_are_rejected_with_a_reason(tmp_path, data, message):
    with pytest.raises(EvaluationDataError, match=message):
        load_questions(write_questions(tmp_path, data))


def test_missing_and_broken_files(tmp_path):
    with pytest.raises(EvaluationDataError, match="not found"):
        load_questions(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(EvaluationDataError, match="Could not read"):
        load_questions(bad)


# ---- the REAL question file against the REAL document ------------------------------------------------
@pytest.fixture(scope="module")
def questions():
    return load_questions(QUESTIONS_PATH)


def test_the_real_dataset_covers_all_five_question_types(questions):
    counts = {}
    for q in questions:
        counts[q.type] = counts.get(q.type, 0) + 1
    assert counts["direct"] >= 10 and counts["paraphrase"] >= 10 and counts["multi_section"] >= 3
    assert counts["unsupported"] >= 5 and counts["near_miss"] >= 5
    assert len(questions) == 49


def test_the_real_dataset_matches_the_real_document(questions, chunks):
    """If someone edits company.html or questions.json and they disagree, this fails and says where."""
    assert check_against_chunks(questions, chunks) == []


def test_the_documents_own_unsupported_example_is_in_the_dataset(questions):
    assert any(q.question == "What is the company's stock price?" and not q.answer_available for q in questions)


def chunk(section, text):
    return SimpleNamespace(section=section, text=text)


def q_answerable(facts, sections=("S",), qid="X1", text="Q?"):
    return EvalQuestion(id=qid, type="direct", question=text, answer_available=True,
                        expected_sections=list(sections), facts=facts)


def test_check_finds_evidence_that_is_not_in_the_document():
    problems = check_against_chunks([q_answerable([["missing text"]])], [chunk("S", "something else")])
    assert any("not in any chunk" in p for p in problems)


def test_check_finds_evidence_in_the_wrong_section():
    problems = check_against_chunks([q_answerable([["alpha"]], sections=("S",))], [chunk("Other", "alpha")])
    assert any("not in the expected section" in p for p in problems)


def test_check_finds_evidence_that_is_too_generic():
    many = [chunk("S", "the common phrase") for _ in range(5)]
    problems = check_against_chunks([q_answerable([["common phrase"]])], many)
    assert any("too generic" in p for p in problems)


def test_check_finds_an_unanswerable_question_that_is_actually_answerable():
    q = EvalQuestion(id="U9", type="unsupported", question="Revenue?", answer_available=False, absent_terms=["revenue"])
    problems = check_against_chunks([q], [chunk("S", "Our Revenue grew")])
    assert any("IS in the document" in p for p in problems)


def test_check_finds_duplicate_question_text():
    a, b = q_answerable([["alpha"]], qid="A"), q_answerable([["alpha"]], qid="B")
    assert any("twice" in p for p in check_against_chunks([a, b], [chunk("S", "alpha")]))


# =============================================================================
# C) evaluate() with a fake retriever (hand-computed expectations)
# =============================================================================
def hit(cid, text, similarity, position):
    return SearchResult(chunk_id=cid, text=text, section="S", subsection=None, source="x.html",
                        document_version="1.0", position=position, similarity=similarity)


class FakeRetriever:
    def __init__(self, table):
        self.table, self.calls = table, []

    def retrieve(self, question, top_k=None):
        self.calls.append((question, top_k))
        hits = [hit(*spec, position=i) for i, spec in enumerate(self.table[question])]
        return RetrievalResponse(question, hits[:top_k], top_k, "fake-model", "1.0", 0.0)


def build_questions(near_miss_score=0.45):
    return [
        EvalQuestion("D1", "direct", "d1?", True, ["S"], [["alpha"]]),
        EvalQuestion("P1", "paraphrase", "p1?", True, ["S"], [["beta", "bravo"]]),
        EvalQuestion("M1", "multi_section", "m1?", True, ["S"], [["gamma"], ["delta"]]),
        EvalQuestion("D2", "direct", "d2?", True, ["S"], [["omega"]]),
        EvalQuestion("U1", "unsupported", "u1?", False, absent_terms=["zzz"]),
        EvalQuestion("N1", "near_miss", "n1?", False, absent_terms=["yyy"]),
    ]


def build_table(near_miss_score=0.45):
    return {
        "d1?": [("c1", "nothing here", 0.90), ("c2", "contains ALPHA text", 0.80), ("c3", "zzz", 0.50)],
        "p1?": [("c4", "BRAVO!", 0.70), ("c5", "x", 0.40)],
        "m1?": [("c6", "gamma", 0.60), ("c7", "x", 0.55), ("c8", "delta", 0.52)],
        "d2?": [("c9", "x", 0.30)],
        "u1?": [("c10", "x", 0.20)],
        "n1?": [("c11", "x", near_miss_score)],
    }


@pytest.fixture
def run():
    return evaluate(FakeRetriever(build_table()), build_questions(), ks=(1, 3, 5))


def result(run, qid):
    return next(r for r in run.results if r.question.id == qid)


def test_each_question_is_retrieved_exactly_once_with_a_generous_k():
    fake = FakeRetriever(build_table())
    evaluate(fake, build_questions(), ks=(1, 3, 5))
    assert len(fake.calls) == 6 and all(k == 10 for _, k in fake.calls)
    fake2 = FakeRetriever(build_table())
    evaluate(fake2, build_questions(), ks=(1, 3, 15))
    assert all(k == 15 for _, k in fake2.calls)


def test_per_question_numbers(run):
    d1 = result(run, "D1")
    assert (d1.recall[1], d1.recall[3]) == (0.0, 1.0)
    assert (d1.precision[1], d1.precision[3], d1.precision[5]) == (0.0, pytest.approx(1 / 3), 0.2)
    assert (d1.reciprocal_rank, d1.first_relevant_rank) == (0.5, 2)
    assert [i.relevant for i in d1.retrieved] == [False, True, False]

    p1 = result(run, "P1")                                         # alternative phrasing "bravo" counts
    assert (p1.recall[1], p1.reciprocal_rank, p1.first_relevant_rank) == (1.0, 1.0, 1)

    m1 = result(run, "M1")
    assert (m1.recall[1], m1.recall[3], m1.precision[3]) == (0.5, 1.0, pytest.approx(2 / 3))

    d2 = result(run, "D2")
    assert (d2.recall[5], d2.reciprocal_rank, d2.first_relevant_rank) == (0.0, 0.0, None)


def test_overall_means(run):
    overall = run.overall()
    assert overall["count"] == 4
    assert overall["recall"][1] == pytest.approx(0.375)            # (0 + 1 + 0.5 + 0) / 4
    assert overall["recall"][3] == pytest.approx(0.75)             # (1 + 1 + 1 + 0) / 4
    assert overall["recall"][5] == pytest.approx(0.75)
    assert overall["precision"][1] == pytest.approx(0.5)           # (0 + 1 + 1 + 0) / 4
    assert overall["precision"][3] == pytest.approx(1 / 3)         # (1/3 + 1/3 + 2/3 + 0) / 4
    assert overall["precision"][5] == pytest.approx(0.2)           # (.2 + .2 + .4 + 0) / 4
    assert overall["mrr"] == pytest.approx(0.625)                  # (0.5 + 1 + 1 + 0) / 4


def test_means_by_question_type(run):
    by_type = run.by_type()
    assert set(by_type) == {"direct", "paraphrase", "multi_section"}      # unanswerable types are not scored
    assert by_type["direct"]["recall"][3] == pytest.approx(0.5) and by_type["direct"]["mrr"] == pytest.approx(0.25)
    assert by_type["paraphrase"]["recall"][1] == 1.0
    assert by_type["multi_section"]["recall"][1] == 0.5 and by_type["multi_section"]["recall"][3] == 1.0


def test_failures_depend_on_k(run):
    assert [r.question.id for r in run.failures(5)] == ["D2"]
    assert sorted(r.question.id for r in run.failures(1)) == ["D1", "D2", "M1"]


def test_unanswerable_questions_are_not_scored_but_keep_their_scores(run):
    u1 = result(run, "U1")
    assert u1.recall[3] is None and u1.reciprocal_rank is None and u1.first_relevant_rank is None
    assert [i.relevant for i in u1.retrieved] == [None]
    assert u1.top1_similarity == 0.20


def test_score_summary(run):
    summary = run.score_summary()
    assert summary["answerable"] == {"n": 4, "min": 0.30, "median": pytest.approx(0.65), "max": 0.90}
    assert summary["unsupported"]["max"] == 0.20 and summary["near_miss"]["max"] == 0.45


def test_threshold_sweep(run):
    rows = {row["threshold"]: row for row in run.threshold_sweep([0.3, 0.5, 0.8])}
    assert rows[0.5] == {"threshold": 0.5, "answerable_kept": 0.75, "unsupported_refused": 1.0, "near_miss_refused": 1.0}
    assert rows[0.3]["answerable_kept"] == 1.0 and rows[0.3]["near_miss_refused"] == 0.0     # 0.45 >= 0.3
    assert rows[0.8]["answerable_kept"] == 0.25                    # only 0.90 survives


def test_default_sweep_runs_from_0_10_to_0_70(run):
    thresholds = [row["threshold"] for row in run.threshold_sweep()]
    assert thresholds[0] == 0.10 and thresholds[-1] == 0.70 and len(thresholds) == 13


def test_separation_detects_a_clean_gap_and_an_overlap(run):
    sep = run.separation()
    assert sep["clean_gap_unsupported"] is True        # lowest answerable 0.30 > highest unsupported 0.20
    assert sep["clean_gap_near_miss"] is False         # near-miss 0.45 is ABOVE the lowest answerable 0.30
    separable = evaluate(FakeRetriever(build_table(near_miss_score=0.25)), build_questions(), ks=(1, 3, 5))
    assert separable.separation()["clean_gap_near_miss"] is True      # 0.25 < 0.30: a threshold would work now


@pytest.mark.parametrize("bad", [(), (0,), (-1, 3)])
def test_bad_ks_are_rejected(bad):
    with pytest.raises(ValueError):
        evaluate(FakeRetriever(build_table()), build_questions(), ks=bad)


def test_results_serialise_to_json(run):
    data = json.loads(json.dumps(run.to_dict()))
    assert data["overall"]["recall"]["3"] == 0.75 and data["overall"]["mrr"] == 0.625
    questions_by_id = {q["id"]: q for q in data["questions"]}
    assert questions_by_id["D1"]["retrieved"][1]["relevant"] is True
    assert questions_by_id["U1"]["retrieved"][0]["relevant"] is None
    assert questions_by_id["U1"]["recall"]["3"] is None
    assert "threshold_sweep" in data and "separation" in data


# =============================================================================
# D) The report, and a full run on the real document
# =============================================================================
def test_report_contents(run):
    run.meta = {"embedding_model": "fake-model", "document_version": "1.0"}
    report = render_markdown(run)
    assert report.startswith("# Retrieval Evaluation Report")
    for heading in ("## Setup", "## Results for answerable questions", "## Threshold sweep",
                    "## Questions with no answer in the document"):
        assert heading in report
    assert "**D2**" in report and "no relevant chunk in the top 10" in report      # the failing question is listed
    assert "OVERLAP" in report                                                    # near-miss scores overlap
    assert "separates these two groups" in report                                 # unsupported ones do not
    assert "hashing" not in report


def test_report_warns_when_the_stand_in_embedder_was_used(run):
    run.meta = {"embedding_model": "hashing-bag-of-words"}
    assert "NOT the quality of your real system" in render_markdown(run)


def test_report_says_so_when_nothing_failed():
    perfect = evaluate(FakeRetriever(build_table()), build_questions()[:1], ks=(1, 3, 5))
    assert "None. Every answerable question had all its evidence" in render_markdown(perfect)


@pytest.fixture(scope="module")
def full_run(questions):
    from backend.rag.embeddings import HashingEmbeddingService
    retriever, temp_dir = build_temporary_index(HashingEmbeddingService(), PROJECT_ROOT / "data" / "raw" / "company.html",
                                                150, 20)
    try:
        yield retriever, evaluate(retriever, questions, ks=(1, 3, 5))
    finally:
        temp_dir.cleanup()


def test_temporary_index_does_not_touch_the_real_index(full_run):
    retriever, _ = full_run
    assert str(retriever.vector_store.path).startswith(tempfile.gettempdir())
    assert retriever.vector_store.count() == 56


def test_full_run_scores_every_question(full_run, questions):
    _, run = full_run
    assert len(run.results) == len(questions) == 49
    assert run.overall()["count"] == 36


def test_full_run_basic_quality_floor_with_the_stand_in(full_run):
    """The stand-in matches shared words, so direct questions must work well. A drop means the pipeline broke."""
    _, run = full_run
    by_type = run.by_type()
    assert by_type["direct"]["recall"][5] >= 0.8
    assert run.overall()["mrr"] >= 0.6


def test_paraphrases_are_harder_than_direct_questions_for_a_word_matcher(full_run):
    """This gap is the reason the real embedding model exists."""
    _, run = full_run
    by_type = run.by_type()
    assert by_type["paraphrase"]["recall"][5] < by_type["direct"]["recall"][5]


def test_recall_never_decreases_as_k_grows(full_run):
    _, run = full_run
    for r in run.of_type("direct", "paraphrase", "multi_section"):
        assert r.recall[1] <= r.recall[3] <= r.recall[5]
        if r.recall[1] == 1.0:
            assert r.reciprocal_rank == 1.0


def test_a_smaller_chunk_limit_gives_more_chunks(doc_copy, embedder):
    big, big_dir = build_temporary_index(embedder, doc_copy, 150, 20)
    small, small_dir = build_temporary_index(embedder, doc_copy, 60, 10)
    try:
        assert small.vector_store.count() > big.vector_store.count()
    finally:
        big_dir.cleanup()
        small_dir.cleanup()


# ---- the command-line tool -------------------------------------------------------------------------------
def run_script(args, tmp_path):
    env = dict(os.environ, CHROMA_PATH=str(tmp_path / "db"), EMBEDDING_PROVIDER="hashing")
    return subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_retrieval.py"), *args],
                          cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300)


def test_script_runs_on_a_temporary_index(tmp_path):
    result = run_script(["--temp-index", "--no-save"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "ALL ANSWERABLE" in result.stdout and "dataset checked against the document: OK" in result.stdout


def test_script_compares_chunk_sizes(tmp_path):
    result = run_script(["--sweep-chunk-sizes", "60,150"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "CHUNK SIZE COMPARISON" in result.stdout
    rows = [l for l in result.stdout.splitlines() if l.strip().startswith(("60 ", "150 "))]
    assert len(rows) == 2


def test_script_refuses_a_dataset_that_disagrees_with_the_document(tmp_path):
    bad = write_questions(tmp_path, [
        {"id": "X1", "type": "direct", "question": "Q?", "answer_available": True,
         "expected_section": "Leave Policy", "evidence": ["this text is nowhere"]}])
    result = run_script(["--questions", str(bad), "--no-save"], tmp_path)
    assert result.returncode == 2
    assert "does not match the document" in result.stdout


def test_script_on_an_empty_real_index_explains_what_to_do(tmp_path):
    result = run_script(["--no-save"], tmp_path)
    assert result.returncode != 0
    assert "index is empty" in result.stdout + result.stderr


def test_script_saves_the_report_files_when_run_normally(tmp_path, monkeypatch, doc_copy, embedder, store):
    """Run main() in-process with the output folder redirected, so your real report is never overwritten."""
    from backend.config import Config
    spec = importlib.util.spec_from_file_location("evaluate_retrieval_script",
                                                  PROJECT_ROOT / "scripts" / "evaluate_retrieval.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    ingest_document(PROJECT_ROOT / "data" / "raw" / "company.html", embedder, store)
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setattr(script, "EVAL_DIR", tmp_path / "out")
    monkeypatch.setattr(script, "get_vector_store", lambda: store)
    monkeypatch.setattr(sys, "argv", ["evaluate_retrieval.py"])
    script.main()

    results = json.loads((tmp_path / "out" / "retrieval_results.json").read_text(encoding="utf-8"))
    assert len(results["questions"]) == 49 and results["meta"]["embedding_model"] == "hashing-bag-of-words"
    report = (tmp_path / "out" / "evaluation_report.md").read_text(encoding="utf-8")
    assert report.startswith("# Retrieval Evaluation Report") and "NOT the quality of your real system" in report
