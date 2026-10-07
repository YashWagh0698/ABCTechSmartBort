"""Step 9 helper: measure how well the retriever finds the right chunks.

    python scripts/evaluate_retrieval.py                     evaluate your real index (writes the report files)
    python scripts/evaluate_retrieval.py --provider hashing  same, with the offline stand-in embedder
    python scripts/evaluate_retrieval.py --ks 1,3,5,10       look at other values of K
    python scripts/evaluate_retrieval.py --temp-index --max-words 100
                                                             try other chunk settings in a TEMPORARY index
    python scripts/evaluate_retrieval.py --sweep-chunk-sizes 60,100,150,200,300
                                                             compare several chunk sizes (writes nothing)

Output files (normal mode):
    evaluation/retrieval_results.json    every question, what was retrieved, scores
    evaluation/evaluation_report.md      the readable report

Your real index is only READ. Experiments use a temporary index that is deleted afterwards.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                                              # noqa: E402
from backend.rag.chunker import chunk_document                                 # noqa: E402
from backend.rag.embeddings import EmbeddingError, get_embedding_service       # noqa: E402
from backend.rag.evaluation import (EvaluationDataError, TYPE_LABELS, build_temporary_index,  # noqa: E402
                                    check_against_chunks, evaluate, load_questions, render_markdown)
from backend.rag.html_parser import ParseError, parse_html_file                # noqa: E402
from backend.rag.ingestion import index_is_current                             # noqa: E402
from backend.rag.retriever import RetrievalError, Retriever                    # noqa: E402
from backend.rag.vector_store import VectorStoreError, get_vector_store        # noqa: E402

EVAL_DIR = PROJECT_ROOT / "evaluation"                       # where the report files are WRITTEN
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"   # the questions that are READ


def parse_int_list(text: str):
    try:
        values = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"Not a list of whole numbers: {text!r}")
    if not values or min(values) < 1:
        raise SystemExit("Numbers must be 1 or larger")
    return values


def print_summary(run) -> None:
    ks = list(run.ks)
    print("\n  RESULTS (answerable questions)")
    print(f"  {'type':<24}{'N':>3} " + " ".join(f"{'R@' + str(k):>6}" for k in ks) + "   "
          + " ".join(f"{'P@' + str(k):>6}" for k in ks) + f" {'MRR':>6}")
    rows = [(TYPE_LABELS[t], m) for t, m in run.by_type().items()] + [("ALL ANSWERABLE", run.overall())]
    for label, m in rows:
        print(f"  {label:<24}{m['count']:>3} " + " ".join(f"{m['recall'][k]:>6.2f}" for k in ks) + "   "
              + " ".join(f"{m['precision'][k]:>6.2f}" for k in ks) + f" {m['mrr']:>6.2f}")

    k = max(ks)
    failures = run.failures(k)
    print(f"\n  NOT fully found in the top {k}: {len(failures)} question(s)")
    for r in failures:
        top = r.retrieved[0]
        print(f"    {r.question.id}  {r.question.question}")
        print(f"          closest chunk: {top.similarity:+.3f}  {top.heading}")

    summary = run.score_summary()
    print("\n  BEST-SIMILARITY SCORE BY GROUP (what a refusal threshold would look at)")
    print(f"  {'group':<24}{'N':>3} {'lowest':>8} {'median':>8} {'highest':>8}")
    for name, label in (("answerable", "answerable"), ("unsupported", "unsupported"),
                        ("near_miss", "similar but incorrect")):
        s = summary[name]
        if s:
            print(f"  {label:<24}{s['n']:>3} {s['min']:>8.3f} {s['median']:>8.3f} {s['max']:>8.3f}")
    sep = run.separation()
    print("  -> unsupported questions separable from answerable ones by a score:",
          "YES" if sep["clean_gap_unsupported"] else "NO (scores overlap)")
    print("  -> similar-but-incorrect questions separable by a score:",
          "YES" if sep["clean_gap_near_miss"] else "NO (scores overlap)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality")
    ap.add_argument("--questions", default=str(QUESTIONS_PATH))
    ap.add_argument("--ks", default="1,3,5", help="comma-separated values of K (default 1,3,5)")
    ap.add_argument("--temp-index", action="store_true", help="evaluate a temporary index instead of the real one")
    ap.add_argument("--max-words", type=int, default=Config.CHUNK_MAX_WORDS, help="chunk size for --temp-index")
    ap.add_argument("--overlap", type=int, default=Config.CHUNK_OVERLAP_WORDS, help="chunk overlap for --temp-index")
    ap.add_argument("--sweep-chunk-sizes", metavar="LIST", help="compare chunk sizes, e.g. 60,100,150,200,300")
    ap.add_argument("--no-save", action="store_true", help="do not write the report files")
    ap.add_argument("--provider", help="sentence-transformers or hashing (default: from .env)")
    ap.add_argument("--model", help="embedding model name (default: from .env)")
    args = ap.parse_args()

    if args.provider:
        Config.EMBEDDING_PROVIDER = args.provider.strip().lower()
    if args.model:
        Config.EMBEDDING_MODEL = args.model.strip()
    ks = parse_int_list(args.ks)

    line = "=" * 92
    print(line)
    print("RETRIEVAL EVALUATION")
    print(line)

    try:
        questions = load_questions(args.questions)
        parsed = parse_html_file(Config.COMPANY_DOCUMENT_PATH)
        problems = check_against_chunks(questions, chunk_document(parsed))
        embedder = get_embedding_service()
    except (EvaluationDataError, ParseError, EmbeddingError, ValueError) as exc:
        raise SystemExit(f"\nERROR: {exc}")
    if problems:
        print("\nThe question file does not match the document, so the results would be misleading:")
        for p in problems:
            print("  -", p)
        raise SystemExit(2)
    print(f"\n  questions : {len(questions)}   (dataset checked against the document: OK)")
    print(f"  model     : {embedder.model_name}")

    try:
        # ---------------------------------------------------------------- sweep mode
        if args.sweep_chunk_sizes:
            sizes = parse_int_list(args.sweep_chunk_sizes)
            print(f"\n  CHUNK SIZE COMPARISON (temporary indexes, K = {ks})")
            print(f"  {'max words':>9} {'chunks':>7}  " + " ".join(f"{'Recall@' + str(k):>9}" for k in ks)
                  + f" {'MRR':>7}  {'not found@' + str(max(ks)):>14}")
            for size in sizes:
                overlap = min(args.overlap, size - 1)
                retriever, temp_dir = build_temporary_index(embedder, Config.COMPANY_DOCUMENT_PATH, size, overlap)
                try:
                    run = evaluate(retriever, questions, ks)
                    overall = run.overall()
                    print(f"  {size:>9} {retriever.vector_store.count():>7}  "
                          + " ".join(f"{overall['recall'][k]:>9.2f}" for k in ks)
                          + f" {overall['mrr']:>7.2f}  {len(run.failures(max(ks))):>14}")
                finally:
                    temp_dir.cleanup()
            print("\n  Pick the size with the best Recall/MRR. If several tie, prefer the one with fewer problems "
                  "at low K.\n  Then set CHUNK_MAX_WORDS in .env and run:  python scripts/ingest_company.py --reindex")
            print(line)
            return

        # --------------------------------------------------------- one evaluation run
        if args.temp_index:
            retriever, temp_dir = build_temporary_index(embedder, Config.COMPANY_DOCUMENT_PATH,
                                                        args.max_words, args.overlap)
            mode = f"temporary index (max {args.max_words} words, overlap {args.overlap})"
            max_words, overlap = args.max_words, args.overlap
        else:
            temp_dir = None
            store = get_vector_store()
            retriever = Retriever(embedder, store)
            info = store.info()
            mode = f"real index ({Config.CHROMA_PATH})"
            max_words, overlap = Config.CHUNK_MAX_WORDS, Config.CHUNK_OVERLAP_WORDS
            if not info.is_empty and index_is_current(Config.COMPANY_DOCUMENT_PATH, info) is False:
                print("\n  WARNING: the index is OUT OF DATE (company.html or the chunk settings changed). "
                      "Results describe the old index.\n           Run: python scripts/ingest_company.py --reindex")

        try:
            meta_info = retriever.vector_store.info()
            meta = {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "embedding_model": embedder.model_name,
                "document_version": meta_info.document_version,
                "chunk_max_words": max_words, "chunk_overlap_words": overlap,
                "chunks_in_index": meta_info.count, "index_mode": mode,
            }
            run = evaluate(retriever, questions, ks, meta=meta)
        finally:
            if temp_dir:
                temp_dir.cleanup()
    except RetrievalError as exc:
        raise SystemExit(f"\nERROR: {exc}")
    except (VectorStoreError, ValueError) as exc:
        raise SystemExit(f"\nERROR: {exc}")

    print(f"  index     : {mode}, {meta['chunks_in_index']} chunks")
    print_summary(run)

    if args.no_save or args.temp_index:
        print("\n  (nothing written" + (": experiments with a temporary index are not saved)" if args.temp_index else ")"))
    else:
        EVAL_DIR.mkdir(exist_ok=True)
        (EVAL_DIR / "retrieval_results.json").write_text(
            json.dumps(run.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        (EVAL_DIR / "evaluation_report.md").write_text(render_markdown(run), encoding="utf-8")
        print("\n  Saved: evaluation/retrieval_results.json")
        print("         evaluation/evaluation_report.md   <- open this one")
    print(line)


if __name__ == "__main__":
    main()
