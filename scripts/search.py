"""Step 8 helper: ask the retriever a question and inspect what it finds.

    python scripts/search.py "How many paid leaves do I get?"
    python scripts/search.py "What is the leave policy?" --top-k 8 --full
    python scripts/search.py                      interactive: type questions, press Enter on an empty line to stop

No LLM is involved. You see exactly what the retriever would hand to an LLM later:
the closest chunks, their similarity scores and their metadata.

Reading the scores: cosine similarity. 1.0 = same meaning, 0.0 = unrelated.
With the default model a good match is usually about 0.3 to 0.6 (NOT close to 1.0).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                                    # noqa: E402
from backend.rag.embeddings import EmbeddingError, get_embedding_service  # noqa: E402
from backend.rag.retriever import Retriever, RetrievalError          # noqa: E402
from backend.rag.vector_store import VectorStoreError, get_vector_store   # noqa: E402


def show(retriever: Retriever, question: str, top_k: int, full: bool) -> None:
    try:
        response = retriever.retrieve(question, top_k)
    except RetrievalError as exc:
        raise SystemExit(f"\nERROR: {exc}")
    except ValueError as exc:
        print(f"\n  Cannot search: {exc}")
        return

    print(f"\nQUESTION: {response.question}")
    print(f"model: {response.embedding_model}   document version: {response.document_version}   "
          f"time: {response.seconds * 1000:.0f} ms")
    for rank, hit in enumerate(response.results, 1):
        bar = "#" * max(0, int(round(hit.similarity * 40)))
        print(f"\n  {rank}. similarity {hit.similarity:+.3f}  {bar}")
        print(f"     {hit.heading}   [{hit.chunk_id}, part {hit.metadata.get('part')}/{hit.metadata.get('parts')}]")
        body = hit.text.split("\n", 1)[1] if "\n" in hit.text else hit.text
        lines = body.splitlines()
        for line in (lines if full else lines[:2]):
            print(f"       {line[:110]}")
        if not full and len(lines) > 2:
            print(f"       ... ({len(lines) - 2} more lines; use --full to see everything)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the index with a question (no LLM)")
    ap.add_argument("question", nargs="?", help="the question (omit for interactive mode)")
    ap.add_argument("--top-k", type=int, default=Config.TOP_K)
    ap.add_argument("--full", action="store_true", help="print the full text of every chunk")
    ap.add_argument("--provider", help="sentence-transformers or hashing (default: from .env)")
    ap.add_argument("--model", help="embedding model name (default: from .env)")
    args = ap.parse_args()

    if args.provider:
        Config.EMBEDDING_PROVIDER = args.provider.strip().lower()
    if args.model:
        Config.EMBEDDING_MODEL = args.model.strip()

    try:
        retriever = Retriever(get_embedding_service(), get_vector_store())
    except (EmbeddingError, VectorStoreError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")

    try:
        if args.question:
            show(retriever, args.question, args.top_k, args.full)
            return
        print("Type a question and press Enter. Press Enter on an empty line to stop.")
        while True:
            question = input("\nquestion> ").strip()
            if not question:
                break
            show(retriever, question, args.top_k, args.full)
    except VectorStoreError as exc:
        raise SystemExit(f"ERROR: {exc}")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
