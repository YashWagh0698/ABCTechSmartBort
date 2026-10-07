"""Step 7 helper: build the search index WITHOUT starting the Flask server.

Does exactly what POST /api/ingest (or /api/reindex) does, but from the terminal:

    python scripts/ingest_company.py                  build the index (does nothing if up to date)
    python scripts/ingest_company.py --reindex        delete the old index and rebuild it
    python scripts/ingest_company.py --provider hashing    use the offline stand-in embedder

If you change the embedding model or edit company.html, use --reindex.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                                    # noqa: E402
from backend.rag.embeddings import EmbeddingError, get_embedding_service  # noqa: E402
from backend.rag.html_parser import ParseError                       # noqa: E402
from backend.rag.ingestion import (IndexBusyError, IndexConflictError,     # noqa: E402
                                   ingest_document, reindex_document)
from backend.rag.vector_store import VectorStoreError, get_vector_store   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the search index from company.html")
    ap.add_argument("--reindex", action="store_true", help="delete the old index and rebuild")
    ap.add_argument("--provider", help="sentence-transformers or hashing (default: from .env)")
    ap.add_argument("--model", help="embedding model name (default: from .env)")
    args = ap.parse_args()

    if args.provider:
        Config.EMBEDDING_PROVIDER = args.provider.strip().lower()
    if args.model:
        Config.EMBEDDING_MODEL = args.model.strip()

    line = "=" * 78
    print(line)
    print("REINDEXING" if args.reindex else "INGESTING", "-", Config.COMPANY_DOCUMENT_PATH.name)
    print(line)
    print(f"  embedding provider : {Config.EMBEDDING_PROVIDER}")
    print(f"  chunk settings     : max {Config.CHUNK_MAX_WORDS} words, overlap {Config.CHUNK_OVERLAP_WORDS}")
    print(f"  vector database    : {Config.CHROMA_PATH}  (collection '{Config.CHROMA_COLLECTION}')")

    try:
        embedder = get_embedding_service()
        store = get_vector_store()
        action = reindex_document if args.reindex else ingest_document
        result = action(Config.COMPANY_DOCUMENT_PATH, embedder, store)
    except IndexConflictError as exc:
        print(f"\n  CONFLICT: {exc}")
        print("  Run again with --reindex.")
        sys.exit(2)
    except (IndexBusyError, ParseError, EmbeddingError, VectorStoreError, ValueError) as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    print(f"\n  status             : {result.status}")
    print(f"  document / version : {result.source} / {result.document_version}")
    print(f"  sections parsed    : {result.sections}")
    print(f"  chunks created     : {result.chunks_created}")
    print(f"  chunks in index    : {result.chunks_in_index}")
    print(f"  embedding model    : {result.embedding_model} ({result.dimension} numbers per chunk)")
    print(f"  fingerprint        : {result.fingerprint}")
    print(f"  time               : {result.seconds} s")
    if result.replaced_previous:
        print(f"  replaced the previous index (version {result.previous_version})")
    if result.status == "already_indexed":
        print("  Nothing to do: the index already matches the document.")
    print(line)


if __name__ == "__main__":
    main()
