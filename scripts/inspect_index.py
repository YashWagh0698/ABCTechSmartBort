"""Step 6/7 helper: look inside the vector database and verify that the records exist.

    python scripts/inspect_index.py                          overview of the index
    python scripts/inspect_index.py --show leave-policy-001  full record (text + metadata) for one chunk
    python scripts/inspect_index.py --query "How many paid leaves do I get?"
                                                             search the index (a preview of Step 8)
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                                    # noqa: E402
from backend.rag.embeddings import EmbeddingError, get_embedding_service  # noqa: E402
from backend.rag.ingestion import index_is_current                   # noqa: E402
from backend.rag.vector_store import VectorStoreError, get_vector_store   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the vector database")
    ap.add_argument("--show", metavar="CHUNK_ID", help="print one stored record in full")
    ap.add_argument("--query", metavar="TEXT", help="search the index with a question")
    ap.add_argument("--top-k", type=int, default=Config.TOP_K)
    ap.add_argument("--provider", help="sentence-transformers or hashing (only used with --query)")
    ap.add_argument("--model", help="embedding model name (only used with --query)")
    args = ap.parse_args()

    if args.provider:
        Config.EMBEDDING_PROVIDER = args.provider.strip().lower()
    if args.model:
        Config.EMBEDDING_MODEL = args.model.strip()

    line = "=" * 92
    store = get_vector_store()
    try:
        info = store.info()
        print(line)
        print(f"VECTOR DATABASE: {Config.CHROMA_PATH}   (collection '{Config.CHROMA_COLLECTION}')")
        print(line)
        if info.is_empty:
            print("\n  The index is EMPTY. Build it with:  python scripts/ingest_company.py")
            return

        print("\n[1] INDEX")
        print(f"    chunks stored   : {info.count}")
        print(f"    document        : {info.source}   version {info.document_version}")
        print(f"    embedding model : {info.embedding_model}")
        print(f"    indexed at      : {info.indexed_at}")
        print(f"    fingerprint     : {info.fingerprint}")
        current = index_is_current(Config.COMPANY_DOCUMENT_PATH, info)
        label = {True: "yes", False: "NO - company.html or chunk settings changed, run --reindex",
                 None: "unknown"}[current]
        print(f"    up to date      : {label}")

        chunks = store.list_chunks()
        per_section = OrderedDict()
        for c in chunks:
            per_section.setdefault(c.section, []).append(c)
        print(f"\n[2] RECORDS PER SECTION ({len(per_section)} sections)")
        for section, items in per_section.items():
            print(f"    {len(items):>2}  {section}")

        sample = store.get_chunk(args.show) if args.show else chunks[0]
        print(f"\n[3] {'RECORD ' + args.show if args.show else 'SAMPLE RECORD (the first chunk)'}")
        if sample is None:
            print(f"    no record with id '{args.show}'. Valid ids look like: {chunks[0].chunk_id}")
        else:
            print(f"    id       : {sample.chunk_id}")
            print("    metadata :")
            for key, value in sorted(sample.metadata.items()):
                print(f"        {key:<16} {value!r}")
            print("    document :")
            for text_line in sample.text.splitlines():
                print("        " + text_line)

        if args.query:
            print(f"\n[4] SEARCH: \"{args.query}\"")
            embedder = get_embedding_service()
            if embedder.model_name != info.embedding_model:
                raise SystemExit(f"    ERROR: the index was built with '{info.embedding_model}' but the configured "
                                 f"model is '{embedder.model_name}'. Their vectors cannot be compared.\n"
                                 "    Run:  python scripts/ingest_company.py --reindex")
            for rank, hit in enumerate(store.search(embedder.embed_query(args.query), args.top_k), 1):
                print(f"    {rank}. {hit.similarity:+.3f}  {hit.chunk_id:<28} {hit.heading}")
        print(line)
    except (VectorStoreError, EmbeddingError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
