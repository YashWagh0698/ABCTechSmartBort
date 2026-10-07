"""Step 4 helper: build chunks from company.html and look at them.

Run from the project root:

    python scripts/build_chunks.py                    build chunks, run quality checks, save chunks.json
    python scripts/build_chunks.py --show "leave"     also print the full text of matching chunks
    python scripts/build_chunks.py --compare          compare several chunk sizes (saves nothing)
    python scripts/build_chunks.py --max-words 100 --overlap 10     try other settings

Output file: data/processed/chunks.json  (open it in your editor and read it!)
"""
import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                                   # noqa: E402
from backend.rag.chunker import chunk_document, find_problems, save_chunks  # noqa: E402
from backend.rag.html_parser import ParseError, parse_html_file     # noqa: E402


def body_words(chunk) -> int:
    """Words in the chunk WITHOUT the heading line (this is what the limit applies to)."""
    return len(chunk.text.split("\n", 1)[1].split())


def print_overview(chunks, max_words, overlap) -> None:
    sizes = [c.word_count for c in chunks]
    split_sections = sum(1 for c in chunks if c.part == 1 and c.parts > 1)
    print(f"\n[1] SETTINGS   max body words = {max_words}   overlap = {overlap}")
    print(f"\n[2] RESULT     {len(chunks)} chunks   "
          f"words per chunk: min {min(sizes)} / median {int(statistics.median(sizes))} / max {max(sizes)}")
    print(f"               sections that had to be split into several chunks: {split_sections}")

    problems = find_problems(chunks, max_words)
    print("\n[3] QUALITY CHECK")
    if problems:
        for p in problems:
            print("    PROBLEM:", p)
    else:
        print("    PASS: ids unique, no empty chunks, none over the limit, metadata complete")

    print("\n[4] CHUNKS")
    print(f"    {'pos':>3}  {'chunk_id':<26} {'heading':<58} {'words':>5}  part")
    for c in chunks:
        heading = c.text.split("\n", 1)[0]
        part = f"{c.part}/{c.parts}" if c.parts > 1 else ""
        print(f"    {c.position:>3}  {c.chunk_id:<26} {heading[:58]:<58} {c.word_count:>5}  {part}")


def print_comparison(doc) -> None:
    print("\nCOMPARING CHUNK SIZES on the same document "
          f"({doc.total_words:,} words in {len(doc.sections)} sections)\n")
    print(f"    {'max words':>9}  {'chunks':>6}  {'min':>4} {'median':>6} {'max':>4}  "
          f"{'sections split':>14}  {'chunks <= 30 words':>18}")
    for size in (60, 100, 150, 200, 300, 800):
        overlap = min(20, size - 1)
        chunks = chunk_document(doc, max_words=size, overlap_words=overlap)
        sizes = [c.word_count for c in chunks]
        split = sum(1 for c in chunks if c.part == 1 and c.parts > 1)
        tiny = sum(1 for s in sizes if s <= 30)
        marker = "   <- default" if size == Config.CHUNK_MAX_WORDS else ""
        print(f"    {size:>9}  {len(chunks):>6}  {min(sizes):>4} {int(statistics.median(sizes)):>6} "
              f"{max(sizes):>4}  {split:>14}  {tiny:>18}{marker}")
    print("\n    Reading the table: a small limit makes MORE, SHORTER chunks (precise, but each has less")
    print("    context). A big limit makes FEWER, LONGER chunks (more context, but several topics per")
    print("    chunk and it may exceed the embedding model's input limit).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build chunks from company.html")
    ap.add_argument("--max-words", type=int, default=Config.CHUNK_MAX_WORDS)
    ap.add_argument("--overlap", type=int, default=Config.CHUNK_OVERLAP_WORDS)
    ap.add_argument("--show", metavar="TEXT", help="print full text of chunks whose heading contains TEXT")
    ap.add_argument("--compare", action="store_true", help="compare several chunk sizes")
    ap.add_argument("--no-save", action="store_true", help="do not write chunks.json")
    args = ap.parse_args()

    try:
        doc = parse_html_file(Config.COMPANY_DOCUMENT_PATH)
    except ParseError as exc:
        raise SystemExit(f"PARSE ERROR: {exc}")

    line = "=" * 100
    print(line)
    print(f"CHUNKING: {doc.source_file}  (version {doc.version})")
    print(line)

    if args.compare:
        print_comparison(doc)
        print(line)
        return

    try:
        chunks = chunk_document(doc, max_words=args.max_words, overlap_words=args.overlap)
    except ValueError as exc:
        raise SystemExit(f"BAD SETTINGS: {exc}")

    print_overview(chunks, args.max_words, args.overlap)

    if args.show:
        needle = args.show.lower()
        matches = [c for c in chunks if needle in c.text.split("\n", 1)[0].lower()]
        print(f"\n[5] FULL TEXT of chunks whose heading contains '{args.show}': {len(matches)} found")
        for c in matches:
            print("\n" + "-" * 100)
            print(f"{c.chunk_id}   (position {c.position}, {c.word_count} words, part {c.part}/{c.parts})")
            print("-" * 100)
            print(c.text)

    if not args.no_save:
        path = save_chunks(chunks, Config.CHUNKS_PATH)
        print(f"\nSaved: {path.relative_to(PROJECT_ROOT)}")
    print(line)


if __name__ == "__main__":
    main()
