"""Step 3 helper: run the HTML parser and look at what it produces.

Run from the project root:

    python scripts/parse_company.py                  overview of all sections
    python scripts/parse_company.py --show leave     print full text of sections matching "leave"
    python scripts/parse_company.py --no-save        do not write the JSON file

The result is also saved to data/processed/parsed_document.json so you can
open it in an editor and read it.
"""
import argparse
import json
import sys
from pathlib import Path

# Allow "from backend..." imports when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                     # noqa: E402
from backend.rag.html_parser import ParseError, parse_html_file  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "parsed_document.json"

# Text that exists ONLY in the noise parts of company.html.
# If any of it appears in the output, the parser failed to clean something.
NOISE_MARKERS = {
    "tracking script": "googletagmanager",
    "tracking function": "dataLayer",
    "css rules": "border-collapse",
    "menu link": "Employee Login",
    "table of contents": "On this page",
    "cookie banner": "Accept all cookies",
    "hidden draft note": "Sports club",
    "footer": "All rights reserved",
    "tracking pixel": "analytics.example.com",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the HTML parser on company.html")
    ap.add_argument("--path", default=str(Config.COMPANY_DOCUMENT_PATH))
    ap.add_argument("--show", metavar="TEXT",
                    help="print the full text of sections whose title contains TEXT")
    ap.add_argument("--no-save", action="store_true", help="do not write the JSON file")
    args = ap.parse_args()

    try:
        doc = parse_html_file(args.path)
    except ParseError as exc:
        raise SystemExit(f"PARSE ERROR: {exc}")

    line = "=" * 74
    print(line)
    print(f"PARSED: {doc.source_file}")
    print(line)
    print(f"  title          : {doc.title}")
    print(f"  version        : {doc.version}")
    print(f"  effective date : {doc.effective_date}")
    print(f"  owner          : {doc.owner}")
    print(f"  sections       : {len(doc.sections)}")
    print(f"  total words    : {doc.total_words:,}")

    print("\n[1] NOISE REMOVED")
    for label, count in sorted(doc.removed_noise.items()):
        print(f"    {label:<22} x{count}")

    print("\n[2] NOISE CHECK (is any noise text left in the output?)")
    all_text = "\n".join(s.text for s in doc.sections)
    leaked = 0
    for name, marker in NOISE_MARKERS.items():
        found = marker.lower() in all_text.lower()
        leaked += found
        print(f"    {'LEAKED' if found else 'clean ':<7} {name}")
    print("    -> " + ("FAIL: noise leaked into the output" if leaked else "PASS: all noise removed"))

    print("\n[3] SECTIONS")
    print(f"    {'#':>2}  {'Section':<30} {'Subsection':<42} {'Blk':>3} {'Words':>5}")
    for s in doc.sections:
        sub = s.subsection or "(intro)"
        print(f"    {s.order:>2}  {s.section[:30]:<30} {sub[:42]:<42} "
              f"{len(s.blocks):>3} {s.word_count:>5}")

    if args.show:
        needle = args.show.lower()
        matches = [s for s in doc.sections
                   if needle in s.section.lower() or needle in (s.subsection or "").lower()]
        print(f"\n[4] FULL TEXT of sections matching '{args.show}': {len(matches)} found")
        for s in matches:
            print("\n" + "-" * 74)
            print(f"#{s.order}  {s.section}  >  {s.subsection or '(intro)'}   [{s.section_id}]")
            print("-" * 74)
            for block in s.blocks:
                print(f"({block.kind})")
                print(block.text)
                print()

    if not args.no_save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"\nSaved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(line)


if __name__ == "__main__":
    main()
