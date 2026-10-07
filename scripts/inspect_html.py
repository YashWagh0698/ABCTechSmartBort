"""Step 2 helper: inspect the structure of the company HTML document.

This is a development tool, NOT part of the chatbot. It answers the questions
Phase 2 asks:  what useful elements exist?  what unwanted elements exist?
What is the heading hierarchy?  What does the structure look like?

It uses only Python's standard library (html.parser), so nothing extra is
installed. In Step 3 the real parser will use BeautifulSoup instead.

Run from the project root:
    python scripts/inspect_html.py
    python scripts/inspect_html.py --path data/raw/company.html
"""
import argparse
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = PROJECT_ROOT / "data" / "raw" / "company.html"

# Tags that have no closing tag (so they must never be pushed on our stack)
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}

# Tags that are ALWAYS noise for a knowledge base
NOISE_TAGS = {"script", "style", "noscript", "nav", "header", "footer"}

# If a class/id contains one of these words, the element is probably noise
NOISE_WORDS = ("cookie", "banner", "tracking", "analytics", "popup", "advert")

CONTENT_TAGS = ["h1", "h2", "h3", "h4", "p", "ul", "ol", "li", "table", "dl", "section"]


class Inspector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []                 # open tags: (tag, noise_label or None)
        self.noise_depth = 0            # >0 means we are inside a noisy element
        self.tag_counts = Counter()     # content tags found
        self.noise_found = Counter()    # noise label -> how many
        self.noise_samples = {}         # noise label -> first text sample
        self.comments = 0
        self.meta = {}
        self.headings = []              # (level, text)
        self._heading_buf = None        # (level, [text parts]) while inside h1-h3
        self.current_h2 = "(before first section)"
        self.tables = []                # dicts: section, headers, rows
        self._in_th = False
        self._th_buf = []
        self.max_list_depth = 0
        self._list_depth = 0
        self.useful_words = 0
        self.noise_words = 0

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _noise_label(tag, attrs):
        a = dict(attrs)
        if tag in NOISE_TAGS:
            return f"<{tag}>"
        if "hidden" in a or "display:none" in (a.get("style") or "").replace(" ", ""):
            return f"hidden element <{tag}>"
        marker = f'{a.get("class", "")} {a.get("id", "")}'.lower()
        for word in NOISE_WORDS:
            if word in marker:
                return f"{word} element <{tag}>"
        return None

    # ---- parser callbacks ---------------------------------------------
    def handle_comment(self, data):
        self.comments += 1

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = dict(attrs)
            if a.get("name"):
                self.meta[a["name"]] = a.get("content", "")
        if tag in VOID_TAGS:
            # still might be noise (e.g. tracking <img> inside <noscript>) - already covered
            return

        label = self._noise_label(tag, attrs)
        self.stack.append((tag, label))
        if label:
            self.noise_depth += 1
            self.noise_found[label] += 1
            return
        if self.noise_depth:
            return  # inside noise: don't count content tags

        if tag in CONTENT_TAGS:
            self.tag_counts[tag] += 1
        if tag in ("ul", "ol"):
            self._list_depth += 1
            self.max_list_depth = max(self.max_list_depth, self._list_depth)
        if tag in ("h1", "h2", "h3", "h4"):
            self._heading_buf = (int(tag[1]), [])
        if tag == "table":
            self.tables.append({"section": self.current_h2, "headers": [], "rows": 0})
        if tag == "tr" and self.tables:
            self.tables[-1]["rows"] += 1
        if tag == "th":
            self._in_th, self._th_buf = True, []

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        # pop until we find the matching open tag (tolerates sloppy HTML)
        while self.stack:
            open_tag, label = self.stack.pop()
            if label:
                self.noise_depth -= 1
            if open_tag == tag:
                break
        if self.noise_depth:
            return
        if tag in ("ul", "ol"):
            self._list_depth -= 1
        if tag in ("h1", "h2", "h3", "h4") and self._heading_buf:
            level, parts = self._heading_buf
            text = " ".join("".join(parts).split())
            self.headings.append((level, text))
            if level == 2:
                self.current_h2 = text
            self._heading_buf = None
        if tag == "th" and self._in_th and self.tables:
            self.tables[-1]["headers"].append(" ".join("".join(self._th_buf).split()))
            self._in_th = False

    def handle_data(self, data):
        words = len(data.split())
        if self.noise_depth:
            self.noise_words += words
            # remember a text sample for the first occurrence of each noise label
            for _tag, lab in reversed(self.stack):
                if lab and lab not in self.noise_samples and data.strip():
                    self.noise_samples[lab] = " ".join(data.split())[:60]
                    break
            return
        self.useful_words += words
        if self._heading_buf is not None:
            self._heading_buf[1].append(data)
        if self._in_th:
            self._th_buf.append(data)


def main():
    ap = argparse.ArgumentParser(description="Inspect the company HTML structure")
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    raw = path.read_text(encoding="utf-8")
    ins = Inspector()
    ins.feed(raw)

    line = "=" * 64
    print(line)
    print(f"INSPECTING: {path.name}   ({len(raw):,} characters)")
    print(line)

    print("\n[1] DOCUMENT METADATA (from <meta> tags)")
    for k, v in ins.meta.items():
        if k != "viewport":
            print(f"    {k}: {v}")

    print("\n[2] USEFUL CONTENT ELEMENTS (what the parser must keep)")
    for tag in CONTENT_TAGS:
        if ins.tag_counts[tag]:
            print(f"    <{tag}>".ljust(14), ins.tag_counts[tag])
    print(f"    deepest list nesting: {ins.max_list_depth} level(s)")
    print(f"    useful words: {ins.useful_words:,}")

    print("\n[3] UNWANTED ELEMENTS (what the parser must remove)")
    for label, n in ins.noise_found.most_common():
        sample = ins.noise_samples.get(label, "")
        print(f"    {label}".ljust(34), f"x{n}".ljust(5), f'e.g. "{sample}"' if sample else "")
    print(f"    HTML comments: {ins.comments}")
    print(f"    noise words: {ins.noise_words:,}")

    print("\n[4] HEADING OUTLINE (document structure)")
    for level, text in ins.headings:
        print("    " + "    " * (level - 1) + f"h{level}  {text}")

    print("\n[5] TABLES (need special handling)")
    for i, t in enumerate(ins.tables, 1):
        print(f"    table {i:>2} in '{t['section']}': {t['rows']} rows, "
              f"columns = {t['headers']}")

    h2 = sum(1 for lv, _ in ins.headings if lv == 2)
    h3 = sum(1 for lv, _ in ins.headings if lv == 3)
    print(f"\nSUMMARY: {h2} sections (h2), {h3} subsections (h3), "
          f"{len(ins.tables)} tables, {ins.useful_words:,} useful words.")
    print(line)


if __name__ == "__main__":
    main()
