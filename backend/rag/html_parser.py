"""Step 3 - HTML parser.

Turns the raw company HTML into clean, structured sections.

What it does, in order:
    1. Read the file (never modifies it).
    2. Read metadata from <meta> tags (version, owner, effective date).
    3. Remove noise: scripts, styles, menus, footers, cookie banners,
       hidden elements and HTML comments.
    4. Walk through the remaining content in document order, tracking the
       current <h2> (section) and <h3> (subsection).
    5. Convert each element into clean text:
         <p>     -> paragraph
         <ul/ol> -> list (numbers kept for ordered lists, nesting kept)
         <table> -> one sentence per row that repeats the column names
         <dl>    -> "Question: ... / Answer: ..." pairs
    6. Return a ParsedDocument. Raise ParseError if the result is unusable.

This file has NO chunking, embedding or database code. That comes later.
"""
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from backend.models.document import ParsedBlock, ParsedDocument, ParsedSection
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# --- Rules for what counts as noise -----------------------------------------
# Tags that never contain knowledge-base content.
NOISE_TAGS = ("script", "style", "noscript", "template", "iframe",
              "nav", "header", "footer")

# If a class or id contains one of these words, the element is noise.
NOISE_WORDS = ("cookie", "banner", "tracking", "analytics", "popup", "advert")

# Tags we never judge by class/id (a body class like "no-cookie" must not delete the page).
NEVER_NOISE = ("html", "head", "body", "main")

# Tags that start a new "block" of content (as opposed to inline text like <strong>).
BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "dl",
              "div", "section", "article", "blockquote", "pre", "hr", "main"}

# Text placed before the first <h2> (e.g. "Version 1.0 | Owner: HR") goes here.
INTRO_SECTION_TITLE = "Document Information"
INTRO_SECTION_ID = "document-info"


class ParseError(Exception):
    """Raised when the HTML file is missing or cannot give usable content."""


# =============================================================================
# Small helpers
# =============================================================================
def clean_text(text: str) -> str:
    """Collapse all whitespace (spaces, tabs, new lines, &nbsp;) into single spaces."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _text_of(tag: Tag) -> str:
    """Visible text of a tag. Inline tags such as <strong> and <a> are flattened."""
    return clean_text(tag.get_text(""))


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


# =============================================================================
# Step A: metadata
# =============================================================================
def _read_metadata(soup: BeautifulSoup) -> dict:
    """Collect <meta name="..." content="..."> pairs."""
    meta = {}
    for tag in soup.find_all("meta"):
        name = tag.get("name")
        if name:
            meta[name] = (tag.get("content") or "").strip()
    return meta


# =============================================================================
# Step B: noise removal
# =============================================================================
def _noise_label(tag: Tag) -> Optional[str]:
    """Return a label such as '<nav>' if the tag is noise, otherwise None."""
    if tag.name in NOISE_TAGS:
        return f"<{tag.name}>"
    if tag.name in NEVER_NOISE:
        return None

    attrs = tag.attrs or {}
    style = (attrs.get("style") or "").replace(" ", "").lower()
    if "hidden" in attrs or "display:none" in style or "visibility:hidden" in style:
        return "hidden element"

    classes = attrs.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    marker = " ".join(classes + [str(attrs.get("id") or "")]).lower()
    for word in NOISE_WORDS:
        if word in marker:
            return f"{word} element"
    return None


def _remove_noise(soup: BeautifulSoup) -> Counter:
    """Delete noise from the soup (in memory only) and count what was removed."""
    removed: Counter = Counter()

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
        removed["html comment"] += 1

    for tag in soup.find_all(True):
        if tag.decomposed:          # its parent was already deleted
            continue
        label = _noise_label(tag)
        if label:
            removed[label] += 1
            tag.decompose()

    # <br> would glue words together ("line1line2"), so turn it into a space.
    for br in soup.find_all("br"):
        br.replace_with(" ")
    return removed


# =============================================================================
# Step C: converting elements to text
# =============================================================================
def _list_to_text(list_tag: Tag, depth: int = 0) -> str:
    """<ul>/<ol> -> lines. Ordered lists keep numbers; nested lists are indented."""
    lines: List[str] = []
    number = 0
    for li in list_tag.find_all("li", recursive=False):
        number += 1
        marker = f"{number}." if list_tag.name == "ol" else "-"

        own_parts: List[str] = []   # the item's own text
        nested_lists: List[Tag] = []  # lists inside this item
        for child in li.children:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                nested_lists.append(child)
            elif isinstance(child, Tag):
                own_parts.append(child.get_text(""))
            else:
                own_parts.append(str(child))

        text = clean_text("".join(own_parts))
        if text:
            lines.append("  " * depth + f"{marker} {text}")
        for nested in nested_lists:
            nested_text = _list_to_text(nested, depth + 1)
            if nested_text:
                lines.append(nested_text)
    return "\n".join(lines)


def _table_to_rows(table: Tag) -> List[str]:
    """Every body row becomes 'Column: value; Column: value; ...'.

    A table cell alone has no meaning ("8"). With its column name it does
    ("Days Per Year: 8"), and that is what a user's question will resemble.
    """
    header_row = None
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
    else:
        for tr in table.find_all("tr"):
            if tr.find("th"):
                header_row = tr
                break

    headers = []
    if header_row:
        headers = [_text_of(cell) for cell in header_row.find_all(["th", "td"])]

    rows: List[str] = []
    for tr in table.find_all("tr"):
        if tr is header_row:
            continue
        cells = [_text_of(cell) for cell in tr.find_all(["th", "td"])]
        if not any(cells):
            continue
        if headers and len(headers) == len(cells):
            parts = [f"{h}: {v}" for h, v in zip(headers, cells) if v]
        else:  # irregular row: keep the values, we cannot label them safely
            parts = [v for v in cells if v]
        rows.append("; ".join(parts))
    return rows


def _dl_to_pairs(dl: Tag) -> List[str]:
    """<dl><dt>question</dt><dd>answer</dd></dl> -> 'Question: ...\\nAnswer: ...'."""
    pairs: List[str] = []
    question: Optional[str] = None
    answers: List[str] = []

    def flush():
        if question is None and not answers:
            return
        lines = []
        if question:
            lines.append(f"Question: {question}")
        if answers:
            lines.append("Answer: " + " ".join(answers))
        pairs.append("\n".join(lines))

    for item in dl.find_all(["dt", "dd"]):
        if item.name == "dt":
            flush()
            question, answers = _text_of(item), []
        else:
            text = _text_of(item)
            if text:
                answers.append(text)
    flush()
    return pairs


# =============================================================================
# Step D: walking the document and building sections
# =============================================================================
class _SectionBuilder:
    """Collects blocks and starts a new section whenever a heading appears."""

    def __init__(self) -> None:
        self.sections: List[ParsedSection] = []
        self._section = INTRO_SECTION_TITLE
        self._section_id = INTRO_SECTION_ID
        self._subsection: Optional[str] = None
        self._blocks: List[ParsedBlock] = []

    def start_section(self, title: str, section_id: str) -> None:
        self.flush()
        self._section, self._section_id, self._subsection = title, section_id, None

    def start_subsection(self, title: str) -> None:
        self.flush()
        self._subsection = title

    def add(self, kind: str, text: str) -> None:
        if text:
            self._blocks.append(ParsedBlock(kind=kind, text=text))

    def flush(self) -> None:
        """Close the current section. Headings with no text of their own are skipped."""
        if self._blocks:
            self.sections.append(ParsedSection(
                order=len(self.sections),
                section_id=self._section_id,
                section=self._section,
                subsection=self._subsection,
                blocks=self._blocks,
            ))
        self._blocks = []


def _has_block_children(tag: Tag) -> bool:
    return any(isinstance(c, Tag) and c.name in BLOCK_TAGS for c in tag.children)


def _walk(container: Tag, builder: _SectionBuilder) -> None:
    """Visit the children of `container` in order and feed the builder."""
    inline_parts: List[str] = []   # loose text / <strong> / <a> waiting to become a paragraph

    def flush_inline() -> None:
        builder.add("paragraph", clean_text("".join(inline_parts)))
        inline_parts.clear()

    for child in list(container.children):
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            inline_parts.append(str(child))
            continue
        if child.name not in BLOCK_TAGS:          # <strong>, <a>, <span> ...
            inline_parts.append(child.get_text(""))
            continue

        flush_inline()
        name = child.name

        if name == "h1":
            continue                               # title is read separately
        elif name == "h2":
            parent = child.find_parent("section")
            title = _text_of(child)
            section_id = (parent.get("id") if parent and parent.get("id") else _slugify(title))
            builder.start_section(title, section_id)
        elif name == "h3":
            builder.start_subsection(_text_of(child))
        elif name in ("h4", "h5", "h6", "p", "blockquote", "pre"):
            builder.add("paragraph", _text_of(child))
        elif name in ("ul", "ol"):
            builder.add("list", _list_to_text(child))
        elif name == "table":
            for row in _table_to_rows(child):
                builder.add("table_row", row)
        elif name == "dl":
            for pair in _dl_to_pairs(child):
                builder.add("qa", pair)
        elif name == "hr":
            continue
        else:                                      # div, section, article, main
            if _has_block_children(child):
                _walk(child, builder)              # look inside
            else:
                builder.add("paragraph", _text_of(child))   # e.g. <div class="note">

    flush_inline()


# =============================================================================
# Public API
# =============================================================================
def parse_html_string(html: str, source_file: str = "document.html") -> ParsedDocument:
    """Parse an HTML string into a ParsedDocument."""
    soup = BeautifulSoup(html, "html.parser")

    meta = _read_metadata(soup)               # read BEFORE removing anything
    removed = _remove_noise(soup)

    root = soup.find("main") or soup.body
    if root is None:
        raise ParseError("The HTML has no <main> or <body> element to read.")

    h1 = root.find("h1")
    if h1 is not None:
        title = _text_of(h1)
    elif soup.title and soup.title.string:
        title = clean_text(soup.title.string)
    else:
        title = source_file

    builder = _SectionBuilder()
    _walk(root, builder)
    builder.flush()

    if not builder.sections:
        raise ParseError("No readable content was found after removing noise.")

    version = meta.get("document-version", "")
    if not version:
        logger.warning("No <meta name='document-version'> found; using 'unknown'.")
        version = "unknown"

    document = ParsedDocument(
        title=title,
        version=version,
        effective_date=meta.get("effective-date", ""),
        owner=meta.get("document-owner", ""),
        source_file=source_file,
        sections=builder.sections,
        removed_noise=dict(removed),
    )
    logger.info("Parsed %s: %d sections, %d words, removed noise: %s",
                source_file, len(document.sections), document.total_words, dict(removed))
    return document


def parse_html_file(path) -> ParsedDocument:
    """Read an HTML file from disk and parse it."""
    path = Path(path)
    if not path.is_file():
        raise ParseError(f"Company document not found: {path}")
    try:
        html = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ParseError(f"Could not read {path}: {exc}") from exc
    return parse_html_string(html, source_file=path.name)
