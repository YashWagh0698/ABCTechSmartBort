"""Step 4 - Chunker.

Turns the parser's structured sections into chunks (small pieces of text with
metadata) that are ready to be embedded.

STRATEGY: "structure first, size second"

    1. Start from the document's own structure: one <h3> subsection = one chunk.
       (A subsection is one topic, so a chunk stays about one topic.)
    2. If a section is bigger than the limit, split it BETWEEN blocks
       (between table rows, FAQ pairs, paragraphs, lists) - never in the middle
       of a row or an answer.
    3. Only if a single block is still too big, split INSIDE it:
       by list items, then by sentences, then (last resort) by words.
       Overlap is used only here, to avoid cutting a thought in half.
    4. Put the heading in front of every chunk:  "Leave Policy > Annual Paid Leave".
       Without it, a chunk like "Casual Leave: 8 days" would not say what it is about.

Sizes are measured in WORDS (about 0.75 words per token), see Config.
Nothing here talks to embeddings, ChromaDB or an LLM.
"""
import json
import math
import re
from pathlib import Path
from typing import List, Optional

from backend.config import Config
from backend.models.chunk import Chunk
from backend.models.document import ParsedDocument, ParsedSection
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# A sentence ends at . ! or ? followed by whitespace.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


# =============================================================================
# Helpers
# =============================================================================
def count_words(text: str) -> int:
    return len(text.split())


def _validate_limits(max_words: int, overlap_words: int) -> None:
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")


def heading_line(section: ParsedSection) -> str:
    """'Leave Policy > Annual Paid Leave', or just 'Departments' if there is no <h3>."""
    if section.subsection:
        return f"{section.section} > {section.subsection}"
    return section.section


# =============================================================================
# Splitting and packing
# =============================================================================
def _pack(units: List[str], joiner: str, max_words: int, overlap_words: int) -> List[str]:
    """Group small units into pieces of at most `max_words` words.

    Units are never cut. If overlap_words > 0, the last unit(s) of a piece are
    repeated at the start of the next piece, as long as they fit in the overlap.
    """
    pieces: List[str] = []
    current: List[str] = []
    current_words = 0

    for unit in units:
        unit_words = count_words(unit)
        if current and current_words + unit_words > max_words:
            pieces.append(joiner.join(current))

            carry: List[str] = []           # units we repeat as overlap
            carry_words = 0
            for previous in reversed(current):
                w = count_words(previous)
                if carry_words + w > overlap_words:
                    break
                carry.insert(0, previous)
                carry_words += w
            if carry_words + unit_words > max_words:   # overlap would not leave room
                carry, carry_words = [], 0
            current, current_words = carry, carry_words

        current.append(unit)
        current_words += unit_words

    if current:
        pieces.append(joiner.join(current))
    return pieces


def _pack_balanced(units: List[str], joiner: str, max_words: int) -> List[str]:
    """Like _pack (no overlap), but makes the pieces EVEN in size.

    Plain greedy packing fills each piece to the limit and leaves the scraps
    for the end, e.g. 125 / 147 / 18 words. Here we first count how many pieces
    are needed, then search for the smallest limit that still needs only that
    many pieces. Result: about 94 / 94 / 94 instead of 125 / 147 / 18, so there
    is no tiny leftover chunk.
    """
    pieces = _pack(units, joiner, max_words, overlap_words=0)
    needed = len(pieces)
    if needed <= 1:
        return pieces

    total = sum(count_words(u) for u in units)
    low = max(max(count_words(u) for u in units), math.ceil(total / needed))
    high = max_words
    while low < high:                       # binary search: fewer pieces as the limit grows
        middle = (low + high) // 2
        if len(_pack(units, joiner, middle, overlap_words=0)) <= needed:
            high = middle
        else:
            low = middle + 1
    return _pack(units, joiner, low, overlap_words=0)


def _split_by_words(text: str, max_words: int, overlap_words: int) -> List[str]:
    """Last resort: sliding window over words."""
    words = text.split()
    step = max_words - overlap_words
    pieces = []
    start = 0
    while True:
        pieces.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
        start += step
    return pieces


def _split_long_text(text: str, max_words: int, overlap_words: int) -> List[str]:
    """Split ONE block that is longer than max_words.

    Try the most natural boundary first: list items / lines, then sentences,
    then words.
    """
    if count_words(text) <= max_words:
        return [text]

    lines = text.split("\n")
    if len(lines) > 1:
        units, joiner = lines, "\n"
    else:
        sentences = [s for s in _SENTENCE_BREAK.split(text) if s.strip()]
        if len(sentences) > 1:
            units, joiner = sentences, " "
        else:
            return _split_by_words(text, max_words, overlap_words)

    # A single line or sentence may itself be too long: split it further.
    flat: List[str] = []
    for unit in units:
        if count_words(unit) > max_words:
            flat.extend(_split_long_text(unit, max_words, overlap_words))
        else:
            flat.append(unit)
    return _pack(flat, joiner, max_words, overlap_words)


def _section_bodies(section: ParsedSection, max_words: int, overlap_words: int) -> List[str]:
    """Return the chunk bodies (text without heading) for one parsed section."""
    units: List[str] = []
    for block in section.blocks:
        if count_words(block.text) > max_words:
            units.extend(_split_long_text(block.text, max_words, overlap_words))
        else:
            units.append(block.text)
    # Whole blocks are self-contained (a table row, a Q&A pair), so repeating
    # them as "overlap" adds nothing: overlap is 0 between blocks. We balance
    # the pieces so that no tiny leftover chunk appears.
    return [b for b in _pack_balanced(units, "\n", max_words) if b.strip()]


# =============================================================================
# Public API
# =============================================================================
def chunk_document(
    doc: ParsedDocument,
    max_words: Optional[int] = None,
    overlap_words: Optional[int] = None,
) -> List[Chunk]:
    """Create chunks (with metadata) from a ParsedDocument."""
    max_words = Config.CHUNK_MAX_WORDS if max_words is None else max_words
    overlap_words = Config.CHUNK_OVERLAP_WORDS if overlap_words is None else overlap_words
    _validate_limits(max_words, overlap_words)

    chunks: List[Chunk] = []
    id_counters: dict = {}          # section_id -> how many chunks so far

    for section in doc.sections:
        head = heading_line(section)
        bodies = _section_bodies(section, max_words, overlap_words)

        for part, body in enumerate(bodies, start=1):
            number = id_counters.get(section.section_id, 0) + 1
            id_counters[section.section_id] = number
            text = f"{head}\n{body}"
            chunks.append(Chunk(
                chunk_id=f"{section.section_id}-{number:03d}",
                text=text,
                section=section.section,
                subsection=section.subsection,
                source=doc.source_file,
                document_version=doc.version,
                position=len(chunks),
                section_id=section.section_id,
                word_count=count_words(text),
                part=part,
                parts=len(bodies),
            ))

    logger.info("Created %d chunks from %d sections (max_words=%d, overlap=%d)",
                len(chunks), len(doc.sections), max_words, overlap_words)
    return chunks


def find_problems(chunks: List[Chunk], max_words: Optional[int] = None) -> List[str]:
    """A quality gate. Returns a list of human-readable problems (empty = all good)."""
    max_words = Config.CHUNK_MAX_WORDS if max_words is None else max_words
    problems: List[str] = []

    seen = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            problems.append(f"duplicate chunk_id: {chunk.chunk_id}")
        seen.add(chunk.chunk_id)

        body = chunk.text.split("\n", 1)[1] if "\n" in chunk.text else ""
        if not body.strip():
            problems.append(f"{chunk.chunk_id}: empty body")
        if count_words(body) > max_words:
            problems.append(f"{chunk.chunk_id}: body has {count_words(body)} words (limit {max_words})")
        if not chunk.section or not chunk.source or not chunk.document_version:
            problems.append(f"{chunk.chunk_id}: missing metadata")
    return problems


def save_chunks(chunks: List[Chunk], path) -> Path:
    """Write chunks to a JSON file (a list of objects) for inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_chunks(path) -> List[Chunk]:
    """Read chunks back from a JSON file written by save_chunks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {path}. Run scripts/build_chunks.py first.")
    return [Chunk.from_dict(item) for item in json.loads(path.read_text(encoding="utf-8"))]
