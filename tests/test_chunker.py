"""Step 4 tests: the chunker.

Covers what the project documentation asks for (section 58):
chunk size, overlap, empty chunks, very large sections,
heading preservation and metadata preservation.

A) Real document tests  - chunk data/raw/company.html
B) Small synthetic tests - tiny hand-made documents, one rule each
"""
import json
import re
from pathlib import Path

import pytest

from backend.models.chunk import Chunk
from backend.models.document import ParsedBlock, ParsedDocument, ParsedSection
from backend.rag.chunker import (
    _pack,
    _pack_balanced,
    chunk_document,
    count_words,
    find_problems,
    load_chunks,
    save_chunks,
)
from backend.rag.html_parser import parse_html_file

DOC_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "company.html"
MAX_WORDS, OVERLAP = 150, 20          # the project defaults


def body_of(chunk: Chunk) -> str:
    """Chunk text without the heading line."""
    return chunk.text.split("\n", 1)[1]


# =============================================================================
# A) Real document
# =============================================================================
@pytest.fixture(scope="module")
def parsed():
    return parse_html_file(DOC_PATH)


@pytest.fixture(scope="module")
def chunks(parsed):
    return chunk_document(parsed, max_words=MAX_WORDS, overlap_words=OVERLAP)


def test_default_settings_give_56_chunks(chunks):
    assert len(chunks) == 56


def test_quality_gate_passes(chunks):
    assert find_problems(chunks, MAX_WORDS) == []


def test_no_chunk_body_exceeds_the_limit(chunks):
    assert max(count_words(body_of(c)) for c in chunks) <= MAX_WORDS


def test_no_empty_chunks(chunks):
    assert all(body_of(c).strip() for c in chunks)


def test_every_chunk_starts_with_its_heading(chunks):
    for c in chunks:
        expected = f"{c.section} > {c.subsection}" if c.subsection else c.section
        assert c.text.split("\n", 1)[0] == expected


def test_metadata_is_preserved(chunks):
    for c in chunks:
        assert c.source == "company.html"
        assert c.document_version == "1.0"
        assert c.section and c.section_id
        assert c.word_count == count_words(c.text)


def test_positions_are_0_to_n_minus_1(chunks):
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_unique_and_readable(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9-]+-\d{3}", i) for i in ids)


def test_ids_count_per_section_starting_at_001(chunks):
    leave = [c.chunk_id for c in chunks if c.section_id == "leave-policy"]
    assert leave == [f"leave-policy-00{n}" for n in range(1, 7)]


def test_only_the_three_large_sections_are_split(chunks):
    split = {(c.section, c.subsection) for c in chunks if c.parts > 1}
    assert split == {
        ("Departments", None),
        ("Leave Policy", "Special Leave"),
        ("Frequently Asked Questions", None),
    }


def test_split_sections_have_no_tiny_leftover_chunk(chunks):
    for c in chunks:
        if c.parts > 1:
            assert count_words(body_of(c)) >= 50, c.chunk_id


def test_part_numbers_are_consistent(chunks):
    for c in chunks:
        assert 1 <= c.part <= c.parts
    faq = [c for c in chunks if c.section_id == "faq"]
    assert [(c.part, c.parts) for c in faq] == [(1, 3), (2, 3), (3, 3)]


def test_faq_pairs_are_never_cut_in_half(chunks):
    for c in chunks:
        if c.section_id == "faq":
            assert c.text.count("Question:") == c.text.count("Answer:") > 0


def test_every_block_survives_intact_in_some_chunk(parsed, chunks):
    """No information is lost: each block (table row, FAQ pair, list...) appears whole."""
    text_by_section = {}
    for c in chunks:
        text_by_section.setdefault(c.section_id, []).append(c.text)
    for s in parsed.sections:
        joined = "\n".join(text_by_section[s.section_id])
        for block in s.blocks:
            assert block.text in joined, f"lost: {s.section} > {s.subsection}: {block.text[:50]}"


def test_key_facts_are_in_the_right_chunk(chunks):
    def find(fact):
        return [c for c in chunks if fact in c.text]

    leave = find("24 paid leave days per year")
    assert any(c.text.startswith("Leave Policy > Annual Paid Leave") for c in leave)
    hours = find("9:00 AM to 6:00 PM")
    assert any(c.text.startswith("Working Hours and Attendance > Standard Working Hours")
               for c in hours)


def test_unsupported_topics_are_not_in_any_chunk(chunks):
    text = " ".join(c.text for c in chunks).lower()
    for topic in ("revenue", "stock price", "dental", "sabbatical", "esop", "sports club"):
        assert topic not in text


def test_chunking_is_deterministic(parsed):
    a = [c.to_dict() for c in chunk_document(parsed, MAX_WORDS, OVERLAP)]
    b = [c.to_dict() for c in chunk_document(parsed, MAX_WORDS, OVERLAP)]
    assert a == b


def test_huge_limit_gives_one_chunk_per_parsed_section(parsed):
    result = chunk_document(parsed, max_words=800, overlap_words=100)
    assert len(result) == len(parsed.sections) == 52
    assert all(c.parts == 1 for c in result)


@pytest.mark.parametrize("size", [30, 60, 100, 150, 200, 800])
def test_any_limit_respects_size_and_loses_no_words(parsed, size):
    result = chunk_document(parsed, max_words=size, overlap_words=min(10, size - 1))
    assert find_problems(result, size) == []
    words_by_section = {}
    for c in result:
        words_by_section.setdefault(c.section_id, set()).update(c.text.split())
    for s in parsed.sections:
        for block in s.blocks:
            missing = set(block.text.split()) - words_by_section[s.section_id]
            assert not missing, f"words lost at size {size}: {list(missing)[:3]}"


def test_smaller_limit_means_more_chunks(parsed):
    counts = [len(chunk_document(parsed, max_words=n, overlap_words=5)) for n in (200, 100, 60)]
    assert counts[0] < counts[1] < counts[2]


def test_save_and_load_round_trip(chunks, tmp_path):
    path = save_chunks(chunks, tmp_path / "out" / "chunks.json")   # creates the folder too
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and len(raw) == len(chunks)
    for key in ("chunk_id", "text", "section", "subsection", "source",
                "document_version", "position"):               # fields the doc requires
        assert key in raw[0]
    assert load_chunks(path) == chunks


def test_load_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_chunks"):
        load_chunks(tmp_path / "nope.json")


# =============================================================================
# B) Small synthetic documents
# =============================================================================
def make_doc(*sections) -> ParsedDocument:
    """sections: tuples of (section, subsection, [block texts]) or (..., [(kind, text)])."""
    parsed_sections = []
    for i, (section, subsection, blocks) in enumerate(sections):
        parsed_blocks = [b if isinstance(b, ParsedBlock) else ParsedBlock("paragraph", b)
                         for b in blocks]
        parsed_sections.append(ParsedSection(
            order=i, section_id=section.lower().replace(" ", "-"),
            section=section, subsection=subsection, blocks=parsed_blocks))
    return ParsedDocument(title="T", version="9.9", effective_date="", owner="",
                          source_file="x.html", sections=parsed_sections)


def words(n: int, tag: str = "w") -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def test_small_section_is_one_chunk_with_heading():
    doc = make_doc(("Leave", "Casual", ["Eight days."]))
    [c] = chunk_document(doc, 50, 5)
    assert c.text == "Leave > Casual\nEight days."
    assert (c.section, c.subsection, c.source, c.document_version) == ("Leave", "Casual", "x.html", "9.9")


def test_heading_without_subsection_is_just_the_section():
    [c] = chunk_document(make_doc(("Departments", None, ["Engineering."])), 50, 5)
    assert c.text == "Departments\nEngineering."


def test_whitespace_only_section_creates_no_chunk():
    doc = make_doc(("Empty", None, ["   "]), ("Real", None, ["Text."]))
    result = chunk_document(doc, 50, 5)
    assert [c.section for c in result] == ["Real"]
    assert result[0].position == 0


def test_blocks_are_grouped_up_to_the_limit_and_never_cut():
    blocks = [words(20, f"b{i}_") for i in range(10)]            # ten 20-word blocks
    result = chunk_document(make_doc(("S", None, blocks)), max_words=50, overlap_words=10)
    assert all(count_words(body_of(c)) <= 50 for c in result)
    for block in blocks:                                          # each block whole, exactly once
        assert sum(block in c.text for c in result) == 1


def test_overlap_repeats_last_sentence_when_a_block_is_split():
    sentences = [f"Sentence {i:02d} is made of exactly ten simple words here." for i in range(1, 31)]
    text = " ".join(sentences)                                    # one 300-word paragraph
    result = chunk_document(make_doc(("S", None, [text])), max_words=50, overlap_words=10)
    bodies = [body_of(c) for c in result]
    assert len(bodies) > 2
    assert all(count_words(b) <= 50 for b in bodies)
    for i in range(1, len(bodies)):                               # overlap: previous last sentence
        assert bodies[i].startswith(bodies[i - 1].split(". ")[-1].rstrip(".")[:11])
    assert all(any(s in b for b in bodies) for s in sentences)    # nothing lost


def test_zero_overlap_means_no_repeated_sentence():
    sentences = [f"Sentence {i:02d} is made of exactly ten simple words here." for i in range(1, 31)]
    result = chunk_document(make_doc(("S", None, [" ".join(sentences)])), 50, 0)
    assert all(sum(s in body_of(c) for c in result) == 1 for s in sentences)


def test_long_list_is_split_between_items_not_inside_them():
    items = [f"- item {i:02d} " + words(8, f"i{i}_") for i in range(20)]   # 20 lines, 10 words each
    result = chunk_document(make_doc(("S", None, [ParsedBlock("list", "\n".join(items))])), 50, 0)
    assert len(result) > 1
    for c in result:
        for line in body_of(c).split("\n"):
            assert line in items                                  # every line is a whole, original item


def test_single_very_long_sentence_falls_back_to_word_windows():
    text = words(100)                                             # 100 words, no punctuation
    result = chunk_document(make_doc(("S", None, [text])), max_words=30, overlap_words=5)
    w = text.split()
    assert [body_of(c) for c in result] == [" ".join(w[start:start + 30]) for start in (0, 25, 50, 75)]


def test_very_large_section_is_split_into_many_valid_chunks():
    blocks = [words(40, f"p{i}_") for i in range(100)]            # 4,000 words
    result = chunk_document(make_doc(("Big", None, blocks)), max_words=150, overlap_words=20)
    assert len(result) >= 27
    assert find_problems(result, 150) == []
    assert all(c.parts == len(result) for c in result)


def test_ids_stay_unique_when_two_sections_share_an_id():
    doc = make_doc(("Same", "A", ["one"]), ("Same", "B", ["two"]))
    assert [c.chunk_id for c in chunk_document(doc, 50, 5)] == ["same-001", "same-002"]


# ---- balancing --------------------------------------------------------------
def test_balanced_packing_avoids_a_tiny_leftover_piece():
    sizes = [15, 20, 15, 15, 35, 20, 30, 35, 15, 15]
    units = [words(n, f"u{k}_") for k, n in enumerate(sizes)]
    greedy = [count_words(p) for p in _pack(units, "\n", 100, 0)]
    balanced = [count_words(p) for p in _pack_balanced(units, "\n", 100)]
    assert greedy == [100, 100, 15]          # the problem: a 15-word orphan
    assert balanced == [65, 85, 65]          # the fix: same number of pieces, evenly sized
    assert "\n".join(balanced_piece for balanced_piece in _pack_balanced(units, "\n", 100)).split() \
        == "\n".join(units).split()          # same content, same order


def test_balanced_packing_never_exceeds_the_limit():
    units = [words(n, f"u{k}_") for k, n in enumerate([30, 45, 10, 60, 25, 40, 5, 55])]
    assert all(count_words(p) <= 100 for p in _pack_balanced(units, "\n", 100))


# ---- validation and the quality gate ---------------------------------------
@pytest.mark.parametrize("max_words, overlap", [(0, 0), (-5, 0), (50, -1), (50, 50), (50, 80)])
def test_bad_limits_are_rejected(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_document(make_doc(("S", None, ["x"])), max_words, overlap)


def _chunk(**override) -> Chunk:
    base = dict(chunk_id="a-001", text="H\nbody", section="S", subsection=None, source="x.html",
                document_version="1", position=0, section_id="a", word_count=2)
    base.update(override)
    return Chunk(**base)


def test_quality_gate_detects_each_kind_of_problem():
    assert find_problems([_chunk()], 10) == []
    assert any("duplicate" in p for p in find_problems([_chunk(), _chunk(position=1)], 10))
    assert any("empty body" in p for p in find_problems([_chunk(text="H\n   ")], 10))
    assert any("limit" in p for p in find_problems([_chunk(text="H\n" + words(20))], 10))
    assert any("metadata" in p for p in find_problems([_chunk(source="")], 10))
    assert any("metadata" in p for p in find_problems([_chunk(document_version="")], 10))
