"""Step 3 tests: the HTML parser.

Two kinds of tests:
  A) "Real document" tests  - parse data/raw/company.html and check the result.
  B) "Small HTML" tests     - tiny hand-written HTML strings that test one
                              rule each (edge cases, errors).
"""
import json
from pathlib import Path

import pytest

from backend.rag.html_parser import ParseError, parse_html_file, parse_html_string

DOC_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "company.html"


# =============================================================================
# A) Real document
# =============================================================================
@pytest.fixture(scope="module")
def doc():
    return parse_html_file(DOC_PATH)


def _find(doc, section, subsection=None):
    for s in doc.sections:
        if s.section == section and s.subsection == subsection:
            return s
    raise AssertionError(f"section not found: {section} > {subsection}")


def test_metadata_is_read(doc):
    assert doc.title == "ABC Tech Solutions - Company Knowledge Base"
    assert doc.version == "1.0"
    assert doc.effective_date == "2026-04-01"
    assert doc.owner.startswith("Human Resources")
    assert doc.source_file == "company.html"


def test_all_14_main_sections_are_found_in_order(doc):
    names = []
    for s in doc.sections:
        if s.section != "Document Information" and (not names or names[-1] != s.section):
            names.append(s.section)
    assert names == [
        "Company Overview", "Departments", "Job Roles and Career Levels",
        "Working Hours and Attendance", "Work Model", "Leave Policy",
        "Salary and Compensation", "Benefits", "Employee Policies",
        "HR Information", "Internal Processes", "Company Rules",
        "Frequently Asked Questions", "Contact Directory",
    ]


def test_section_count_and_order_numbers(doc):
    assert len(doc.sections) == 52
    assert [s.order for s in doc.sections] == list(range(52))


def test_section_ids_come_from_html(doc):
    assert _find(doc, "Leave Policy", "Annual Paid Leave").section_id == "leave-policy"
    assert _find(doc, "Frequently Asked Questions").section_id == "faq"


def test_every_section_has_text_and_no_empty_blocks(doc):
    for s in doc.sections:
        assert s.blocks, f"empty section: {s.section} > {s.subsection}"
        assert all(b.text.strip() for b in s.blocks)


# ---- noise removal ----------------------------------------------------------
NOISE_TEXT = [
    "googletagmanager", "dataLayer", "gtag", "border-collapse",
    "Employee Login", "On this page", "Accept all cookies",
    "Sports club", "All rights reserved", "analytics.example.com",
    "Privacy Policy",
]


@pytest.mark.parametrize("noise", NOISE_TEXT)
def test_noise_text_is_not_in_output(doc, noise):
    all_text = "\n".join(s.text for s in doc.sections)
    assert noise.lower() not in all_text.lower()


@pytest.mark.parametrize("noise", NOISE_TEXT)
def test_noise_is_removed_even_without_main_tag_protection(noise):
    """In company.html the header, banner, hidden note and footer sit OUTSIDE <main>,
    so reading only <main> already hides them. This test removes that safety net
    (turns <main> into a plain <div>) so the noise-removal RULES must do the work."""
    html = DOC_PATH.read_text(encoding="utf-8")
    html = html.replace("<main>", "<div>").replace("</main>", "</div>")
    doc_without_main = parse_html_string(html)
    all_text = "\n".join(s.text for s in doc_without_main.sections)
    assert noise.lower() not in all_text.lower()


def test_noise_removal_is_reported(doc):
    removed = doc.removed_noise
    assert removed["<script>"] == 2
    assert removed["<style>"] == 1
    assert removed["<footer>"] == 1
    assert removed["<noscript>"] == 1
    assert removed["hidden element"] == 1
    assert removed["cookie element"] == 1
    assert removed["html comment"] > 0


def test_no_html_tags_or_entities_left_in_text(doc):
    for s in doc.sections:
        assert "<" not in s.text and ">" not in s.text, s.subsection
        assert "&#" not in s.text and "&amp;" not in s.text


def test_raw_file_is_not_modified():
    before = DOC_PATH.read_bytes()
    parse_html_file(DOC_PATH)
    assert DOC_PATH.read_bytes() == before


# ---- conversion rules -------------------------------------------------------
def test_table_rows_repeat_column_names(doc):
    s = _find(doc, "Leave Policy", "Annual Paid Leave")
    rows = [b.text for b in s.blocks if b.kind == "table_row"]
    assert len(rows) == 4
    assert rows[0] == (
        "Leave Type: Casual Leave; Days Per Year: 8; Rules: Can be taken in half-day "
        "units. Maximum 3 consecutive days. Apply at least 1 day in advance except in emergencies."
    )
    assert rows[3] == ("Leave Type: Total; Days Per Year: 24; "
                       "Rules: Casual, sick and earned leave combined.")


def test_table_in_section_without_h3_is_kept(doc):
    s = _find(doc, "Departments")           # h2 followed directly by a table
    rows = [b for b in s.blocks if b.kind == "table_row"]
    assert len(rows) == 12
    assert rows[0].text.startswith("Department: Engineering; Main Function:")


def test_ordered_list_keeps_numbers(doc):
    s = _find(doc, "Internal Processes", "How to Apply for Leave")
    lines = s.blocks[0].text.split("\n")
    assert s.blocks[0].kind == "list"
    assert len(lines) == 5
    assert lines[0] == "1. Log in to the ABC People portal."
    assert lines[4].startswith("5. ")


def test_unordered_list_uses_dashes_and_nested_items_are_indented(doc):
    s = _find(doc, "Job Roles and Career Levels", "Role Responsibilities")
    lines = s.blocks[0].text.split("\n")
    assert lines[0].startswith("- Associate Engineer and Software Engineer:")
    assert "  - Write clean, tested and documented code." in lines   # 2-space indent


def test_faq_becomes_question_answer_pairs(doc):
    s = _find(doc, "Frequently Asked Questions")
    assert len(s.blocks) == 12
    assert all(b.kind == "qa" for b in s.blocks)
    assert s.blocks[0].text == (
        "Question: How many paid leave days do employees get in a year?\n"
        "Answer: Employees get 24 paid leave days per year: 8 casual, 8 sick and "
        "8 earned leave days."
    )


def test_callout_div_is_kept_as_paragraph(doc):
    s = _find(doc, "Leave Policy", "Special Leave")
    assert any(b.kind == "paragraph" and b.text.startswith("Note: Special leave is separate")
               for b in s.blocks)


def test_html_entity_is_decoded(doc):
    s = _find(doc, "Benefits", "Health and Insurance")
    assert "₹5,00,000" in s.text


def test_inline_tags_are_flattened(doc):
    s = _find(doc, "Company Overview", "Mission and Vision")
    assert s.blocks[0].text.startswith("Mission: To help organisations")


def test_text_before_first_h2_is_kept_as_document_information(doc):
    first = doc.sections[0]
    assert first.section == "Document Information"
    assert "Applies to: all full-time employees" in first.text


def test_key_facts_survive_parsing(doc):
    all_text = "\n".join(s.text for s in doc.sections)
    for fact in ("24 paid leave days per year", "9:00 AM to 6:00 PM",
                 "Tuesday, Wednesday and Thursday", "itsupport@abctech.example",
                 "Salary is credited on the last working day"):
        assert fact in all_text, fact


def test_unsupported_topics_stay_absent_after_parsing(doc):
    all_text = "\n".join(s.text for s in doc.sections).lower()
    for topic in ("revenue", "stock price", "dental", "sabbatical", "esop"):
        assert topic not in all_text


def test_result_is_json_serialisable_and_deterministic():
    first = parse_html_file(DOC_PATH).to_dict()
    second = parse_html_file(DOC_PATH).to_dict()
    assert first == second
    assert json.loads(json.dumps(first, ensure_ascii=False))["section_count"] == 52


# =============================================================================
# B) Small HTML strings - one rule per test
# =============================================================================
def wrap(body: str, head: str = "") -> str:
    return f"<html><head>{head}</head><body><main>{body}</main></body></html>"


def test_nested_noise_is_removed_without_crashing():
    html = wrap("""
        <nav><ul><li><a>Menu item</a></li></ul><script>var x=1;</script></nav>
        <h2>Real</h2><p>Keep me.</p>
    """)
    doc = parse_html_string(html)
    assert doc.sections[0].text == "Keep me."
    assert "Menu item" not in doc.sections[0].text


def test_hidden_attribute_and_visibility_hidden_are_removed():
    html = wrap("""
        <h2>Real</h2><p>Visible.</p>
        <p hidden>Secret one.</p>
        <p style="visibility: hidden">Secret two.</p>
        <p style="display: none;">Secret three.</p>
    """)
    text = parse_html_string(html).sections[0].text
    assert text == "Visible."


def test_class_keyword_noise_is_removed():
    html = wrap('<h2>Real</h2><p>Keep.</p><div class="promo-popup">Buy now</div>')
    assert parse_html_string(html).sections[0].text == "Keep."


def test_body_with_noisy_class_is_not_deleted():
    html = '<html><body class="no-cookie"><main><h2>A</h2><p>Text.</p></main></body></html>'
    assert parse_html_string(html).sections[0].text == "Text."


def test_table_without_thead_uses_first_th_row_as_header():
    html = wrap("""<h2>T</h2><table>
        <tr><th>Name</th><th>Qty</th></tr>
        <tr><td>Pens</td><td>10</td></tr></table>""")
    assert parse_html_string(html).sections[0].text == "Name: Pens; Qty: 10"


def test_table_without_any_header_keeps_values():
    html = wrap("<h2>T</h2><table><tr><td>Pens</td><td>10</td></tr></table>")
    assert parse_html_string(html).sections[0].text == "Pens; 10"


def test_irregular_row_does_not_get_wrong_labels():
    html = wrap("""<h2>T</h2><table><thead><tr><th>A</th><th>B</th></tr></thead>
        <tbody><tr><td colspan="2">Only one cell</td></tr></tbody></table>""")
    assert parse_html_string(html).sections[0].text == "Only one cell"


def test_h2_without_text_is_skipped_but_its_subsections_are_kept():
    html = wrap("<h2>Empty</h2><h3>Sub</h3><p>Body.</p>")
    sections = parse_html_string(html).sections
    assert len(sections) == 1
    assert (sections[0].section, sections[0].subsection) == ("Empty", "Sub")


def test_h2_resets_the_subsection():
    html = wrap("<h2>One</h2><h3>Sub</h3><p>a</p><h2>Two</h2><p>b</p>")
    s = parse_html_string(html).sections
    assert s[1].section == "Two" and s[1].subsection is None


def test_section_id_falls_back_to_slug_of_title():
    html = wrap("<h2>Leave Policy</h2><p>x</p>")
    assert parse_html_string(html).sections[0].section_id == "leave-policy"


def test_whitespace_nbsp_and_br_are_cleaned():
    html = wrap("<h2>A</h2><p>Line one<br>line\u00a0two\n\n   end</p>")
    assert parse_html_string(html).sections[0].text == "Line one line two end"


def test_loose_text_and_inline_tags_in_a_div_become_a_paragraph():
    html = wrap('<h2>A</h2><div class="note"><strong>Note:</strong> Be kind.</div>')
    assert parse_html_string(html).sections[0].text == "Note: Be kind."


def test_missing_version_falls_back_to_unknown():
    assert parse_html_string(wrap("<h2>A</h2><p>x</p>")).version == "unknown"


def test_title_falls_back_to_title_tag_then_filename():
    with_title = parse_html_string(wrap("<h2>A</h2><p>x</p>", "<title>My Page</title>"))
    assert with_title.title == "My Page"
    nothing = parse_html_string("<body><h2>A</h2><p>x</p></body>", source_file="f.html")
    assert nothing.title == "f.html"


def test_works_without_main_tag():
    doc = parse_html_string("<html><body><h2>A</h2><p>x</p></body></html>")
    assert doc.sections[0].text == "x"


def test_definition_list_with_two_answers():
    html = wrap("<h2>F</h2><dl><dt>Q?</dt><dd>One.</dd><dd>Two.</dd></dl>")
    assert parse_html_string(html).sections[0].text == "Question: Q?\nAnswer: One. Two."


# ---- error handling ---------------------------------------------------------
def test_missing_file_raises_parse_error(tmp_path):
    with pytest.raises(ParseError, match="not found"):
        parse_html_file(tmp_path / "nope.html")


def test_page_with_only_noise_raises_parse_error():
    html = wrap("<script>x</script><nav>menu</nav><footer>foot</footer>")
    with pytest.raises(ParseError, match="No readable content"):
        parse_html_string(html)


def test_empty_string_raises_parse_error():
    with pytest.raises(ParseError):
        parse_html_string("")


def test_non_utf8_file_raises_parse_error(tmp_path):
    bad = tmp_path / "bad.html"
    bad.write_bytes(b"<html>\xff\xfe\x00bad</html>")
    with pytest.raises(ParseError, match="Could not read"):
        parse_html_file(bad)
