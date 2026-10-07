"""Step 2 tests: is data/raw/company.html a good knowledge source?

These tests protect the *input data* of the RAG system. If someone edits
company.html and accidentally breaks its structure, or adds information that
we deliberately left out (revenue, stock price...), these tests tell us.
Only the standard library is used.
"""
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

DOC_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "company.html"
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}


@pytest.fixture(scope="module")
def html() -> str:
    assert DOC_PATH.exists(), f"Company document not found: {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


def test_document_is_substantial(html):
    assert len(html) > 20_000


def test_document_version_metadata(html):
    assert '<meta name="document-version" content="1.0">' in html


def test_exactly_one_h1(html):
    assert len(re.findall(r"<h1[ >]", html)) == 1


def test_has_all_expected_sections(html):
    ids = re.findall(r'<section id="([^"]+)"', html)
    assert len(ids) == len(set(ids)), "section ids must be unique"
    expected = {"company-overview", "departments", "job-levels", "working-hours",
                "work-model", "leave-policy", "salary", "benefits",
                "employee-policies", "hr-information", "internal-processes",
                "company-rules", "faq", "contacts"}
    assert expected <= set(ids)


def test_every_section_has_an_h2(html):
    sections = re.findall(r"<section .*?</section>", html, flags=re.S)
    assert len(sections) == 14
    for sec in sections:
        assert len(re.findall(r"<h2>", sec)) == 1


def test_h3_never_appears_before_first_h2(html):
    assert html.index("<h2>") < html.index("<h3>")


def test_every_table_has_header_row(html):
    assert html.count("<table") == html.count("<thead>") >= 10


def test_noise_elements_exist_for_the_parser_to_remove(html):
    # Step 3's parser must have something to clean.
    for fragment in ("<script", "<style", "<nav", "<header", "<footer",
                     "<noscript", "display:none", "cookie-banner"):
        assert fragment in html, f"expected noise missing: {fragment}"


def test_html_is_well_formed(html):
    """Every opening tag must be closed in the right order."""
    problems = []

    class Checker(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []

        def handle_starttag(self, tag, attrs):
            if tag not in VOID_TAGS:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in VOID_TAGS:
                return
            if not self.stack or self.stack[-1] != tag:
                problems.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            else:
                self.stack.pop()

    checker = Checker()
    checker.feed(html)
    assert not problems, problems[:3]
    assert not checker.stack, f"unclosed tags: {checker.stack}"


# ---- the facts our later tests will rely on ---------------------------------
KNOWN_FACTS = [
    "24 paid leave days per year",
    "9:00 AM to 6:00 PM",
    "last working day",
    "Tuesday, Wednesday and Thursday",
    "60 days",
    "&#8377;5,00,000",
    "26 weeks",
    "itsupport@abctech.example",
    "within 30 days",
    "1 April",
]


@pytest.mark.parametrize("fact", KNOWN_FACTS)
def test_known_fact_is_present(html, fact):
    assert fact in html


# ---- topics that must stay ABSENT so refusal behaviour can be tested --------
MUST_BE_ABSENT = ["revenue", "profit", "stock price", "share price",
                  "valuation", "esop", "sabbatical", "dental"]


@pytest.mark.parametrize("topic", MUST_BE_ABSENT)
def test_unsupported_topic_is_absent(html, topic):
    assert topic not in html.lower(), (
        f"'{topic}' must NOT appear in company.html - it is used to test that "
        "the bot refuses when there is no evidence."
    )
