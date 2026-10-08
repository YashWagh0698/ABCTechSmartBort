"""Step 15 - Validate source citations returned by the LLM.

The prompt gives each retrieved passage a stable label such as [1], [2], [3].
This module checks that citations in the answer refer only to labels that were
actually supplied to the model.
"""

import re
from dataclasses import dataclass
from typing import Iterable


_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class CitationDecision:
    """Result of citation validation."""

    valid: bool
    citations: list[int]
    invalid_citations: list[int]
    reason: str = ""


def extract_citations(answer: str) -> list[int]:
    """Return citation labels appearing in an answer, in first-seen order."""

    if not isinstance(answer, str):
        raise TypeError("answer must be text")

    found = []
    seen = set()

    for match in _CITATION.finditer(answer):
        label = int(match.group(1))

        if label not in seen:
            seen.add(label)
            found.append(label)

    return found


def validate_citations(
    answer: str,
    valid_labels: Iterable[int],
) -> CitationDecision:
    """Check that every citation in the answer exists in the prompt sources."""

    citations = extract_citations(answer)
    valid = {int(label) for label in valid_labels}

    invalid = [
        label
        for label in citations
        if label not in valid
    ]

    if invalid:
        return CitationDecision(
            valid=False,
            citations=citations,
            invalid_citations=invalid,
            reason=(
                "the answer contains citation labels that were not present "
                "in the grounded context: "
                + ", ".join(f"[{label}]" for label in invalid)
            ),
        )

    return CitationDecision(
        valid=True,
        citations=citations,
        invalid_citations=[],
        reason="all citation labels refer to supplied passages",
    )
