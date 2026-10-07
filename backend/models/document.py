"""Data structures produced by the HTML parser (Step 3).

Think of these as the "shape" of the parser's output:

    ParsedDocument                      one per HTML file
      ├─ title, version, effective_date, owner, source_file
      └─ sections: [ParsedSection, ...]     one per <h3> block (or per <h2> intro)
            ├─ section      e.g. "Leave Policy"             (from <h2>)
            ├─ subsection   e.g. "Carry Forward and Encashment" (from <h3>)
            └─ blocks: [ParsedBlock, ...]   the actual content, kept separate
                  kind = paragraph | list | table_row | qa

Why keep "blocks" instead of one big string?
Step 4 (chunking) needs to know where a table row or a FAQ pair starts and
ends. If everything were glued into one string, that information would be lost.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ParsedBlock:
    kind: str   # "paragraph", "list", "table_row" or "qa"
    text: str   # clean text, no HTML tags

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text}


@dataclass
class ParsedSection:
    order: int                   # position in the document, starting at 0
    section_id: str              # e.g. "leave-policy" (from <section id="...">)
    section: str                 # the <h2> title
    subsection: Optional[str]    # the <h3> title, or None for text directly under <h2>
    blocks: List[ParsedBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        """All blocks joined with new lines."""
        return "\n".join(block.text for block in self.blocks)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "section_id": self.section_id,
            "section": self.section,
            "subsection": self.subsection,
            "word_count": self.word_count,
            "blocks": [block.to_dict() for block in self.blocks],
            "text": self.text,
        }


@dataclass
class ParsedDocument:
    title: str
    version: str
    effective_date: str
    owner: str
    source_file: str
    sections: List[ParsedSection] = field(default_factory=list)
    removed_noise: Dict[str, int] = field(default_factory=dict)  # what we threw away

    @property
    def total_words(self) -> int:
        return sum(section.word_count for section in self.sections)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "version": self.version,
            "effective_date": self.effective_date,
            "owner": self.owner,
            "source_file": self.source_file,
            "section_count": len(self.sections),
            "total_words": self.total_words,
            "removed_noise": self.removed_noise,
            "sections": [section.to_dict() for section in self.sections],
        }
