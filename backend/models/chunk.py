"""The Chunk data structure (Step 4).

A chunk is the unit that will be embedded, stored in ChromaDB, retrieved and
finally shown to the LLM. Every chunk carries its own metadata, so that when
it is retrieved we still know WHERE it came from (needed for citations).

Fields follow the project documentation (section 23), plus a few helpers:

    chunk_id         "leave-policy-001"   unique, human-readable
    text             heading line + content (this is what gets embedded)
    section          "Leave Policy"                (the <h2>)
    subsection       "Annual Paid Leave" or None   (the <h3>)
    source           "company.html"
    document_version "1.0"
    position         0, 1, 2 ...          order in the whole document
    -- helpers --
    section_id       "leave-policy"       id of the HTML <section>
    word_count       words in `text`
    part / parts     1 / 2                set when one section had to be split
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section: str
    subsection: Optional[str]
    source: str
    document_version: str
    position: int
    section_id: str
    word_count: int
    part: int = 1
    parts: int = 1

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "section": self.section,
            "subsection": self.subsection,
            "source": self.source,
            "document_version": self.document_version,
            "position": self.position,
            "section_id": self.section_id,
            "word_count": self.word_count,
            "part": self.part,
            "parts": self.parts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)
