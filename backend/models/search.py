"""Data structures returned by the vector store (Steps 6-7).

The rest of the application never sees ChromaDB's raw dictionaries. It only
sees these plain objects, so ChromaDB could be replaced without changing
the retriever, the routes or the tests.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchResult:
    """One stored chunk, optionally with how close it was to a question."""
    chunk_id: str
    text: str
    section: str
    subsection: Optional[str]
    source: str
    document_version: str
    position: int
    similarity: Optional[float] = None   # 1.0 = same meaning, 0.0 = unrelated (set by search())
    distance: Optional[float] = None     # cosine distance = 1 - similarity (set by search())
    metadata: dict = field(default_factory=dict)   # everything stored with the chunk

    @property
    def heading(self) -> str:
        return f"{self.section} > {self.subsection}" if self.subsection else self.section


@dataclass
class IndexInfo:
    """A summary of what is currently stored in the vector database."""
    count: int = 0
    source: Optional[str] = None
    document_version: Optional[str] = None
    embedding_model: Optional[str] = None
    fingerprint: Optional[str] = None
    indexed_at: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return self.count == 0


@dataclass
class RetrievalResponse:
    """Everything the retriever found for ONE question."""
    question: str
    results: List[SearchResult]          # best first
    top_k: int                           # how many were asked for
    embedding_model: str
    document_version: Optional[str]
    seconds: float

    @property
    def best_similarity(self) -> Optional[float]:
        """Similarity of the closest chunk (None if nothing was found)."""
        return self.results[0].similarity if self.results else None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "document_version": self.document_version,
            "seconds": round(self.seconds, 4),
            "results": [
                {
                    "rank": rank,
                    "chunk_id": r.chunk_id,
                    "heading": r.heading,
                    "similarity": round(r.similarity, 4) if r.similarity is not None else None,
                    "section": r.section,
                    "subsection": r.subsection,
                    "source": r.source,
                    "document_version": r.document_version,
                    "text": r.text,
                }
                for rank, r in enumerate(self.results, 1)
            ],
        }
