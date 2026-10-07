"""Step 6 - Vector store (ChromaDB).

A vector database stores, for every chunk:

    ID         "leave-policy-001"
    Document   the chunk text
    Embedding  [0.12, -0.32, ...]    (384 numbers)
    Metadata   section, subsection, source, document_version, ...

and can quickly answer: "which stored vectors are closest to THIS vector?"

This is the ONLY file that knows ChromaDB exists. The rest of the application
uses the simple operations below (add_documents, search, delete_collection,
get_collection ...), exactly as the project documentation asks.

Safety rules this file enforces (each was discovered by testing ChromaDB):
  * ChromaDB silently IGNORES a record whose id already exists. That could leave
    stale chunks in the index without any error, so we check for it ourselves.
  * ChromaDB refuses `None` in metadata, so "no subsection" is stored as "".
  * A vector of a different size means a different embedding model was used;
    we turn ChromaDB's error into an instruction ("run reindex").
"""
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings

from backend.config import Config
from backend.models.chunk import Chunk
from backend.models.search import IndexInfo, SearchResult
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_NAME_RULE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$")   # ChromaDB's rule: 3-512 chars
_BATCH_SIZE = 500


class VectorStoreError(Exception):
    """Raised when the vector database cannot do what was asked."""


def _check_extra_metadata(extra: Dict) -> Dict:
    for key, value in extra.items():
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"metadata '{key}' must be str, int, float or bool, got {type(value).__name__}")
    return dict(extra)


class VectorStore:
    def __init__(self, path=None, collection_name: Optional[str] = None):
        self.path = Path(path) if path else Path(Config.CHROMA_PATH)
        self.collection_name = collection_name or Config.CHROMA_COLLECTION
        if not _NAME_RULE.match(self.collection_name):
            raise ValueError(
                f"Invalid collection name '{self.collection_name}'. Use 3-512 letters, digits, "
                "'.', '_' or '-' (CHROMA_COLLECTION in .env)."
            )
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------ setup
    def _get_client(self):
        if self._client is None:
            self.path.mkdir(parents=True, exist_ok=True)
            # anonymized_telemetry=False: do not send usage statistics anywhere.
            self._client = chromadb.PersistentClient(
                path=str(self.path), settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    def get_collection(self):
        """The ChromaDB collection (created on first use). Internal use only.

        "hnsw:space": "cosine" makes ChromaDB measure cosine distance, the same
        similarity our embeddings are designed for. embedding_function=None
        means: ChromaDB never embeds anything itself, we always give it vectors,
        so the SAME model is guaranteed for chunks and questions.
        """
        if self._collection is None:
            try:
                self._collection = self._get_client().get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,
                )
            except Exception as exc:
                raise VectorStoreError(f"Could not open the vector database: {exc}") from exc
        return self._collection

    def delete_collection(self) -> None:
        """Remove the collection and everything in it (no error if it does not exist)."""
        try:
            self._get_client().delete_collection(self.collection_name)
        except Exception as exc:
            # "does not exist" is fine; anything else is a real problem
            if "not exist" not in str(exc).lower() and "notfound" not in type(exc).__name__.lower():
                raise VectorStoreError(f"Could not delete the collection: {exc}") from exc
        self._collection = None
        logger.info("Collection '%s' deleted", self.collection_name)

    # ----------------------------------------------------------------- writing
    def add_documents(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        extra_metadata: Optional[Dict] = None,
    ) -> int:
        """Store chunks with their vectors. Returns how many were stored.

        extra_metadata (for example the document fingerprint) is added to every record.
        """
        chunks = list(chunks)
        if not chunks:
            raise ValueError("no chunks to store")
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        ids = [c.chunk_id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("chunk ids are not unique")
        size = len(embeddings[0])
        for vector in embeddings:
            if len(vector) != size:
                raise ValueError("embeddings have different lengths")
            if not all(math.isfinite(x) for x in vector):
                raise ValueError("an embedding contains NaN or infinity")
        extra = _check_extra_metadata(extra_metadata or {})

        collection = self.get_collection()

        # ChromaDB would silently ignore ids that already exist. Refuse instead.
        clash = self._existing_ids(ids)
        if clash:
            raise VectorStoreError(
                f"{len(clash)} chunk id(s) are already stored (for example '{clash[0]}'). "
                "Delete the collection or run a reindex instead of adding twice."
            )

        metadatas = []
        for chunk in chunks:
            record = {
                "section": chunk.section,
                "subsection": chunk.subsection or "",       # ChromaDB does not accept None
                "source": chunk.source,
                "document_version": chunk.document_version,
                "position": chunk.position,
                "section_id": chunk.section_id,
                "word_count": chunk.word_count,
                "part": chunk.part,
                "parts": chunk.parts,
            }
            record.update(extra)
            metadatas.append(record)

        try:
            for start in range(0, len(chunks), _BATCH_SIZE):
                end = start + _BATCH_SIZE
                collection.add(
                    ids=ids[start:end],
                    embeddings=[list(map(float, v)) for v in embeddings[start:end]],
                    documents=[c.text for c in chunks[start:end]],
                    metadatas=metadatas[start:end],
                )
        except Exception as exc:
            if "dimension" in str(exc).lower():
                raise VectorStoreError(
                    "The vectors have a different size than the ones already stored. This usually means "
                    "the embedding model was changed. Rebuild the index with POST /api/reindex."
                ) from exc
            raise VectorStoreError(f"Could not store the chunks: {exc}") from exc

        logger.info("Stored %d chunks in collection '%s'", len(chunks), self.collection_name)
        return len(chunks)

    def _existing_ids(self, ids: List[str]) -> List[str]:
        found = self.get_collection().get(ids=ids, include=[])
        return list(found["ids"])

    # ----------------------------------------------------------------- reading
    def count(self) -> int:
        return self.get_collection().count()

    def search(self, query_embedding: Sequence[float], top_k: Optional[int] = None) -> List[SearchResult]:
        """The `top_k` stored chunks closest to the question vector, best first."""
        top_k = Config.TOP_K if top_k is None else top_k
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        total = self.count()
        if total == 0:
            return []
        try:
            raw = self.get_collection().query(
                query_embeddings=[list(map(float, query_embedding))],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if "dimension" in str(exc).lower():
                raise VectorStoreError(
                    "The question vector has a different size than the stored vectors. The embedding "
                    "model in .env is not the one the index was built with. Run POST /api/reindex."
                ) from exc
            raise VectorStoreError(f"Search failed: {exc}") from exc

        results = []
        for chunk_id, text, meta, distance in zip(raw["ids"][0], raw["documents"][0],
                                                  raw["metadatas"][0], raw["distances"][0]):
            result = self._to_result(chunk_id, text, meta)
            result.distance = float(distance)
            result.similarity = max(-1.0, min(1.0, 1.0 - float(distance)))   # cosine distance -> similarity
            results.append(result)
        return results

    def get_chunk(self, chunk_id: str) -> Optional[SearchResult]:
        """One stored chunk by its id, or None. (Used to verify that records exist.)"""
        raw = self.get_collection().get(ids=[chunk_id], include=["documents", "metadatas"])
        if not raw["ids"]:
            return None
        return self._to_result(raw["ids"][0], raw["documents"][0], raw["metadatas"][0])

    def list_chunks(self) -> List[SearchResult]:
        """Every stored chunk in document order."""
        raw = self.get_collection().get(include=["documents", "metadatas"])
        items = [self._to_result(i, d, m) for i, d, m in zip(raw["ids"], raw["documents"], raw["metadatas"])]
        return sorted(items, key=lambda r: r.position)

    def info(self) -> IndexInfo:
        """What is stored: how many chunks, from which document, built with which model."""
        collection = self.get_collection()
        total = collection.count()
        if total == 0:
            return IndexInfo()
        meta = collection.get(limit=1, include=["metadatas"])["metadatas"][0]
        return IndexInfo(
            count=total,
            source=meta.get("source"),
            document_version=meta.get("document_version"),
            embedding_model=meta.get("embedding_model"),
            fingerprint=meta.get("fingerprint"),
            indexed_at=meta.get("indexed_at"),
        )

    @staticmethod
    def _to_result(chunk_id: str, text: str, meta: dict) -> SearchResult:
        meta = dict(meta or {})
        return SearchResult(
            chunk_id=chunk_id,
            text=text,
            section=meta.get("section", ""),
            subsection=meta.get("subsection") or None,       # "" -> None again
            source=meta.get("source", ""),
            document_version=meta.get("document_version", ""),
            position=int(meta.get("position", 0)),
            metadata=meta,
        )


@lru_cache(maxsize=None)
def _build_store(path: str, collection_name: str) -> VectorStore:
    return VectorStore(path, collection_name)


def get_vector_store() -> VectorStore:
    """The ONE place the rest of the app gets its vector store from (built once, reused)."""
    return _build_store(str(Config.CHROMA_PATH), Config.CHROMA_COLLECTION)
