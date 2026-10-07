"""Step 7 - Ingestion: from company.html to a searchable index.

    company.html -> parse -> chunk -> embed -> store in ChromaDB

Two operations (they back POST /api/ingest and POST /api/reindex):

    ingest_document   Build the index. Safe to repeat:
                        - index empty                  -> build it
                        - index already matches        -> do nothing ("already_indexed")
                        - index is from another version/model/settings
                                                       -> refuse (IndexConflictError):
                          mixing old and new chunks would give wrong answers.
    reindex_document  Always rebuild from scratch (document changed, model changed ...).

THE FINGERPRINT answers "is the index still up to date?". It is a short hash of:
the document text + the embedding model name + the chunk settings. If any of them
changes, the stored chunks would no longer match what we would build today.

ORDER MATTERS: everything that can fail (reading, parsing, embedding) is done
BEFORE the old index is touched. A broken document or an unavailable model
therefore leaves the existing index intact.
"""
import hashlib
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.config import Config
from backend.models.search import IndexInfo
from backend.rag.chunker import chunk_document
from backend.rag.embeddings import EmbeddingService
from backend.rag.html_parser import ParseError, parse_html_string
from backend.rag.vector_store import VectorStore
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_index_lock = threading.Lock()


class IngestionError(Exception):
    """Base class for ingestion problems."""


class IndexConflictError(IngestionError):
    """The existing index does not match the document / model / settings. Use reindex."""


class IndexBusyError(IngestionError):
    """Another ingest or reindex is already running."""


@dataclass
class IngestResult:
    status: str                    # "success" or "already_indexed"
    source: str
    document_version: str
    sections: int
    chunks_created: int            # 0 when nothing had to be done
    chunks_in_index: int
    embedding_model: str
    dimension: int
    fingerprint: str
    seconds: float
    replaced_previous: bool = False
    previous_version: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Helpers
# =============================================================================
def compute_fingerprint(document_bytes: bytes, embedding_model: str, max_words: int, overlap_words: int) -> str:
    """Short hash identifying 'this document, embedded by this model, chunked with these settings'."""
    normalised = document_bytes.replace(b"\r\n", b"\n")      # Windows vs Linux line endings must not matter
    digest = hashlib.sha256()
    digest.update(normalised)
    digest.update(f"\0{embedding_model}|{max_words}|{overlap_words}".encode("utf-8"))
    return digest.hexdigest()[:16]


@contextmanager
def _exclusive():
    """Only one ingest/reindex at a time (two at once would corrupt the index)."""
    if not _index_lock.acquire(blocking=False):
        raise IndexBusyError("Indexing is already in progress. Try again in a moment.")
    try:
        yield
    finally:
        _index_lock.release()


def _read_document(path) -> bytes:
    path = Path(path)
    if not path.is_file():
        raise ParseError(f"Company document not found: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ParseError(f"Could not read {path}: {exc}") from exc


def _settings(max_words, overlap_words):
    return (Config.CHUNK_MAX_WORDS if max_words is None else max_words,
            Config.CHUNK_OVERLAP_WORDS if overlap_words is None else overlap_words)


def _parse_and_chunk(path, max_words: int, overlap_words: int):
    data = _read_document(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(f"Could not read {Path(path).name}: it is not valid UTF-8 text") from exc
    doc = parse_html_string(text, source_file=Path(path).name)
    chunks = chunk_document(doc, max_words, overlap_words)
    return data, doc, chunks


def index_is_current(path, info: IndexInfo, max_words: Optional[int] = None,
                     overlap_words: Optional[int] = None) -> Optional[bool]:
    """Does the stored index still match the file on disk? None = cannot tell (no index / no file)."""
    if info.is_empty or not info.fingerprint or not info.embedding_model:
        return None
    try:
        data = _read_document(path)
    except ParseError:
        return None
    max_words, overlap_words = _settings(max_words, overlap_words)
    return info.fingerprint == compute_fingerprint(data, info.embedding_model, max_words, overlap_words)


def _store(store: VectorStore, embedder: EmbeddingService, chunks, vectors, fingerprint: str) -> None:
    extra = {
        "fingerprint": fingerprint,
        "embedding_model": embedder.model_name,
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        store.add_documents(chunks, vectors, extra_metadata=extra)
    except Exception:
        # Never leave a half-filled index behind: it would look valid but miss chunks.
        logger.error("Storing failed: removing the incomplete index")
        store.delete_collection()
        raise


# =============================================================================
# Public operations
# =============================================================================
def ingest_document(path, embedder: EmbeddingService, store: VectorStore,
                    max_words: Optional[int] = None, overlap_words: Optional[int] = None) -> IngestResult:
    """Build the index if it does not exist; do nothing if it is already up to date."""
    started = time.perf_counter()
    max_words, overlap_words = _settings(max_words, overlap_words)
    with _exclusive():
        data, doc, chunks = _parse_and_chunk(path, max_words, overlap_words)
        fingerprint = compute_fingerprint(data, embedder.model_name, max_words, overlap_words)
        info = store.info()

        def result(status, created):
            return IngestResult(
                status=status, source=doc.source_file, document_version=doc.version,
                sections=len(doc.sections), chunks_created=created, chunks_in_index=store.count(),
                embedding_model=embedder.model_name, dimension=embedder.dimension,
                fingerprint=fingerprint, seconds=round(time.perf_counter() - started, 2),
            )

        if info.is_empty:
            vectors = embedder.embed_chunks(chunks)
            _store(store, embedder, chunks, vectors, fingerprint)
            logger.info("Ingested %d chunks (version %s)", len(chunks), doc.version)
            return result("success", len(chunks))

        if info.fingerprint == fingerprint and info.count == len(chunks):
            logger.info("Index is already up to date (%d chunks): nothing to do", info.count)
            return result("already_indexed", 0)

        reasons = []
        if info.document_version != doc.version:
            reasons.append(f"document version {info.document_version} in the index, {doc.version} on disk")
        if info.embedding_model != embedder.model_name:
            reasons.append(f"embedding model '{info.embedding_model}' in the index, '{embedder.model_name}' configured")
        if info.fingerprint == fingerprint and info.count != len(chunks):
            reasons.append(f"the index is incomplete ({info.count} of {len(chunks)} chunks)")
        if not reasons:
            reasons.append("the document text or the chunk settings changed")
        raise IndexConflictError(
            "The existing index does not match: " + "; ".join(reasons) + ". "
            "Use POST /api/reindex to rebuild it."
        )


def reindex_document(path, embedder: EmbeddingService, store: VectorStore,
                     max_words: Optional[int] = None, overlap_words: Optional[int] = None) -> IngestResult:
    """Replace the whole index with a fresh one built from the current document."""
    started = time.perf_counter()
    max_words, overlap_words = _settings(max_words, overlap_words)
    with _exclusive():
        data, doc, chunks = _parse_and_chunk(path, max_words, overlap_words)       # may raise: index untouched
        fingerprint = compute_fingerprint(data, embedder.model_name, max_words, overlap_words)
        vectors = embedder.embed_chunks(chunks)                                    # may raise: index untouched

        previous = store.info()
        if not previous.is_empty:
            store.delete_collection()                                              # only now do we change anything
        _store(store, embedder, chunks, vectors, fingerprint)
        logger.info("Reindexed %d chunks (version %s, replaced %s)", len(chunks), doc.version,
                    previous.document_version if not previous.is_empty else "nothing")
        return IngestResult(
            status="success", source=doc.source_file, document_version=doc.version,
            sections=len(doc.sections), chunks_created=len(chunks), chunks_in_index=store.count(),
            embedding_model=embedder.model_name, dimension=embedder.dimension,
            fingerprint=fingerprint, seconds=round(time.perf_counter() - started, 2),
            replaced_previous=not previous.is_empty,
            previous_version=None if previous.is_empty else previous.document_version,
        )
