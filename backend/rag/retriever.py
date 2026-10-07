"""Step 8 - Retriever: from a question to the closest chunks.

    question -> embedding -> ChromaDB nearest neighbours -> top-K chunks + similarity scores

This is the "R" in RAG (Retrieval). It does NOT decide whether the chunks are
good enough to answer from (that is relevance.py, Step 12) and it does NOT talk
to an LLM (Step 10). It only finds candidates and reports how close they are.
The project documentation is explicit: inspect retrieval on its own first.

Two safety checks happen on EVERY call, because a wrong answer here silently
poisons everything after it:

  * EMPTY INDEX     -> IndexNotReadyError. "Nothing indexed yet" must never look
                       like "no relevant chunk exists".
  * MODEL MISMATCH  -> ModelMismatchError. If the question is embedded by a different
                       model than the one that built the index, the numbers cannot be
                       compared. Without this check we would still get results back,
                       they would just be meaningless.
"""
import time
from typing import Optional

from backend.config import Config
from backend.models.search import RetrievalResponse
from backend.rag.embeddings import EmbeddingService
from backend.rag.vector_store import VectorStore
from backend.utils.logger import get_logger

logger = get_logger(__name__)

MAX_TOP_K = 50


class RetrievalError(Exception):
    """Base class for retrieval problems (not the user's fault)."""


class IndexNotReadyError(RetrievalError):
    """The vector database is empty: nothing has been indexed yet."""


class ModelMismatchError(RetrievalError):
    """The index was built with a different embedding model than the one configured now."""


class Retriever:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore,
                 top_k: Optional[int] = None):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.default_top_k = Config.TOP_K if top_k is None else top_k
        self._validate_top_k(self.default_top_k)

    # ------------------------------------------------------------- validation
    @staticmethod
    def _validate_top_k(top_k) -> int:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be a whole number")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        return top_k

    @staticmethod
    def _clean_question(question) -> str:
        if not isinstance(question, str):
            raise TypeError("the question must be text")
        question = " ".join(question.split())             # collapse spaces, tabs, new lines
        if not question:
            raise ValueError("the question is empty")
        if len(question) > Config.MAX_QUESTION_CHARS:
            raise ValueError(f"the question is too long (maximum {Config.MAX_QUESTION_CHARS} characters)")
        return question

    def _check_index(self):
        info = self.vector_store.info()
        if info.is_empty:
            raise IndexNotReadyError(
                "The search index is empty. Build it first: POST /api/ingest "
                "(or run: python scripts/ingest_company.py)."
            )
        if info.embedding_model != self.embedding_service.model_name:
            raise ModelMismatchError(
                f"The index was built with the embedding model '{info.embedding_model}', but the "
                f"configured model is '{self.embedding_service.model_name}'. Their vectors cannot be "
                "compared. Rebuild the index: POST /api/reindex "
                "(or run: python scripts/ingest_company.py --reindex)."
            )
        return info

    # --------------------------------------------------------------- the work
    def retrieve(self, question: str, top_k: Optional[int] = None) -> RetrievalResponse:
        """Return the `top_k` chunks closest to the question, best first, with similarity scores."""
        started = time.perf_counter()
        question = self._clean_question(question)
        top_k = self.default_top_k if top_k is None else self._validate_top_k(top_k)

        info = self._check_index()
        results = self.vector_store.search(self.embedding_service.embed_query(question), top_k)

        response = RetrievalResponse(
            question=question,
            results=results,
            top_k=top_k,
            embedding_model=self.embedding_service.model_name,
            document_version=info.document_version,
            seconds=time.perf_counter() - started,
        )
        # Log sizes and scores, never the question text itself (it may be sensitive).
        logger.info("Retrieved %d chunks for a question of %d characters; scores %s",
                    len(results), len(question), [round(r.similarity, 3) for r in results])
        return response
