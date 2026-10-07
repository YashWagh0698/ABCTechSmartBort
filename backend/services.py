"""Shared services (embedding model, vector database, retriever, LLM) for the Flask app.

They are created LAZILY - only the first time a route needs them - because
loading the embedding model takes seconds. This keeps `GET /api/health`
instant and lets the tests hand in fake services:

    create_app(services=Services(embedding_service=FakeEmbedder(), vector_store=TempStore()))
"""
from typing import Optional

from backend.rag.embeddings import EmbeddingService, get_embedding_service
from backend.rag.llm import LLMService, get_llm_service
from backend.rag.rag_service import RAGService
from backend.rag.retriever import Retriever
from backend.rag.vector_store import VectorStore, get_vector_store


class Services:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None,
                 vector_store: Optional[VectorStore] = None,
                 llm_service: Optional[LLMService] = None):
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._llm_service = llm_service
        self._retriever = None
        self._rag_service = None

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def retriever(self) -> Retriever:
        """Built from the two services above (so it is lazy too)."""
        if self._retriever is None:
            self._retriever = Retriever(self.embedding_service, self.vector_store)
        return self._retriever

    @property
    def llm_service(self) -> LLMService:
        """Created only when first needed (it reads .env and may raise LLMConfigError)."""
        if self._llm_service is None:
            self._llm_service = get_llm_service()
        return self._llm_service

    @property
    def rag_service(self) -> RAGService:
        """The whole question-answering pipeline (Step 12), built from the services above."""
        if self._rag_service is None:
            self._rag_service = RAGService(self.retriever, self.llm_service)
        return self._rag_service
