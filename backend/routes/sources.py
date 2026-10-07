"""GET /api/sources - what the knowledge base currently contains."""
from flask import Blueprint, current_app, jsonify

from backend.config import Config
from backend.rag.ingestion import index_is_current
from backend.rag.vector_store import VectorStoreError
from backend.utils.logger import get_logger

logger = get_logger(__name__)
sources_bp = Blueprint("sources", __name__)


@sources_bp.get("/sources")
def sources():
    store = current_app.extensions["services"].vector_store
    try:
        info = store.info()
    except VectorStoreError:
        logger.exception("Could not read the index")
        return jsonify({"status": "error", "error": "The vector database reported an error."}), 500

    return jsonify({
        "indexed": not info.is_empty,
        "document": info.source,
        "version": info.document_version,
        "chunks": info.count,
        "embedding_model": info.embedding_model,
        "indexed_at": info.indexed_at,
        # False means: company.html (or the chunk settings) changed after indexing -> run POST /api/reindex
        "up_to_date": index_is_current(Config.COMPANY_DOCUMENT_PATH, info),
    })
