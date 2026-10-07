"""POST /api/ingest and POST /api/reindex.

These routes are deliberately thin: they call the ingestion functions and turn
the outcome into an HTTP response. All the real work lives in backend/rag/ingestion.py.

Error responses never contain file paths or stack traces (those go to the log).
"""
from flask import Blueprint, current_app, jsonify

from backend.config import Config
from backend.rag.embeddings import EmbeddingError
from backend.rag.html_parser import ParseError
from backend.rag.ingestion import (IndexBusyError, IndexConflictError,
                                   ingest_document, reindex_document)
from backend.rag.vector_store import VectorStoreError
from backend.utils.logger import get_logger

logger = get_logger(__name__)
ingestion_bp = Blueprint("ingestion", __name__)


def _error(status_code: int, status: str, message: str):
    return jsonify({"status": status, "error": message}), status_code


def _run(action):
    services = current_app.extensions["services"]
    try:
        result = action(Config.COMPANY_DOCUMENT_PATH, services.embedding_service, services.vector_store)
    except IndexBusyError as exc:
        return _error(409, "busy", str(exc))
    except IndexConflictError as exc:
        return _error(409, "conflict", str(exc))
    except ParseError:
        logger.exception("The company document could not be processed")
        return _error(500, "error", "The company document could not be read or contained no usable content.")
    except EmbeddingError:
        logger.exception("The embedding model is not available")
        return _error(503, "error", "The embedding model is not available. Check the server log.")
    except VectorStoreError:
        logger.exception("The vector database reported an error")
        return _error(500, "error", "The vector database reported an error. Check the server log.")
    return jsonify(result.to_dict())


@ingestion_bp.post("/ingest")
def ingest():
    """Build the index from the company document (does nothing if it is already up to date)."""
    return _run(ingest_document)


@ingestion_bp.post("/reindex")
def reindex():
    """Delete the old index and rebuild it from the current document."""
    return _run(reindex_document)
