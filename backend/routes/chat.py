
"""POST /api/chat - answer a question using the RAG pipeline.

The route is deliberately thin.

Flow:
    request
      -> retrieve
      -> relevance check
      -> grounded prompt
      -> LLM
      -> JSON response

The route does not contain embedding, retrieval, prompting, or LLM logic.
"""

from flask import Blueprint, current_app, jsonify, request

from backend.rag.llm import (
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from backend.rag.prompt_builder import (
    NO_ANSWER_GENERIC,
    PromptBuildError,
    build_prompt,
)
from backend.rag.relevance import assess_relevance
from backend.rag.retriever import (
    IndexNotReadyError,
    ModelMismatchError,
    RetrievalError,
)
from backend.utils.logger import get_logger


logger = get_logger(__name__)

chat_bp = Blueprint("chat", __name__)


def _error(status_code: int, status: str, message: str):
    return jsonify({
        "status": status,
        "error": message,
    }), status_code


@chat_bp.post("/chat")
def chat():
    """Answer one employee question using the company knowledge base."""

    # ---------------------------------------------------------
    # 1. Validate request
    # ---------------------------------------------------------

    if not request.is_json:
        return _error(
            400,
            "invalid_request",
            "Request body must be JSON.",
        )

    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return _error(
            400,
            "invalid_request",
            "Request body must be a JSON object.",
        )

    question = body.get("question")

    if not isinstance(question, str):
        return _error(
            400,
            "invalid_request",
            "The 'question' field must be text.",
        )

    if not question.strip():
        return _error(
            400,
            "invalid_request",
            "The 'question' field cannot be empty.",
        )

    # ---------------------------------------------------------
    # 2. Get shared services
    # ---------------------------------------------------------

    services = current_app.extensions["services"]

    # ---------------------------------------------------------
    # 3. Retrieve candidate passages
    # ---------------------------------------------------------

    try:
        retrieval = services.retriever.retrieve(question)

    except (IndexNotReadyError, ModelMismatchError) as exc:
        logger.warning("Chat retrieval unavailable: %s", type(exc).__name__)

        return _error(
            503,
            "index_unavailable",
            str(exc),
        )

    except (RetrievalError, ValueError, TypeError) as exc:
        logger.warning("Chat retrieval failed: %s", type(exc).__name__)

        return _error(
            400,
            "invalid_question",
            str(exc),
        )

    # ---------------------------------------------------------
    # 4. Step 12 - decide whether there is enough evidence
    # ---------------------------------------------------------

    decision = assess_relevance(retrieval.results)

    if not decision.sufficient:
        return jsonify({
            "status": "ok",
            "answer": NO_ANSWER_GENERIC,
            "sources": [],
            "meta": {
                "retrieved": len(retrieval.results),
                "best_similarity": decision.best_similarity,
                "threshold": decision.threshold,
            },
        })

    # ---------------------------------------------------------
    # 5. Build grounded prompt
    # ---------------------------------------------------------

    try:
        prompt = build_prompt(
            question=retrieval.question,
            chunks=decision.passages,
        )

    except PromptBuildError:
        logger.exception("Could not build the grounded prompt")

        return _error(
            500,
            "prompt_error",
            "The question could not be prepared for the language model.",
        )

    # ---------------------------------------------------------
    # 6. Ask the LLM
    # ---------------------------------------------------------

    try:
        response = services.llm_service.generate(
            system=prompt.system,
            user=prompt.user,
        )

    except LLMConfigError:
        logger.exception("LLM configuration is invalid")

        return _error(
            503,
            "llm_unavailable",
            "The language model is not configured correctly.",
        )

    except LLMAuthError:
        logger.exception("LLM authentication failed")

        return _error(
            503,
            "llm_unavailable",
            "The language model could not be authenticated.",
        )

    except LLMRateLimitError:
        logger.exception("LLM rate limit reached")

        return _error(
            503,
            "llm_unavailable",
            "The language model is temporarily unavailable.",
        )

    except (LLMTimeoutError, LLMUnavailableError):
        logger.exception("LLM temporarily unavailable")

        return _error(
            503,
            "llm_unavailable",
            "The language model is temporarily unavailable.",
        )

    except LLMRequestError:
        logger.exception("LLM request failed")

        return _error(
            502,
            "llm_error",
            "The language model could not process the request.",
        )

    except LLMError:
        logger.exception("LLM generation failed")

        return _error(
            502,
            "llm_error",
            "The language model could not generate an answer.",
        )

    # ---------------------------------------------------------
    # 7. Return answer + source metadata
    # ---------------------------------------------------------

    sources = [
        {
            "citation": f"[{source.label}]",
            "label": source.label,
            "chunk_id": source.chunk_id,
            "heading": source.heading,
            "source": source.source,
            "document_version": source.document_version,
        }
        for source in prompt.sources
    ]

    return jsonify({
        "status": "ok",
        "answer": response.text,
        "sources": sources,
        "meta": {
            "retrieved": len(retrieval.results),
            "used_passages": len(prompt.sources),
            "best_similarity": decision.best_similarity,
            "threshold": decision.threshold,
            "llm_provider": response.provider,
            "llm_model": response.model,
        },
    })