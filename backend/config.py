"""Central configuration.

Every setting lives here. Other modules import from this file instead of
reading environment variables themselves. Secrets (like LLM_API_KEY) come
from the .env file and are never hard-coded.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains backend/, data/, frontend/ ...
BASE_DIR = Path(__file__).resolve().parent.parent

# Read variables from .env (if the file exists) into os.environ
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _optional_float(name: str, default: float):
    """Unset -> default. Set but empty -> None (meaning: do not use this setting)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return None if raw == "" else float(raw)


def _path(name: str, default: str) -> Path:
    """Resolve a path from .env relative to the project root."""
    return (BASE_DIR / os.getenv(name, default)).resolve()


class Config:
    # --- Application ---
    APPLICATION_ENV = os.getenv("APPLICATION_ENV", "development")
    DEBUG = _bool("DEBUG", True)
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))

    # --- Data / vector store ---
    COMPANY_DOCUMENT_PATH = _path("COMPANY_DOCUMENT_PATH", "data/raw/company.html")
    CHUNKS_PATH = _path("CHUNKS_PATH", "data/processed/chunks.json")
    CHROMA_PATH = _path("CHROMA_PATH", "vector_db/chroma")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "abc_company")

    # --- Models ---
    # "sentence-transformers" = real local model (default); "hashing" = tiny offline
    # stand-in used for tests and for developing without downloading a model.
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").strip().lower()
    # Empty means "use the default model" (see backend/rag/embeddings.py).
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    # LLM (Step 10). Provider: fake | anthropic | openai | openai-compatible | ollama
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
    LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
    LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()  # SECRET: never expose, never log
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()   # only for gateways / local servers
    LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    LLM_TEMPERATURE = _optional_float("LLM_TEMPERATURE", 0.0)   # empty in .env = do not send it
    LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "600"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))              # extra attempts, transient errors only
    LLM_RETRY_BACKOFF_SECONDS = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "1.0"))

    # --- RAG tuning ---
    # TOP_K: how many chunks the retriever returns (Step 8). Tested with real data in Step 9.
    TOP_K = int(os.getenv("TOP_K", "5"))
    # RELEVANCE_THRESHOLD (Step 12): the minimum COSINE SIMILARITY (-1..1, higher = closer) a retrieved passage needs
    # to count as evidence. If NO passage reaches it, the LLM is not called at all.
    # 0.20 is PROVISIONAL and deliberately LOW: it only removes clearly unrelated questions. It was chosen from the few
    # real-model scores seen so far (correct answers 0.27-0.51, off-topic 0.22-0.30). Similar-but-incorrect questions
    # (dental insurance 0.43) score as high as correct ones, so no threshold can catch them: the prompt rules and the
    # answer validator do. Re-tune it with the threshold sweep in evaluation/evaluation_report.md.
    # (The old placeholder 0.75 would refuse almost everything; if your .env still says 0.75, change it.)
    RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.20"))
    # Chunk limits are measured in WORDS (1 token is roughly 0.75 of a word).
    CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "150"))
    CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "20"))

    # --- Safety limits ---
    MAX_CONTENT_LENGTH = 16 * 1024  # reject request bodies bigger than 16 KB
    MAX_QUESTION_CHARS = 1000        # longest user question we accept
    # Step 11: the most characters of retrieved text placed in a prompt (lowest-ranked chunks are dropped first).
    CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "12000"))
