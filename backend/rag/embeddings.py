"""Step 5 - Embeddings.

An EMBEDDING is a list of numbers (a vector) that represents the MEANING of a
text. Texts with similar meaning get vectors that point in similar directions:

    "How many annual leaves are available?"  ->  [0.12, -0.28, 0.93, ...]
    "Employees receive 24 paid leave days"   ->  [0.10, -0.25, 0.90, ...]   (close)
    "Salary is paid on the last working day" ->  [-0.70, 0.31, 0.05, ...]   (far)

How "close" is measured: COSINE SIMILARITY (1.0 = same direction, 0 = unrelated).

IMPORTANT RULE: the SAME model must embed the chunks AND the user's questions.
Vectors from different models live in different "spaces" and cannot be compared.
That is why everything goes through one object: an EmbeddingService.

Three classes live here:

    EmbeddingService                abstract interface (what the rest of the app uses)
      ├─ SentenceTransformerService the real local model (default)
      └─ HashingEmbeddingService    tiny offline stand-in for tests / development

The rest of the application only calls get_embedding_service(), so the model can
be changed in .env without rewriting any other code.
"""
import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional, Sequence

from backend.config import Config
from backend.models.chunk import Chunk
from backend.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # small (about 90 MB), 384 numbers per text
HASHING_DIMENSION = 384                                     # same size as the default model
PROVIDERS = ("sentence-transformers", "hashing")

Vector = List[float]


class EmbeddingError(Exception):
    """Raised when the embedding model cannot be loaded or used."""


# =============================================================================
# Helpers
# =============================================================================
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """1.0 = same direction, 0.0 = unrelated, -1.0 = opposite."""
    if len(a) != len(b):
        raise ValueError(f"vectors have different lengths: {len(a)} and {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _check_texts(texts: Sequence[str]) -> List[str]:
    """Reject anything that is not a non-empty string."""
    cleaned = []
    for item in texts:
        if not isinstance(item, str):
            raise TypeError(f"expected text (str), got {type(item).__name__}")
        if not item.strip():
            raise ValueError("cannot embed an empty text")
        cleaned.append(item)
    return cleaned


# =============================================================================
# The interface
# =============================================================================
class EmbeddingService(ABC):
    """What every embedding backend must provide."""

    model_name: str = "unknown"

    @property
    @abstractmethod
    def dimension(self) -> int:
        """How many numbers are in each vector."""

    @abstractmethod
    def _embed(self, texts: List[str]) -> List[Vector]:
        """Convert already-validated texts into vectors (one per text, same order)."""

    # ---- optional: only models with a tokenizer implement these -------------
    @property
    def max_tokens(self) -> Optional[int]:
        """Longest input the model reads (longer text is silently cut). None = unknown."""
        return None

    def token_counts(self, texts: Sequence[str]) -> Optional[List[int]]:
        return None

    # ---- the public API used by the rest of the application -----------------
    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        """Vectors for many chunk texts. Order is preserved."""
        texts = _check_texts(texts)
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> Vector:
        """Vector for ONE user question. Uses exactly the same model as the chunks."""
        return self._embed(_check_texts([text]))[0]

    def embed_chunks(self, chunks: Sequence[Chunk]) -> List[Vector]:
        """Vectors for Chunk objects (the heading is part of chunk.text)."""
        return self.embed_documents([chunk.text for chunk in chunks])


# =============================================================================
# Real model
# =============================================================================
class SentenceTransformerService(EmbeddingService):
    """Runs a Sentence Transformers model locally (no API key, no per-call cost).

    The model is downloaded once, on first use (internet needed), and cached on
    your computer afterwards.
    """

    def __init__(self, model_name: Optional[str] = None, model=None, batch_size: Optional[int] = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL or DEFAULT_MODEL
        self.batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        # `model` can be injected (used by the tests); normally we load it.
        self._model = model if model is not None else self._load(self.model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        logger.info("Embedding model ready: %s (%d numbers per text)", self.model_name, self._dimension)

    @staticmethod
    def _load(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "The 'sentence-transformers' package is not installed. "
                "Run:  pip install sentence-transformers"
            ) from exc
        try:
            logger.info("Loading embedding model %s (the first run downloads it)...", model_name)
            return SentenceTransformer(model_name)
        except Exception as exc:  # network, disk, wrong name ... all end up here
            raise EmbeddingError(
                f"Could not load the embedding model '{model_name}': {exc}. "
                "The first run needs an internet connection to download the model, and the "
                "name in EMBEDDING_MODEL must be correct."
            ) from exc

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_tokens(self) -> Optional[int]:
        value = getattr(self._model, "max_seq_length", None)
        return int(value) if value else None

    def token_counts(self, texts: Sequence[str]) -> Optional[List[int]]:
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            return None
        encoded = tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def _embed(self, texts: List[str]) -> List[Vector]:
        # normalize_embeddings=True makes every vector length 1, so a plain dot
        # product equals cosine similarity (and ChromaDB distances behave).
        rows = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in rows]


# =============================================================================
# Offline stand-in
# =============================================================================
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("""
    a an the and or of to in on at for is are was were be been do does did i my me we our
    you your it its this that what which how many much when where who can will with from by
    as if per there have has any
""".split())


def _stem(word: str) -> str:
    """Very crude plural removal so 'leaves' matches 'leave' and 'days' matches 'day'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class HashingEmbeddingService(EmbeddingService):
    """A tiny, offline, deterministic embedder ("feature hashing" of words).

    It turns every word into a position in a 384-number vector. Two texts that
    share words get similar vectors. That is ALL it understands: it knows nothing
    about meaning ("vacation" and "leave" look unrelated to it).

    Use it for: unit tests, and developing the pipeline without downloading a model.
    Do NOT use it for real answers. Use SentenceTransformerService for that.
    """

    model_name = "hashing-bag-of-words"

    def __init__(self, dimension: int = HASHING_DIMENSION):
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> Vector:
        words = _WORD.findall(text.lower())
        if not words:
            raise ValueError("cannot embed a text without letters or digits")
        tokens = [_stem(w) for w in words if w not in _STOPWORDS]
        if not tokens:                       # text made only of stopwords: keep them
            tokens = [_stem(w) for w in words]

        counts: dict = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        vector = [0.0] * self._dimension
        for token, count in counts.items():
            # NOT Python's hash(): it changes on every run. blake2b is stable.
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]

    def _embed(self, texts: List[str]) -> List[Vector]:
        return [self._embed_one(text) for text in texts]


# =============================================================================
# Factory
# =============================================================================
@lru_cache(maxsize=None)
def _build(provider: str, model_name: str) -> EmbeddingService:
    if provider == "sentence-transformers":
        return SentenceTransformerService(model_name or None)
    if provider == "hashing":
        return HashingEmbeddingService()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{provider}'. Valid values: {', '.join(PROVIDERS)}")


def get_embedding_service() -> EmbeddingService:
    """The ONE place the rest of the app gets its embedding model from.

    Built once and reused (loading a model takes seconds, so we must not do it per request).
    """
    return _build(Config.EMBEDDING_PROVIDER, Config.EMBEDDING_MODEL)
