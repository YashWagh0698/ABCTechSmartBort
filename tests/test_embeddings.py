"""Step 5 tests: the embedding service.

IMPORTANT: no test downloads a model or needs internet.
  - HashingEmbeddingService is tested for real (it is pure Python).
  - SentenceTransformerService is tested with a STUB model that has the same
    methods as the real one, so we verify OUR code around it (settings passed,
    output format, error messages). Whether the real model gives good answers is
    checked by you with:  python scripts/check_embeddings.py
"""
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from backend.config import Config
from backend.rag.chunker import chunk_document
from backend.rag.embeddings import (
    DEFAULT_MODEL,
    EmbeddingError,
    EmbeddingService,
    HashingEmbeddingService,
    SentenceTransformerService,
    _build,
    cosine_similarity,
    get_embedding_service,
)
from backend.rag.html_parser import parse_html_file

DOC_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "company.html"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def norm(vector) -> float:
    return math.sqrt(sum(x * x for x in vector))


@pytest.fixture(scope="module")
def chunks():
    return chunk_document(parse_html_file(DOC_PATH), max_words=150, overlap_words=20)


@pytest.fixture
def svc():
    return HashingEmbeddingService()


# =============================================================================
# cosine_similarity
# =============================================================================
def test_cosine_identical_orthogonal_opposite():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_ignores_vector_length():
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_cosine_with_zero_vector_is_zero_not_a_crash():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_cosine_rejects_different_lengths():
    with pytest.raises(ValueError, match="different lengths"):
        cosine_similarity([1, 2], [1, 2, 3])


# =============================================================================
# HashingEmbeddingService (real code, no download)
# =============================================================================
def test_hashing_vector_has_right_size_and_length_one(svc):
    [v] = svc.embed_documents(["Employees receive 24 paid leave days per year."])
    assert len(v) == svc.dimension == 384
    assert norm(v) == pytest.approx(1.0)
    assert all(isinstance(x, float) for x in v)


def test_hashing_is_deterministic(svc):
    text = "Salary is credited on the last working day."
    assert svc.embed_query(text) == svc.embed_query(text)


def test_hashing_is_stable_across_python_processes():
    """Python's built-in hash() changes on every run. Ours must not."""
    code = ("import json; from backend.rag.embeddings import HashingEmbeddingService as S; "
            "print(json.dumps(S().embed_query('paid leave days per year')))")
    outputs = []
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                                capture_output=True, text=True, check=True)
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]


def test_same_model_for_documents_and_queries(svc):
    """The project's golden rule: query and document vectors come from the same model."""
    text = "How many paid leave days are there?"
    assert svc.embed_query(text) == svc.embed_documents([text])[0]


def test_one_chunk_gives_one_embedding(svc, chunks):
    """Phase 5 test from the documentation: one chunk -> one embedding."""
    result = svc.embed_chunks([chunks[18]])
    assert len(result) == 1 and len(result[0]) == svc.dimension


def test_all_chunks_give_one_embedding_each_in_order(svc, chunks):
    vectors = svc.embed_chunks(chunks)
    assert len(vectors) == len(chunks) == 56
    assert vectors[10] == svc.embed_documents([chunks[10].text])[0]     # order preserved
    assert all(norm(v) == pytest.approx(1.0) for v in vectors)


def test_empty_list_gives_empty_list(svc):
    assert svc.embed_documents([]) == []


def test_same_text_has_similarity_one(svc):
    v = svc.embed_query("working hours")
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_case_punctuation_and_plurals_do_not_matter(svc):
    a = svc.embed_query("Paid LEAVES!")
    b = svc.embed_query("paid leave")
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_shared_words_score_higher_than_unrelated_text(svc):
    q = svc.embed_query("How many paid leave days do employees get?")
    leave = svc.embed_query("Employees receive 24 paid leave days per year.")
    salary = svc.embed_query("Salary is credited on the last working day of the month.")
    assert cosine_similarity(q, leave) > cosine_similarity(q, salary) + 0.3


def test_hashing_cannot_understand_meaning_only_words(svc):
    """Documents the LIMIT of the stand-in, which is why a real model is needed."""
    q = svc.embed_query("vacation")
    leave = svc.embed_query("leave")
    assert cosine_similarity(q, leave) == pytest.approx(0.0)


def test_text_made_only_of_stopwords_still_works(svc):
    assert norm(svc.embed_query("what is the")) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_text_is_rejected(svc, bad):
    with pytest.raises(ValueError, match="empty"):
        svc.embed_query(bad)
    with pytest.raises(ValueError, match="empty"):
        svc.embed_documents(["fine text", bad])


def test_text_without_letters_or_digits_is_rejected(svc):
    with pytest.raises(ValueError, match="letters or digits"):
        svc.embed_query("?!?!")


@pytest.mark.parametrize("bad", [None, 42, ["list"]])
def test_non_text_is_rejected(svc, bad):
    with pytest.raises(TypeError):
        svc.embed_query(bad)


def test_dimension_must_be_reasonable():
    with pytest.raises(ValueError):
        HashingEmbeddingService(dimension=2)


def test_obvious_questions_find_their_chunk_with_hashing(svc, chunks):
    """A small preview of retrieval (Step 8): the right chunk is in the top 3."""
    vectors = svc.embed_chunks(chunks)
    cases = {
        "What are the working hours?": "Working Hours and Attendance > Standard Working Hours",
        "When is my salary credited?": "Salary and Compensation > Pay Cycle",
        "How long is the notice period?": "HR Information > Notice Period",
        "How much is the health insurance cover?": "Benefits > Health and Insurance",
    }
    for question, expected in cases.items():
        q = svc.embed_query(question)
        top3 = sorted(range(len(chunks)), key=lambda i: cosine_similarity(q, vectors[i]),
                      reverse=True)[:3]
        headings = [chunks[i].text.split("\n", 1)[0] for i in top3]
        assert expected in headings, (question, headings)


# =============================================================================
# SentenceTransformerService with a STUB model
# =============================================================================
class StubModel:
    """Has the same methods the real SentenceTransformer has (and that we use)."""
    max_seq_length = 256

    def __init__(self):
        self.calls = []

        class _Tokenizer:
            def __call__(inner, texts, add_special_tokens, truncation):
                assert add_special_tokens is True and truncation is False
                return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}
        self.tokenizer = _Tokenizer()

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
        self.calls.append({"n": len(texts), "batch_size": batch_size,
                           "normalize": normalize_embeddings, "progress": show_progress_bar})
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def test_wrapper_passes_the_right_settings_to_the_model():
    stub = StubModel()
    service = SentenceTransformerService(model_name="my/model", model=stub, batch_size=16)
    service.embed_documents(["a b", "c d", "e f"])
    assert stub.calls == [{"n": 3, "batch_size": 16, "normalize": True, "progress": False}]


def test_wrapper_returns_plain_lists_of_floats():
    service = SentenceTransformerService(model=StubModel())
    vectors = service.embed_documents(["x y"])
    assert vectors == [[1.0, 0.0, 0.0, 0.0]]
    assert type(vectors) is list and type(vectors[0]) is list and type(vectors[0][0]) is float


def test_wrapper_sends_a_whole_batch_in_one_call():
    stub = StubModel()
    SentenceTransformerService(model=stub, batch_size=32).embed_documents(["t"] * 70)
    assert len(stub.calls) == 1 and stub.calls[0]["n"] == 70      # the library does the batching


def test_wrapper_query_uses_the_same_path_as_documents():
    stub = StubModel()
    service = SentenceTransformerService(model=stub)
    assert service.embed_query("hello there") == service.embed_documents(["hello there"])[0]


def test_wrapper_dimension_and_default_model_name(monkeypatch):
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "")
    service = SentenceTransformerService(model=StubModel())
    assert service.dimension == 4
    assert service.model_name == DEFAULT_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_wrapper_model_name_from_config(monkeypatch):
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "org/other-model")
    assert SentenceTransformerService(model=StubModel()).model_name == "org/other-model"


def test_wrapper_token_counts_and_limit():
    service = SentenceTransformerService(model=StubModel())
    assert service.max_tokens == 256
    assert service.token_counts(["one two three", "a"]) == [5, 3]


def test_wrapper_still_validates_input():
    service = SentenceTransformerService(model=StubModel())
    with pytest.raises(ValueError):
        service.embed_query("   ")


def test_missing_library_gives_install_instructions(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)    # makes `import` fail
    with pytest.raises(EmbeddingError, match="pip install sentence-transformers"):
        SentenceTransformerService(model_name="any/model")


def test_model_load_failure_gives_helpful_message(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")

    def boom(name):
        raise OSError("no internet")
    fake_module.SentenceTransformer = boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    with pytest.raises(EmbeddingError) as info:
        SentenceTransformerService(model_name="some/model")
    assert "some/model" in str(info.value) and "internet" in str(info.value)


def test_wrapper_loads_the_model_by_name(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")
    requested = []

    def factory(name):
        requested.append(name)
        return StubModel()
    fake_module.SentenceTransformer = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    SentenceTransformerService(model_name="org/model-x")
    assert requested == ["org/model-x"]


# =============================================================================
# Factory and interface
# =============================================================================
@pytest.fixture(autouse=False)
def clean_cache():
    _build.cache_clear()
    yield
    _build.cache_clear()


def test_factory_returns_hashing_service(monkeypatch, clean_cache):
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "hashing")
    assert isinstance(get_embedding_service(), HashingEmbeddingService)


def test_factory_builds_the_service_only_once(monkeypatch, clean_cache):
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "hashing")
    assert get_embedding_service() is get_embedding_service()


def test_factory_passes_model_name_to_real_service(monkeypatch, clean_cache):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = lambda name: StubModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "sentence-transformers")
    monkeypatch.setattr(Config, "EMBEDDING_MODEL", "org/model-y")
    service = get_embedding_service()
    assert isinstance(service, SentenceTransformerService)
    assert service.model_name == "org/model-y"


def test_factory_rejects_unknown_provider(monkeypatch, clean_cache):
    monkeypatch.setattr(Config, "EMBEDDING_PROVIDER", "magic")
    with pytest.raises(ValueError, match="sentence-transformers, hashing"):
        get_embedding_service()


def test_both_services_share_the_same_interface():
    assert issubclass(HashingEmbeddingService, EmbeddingService)
    assert issubclass(SentenceTransformerService, EmbeddingService)


def test_interface_cannot_be_used_directly():
    with pytest.raises(TypeError):
        EmbeddingService()
