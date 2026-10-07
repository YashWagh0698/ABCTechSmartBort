"""Step 8 tests: the retriever.

Uses a real temporary ChromaDB filled by the real ingestion code, with the
offline embedder. The retriever's job is to find candidates and report scores;
deciding whether they are good enough is NOT its job (that is Step 12).
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.config import Config
from backend.models.search import RetrievalResponse
from backend.rag.ingestion import ingest_document, reindex_document
from backend.rag.retriever import (IndexNotReadyError, ModelMismatchError, RetrievalError,
                                   Retriever, MAX_TOP_K)
from backend.services import Services

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def retriever(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    return Retriever(embedder, store)


# ---- the basic job ------------------------------------------------------------------
def test_retrieve_returns_ranked_chunks_with_scores(retriever):
    response = retriever.retrieve("What are the working hours?")
    assert isinstance(response, RetrievalResponse)
    assert len(response.results) == 5                                   # Config.TOP_K
    scores = [r.similarity for r in response.results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)
    assert response.best_similarity == scores[0]


def test_response_describes_where_the_answer_came_from(retriever):
    response = retriever.retrieve("What are the working hours?")
    assert response.embedding_model == "hashing-bag-of-words"
    assert response.document_version == "1.0"
    assert response.seconds >= 0
    best = response.results[0]
    assert best.heading == "Working Hours and Attendance > Standard Working Hours"
    assert (best.source, best.document_version) == ("company.html", "1.0")


@pytest.mark.parametrize("question, expected", [
    ("What are the working hours?", "Working Hours and Attendance > Standard Working Hours"),
    ("When is my salary credited?", "Salary and Compensation > Pay Cycle"),
    ("How long is the notice period?", "HR Information > Notice Period"),
    ("How much is the health insurance cover?", "Benefits > Health and Insurance"),
    ("How many paid leave days do employees get per year?", "Leave Policy > Annual Paid Leave"),
])
def test_known_questions_find_their_chunk_in_the_top_3(retriever, question, expected):
    """The documentation's retrieval test: known question -> expected chunk in the Top-K."""
    headings = [r.heading for r in retriever.retrieve(question, top_k=3).results]
    assert expected in headings


def test_same_question_gives_the_same_answer_twice(retriever):
    a = [(r.chunk_id, r.similarity) for r in retriever.retrieve("salary credited").results]
    b = [(r.chunk_id, r.similarity) for r in retriever.retrieve("salary credited").results]
    assert a == b


# ---- top_k ----------------------------------------------------------------------------------
def test_top_k_default_comes_from_config(doc_copy, embedder, store, monkeypatch):
    ingest_document(doc_copy, embedder, store)
    monkeypatch.setattr(Config, "TOP_K", 3)
    assert len(Retriever(embedder, store).retrieve("leave").results) == 3


def test_top_k_can_be_set_per_retriever_and_per_call(retriever, embedder, store):
    assert len(Retriever(embedder, store, top_k=2).retrieve("leave").results) == 2
    assert len(retriever.retrieve("leave", top_k=8).results) == 8
    assert retriever.retrieve("leave", top_k=8).top_k == 8


@pytest.mark.parametrize("bad", [0, -1, MAX_TOP_K + 1, True, "5", 2.5, [5]])
def test_bad_top_k_is_rejected_per_call(retriever, bad):
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("leave", top_k=bad)


def test_bad_default_top_k_is_rejected_at_creation(embedder, store):
    with pytest.raises(ValueError, match="top_k"):
        Retriever(embedder, store, top_k=0)


# ---- the question ------------------------------------------------------------------------------
def test_question_whitespace_is_cleaned(retriever):
    response = retriever.retrieve("   What   are\tthe\nworking hours?  ")
    assert response.question == "What are the working hours?"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_empty_question_is_rejected(retriever, bad):
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve(bad)


@pytest.mark.parametrize("bad", [None, 42, ["leave"]])
def test_non_text_question_is_rejected(retriever, bad):
    with pytest.raises(TypeError):
        retriever.retrieve(bad)


def test_overlong_question_is_rejected(retriever):
    with pytest.raises(ValueError, match="too long"):
        retriever.retrieve("a " * (Config.MAX_QUESTION_CHARS // 2 + 5))


def test_question_at_the_limit_is_accepted(retriever):
    assert retriever.retrieve("a" * Config.MAX_QUESTION_CHARS).results


# ---- the two safety checks ----------------------------------------------------------------------------
def test_empty_index_is_reported_as_not_ready_not_as_no_results(embedder, store):
    with pytest.raises(IndexNotReadyError, match="POST /api/ingest"):
        Retriever(embedder, store).retrieve("What are the working hours?")


def test_a_different_embedding_model_is_refused(doc_copy, embedder, other_embedder, store):
    ingest_document(doc_copy, embedder, store)
    with pytest.raises(ModelMismatchError) as info:
        Retriever(other_embedder, store).retrieve("What are the working hours?")
    message = str(info.value)
    assert "hashing-bag-of-words" in message and "other-model" in message and "reindex" in message.lower()


def test_an_index_that_does_not_say_which_model_built_it_is_refused(embedder, store, chunks):
    store.add_documents(chunks[:3], embedder.embed_chunks(chunks[:3]))        # no embedding_model metadata
    with pytest.raises(ModelMismatchError):
        Retriever(embedder, store).retrieve("working hours")


def test_after_reindexing_with_the_new_model_it_works_again(doc_copy, embedder, other_embedder, store):
    ingest_document(doc_copy, embedder, store)
    reindex_document(doc_copy, other_embedder, store)
    assert Retriever(other_embedder, store).retrieve("working hours").embedding_model == "other-model"
    with pytest.raises(ModelMismatchError):
        Retriever(embedder, store).retrieve("working hours")


def test_both_errors_are_retrieval_errors(embedder, store):
    assert issubclass(IndexNotReadyError, RetrievalError) and issubclass(ModelMismatchError, RetrievalError)


def test_the_check_notices_a_reindex_that_happens_later(doc_copy, embedder, other_embedder, store):
    retriever = Retriever(embedder, store)
    ingest_document(doc_copy, embedder, store)
    assert retriever.retrieve("working hours").results
    reindex_document(doc_copy, other_embedder, store)                  # index changes under a live retriever
    with pytest.raises(ModelMismatchError):
        retriever.retrieve("working hours")


# ---- output and logging ---------------------------------------------------------------------------------
def test_response_can_be_turned_into_json(retriever):
    data = retriever.retrieve("When is my salary credited?", top_k=3).to_dict()
    json.dumps(data)                                                    # must not raise
    assert data["top_k"] == 3 and len(data["results"]) == 3
    first = data["results"][0]
    assert first["rank"] == 1
    assert {"chunk_id", "heading", "similarity", "section", "subsection", "source",
            "document_version", "text"} <= set(first)
    assert [r["rank"] for r in data["results"]] == [1, 2, 3]


def test_an_empty_response_has_no_best_similarity():
    empty = RetrievalResponse("q", [], 5, "m", "1.0", 0.0)
    assert empty.best_similarity is None and empty.to_dict()["results"] == []


def test_the_log_records_sizes_and_scores_but_not_the_question_text(retriever, caplog):
    secret = "my private question about zebras"
    with caplog.at_level(logging.INFO, logger="backend.rag.retriever"):
        retriever.retrieve(secret)
    assert "zebras" not in caplog.text
    assert f"{len(secret)} characters" in caplog.text


# ---- wiring ---------------------------------------------------------------------------------------------
def test_services_builds_the_retriever_lazily_and_once(embedder, store):
    services = Services(embedding_service=embedder, vector_store=store)
    assert services._retriever is None
    first = services.retriever
    assert first is services.retriever
    assert first.embedding_service is embedder and first.vector_store is store


# ---- the command-line tool -------------------------------------------------------------------------------
def run_script(name, args, chroma_path):
    env = dict(os.environ, CHROMA_PATH=str(chroma_path), EMBEDDING_PROVIDER="hashing")
    return subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / name), *args],
                          cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=120)


def test_search_script_explains_an_empty_index(tmp_path):
    result = run_script("search.py", ["What are the working hours?"], tmp_path / "empty_db")
    assert result.returncode != 0
    assert "index is empty" in (result.stdout + result.stderr)


def test_search_script_prints_ranked_results(tmp_path):
    run_script("ingest_company.py", [], tmp_path / "db")
    result = run_script("search.py", ["What are the working hours?", "--top-k", "3"], tmp_path / "db")
    assert result.returncode == 0, result.stderr
    assert "Standard Working Hours" in result.stdout and "similarity" in result.stdout
