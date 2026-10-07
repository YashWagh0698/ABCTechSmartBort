"""Step 7 tests: the ingestion pipeline (parse -> chunk -> embed -> store).

The important question for every test: "can the index end up WRONG or HALF-BUILT?"
"""
import re
import threading

import pytest

from backend.config import Config
from backend.rag import ingestion
from backend.rag.embeddings import EmbeddingError, HashingEmbeddingService
from backend.rag.html_parser import ParseError
from backend.rag.ingestion import (IndexBusyError, IndexConflictError, compute_fingerprint,
                                   index_is_current, ingest_document, reindex_document)
from backend.rag.vector_store import VectorStoreError


def edit(path, old, new):
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


# ---- the fingerprint -----------------------------------------------------------
def test_fingerprint_is_stable_and_short():
    a = compute_fingerprint(b"<html>x</html>", "model", 150, 20)
    assert a == compute_fingerprint(b"<html>x</html>", "model", 150, 20)
    assert re.fullmatch(r"[0-9a-f]{16}", a)


@pytest.mark.parametrize("args", [
    (b"<html>y</html>", "model", 150, 20),     # document changed
    (b"<html>x</html>", "model2", 150, 20),    # embedding model changed
    (b"<html>x</html>", "model", 100, 20),     # chunk size changed
    (b"<html>x</html>", "model", 150, 10),     # overlap changed
])
def test_fingerprint_changes_when_anything_it_covers_changes(args):
    assert compute_fingerprint(b"<html>x</html>", "model", 150, 20) != compute_fingerprint(*args)


def test_fingerprint_ignores_windows_line_endings():
    assert compute_fingerprint(b"a\nb\n", "m", 1, 0) == compute_fingerprint(b"a\r\nb\r\n", "m", 1, 0)


# ---- ingest ------------------------------------------------------------------------
def test_ingest_builds_the_index(doc_copy, embedder, store):
    result = ingest_document(doc_copy, embedder, store)
    assert result.status == "success"
    assert (result.chunks_created, result.chunks_in_index, result.sections) == (56, 56, 52)
    assert (result.source, result.document_version) == ("company.html", "1.0")
    assert (result.embedding_model, result.dimension) == ("hashing-bag-of-words", 384)
    assert store.count() == 56 and not result.replaced_previous


def test_every_record_carries_fingerprint_model_and_time(doc_copy, embedder, store):
    result = ingest_document(doc_copy, embedder, store)
    for record in store.list_chunks():
        assert record.metadata["fingerprint"] == result.fingerprint
        assert record.metadata["embedding_model"] == "hashing-bag-of-words"
        assert record.metadata["indexed_at"].endswith("+00:00")


def test_ingesting_twice_does_nothing_the_second_time(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    again = ingest_document(doc_copy, embedder, store)
    assert again.status == "already_indexed"
    assert (again.chunks_created, again.chunks_in_index) == (0, 56)     # no duplicates


def test_ingest_refuses_when_the_document_changed(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    edit(doc_copy, "24 paid leave days per year", "30 paid leave days per year")
    edit(doc_copy, 'content="1.0"', 'content="1.1"')
    with pytest.raises(IndexConflictError) as info:
        ingest_document(doc_copy, embedder, store)
    message = str(info.value)
    assert "reindex" in message and "1.0" in message and "1.1" in message
    assert "24 paid leave days per year" in store.get_chunk("leave-policy-001").text     # old index untouched


def test_ingest_refuses_when_the_text_changed_but_the_version_did_not(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    edit(doc_copy, "24 paid leave days per year", "30 paid leave days per year")
    with pytest.raises(IndexConflictError, match="text or the chunk settings changed"):
        ingest_document(doc_copy, embedder, store)


def test_ingest_refuses_when_the_embedding_model_changed(doc_copy, embedder, other_embedder, store):
    ingest_document(doc_copy, embedder, store)
    with pytest.raises(IndexConflictError, match="other-model"):
        ingest_document(doc_copy, other_embedder, store)


def test_ingest_refuses_when_chunk_settings_changed(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    with pytest.raises(IndexConflictError):
        ingest_document(doc_copy, embedder, store, max_words=100, overlap_words=10)


def test_ingest_detects_an_incomplete_index(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    store.get_collection().delete(ids=["faq-003"])                  # simulate a crash that lost a record
    with pytest.raises(IndexConflictError, match="incomplete"):
        ingest_document(doc_copy, embedder, store)


def test_ingest_with_other_chunk_settings_on_an_empty_index(doc_copy, embedder, store):
    result = ingest_document(doc_copy, embedder, store, max_words=60, overlap_words=10)
    assert result.chunks_created > 56


# ---- reindex ---------------------------------------------------------------------------
def test_reindex_works_on_an_empty_index(doc_copy, embedder, store):
    result = reindex_document(doc_copy, embedder, store)
    assert (result.status, result.chunks_created, result.replaced_previous) == ("success", 56, False)
    assert result.previous_version is None


def test_reindex_replaces_old_content_with_new(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    edit(doc_copy, "24 paid leave days per year", "30 paid leave days per year")
    edit(doc_copy, 'content="1.0"', 'content="1.1"')
    result = reindex_document(doc_copy, embedder, store)
    assert (result.replaced_previous, result.previous_version, result.document_version) == (True, "1.0", "1.1")
    text = store.get_chunk("leave-policy-001").text
    assert "30 paid leave days per year" in text and "24 paid leave days" not in text
    assert store.info().document_version == "1.1"
    assert store.count() == 56                                       # replaced, not added to


def test_reindex_leaves_no_stale_chunks_of_deleted_sections(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    assert store.get_chunk("company-rules-001") is not None
    html = doc_copy.read_text(encoding="utf-8")
    doc_copy.write_text(re.sub(r'<section id="company-rules">.*?</section>', "", html, flags=re.S), encoding="utf-8")
    reindex_document(doc_copy, embedder, store)
    assert store.get_chunk("company-rules-001") is None
    assert store.count() == 55


def test_reindex_after_a_model_with_different_vector_size(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)                       # 384 numbers per chunk
    smaller = HashingEmbeddingService(dimension=64)
    result = reindex_document(doc_copy, smaller, store)
    assert (result.dimension, store.count()) == (64, 56)
    assert len(store.search(smaller.embed_query("working hours"), top_k=1)) == 1


def test_a_failing_embedding_model_keeps_the_old_index_on_reindex(doc_copy, embedder, failing_embedder, store):
    ingest_document(doc_copy, embedder, store)
    before = store.info()
    with pytest.raises(EmbeddingError):
        reindex_document(doc_copy, failing_embedder, store)
    assert store.count() == 56
    assert store.info().fingerprint == before.fingerprint
    assert store.get_chunk("leave-policy-001") is not None


def test_a_broken_document_keeps_the_old_index_on_reindex(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    doc_copy.write_text("<html><body><main></main></body></html>", encoding="utf-8")
    with pytest.raises(ParseError):
        reindex_document(doc_copy, embedder, store)
    assert store.count() == 56


def test_a_failing_embedding_model_leaves_a_new_index_empty(doc_copy, failing_embedder, store):
    with pytest.raises(EmbeddingError):
        ingest_document(doc_copy, failing_embedder, store)
    assert store.count() == 0


def test_an_interrupted_write_does_not_leave_a_half_built_index(doc_copy, embedder, store, monkeypatch):
    """A realistic crash: 10 records ARE written, then the write fails."""
    original = store.add_documents

    def write_some_then_crash(chunks, embeddings, extra_metadata=None):
        original(chunks[:10], embeddings[:10], extra_metadata)
        assert store.count() == 10                       # the half-built index really exists
        raise VectorStoreError("disk full")

    monkeypatch.setattr(store, "add_documents", write_some_then_crash)
    with pytest.raises(VectorStoreError, match="disk full"):
        ingest_document(doc_copy, embedder, store)
    assert store.count() == 0                            # ...and it was removed again


def test_an_interrupted_reindex_write_does_not_leave_a_half_built_index(doc_copy, embedder, store, monkeypatch):
    ingest_document(doc_copy, embedder, store)
    original = store.add_documents

    def write_some_then_crash(chunks, embeddings, extra_metadata=None):
        original(chunks[:10], embeddings[:10], extra_metadata)
        raise VectorStoreError("disk full")

    monkeypatch.setattr(store, "add_documents", write_some_then_crash)
    with pytest.raises(VectorStoreError):
        reindex_document(doc_copy, embedder, store)
    assert store.count() == 0                            # empty and honest, not 10 of 56 pretending to be complete
    monkeypatch.undo()
    assert ingest_document(doc_copy, embedder, store).status == "success"      # and recoverable


# ---- bad documents ---------------------------------------------------------------------------
def test_missing_document(tmp_path, embedder, store):
    with pytest.raises(ParseError, match="not found"):
        ingest_document(tmp_path / "nope.html", embedder, store)
    assert store.count() == 0


def test_document_that_is_not_utf8(tmp_path, embedder, store):
    bad = tmp_path / "bad.html"
    bad.write_bytes(b"<html>\xff\xfe\x00</html>")
    with pytest.raises(ParseError, match="UTF-8"):
        ingest_document(bad, embedder, store)


# ---- only one at a time ------------------------------------------------------------------------
def test_a_second_indexing_run_is_refused_while_one_is_running(doc_copy, embedder, store):
    assert ingestion._index_lock.acquire(blocking=False)
    try:
        with pytest.raises(IndexBusyError, match="already in progress"):
            ingest_document(doc_copy, embedder, store)
        with pytest.raises(IndexBusyError):
            reindex_document(doc_copy, embedder, store)
    finally:
        ingestion._index_lock.release()


def test_the_lock_is_released_after_a_failure(doc_copy, failing_embedder, embedder, store):
    with pytest.raises(EmbeddingError):
        ingest_document(doc_copy, failing_embedder, store)
    assert ingest_document(doc_copy, embedder, store).status == "success"      # not stuck


def test_two_simultaneous_runs_never_corrupt_the_index(doc_copy, embedder, store):
    outcomes = []

    def run():
        try:
            outcomes.append(ingest_document(doc_copy, embedder, store).status)
        except IndexBusyError:
            outcomes.append("busy")

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count() == 56
    assert outcomes.count("success") == 1 and set(outcomes) <= {"success", "busy", "already_indexed"}


# ---- is the index up to date? -------------------------------------------------------------------
def test_index_is_current_tracks_the_document(doc_copy, embedder, store):
    assert index_is_current(doc_copy, store.info()) is None                  # nothing indexed yet
    ingest_document(doc_copy, embedder, store)
    assert index_is_current(doc_copy, store.info()) is True
    edit(doc_copy, "24 paid leave days per year", "30 paid leave days per year")
    assert index_is_current(doc_copy, store.info()) is False


def test_index_is_current_is_unknown_when_the_file_is_gone(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    doc_copy.unlink()
    assert index_is_current(doc_copy, store.info()) is None


def test_index_is_current_notices_changed_chunk_settings(doc_copy, embedder, store, monkeypatch):
    ingest_document(doc_copy, embedder, store)
    monkeypatch.setattr(Config, "CHUNK_MAX_WORDS", 100)
    assert index_is_current(doc_copy, store.info()) is False


# ---- end to end ----------------------------------------------------------------------------------
def test_after_ingest_a_question_finds_its_chunk(doc_copy, embedder, store):
    ingest_document(doc_copy, embedder, store)
    best = store.search(embedder.embed_query("When is my salary credited?"), top_k=3)
    assert "Salary and Compensation > Pay Cycle" in [h.heading for h in best]


def test_result_can_be_turned_into_a_dictionary(doc_copy, embedder, store):
    data = ingest_document(doc_copy, embedder, store).to_dict()
    assert {"status", "chunks_created", "chunks_in_index", "document_version", "embedding_model",
            "fingerprint", "seconds", "replaced_previous"} <= set(data)
