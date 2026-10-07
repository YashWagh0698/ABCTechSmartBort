"""Step 6 tests: the ChromaDB vector store.

These use the REAL ChromaDB (in a temporary folder) and the offline embedder.
Several tests protect against traps discovered while probing ChromaDB:
silently ignored duplicate ids, None in metadata, vector-size mismatches.
"""
import pytest

from backend.config import Config
from backend.models.chunk import Chunk
from backend.models.search import SearchResult
from backend.rag.embeddings import cosine_similarity
from backend.rag.vector_store import VectorStore, VectorStoreError, _build_store, get_vector_store


def tiny_chunk(chunk_id: str, text: str = "some text", position: int = 0) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=f"Heading\n{text}", section="Heading", subsection=None,
                 source="x.html", document_version="1", position=position, section_id="heading",
                 word_count=3)


@pytest.fixture
def filled(store, embedder, chunks):
    store.add_documents(chunks, embedder.embed_chunks(chunks))
    return store


# ---- empty store ------------------------------------------------------------
def test_new_store_is_empty(store):
    assert store.count() == 0
    assert store.info().is_empty
    assert store.search([0.1] * 8) == []
    assert store.list_chunks() == []
    assert store.get_chunk("nope") is None


# ---- storing and verifying that records exist (Phase 6 of the documentation) -
def test_all_chunks_are_stored(filled, chunks):
    assert filled.count() == len(chunks) == 56


def test_a_stored_record_can_be_read_back(filled):
    record = filled.get_chunk("leave-policy-001")
    assert record.text.startswith("Leave Policy > Annual Paid Leave\n")
    assert "24 paid leave days per year" in record.text
    assert (record.section, record.subsection) == ("Leave Policy", "Annual Paid Leave")
    assert record.source == "company.html" and record.document_version == "1.0"


def test_subsection_none_survives_the_round_trip(filled):
    record = filled.get_chunk("departments-001")        # an <h2> with no <h3>
    assert record.subsection is None
    assert record.heading == "Departments"
    assert record.metadata["subsection"] == ""           # stored as "" because ChromaDB rejects None


def test_metadata_keeps_numbers_as_numbers(filled):
    meta = filled.get_chunk("faq-002").metadata
    assert isinstance(meta["position"], int) and isinstance(meta["word_count"], int)
    assert (meta["part"], meta["parts"]) == (2, 3)


def test_list_chunks_returns_document_order(filled):
    assert [c.position for c in filled.list_chunks()] == list(range(56))


def test_extra_metadata_is_added_to_every_record(store, embedder, chunks):
    store.add_documents(chunks, embedder.embed_chunks(chunks),
                        extra_metadata={"fingerprint": "abc123", "embedding_model": "m", "indexed_at": "t"})
    assert all(c.metadata["fingerprint"] == "abc123" for c in store.list_chunks())
    info = store.info()
    assert (info.count, info.fingerprint, info.embedding_model) == (56, "abc123", "m")
    assert (info.source, info.document_version) == ("company.html", "1.0")


def test_extra_metadata_must_be_simple_values(store):
    with pytest.raises(ValueError, match="must be str, int, float or bool"):
        store.add_documents([tiny_chunk("a-001")], [[0.1] * 8], extra_metadata={"bad": None})


# ---- searching ---------------------------------------------------------------
def test_search_results_are_plain_objects_with_a_heading(filled, embedder):
    hits = filled.search(embedder.embed_query("working hours"), top_k=3)
    assert all(isinstance(h, SearchResult) for h in hits)
    assert hits[0].heading == "Working Hours and Attendance > Standard Working Hours"


def test_search_scores_equal_true_cosine_similarity(filled, embedder, chunks):
    """Proves ChromaDB is configured for COSINE (a wrong metric would give different numbers)."""
    query = embedder.embed_query("How many paid leave days do employees get?")
    vectors = dict(zip([c.chunk_id for c in chunks], embedder.embed_chunks(chunks)))
    hits = filled.search(query, top_k=8)
    for hit in hits:
        assert hit.similarity == pytest.approx(cosine_similarity(query, vectors[hit.chunk_id]), abs=1e-4)
        assert hit.distance == pytest.approx(1 - hit.similarity, abs=1e-6)


def test_search_is_sorted_best_first(filled, embedder):
    scores = [h.similarity for h in filled.search(embedder.embed_query("salary credited"), top_k=10)]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("question, expected", [
    ("What are the working hours?", "Working Hours and Attendance > Standard Working Hours"),
    ("When is my salary credited?", "Salary and Compensation > Pay Cycle"),
    ("How long is the notice period?", "HR Information > Notice Period"),
    ("How much is the health insurance cover?", "Benefits > Health and Insurance"),
])
def test_obvious_questions_find_their_chunk(filled, embedder, question, expected):
    headings = [h.heading for h in filled.search(embedder.embed_query(question), top_k=3)]
    assert expected in headings


def test_searching_with_a_chunks_own_text_returns_that_chunk_first(filled, embedder, chunks):
    target = chunks[20]
    best = filled.search(embedder.embed_query(target.text), top_k=1)[0]
    assert best.chunk_id == target.chunk_id
    assert best.similarity == pytest.approx(1.0, abs=1e-4)


def test_top_k_limits_results_and_is_clamped_to_the_index_size(filled, embedder):
    q = embedder.embed_query("leave")
    assert len(filled.search(q, top_k=2)) == 2
    assert len(filled.search(q, top_k=500)) == 56


def test_top_k_default_comes_from_config(filled, embedder, monkeypatch):
    monkeypatch.setattr(Config, "TOP_K", 4)
    assert len(filled.search(embedder.embed_query("leave"))) == 4


@pytest.mark.parametrize("bad", [0, -1])
def test_top_k_must_be_positive(filled, embedder, bad):
    with pytest.raises(ValueError, match="top_k"):
        filled.search(embedder.embed_query("leave"), top_k=bad)


# ---- persistence and deletion ------------------------------------------------
def test_data_survives_a_new_store_object_on_the_same_folder(filled):
    reopened = VectorStore(filled.path, filled.collection_name)
    assert reopened.count() == 56
    assert reopened.get_chunk("leave-policy-001") is not None


def test_delete_collection_empties_the_store_and_it_can_be_refilled(filled, embedder, chunks):
    filled.delete_collection()
    assert filled.count() == 0
    filled.add_documents(chunks[:3], embedder.embed_chunks(chunks[:3]))
    assert filled.count() == 3


def test_deleting_a_collection_that_does_not_exist_is_fine(store):
    store.delete_collection()
    store.delete_collection()
    assert store.count() == 0


def test_collections_are_isolated_from_each_other(tmp_path):
    a, b = VectorStore(tmp_path / "db", "coll_one"), VectorStore(tmp_path / "db", "coll_two")
    a.add_documents([tiny_chunk("a-001")], [[0.1] * 8])
    assert (a.count(), b.count()) == (1, 0)


def test_many_chunks_are_stored_in_batches(store):
    many = [tiny_chunk(f"c-{i:04d}", f"text {i}", position=i) for i in range(1200)]    # more than one batch
    vectors = [[(i % 7) + 1.0, (i % 5) + 1.0, 1.0, 0.5] for i in range(1200)]
    assert store.add_documents(many, vectors) == 1200
    assert store.count() == 1200


# ---- the traps discovered while probing ChromaDB -------------------------------
def test_adding_the_same_chunks_twice_is_refused_not_silently_ignored(filled, embedder, chunks):
    with pytest.raises(VectorStoreError, match="already stored"):
        filled.add_documents(chunks[:2], embedder.embed_chunks(chunks[:2]))
    assert filled.count() == 56


def test_a_changed_text_with_an_old_id_cannot_slip_through(store):
    store.add_documents([tiny_chunk("same-001", "OLD text")], [[0.1] * 8])
    with pytest.raises(VectorStoreError):
        store.add_documents([tiny_chunk("same-001", "NEW text")], [[0.1] * 8])
    assert "OLD text" in store.get_chunk("same-001").text          # still the original


def test_a_different_vector_size_gives_an_instruction_not_a_crash(store):
    store.add_documents([tiny_chunk("a-001")], [[0.1] * 8])
    with pytest.raises(VectorStoreError, match="reindex"):
        store.add_documents([tiny_chunk("b-001")], [[0.1] * 4])
    with pytest.raises(VectorStoreError, match="reindex"):
        store.search([0.1] * 4)


# ---- argument validation ---------------------------------------------------------
@pytest.mark.parametrize("chunk_list, vectors, message", [
    ([], [], "no chunks"),
    ([tiny_chunk("a-001")], [], "1 chunks but 0 embeddings"),
    ([tiny_chunk("a-001"), tiny_chunk("a-001")], [[1.0] * 4, [1.0] * 4], "not unique"),
    ([tiny_chunk("a-001"), tiny_chunk("a-002")], [[1.0] * 4, [1.0] * 3], "different lengths"),
    ([tiny_chunk("a-001")], [[1.0, float("nan"), 0.0, 0.0]], "NaN"),
])
def test_bad_input_is_rejected_before_touching_the_database(store, chunk_list, vectors, message):
    with pytest.raises(ValueError, match=message):
        store.add_documents(chunk_list, vectors)
    assert store.count() == 0


@pytest.mark.parametrize("name", ["ab", "bad name", "-abc", "abc-", "naïve"])
def test_invalid_collection_names_are_rejected_early(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid collection name"):
        VectorStore(tmp_path, name)


def test_factory_returns_one_shared_store(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "CHROMA_PATH", tmp_path / "factory_db")
    monkeypatch.setattr(Config, "CHROMA_COLLECTION", "factory_coll")
    _build_store.cache_clear()
    try:
        assert get_vector_store() is get_vector_store()
    finally:
        _build_store.cache_clear()
