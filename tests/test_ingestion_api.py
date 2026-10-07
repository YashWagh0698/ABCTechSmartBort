"""Step 7 tests: the HTTP endpoints.

    POST /api/ingest     POST /api/reindex     GET /api/sources

The app is created with fake services (a temporary database and the offline
embedder), and Flask's test client plays the role of curl / Postman.
"""
import pytest

from backend.app import create_app
from backend.config import Config
from backend.rag import ingestion
from backend.services import Services


@pytest.fixture
def client(embedder, store, doc_copy, monkeypatch):
    monkeypatch.setattr(Config, "COMPANY_DOCUMENT_PATH", doc_copy)
    app = create_app(Services(embedding_service=embedder, vector_store=store))
    return app.test_client()


def edit(path, old, new):
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


# ---- POST /api/ingest ----------------------------------------------------------------
def test_ingest_returns_success_and_the_chunk_count(client):
    response = client.post("/api/ingest")
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["chunks_created"] == 56            # the field named in the project documentation
    assert body["document_version"] == "1.0" and body["sections"] == 52
    assert response.mimetype == "application/json"


def test_ingest_twice_reports_already_indexed(client):
    client.post("/api/ingest")
    body = client.post("/api/ingest").get_json()
    assert (body["status"], body["chunks_created"], body["chunks_in_index"]) == ("already_indexed", 0, 56)


def test_ingest_ignores_a_request_body(client):
    response = client.post("/api/ingest", json={"anything": "goes"})
    assert response.status_code == 200


def test_oversized_request_body_is_rejected(client):
    response = client.post("/api/ingest", data="x" * 20_000, content_type="application/json")
    assert response.status_code == 413
    assert "error" in response.get_json()


def test_ingest_after_the_document_changed_gives_409_and_explains(client, doc_copy):
    client.post("/api/ingest")
    edit(doc_copy, "24 paid leave days per year", "30 paid leave days per year")
    response = client.post("/api/ingest")
    body = response.get_json()
    assert response.status_code == 409
    assert body["status"] == "conflict" and "/api/reindex" in body["error"]


# ---- POST /api/reindex ---------------------------------------------------------------
def test_reindex_replaces_the_index(client, doc_copy):
    client.post("/api/ingest")
    edit(doc_copy, "24 paid leave days per year", "30 paid leave days per year")
    edit(doc_copy, 'content="1.0"', 'content="1.1"')
    response = client.post("/api/reindex")
    body = response.get_json()
    assert response.status_code == 200
    assert (body["status"], body["replaced_previous"], body["previous_version"]) == ("success", True, "1.0")
    assert client.get("/api/sources").get_json()["version"] == "1.1"


def test_reindex_works_without_a_previous_index(client):
    body = client.post("/api/reindex").get_json()
    assert (body["status"], body["chunks_created"], body["replaced_previous"]) == ("success", 56, False)


# ---- GET /api/sources ----------------------------------------------------------------
def test_sources_before_indexing(client):
    response = client.get("/api/sources")
    body = response.get_json()
    assert response.status_code == 200
    assert (body["indexed"], body["chunks"], body["document"], body["version"]) == (False, 0, None, None)
    assert body["up_to_date"] is None


def test_sources_after_indexing(client):
    client.post("/api/ingest")
    body = client.get("/api/sources").get_json()
    assert (body["indexed"], body["document"], body["version"], body["chunks"]) == (True, "company.html", "1.0", 56)
    assert body["embedding_model"] == "hashing-bag-of-words" and body["up_to_date"] is True


def test_sources_tells_you_when_a_reindex_is_needed(client, doc_copy):
    client.post("/api/ingest")
    edit(doc_copy, "24 paid leave days per year", "30 paid leave days per year")
    assert client.get("/api/sources").get_json()["up_to_date"] is False


# ---- wrong usage ---------------------------------------------------------------------
@pytest.mark.parametrize("method, url", [("get", "/api/ingest"), ("get", "/api/reindex"),
                                         ("post", "/api/sources"), ("delete", "/api/ingest")])
def test_wrong_method_gives_json_405(client, method, url):
    response = getattr(client, method)(url)
    assert response.status_code == 405
    assert response.get_json() == {"error": "Method not allowed for this URL"}


# ---- failures: clean JSON, no internal details ---------------------------------------
def test_busy_gives_409(client):
    assert ingestion._index_lock.acquire(blocking=False)
    try:
        response = client.post("/api/ingest")
    finally:
        ingestion._index_lock.release()
    assert response.status_code == 409 and response.get_json()["status"] == "busy"


def test_missing_document_gives_500_without_leaking_the_path(client, doc_copy):
    doc_copy.unlink()
    response = client.post("/api/ingest")
    text = response.get_data(as_text=True)
    assert response.status_code == 500
    assert response.get_json()["status"] == "error"
    assert str(doc_copy.parent) not in text and "Traceback" not in text


def test_embedding_model_failure_gives_503(embedder, failing_embedder, store, doc_copy, monkeypatch):
    monkeypatch.setattr(Config, "COMPANY_DOCUMENT_PATH", doc_copy)
    client = create_app(Services(embedding_service=failing_embedder, vector_store=store)).test_client()
    response = client.post("/api/ingest")
    assert response.status_code == 503
    assert response.get_json()["status"] == "error"
    assert store.count() == 0


def test_database_failure_gives_500_without_details(embedder, store, doc_copy, monkeypatch):
    from backend.rag.vector_store import VectorStoreError

    def crash(*a, **k):
        raise VectorStoreError("secret internal detail /var/db/file")
    monkeypatch.setattr(Config, "COMPANY_DOCUMENT_PATH", doc_copy)
    monkeypatch.setattr(store, "info", crash)
    client = create_app(Services(embedding_service=embedder, vector_store=store)).test_client()
    for response in (client.post("/api/ingest"), client.get("/api/sources")):
        assert response.status_code == 500
        assert "secret internal detail" not in response.get_data(as_text=True)


# ---- the rest of the app still works ---------------------------------------------------
def test_health_does_not_load_the_model_or_database():
    """Services are lazy: creating the app and calling health must not build anything."""
    class Exploding(Services):
        @property
        def embedding_service(self):
            raise AssertionError("model must not be loaded for /api/health")

        @property
        def vector_store(self):
            raise AssertionError("database must not be opened for /api/health")

    assert create_app(Exploding()).test_client().get("/api/health").get_json() == {"status": "ok"}
