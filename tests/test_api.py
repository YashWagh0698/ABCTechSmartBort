from backend.app import create_app


def test_health_returns_ok():
    client = create_app().test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_unknown_route_returns_json_404():
    client = create_app().test_client()
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.get_json()
