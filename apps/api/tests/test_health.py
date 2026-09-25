from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_health() -> None:
    client = TestClient(create_app(Settings(environment="test")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_configured_origin() -> None:
    settings = Settings(environment="test", cors_origins=["http://web.test"])
    client = TestClient(create_app(settings))

    response = client.get("/health", headers={"Origin": "http://web.test"})

    assert response.headers["access-control-allow-origin"] == "http://web.test"


def test_docs_disabled_in_production() -> None:
    client = TestClient(create_app(Settings(environment="production")))

    assert client.get("/docs").status_code == 404
