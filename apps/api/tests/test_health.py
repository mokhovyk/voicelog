import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app

PRODUCTION = {
    "environment": "production",
    "database_url": "postgresql+asyncpg://user:pass@db.example/voicelog",
    "openai_api_key": "sk-test",
}


def test_health() -> None:
    client = TestClient(create_app(Settings(environment="test")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_database_reachable(database_url: str) -> None:
    with TestClient(create_app(Settings(environment="test", database_url=database_url))) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200


def test_not_ready_when_database_unreachable() -> None:
    settings = Settings(
        environment="test", database_url="postgresql+asyncpg://u:p@127.0.0.1:1/nothing"
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503


def test_cors_allows_configured_origin() -> None:
    settings = Settings(environment="test", cors_origins=["http://web.test"])
    client = TestClient(create_app(settings))

    response = client.get("/health", headers={"Origin": "http://web.test"})

    assert response.headers["access-control-allow-origin"] == "http://web.test"


def test_docs_disabled_in_production() -> None:
    client = TestClient(create_app(Settings(**PRODUCTION)))

    assert client.get("/docs").status_code == 404


@pytest.mark.parametrize("unset", ["database_url", "openai_api_key"])
def test_production_requires_real_settings(unset: str) -> None:
    fields = {key: value for key, value in PRODUCTION.items() if key != unset}

    with pytest.raises(ValidationError, match=unset.upper()):
        Settings(_env_file=None, **fields)
