import asyncio
import os
from collections.abc import AsyncIterator, Sequence

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text as sa_text
from sqlalchemy.engine import make_url

from app.ai.answers import Answer, SourceNote
from app.ai.embeddings import EMBEDDING_DIMENSIONS
from app.ai.extraction import NoteMetadata
from app.ai.services import AIServices
from app.core.config import Settings
from app.main import create_app

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://voicelog:voicelog_local@127.0.0.1:5432/voicelog_test",
)


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


async def _recreate_database(url: str) -> None:
    target = make_url(url)
    admin_dsn = target.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    try:
        conn = await asyncpg.connect(admin_dsn)
    except (OSError, asyncpg.PostgresError) as exc:
        raise RuntimeError(
            f"Cannot reach Postgres for tests at {target.host}:{target.port}. "
            "Start it from the repo root with `docker compose up -d --wait db`, "
            "or set TEST_DATABASE_URL."
        ) from exc
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{target.database}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{target.database}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def database_url() -> str:
    """A fresh test database, migrated with Alembic (so migrations are tested too)."""
    asyncio.run(_recreate_database(TEST_DATABASE_URL))
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL.replace("%", "%%"))
    command.upgrade(config, "head")
    return TEST_DATABASE_URL


class FakeExtractor:
    def __init__(self, result: NoteMetadata | Exception) -> None:
        self.result = result
        self.calls: list[str] = []

    async def extract(self, transcript: str) -> NoteMetadata:
        self.calls.append(transcript)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeEmbedder:
    model = "fake-embedding"

    def __init__(self, result: list[float] | Exception) -> None:
        self.result = result

    async def embed(self, text: str) -> list[float]:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeTranscriber:
    def __init__(self, result: str | Exception) -> None:
        self.result = result
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, audio: bytes, audio_format: str) -> str:
        self.calls.append((audio, audio_format))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeAnswerer:
    def __init__(self, result: Answer | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, list[SourceNote]]] = []

    async def answer(self, question: str, notes: Sequence[SourceNote]) -> Answer:
        self.calls.append((question, list(notes)))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class KeywordEmbedder:
    """Embeds text as keyword counts, one dimension per keyword, so similarity is
    predictable: texts sharing more keywords are closer."""

    model = "keyword-embedding"

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords

    async def embed(self, text: str) -> list[float]:
        words = text.lower().split()
        vector = [float(words.count(keyword)) for keyword in self.keywords]
        vector += [0.0] * (EMBEDDING_DIMENSIONS - len(vector))
        vector[-1] = 0.1  # never the zero vector, whose cosine distance is undefined
        return vector


@pytest.fixture
def ai() -> AIServices:
    """No AI by default, so tests never call OpenAI. Override in a test module."""
    return AIServices()


@pytest.fixture
async def client(database_url: str, ai: AIServices) -> AsyncIterator[AsyncClient]:
    app = create_app(Settings(environment="test", database_url=database_url), ai=ai)
    async with app.router.lifespan_context(app):
        async with app.state.sessionmaker() as session:
            await session.execute(sa_text("TRUNCATE notes"))
            await session.commit()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
