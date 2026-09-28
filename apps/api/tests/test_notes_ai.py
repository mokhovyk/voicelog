import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.ai.extraction import NoteMetadata
from app.ai.services import AIServices
from app.notes.models import EMBEDDING_DIMENSIONS, Note
from app.notes.schemas import ActionItem
from tests.conftest import FakeEmbedder, FakeExtractor

pytestmark = pytest.mark.anyio

METADATA = NoteMetadata(
    summary="Release sync with Sarah; login bug due Friday.",
    category="Work",
    tags=[" Release ", "release", "Login-Bug", "x" * 60, "", "a", "b", "c"],
    action_items=[ActionItem(text="Fix the login bug by Friday")],
)
EMBEDDING = [0.1] * EMBEDDING_DIMENSIONS


async def stored_embedding(database_url: str, note_id: str) -> list[float] | None:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            value = await conn.scalar(select(Note.embedding).where(Note.id == note_id))
    finally:
        await engine.dispose()
    return None if value is None else [float(x) for x in value]


class TestWithAI:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(extractor=FakeExtractor(METADATA), embedder=FakeEmbedder(EMBEDDING))

    async def test_fills_metadata_and_stores_embedding(
        self, client: AsyncClient, ai: AIServices, database_url: str
    ) -> None:
        response = await client.post("/api/v1/notes", json={"raw_transcript": "Sync with Sarah"})

        note = response.json()
        assert response.status_code == 201
        assert note["summary"] == METADATA.summary
        assert note["category"] == "Work"
        assert note["tags"] == ["release", "login-bug", "x" * 50, "a", "b"]
        assert note["action_items"] == [{"text": "Fix the login bug by Friday", "done": False}]
        assert ai.extractor.calls == ["Sync with Sarah"]
        assert await stored_embedding(database_url, note["id"]) == pytest.approx(EMBEDDING)

    async def test_client_category_and_tags_win(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/notes",
            json={"raw_transcript": "text", "category": "Health", "tags": ["gym"]},
        )

        note = response.json()
        assert (note["category"], note["tags"]) == ("Health", ["gym"])
        assert note["summary"] == METADATA.summary


class TestExtractionFails:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(
            extractor=FakeExtractor(RuntimeError("OpenAI down")), embedder=FakeEmbedder(EMBEDDING)
        )

    async def test_note_saved_with_embedding_but_no_metadata(
        self, client: AsyncClient, database_url: str
    ) -> None:
        response = await client.post("/api/v1/notes", json={"raw_transcript": "text"})

        note = response.json()
        assert response.status_code == 201
        assert (note["summary"], note["category"], note["tags"]) == (None, None, [])
        assert await stored_embedding(database_url, note["id"]) is not None


class TestEmbeddingFails:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(
            extractor=FakeExtractor(METADATA), embedder=FakeEmbedder(RuntimeError("OpenAI down"))
        )

    async def test_note_saved_with_metadata_but_no_embedding(
        self, client: AsyncClient, database_url: str
    ) -> None:
        response = await client.post("/api/v1/notes", json={"raw_transcript": "text"})

        note = response.json()
        assert response.status_code == 201
        assert note["summary"] == METADATA.summary
        assert await stored_embedding(database_url, note["id"]) is None


async def test_without_ai_note_saved_plain(client: AsyncClient, database_url: str) -> None:
    response = await client.post("/api/v1/notes", json={"raw_transcript": "text"})

    note = response.json()
    assert response.status_code == 201
    assert note["summary"] is None
    assert await stored_embedding(database_url, note["id"]) is None
