import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from app.ai.embeddings import EMBEDDING_DIMENSIONS
from app.ai.extraction import ExtractedActionItem, NoteMetadata
from app.ai.services import AIServices
from app.notes.models import Note
from tests.conftest import FakeEmbedder, FakeExtractor

pytestmark = pytest.mark.anyio

METADATA = NoteMetadata(
    summary="Release sync with Sarah; login bug due Friday.",
    category="Work",
    tags=[" Release ", "release", "Login-Bug", "x" * 60, "", "a", "b", "c"],
    action_items=[ExtractedActionItem(text="Fix the login bug by Friday")],
)
EMBEDDING = [0.1] * EMBEDDING_DIMENSIONS


async def stored_embedding(
    database_url: str, note_id: str
) -> tuple[list[float] | None, str | None]:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    select(Note.embedding, Note.embedding_model).where(Note.id == note_id)
                )
            ).one()
    finally:
        await engine.dispose()
    vector, model = row
    return (None if vector is None else [float(x) for x in vector]), model


async def create_and_fetch(client: AsyncClient, **fields: object) -> dict:
    """Create a note and read it back once background enrichment has run.

    The ASGI test transport returns only after background tasks finish."""
    response = await client.post("/api/v1/notes", json={"raw_transcript": "text", **fields})
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "pending"
    return (await client.get(f"/api/v1/notes/{response.json()['id']}")).json()


class TestWithAI:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(extractor=FakeExtractor(METADATA), embedder=FakeEmbedder(EMBEDDING))

    async def test_fills_metadata_and_stores_embedding(
        self, client: AsyncClient, ai: AIServices, database_url: str
    ) -> None:
        note = await create_and_fetch(client, raw_transcript="Sync with Sarah")

        assert note["status"] == "ready"
        assert note["summary"] == METADATA.summary
        assert note["category"] == "Work"
        assert note["tags"] == ["release", "login-bug", "x" * 50, "a", "b"]
        assert note["action_items"] == [{"text": "Fix the login bug by Friday", "done": False}]
        assert ai.extractor.calls == ["Sync with Sarah"]
        vector, model = await stored_embedding(database_url, note["id"])
        assert vector == pytest.approx(EMBEDDING)
        assert model == "fake-embedding"

    async def test_client_category_and_tags_win(self, client: AsyncClient) -> None:
        note = await create_and_fetch(client, category="Personal", tags=["gym"])

        assert (note["category"], note["tags"]) == ("Personal", ["gym"])
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
        note = await create_and_fetch(client)

        assert note["status"] == "failed"
        assert (note["summary"], note["category"], note["tags"]) == (None, None, [])
        assert (await stored_embedding(database_url, note["id"]))[0] is not None


class TestEmbeddingFails:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(
            extractor=FakeExtractor(METADATA), embedder=FakeEmbedder(RuntimeError("OpenAI down"))
        )

    async def test_note_saved_with_metadata_but_no_embedding(
        self, client: AsyncClient, database_url: str
    ) -> None:
        note = await create_and_fetch(client)

        assert note["status"] == "failed"
        assert note["summary"] == METADATA.summary
        assert await stored_embedding(database_url, note["id"]) == (None, None)


async def test_without_ai_note_stays_pending(client: AsyncClient, database_url: str) -> None:
    note = await create_and_fetch(client)

    assert note["status"] == "pending"
    assert note["summary"] is None
    assert await stored_embedding(database_url, note["id"]) == (None, None)
