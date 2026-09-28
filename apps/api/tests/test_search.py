from datetime import UTC, datetime

import pytest
from httpx import AsyncClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import create_async_engine

from app.ai.services import AIServices
from app.notes.models import Note
from tests.conftest import FakeEmbedder, KeywordEmbedder

pytestmark = pytest.mark.anyio

KEYWORDS = ["bug", "login", "sarah", "gym", "groceries"]


async def create(client: AsyncClient, transcript: str, **fields: object) -> str:
    response = await client.post("/api/v1/notes", json={"raw_transcript": transcript, **fields})
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def search(client: AsyncClient, query: str, **params: object) -> list[dict]:
    response = await client.post("/api/v1/search", json={"query": query, **params})
    assert response.status_code == 200, response.text
    return response.json()["items"]


async def update_note(database_url: str, note_id: str, **values: object) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(update(Note).where(Note.id == note_id).values(**values))
    finally:
        await engine.dispose()


class TestSearch:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(embedder=KeywordEmbedder(KEYWORDS))

    async def test_ranks_most_similar_first(self, client: AsyncClient) -> None:
        gym = await create(client, "gym at seven")
        bug = await create(client, "fix the login bug for sarah")
        partial = await create(client, "sarah called about groceries")

        hits = await search(client, "what login bug did sarah report")

        assert [hit["note"]["id"] for hit in hits] == [bug, partial, gym]
        scores = [hit["score"] for hit in hits]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0, abs=0.01)
        assert hits[0]["note"]["raw_transcript"] == "fix the login bug for sarah"

    async def test_respects_limit(self, client: AsyncClient) -> None:
        for transcript in ["bug one", "bug two", "gym"]:
            await create(client, transcript)

        assert len(await search(client, "bug", limit=2)) == 2

    async def test_filters_by_category_and_tag(self, client: AsyncClient) -> None:
        work = await create(client, "login bug", category="Work", tags=["release"])
        await create(client, "login bug", category="Personal", tags=["release"])
        await create(client, "login bug", category="Work")

        hits = await search(client, "bug", category="work", tag="Release")

        assert [hit["note"]["id"] for hit in hits] == [work]

    async def test_filters_by_creation_time(self, client: AsyncClient, database_url: str) -> None:
        old = await create(client, "login bug")
        new = await create(client, "login bug")
        await update_note(database_url, old, created_at=datetime(2026, 1, 1, tzinfo=UTC))

        before = await search(client, "bug", created_before="2026-06-01T00:00:00Z")
        after = await search(client, "bug", created_after="2026-06-01T00:00:00Z")

        assert [hit["note"]["id"] for hit in before] == [old]
        assert [hit["note"]["id"] for hit in after] == [new]

    async def test_skips_notes_without_a_comparable_embedding(
        self, client: AsyncClient, database_url: str
    ) -> None:
        searchable = await create(client, "login bug")
        unembedded = await create(client, "login bug")
        other_model = await create(client, "login bug")
        await update_note(database_url, unembedded, embedding=None, embedding_model=None)
        await update_note(database_url, other_model, embedding_model="old-model")

        hits = await search(client, "bug")

        assert [hit["note"]["id"] for hit in hits] == [searchable]

    async def test_no_notes_returns_empty(self, client: AsyncClient) -> None:
        assert await search(client, "bug") == []

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"query": "   "},
            {"query": "x" * 2001},
            {"query": "bug", "limit": 0},
            {"query": "bug", "limit": 51},
            {"query": "bug", "category": "Health"},
            {"query": "bug", "created_after": "2026-01-01T00:00:00"},
        ],
    )
    async def test_rejects_invalid_input(self, client: AsyncClient, payload: dict) -> None:
        response = await client.post("/api/v1/search", json=payload)
        assert response.status_code == 422


async def test_without_embedder_returns_503(client: AsyncClient) -> None:
    response = await client.post("/api/v1/search", json={"query": "bug"})
    assert response.status_code == 503


class TestEmbeddingFailure:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(embedder=FakeEmbedder(RuntimeError("OpenAI is down")))

    async def test_returns_502(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/search", json={"query": "bug"})
        assert response.status_code == 502
