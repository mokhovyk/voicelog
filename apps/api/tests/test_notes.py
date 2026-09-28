import uuid

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.anyio


async def create(client: AsyncClient, **fields: object) -> dict:
    response = await client.post("/api/v1/notes", json={"raw_transcript": "A note", **fields})
    assert response.status_code == 201, response.text
    return response.json()


async def list_ids(client: AsyncClient, **params: object) -> list[str]:
    response = await client.get("/api/v1/notes", params=params)
    assert response.status_code == 200, response.text
    return [note["id"] for note in response.json()["items"]]


async def test_create_note_returns_stored_note(client: AsyncClient) -> None:
    note = await create(
        client, raw_transcript="  Sync with Sarah.  ", category="Work", tags=["release"]
    )

    assert uuid.UUID(note["id"])
    assert note["raw_transcript"] == "Sync with Sarah."
    assert note["status"] == "pending"
    assert note["category"] == "Work"
    assert note["tags"] == ["release"]
    assert note["summary"] is None
    assert note["action_items"] == []
    assert note["created_at"] and note["updated_at"]
    assert "embedding" not in note


async def test_create_note_normalizes_category_and_tags(client: AsyncClient) -> None:
    note = await create(client, category=" work ", tags=["Release", "LOGIN-bug"])

    assert (note["category"], note["tags"]) == ("Work", ["release", "login-bug"])


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"raw_transcript": "   "},
        {"raw_transcript": "ok", "category": "Health"},
        {"raw_transcript": "ok", "tags": ["x" * 51]},
    ],
)
async def test_create_note_rejects_invalid_input(client: AsyncClient, payload: dict) -> None:
    response = await client.post("/api/v1/notes", json=payload)

    assert response.status_code == 422


async def test_get_note(client: AsyncClient) -> None:
    created = await create(client)

    response = await client.get(f"/api/v1/notes/{created['id']}")

    assert response.status_code == 200
    assert response.json() == created


async def test_get_missing_note_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/notes/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_list_notes_newest_first_with_pagination(client: AsyncClient) -> None:
    ids = [(await create(client, raw_transcript=f"note {i}"))["id"] for i in range(3)]

    first = (await client.get("/api/v1/notes", params={"limit": 2})).json()
    second = (await client.get("/api/v1/notes", params={"limit": 2, "offset": 2})).json()

    assert [n["id"] for n in first["items"] + second["items"]] == ids[::-1]
    assert first["total"] == second["total"] == 3
    assert (first["limit"], first["offset"]) == (2, 0)


async def test_list_notes_filters_by_category(client: AsyncClient) -> None:
    work = await create(client, category="Work")
    await create(client, category="Personal")
    await create(client)

    body = (await client.get("/api/v1/notes", params={"category": "Work"})).json()

    assert [n["id"] for n in body["items"]] == [work["id"]]
    assert body["total"] == 1


async def test_list_notes_filters_by_tag(client: AsyncClient) -> None:
    tagged = await create(client, tags=["release", "bug"])
    await create(client, tags=["gym"])

    body = (await client.get("/api/v1/notes", params={"tag": "Release"})).json()

    assert [n["id"] for n in body["items"]] == [tagged["id"]]


async def test_list_notes_filters_by_creation_time(client: AsyncClient) -> None:
    older = await create(client)
    newer = await create(client)

    assert await list_ids(client, created_after=newer["created_at"]) == [newer["id"]]
    assert await list_ids(client, created_before=newer["created_at"]) == [older["id"]]


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
        {"category": "Health"},
        {"created_after": "2026-01-01T00:00:00"},  # naive datetimes are ambiguous
    ],
)
async def test_list_notes_rejects_invalid_params(client: AsyncClient, params: dict) -> None:
    response = await client.get("/api/v1/notes", params=params)

    assert response.status_code == 422
