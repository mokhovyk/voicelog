import pytest
from httpx import AsyncClient

from app.ai.answers import Answer
from app.ai.services import AIServices
from tests.conftest import FakeAnswerer, FakeEmbedder, KeywordEmbedder

pytestmark = pytest.mark.anyio

KEYWORDS = ["bug", "login", "sarah", "gym", "groceries"]
ANSWER = "You needed to fix the login bug for Sarah by Friday."


async def create(client: AsyncClient, transcript: str, **fields: object) -> str:
    response = await client.post("/api/v1/notes", json={"raw_transcript": transcript, **fields})
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def query(client: AsyncClient, question: str, **params: object) -> dict:
    response = await client.post("/api/v1/query/text", json={"query": question, **params})
    assert response.status_code == 200, response.text
    return response.json()


def ai_with(answer: Answer | Exception) -> AIServices:
    return AIServices(embedder=KeywordEmbedder(KEYWORDS), answerer=FakeAnswerer(answer))


class TestQuery:
    @pytest.fixture
    def ai(self) -> AIServices:
        # Cites the 1st and 3rd most similar notes, plus numbers that do not exist.
        return ai_with(Answer(answer=ANSWER, note_numbers=[3, 1, 1, 0, 9]))

    async def test_answers_from_similar_notes_and_cites_sources(
        self, client: AsyncClient, ai: AIServices
    ) -> None:
        gym = await create(client, "gym at seven")
        bug = await create(client, "fix the login bug for sarah by friday")
        await create(client, "sarah called about groceries")

        result = await query(client, "what login bug did sarah report")

        assert result["answer"] == ANSWER
        # Cited notes only, deduplicated, most similar first; invalid numbers dropped.
        assert [source["note"]["id"] for source in result["sources"]] == [bug, gym]
        assert result["sources"][0]["score"] == pytest.approx(1.0, abs=0.01)
        [(question, notes)] = ai.answerer.calls
        assert question == "what login bug did sarah report"
        assert [note.text for note in notes] == [
            "fix the login bug for sarah by friday",
            "sarah called about groceries",
            "gym at seven",
        ]
        assert all(note.created_at.tzinfo is not None for note in notes)

    async def test_limit_and_filters_bound_the_context(
        self, client: AsyncClient, ai: AIServices
    ) -> None:
        for transcript in ["gym", "bug", "login bug"]:
            await create(client, transcript, category="Work")
        await create(client, "login bug sarah", category="Personal")

        await query(client, "login bug sarah", category="Work", limit=2)

        [(_, notes)] = ai.answerer.calls
        assert [note.text for note in notes] == ["login bug", "bug"]

    async def test_no_matching_notes_skips_the_model(
        self, client: AsyncClient, ai: AIServices
    ) -> None:
        await create(client, "login bug", category="Personal")

        result = await query(client, "login bug", category="Work")

        assert result == {"answer": None, "sources": []}
        assert ai.answerer.calls == []

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"query": "   "},
            {"query": "x" * 2001},
            {"query": "bug", "limit": 0},
            {"query": "bug", "limit": 21},
            {"query": "bug", "category": "Health"},
        ],
    )
    async def test_rejects_invalid_input(self, client: AsyncClient, payload: dict) -> None:
        response = await client.post("/api/v1/query/text", json=payload)
        assert response.status_code == 422


class TestUncitedAnswer:
    @pytest.fixture
    def ai(self) -> AIServices:
        return ai_with(Answer(answer="Your notes don't mention that.", note_numbers=[]))

    async def test_returns_answer_without_sources(self, client: AsyncClient) -> None:
        await create(client, "gym at seven")

        result = await query(client, "what did sarah say")

        assert result == {"answer": "Your notes don't mention that.", "sources": []}


@pytest.mark.parametrize(
    "ai",
    [
        AIServices(),
        AIServices(embedder=KeywordEmbedder(KEYWORDS)),
        AIServices(answerer=FakeAnswerer(Answer(answer=ANSWER, note_numbers=[]))),
    ],
)
async def test_without_ai_returns_503(client: AsyncClient) -> None:
    response = await client.post("/api/v1/query/text", json={"query": "bug"})
    assert response.status_code == 503


class TestAnswerFailure:
    @pytest.fixture
    def ai(self) -> AIServices:
        return ai_with(RuntimeError("OpenAI is down"))

    async def test_returns_502(self, client: AsyncClient) -> None:
        await create(client, "login bug")

        response = await client.post("/api/v1/query/text", json={"query": "bug"})

        assert response.status_code == 502
        assert response.json()["detail"] == "Answer generation failed"


class TestEmbeddingFailure:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(
            embedder=FakeEmbedder(RuntimeError("OpenAI is down")),
            answerer=FakeAnswerer(Answer(answer=ANSWER, note_numbers=[])),
        )

    async def test_returns_502(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/query/text", json={"query": "bug"})

        assert response.status_code == 502
        assert response.json()["detail"] == "Query embedding failed"
