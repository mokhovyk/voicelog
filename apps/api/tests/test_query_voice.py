import pytest
from httpx import AsyncClient

from app.ai.answers import Answer
from app.ai.services import AIServices
from tests.conftest import FakeAnswerer, FakeTranscriber, KeywordEmbedder

pytestmark = pytest.mark.anyio

KEYWORDS = ["bug", "login", "sarah", "gym", "groceries"]
QUESTION = "what login bug did sarah report"
ANSWER = "You needed to fix the login bug for Sarah by Friday."
AUDIO = b"fake audio bytes"


async def create(client: AsyncClient, transcript: str, **fields: object) -> str:
    response = await client.post("/api/v1/notes", json={"raw_transcript": transcript, **fields})
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def ask(
    client: AsyncClient,
    content_type: str = "audio/webm;codecs=opus",
    audio: bytes = AUDIO,
    **form: object,
):
    return await client.post(
        "/api/v1/query/voice", files={"file": ("blob", audio, content_type)}, data=form
    )


def ai_with(transcript: str | Exception, answer: Answer | Exception | None = None) -> AIServices:
    return AIServices(
        embedder=KeywordEmbedder(KEYWORDS),
        answerer=FakeAnswerer(answer or Answer(answer=ANSWER, note_numbers=[1])),
        transcriber=FakeTranscriber(transcript),
    )


class TestVoiceQuery:
    @pytest.fixture
    def ai(self) -> AIServices:
        return ai_with(f"  {QUESTION}\n")

    async def test_answers_the_transcribed_question(
        self, client: AsyncClient, ai: AIServices
    ) -> None:
        await create(client, "gym at seven")
        bug = await create(client, "fix the login bug for sarah by friday")

        response = await ask(client)

        assert response.status_code == 200, response.text
        result = response.json()
        assert result["query"] == QUESTION
        assert result["answer"] == ANSWER
        assert [source["note"]["id"] for source in result["sources"]] == [bug]
        assert ai.transcriber.calls == [(AUDIO, "webm")]
        [(question, _)] = ai.answerer.calls
        assert question == QUESTION

    async def test_limit_and_filters_bound_the_context(
        self, client: AsyncClient, ai: AIServices
    ) -> None:
        for transcript in ["gym", "bug", "login bug"]:
            await create(client, transcript, category="Work", tags=["release"])
        await create(client, "login bug sarah", category="Personal", tags=["release"])
        await create(client, "sarah login bug", category="Work")

        response = await ask(client, category="work", tag="Release", limit="2")

        assert response.status_code == 200, response.text
        [(_, notes)] = ai.answerer.calls
        assert [note.text for note in notes] == ["login bug", "bug"]

    async def test_created_after_filter(self, client: AsyncClient, ai: AIServices) -> None:
        await create(client, "login bug")

        response = await ask(client, created_after="2999-01-01T00:00:00Z")

        assert response.status_code == 200, response.text
        assert response.json() == {"answer": None, "sources": [], "query": QUESTION}
        assert ai.answerer.calls == []

    @pytest.mark.parametrize(
        "form",
        [
            {"limit": "0"},
            {"limit": "21"},
            {"category": "Health"},
            {"created_after": "2025-01-01T00:00:00"},  # no timezone
        ],
    )
    async def test_rejects_invalid_fields_before_transcribing(
        self, client: AsyncClient, ai: AIServices, form: dict
    ) -> None:
        response = await ask(client, **form)

        assert response.status_code == 422
        assert ai.transcriber.calls == []

    async def test_rejects_unsupported_format(self, client: AsyncClient, ai: AIServices) -> None:
        response = await ask(client, content_type="text/plain")

        assert response.status_code == 415
        assert ai.transcriber.calls == []

    async def test_rejects_empty_audio(self, client: AsyncClient, ai: AIServices) -> None:
        response = await ask(client, audio=b"")

        assert response.status_code == 422
        assert ai.transcriber.calls == []

    async def test_requires_file(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/query/voice", data={"limit": "3"})

        assert response.status_code == 422


class TestQuestionTooLong:
    @pytest.fixture
    def ai(self) -> AIServices:
        return ai_with("bug " * 501)

    async def test_returns_422(self, client: AsyncClient, ai: AIServices) -> None:
        response = await ask(client)

        assert response.status_code == 422
        assert response.json()["detail"] == "Question is longer than 2000 characters"
        assert ai.answerer.calls == []


class TestNoSpeech:
    @pytest.fixture
    def ai(self) -> AIServices:
        return ai_with("  \n")

    async def test_returns_422(self, client: AsyncClient, ai: AIServices) -> None:
        response = await ask(client)

        assert response.status_code == 422
        assert response.json()["detail"] == "No speech recognized"
        assert ai.answerer.calls == []


class TestTranscriptionFails:
    @pytest.fixture
    def ai(self) -> AIServices:
        return ai_with(RuntimeError("OpenAI down"))

    async def test_returns_502(self, client: AsyncClient, ai: AIServices) -> None:
        response = await ask(client)

        assert response.status_code == 502
        assert response.json()["detail"] == "Transcription failed"
        assert ai.answerer.calls == []


class TestAnswerFailure:
    @pytest.fixture
    def ai(self) -> AIServices:
        return ai_with(QUESTION, answer=RuntimeError("OpenAI down"))

    async def test_returns_502(self, client: AsyncClient) -> None:
        await create(client, "login bug")

        response = await ask(client)

        assert response.status_code == 502
        assert response.json()["detail"] == "Answer generation failed"


@pytest.mark.parametrize(
    "ai",
    [
        AIServices(),
        AIServices(embedder=KeywordEmbedder(KEYWORDS), transcriber=FakeTranscriber(QUESTION)),
        AIServices(
            embedder=KeywordEmbedder(KEYWORDS),
            answerer=FakeAnswerer(Answer(answer=ANSWER, note_numbers=[])),
        ),
    ],
)
async def test_without_ai_returns_503(client: AsyncClient, ai: AIServices) -> None:
    response = await ask(client)

    assert response.status_code == 503
    if ai.transcriber is not None:
        assert ai.transcriber.calls == []
