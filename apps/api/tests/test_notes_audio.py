import pytest
from httpx import AsyncClient

from app.ai.embeddings import EMBEDDING_DIMENSIONS
from app.ai.extraction import NoteMetadata
from app.ai.services import AIServices
from app.notes import router
from tests.conftest import FakeEmbedder, FakeExtractor, FakeTranscriber

pytestmark = pytest.mark.anyio

TRANSCRIPT = "Had a sync with Sarah. Need to fix the login bug by Friday."
AUDIO = b"fake audio bytes"


async def upload(
    client: AsyncClient,
    content_type: str = "audio/webm;codecs=opus",
    filename: str = "blob",
    audio: bytes = AUDIO,
    **form: object,
):
    return await client.post(
        "/api/v1/notes/upload", files={"file": (filename, audio, content_type)}, data=form
    )


async def note_count(client: AsyncClient) -> int:
    return (await client.get("/api/v1/notes")).json()["total"]


class TestUpload:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(transcriber=FakeTranscriber(f"  {TRANSCRIPT}\n"))

    async def test_saves_transcript_as_pending_note(
        self, client: AsyncClient, ai: AIServices
    ) -> None:
        response = await upload(client)

        assert response.status_code == 201, response.text
        note = response.json()
        assert note["raw_transcript"] == TRANSCRIPT
        assert note["status"] == "pending"
        assert ai.transcriber.calls == [(AUDIO, "webm")]
        stored = (await client.get(f"/api/v1/notes/{note['id']}")).json()
        assert stored["raw_transcript"] == TRANSCRIPT

    async def test_applies_category_and_tags(self, client: AsyncClient) -> None:
        response = await upload(client, category="work", tags=["Release", "sarah"])

        assert response.status_code == 201, response.text
        assert (response.json()["category"], response.json()["tags"]) == (
            "Work",
            ["release", "sarah"],
        )

    @pytest.mark.parametrize(
        ("content_type", "filename", "expected"),
        [
            ("audio/mp4", "blob", "m4a"),
            ("audio/x-wav", "note.bin", "wav"),
            ("AUDIO/MPEG", "blob", "mp3"),
            ("application/octet-stream", "Note.M4A", "m4a"),
            ("", "memo.mpga", "mpga"),
        ],
    )
    async def test_detects_format(
        self,
        client: AsyncClient,
        ai: AIServices,
        content_type: str,
        filename: str,
        expected: str,
    ) -> None:
        response = await upload(client, content_type=content_type, filename=filename)

        assert response.status_code == 201, response.text
        assert ai.transcriber.calls[-1][1] == expected

    @pytest.mark.parametrize(
        ("content_type", "filename"),
        [("text/plain", "note.txt"), ("application/octet-stream", "blob"), ("video/avi", "a.avi")],
    )
    async def test_rejects_unsupported_format(
        self, client: AsyncClient, ai: AIServices, content_type: str, filename: str
    ) -> None:
        response = await upload(client, content_type=content_type, filename=filename)

        assert response.status_code == 415
        assert ai.transcriber.calls == []

    async def test_rejects_audio_over_limit(
        self, client: AsyncClient, ai: AIServices, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(router, "MAX_AUDIO_BYTES", 10)

        assert (await upload(client, audio=b"x" * 10)).status_code == 201
        assert (await upload(client, audio=b"x" * 11)).status_code == 413
        assert len(ai.transcriber.calls) == 1

    async def test_rejects_empty_audio(self, client: AsyncClient, ai: AIServices) -> None:
        response = await upload(client, audio=b"")

        assert response.status_code == 422
        assert ai.transcriber.calls == []

    @pytest.mark.parametrize("form", [{"category": "Health"}, {"tags": ["x" * 51]}])
    async def test_rejects_invalid_fields_before_transcribing(
        self, client: AsyncClient, ai: AIServices, form: dict
    ) -> None:
        response = await upload(client, **form)

        assert response.status_code == 422
        assert ai.transcriber.calls == []

    async def test_requires_file(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/notes/upload", data={"category": "Work"})

        assert response.status_code == 422


class TestTranscriptionFails:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(transcriber=FakeTranscriber(RuntimeError("OpenAI down")))

    async def test_returns_502_and_saves_nothing(self, client: AsyncClient) -> None:
        response = await upload(client)

        assert response.status_code == 502
        assert await note_count(client) == 0


class TestNoSpeech:
    @pytest.fixture
    def ai(self) -> AIServices:
        return AIServices(transcriber=FakeTranscriber("  \n"))

    async def test_returns_422_and_saves_nothing(self, client: AsyncClient) -> None:
        response = await upload(client)

        assert response.status_code == 422
        assert await note_count(client) == 0


class TestEnrichment:
    @pytest.fixture
    def ai(self) -> AIServices:
        metadata = NoteMetadata(
            summary="Login bug due Friday.", category="Work", tags=["bug"], action_items=[]
        )
        return AIServices(
            extractor=FakeExtractor(metadata),
            embedder=FakeEmbedder([0.1] * EMBEDDING_DIMENSIONS),
            transcriber=FakeTranscriber(TRANSCRIPT),
        )

    async def test_enriches_the_transcript(self, client: AsyncClient, ai: AIServices) -> None:
        response = await upload(client)
        assert response.status_code == 201, response.text

        # The ASGI test transport returns only after background tasks finish.
        note = (await client.get(f"/api/v1/notes/{response.json()['id']}")).json()
        assert (note["status"], note["summary"]) == ("ready", "Login bug due Friday.")
        assert ai.extractor.calls == [TRANSCRIPT]


async def test_without_ai_returns_503(client: AsyncClient) -> None:
    response = await upload(client)

    assert response.status_code == 503
    assert await note_count(client) == 0
