"""Run the real OpenAI client code against a mocked OpenAI HTTP API."""

import base64
import json
import struct
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import SecretStr

from app.ai.answers import MAX_NOTE_CHARS, Answer, OpenAIAnswerer, SourceNote
from app.ai.embeddings import EMBEDDING_DIMENSIONS, OpenAIEmbedder
from app.ai.extraction import NoteMetadata, OpenAIMetadataExtractor
from app.ai.services import create_ai_services
from app.ai.transcription import OpenAITranscriber
from app.core.config import Settings

pytestmark = pytest.mark.anyio

KEY = SecretStr("sk-test")


def mock_openai(respond: dict) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=respond)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), requests


async def test_extractor_sends_schema_and_parses_response() -> None:
    metadata = {
        "summary": "Fix the login bug.",
        "category": "Work",
        "tags": ["bug"],
        "action_items": [{"text": "Fix login bug", "done": False}],
    }
    client, requests = mock_openai(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(metadata)},
                }
            ],
        }
    )

    result = await OpenAIMetadataExtractor(KEY, "gpt-test", client).extract("Fix the login bug")

    assert result == NoteMetadata.model_validate(metadata)
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert body["model"] == "gpt-test"
    assert body["response_format"]["type"] == "json_schema"
    assert body["messages"][-1]["content"] == "Fix the login bug"


def chat_completion(content: dict) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(content)},
            }
        ],
    }


async def test_answerer_sends_numbered_dated_notes_and_parses_citations() -> None:
    answer = {"answer": "Fix the login bug by Friday.", "note_numbers": [2]}
    client, requests = mock_openai(chat_completion(answer))
    notes = [
        SourceNote(created_at=datetime(2026, 9, 21, 9, tzinfo=UTC), text="Gym at seven."),
        SourceNote(created_at=datetime(2026, 9, 28, 9, tzinfo=UTC), text="x" * 9000),
    ]

    result = await OpenAIAnswerer(KEY, "gpt-test", client).answer("What bug?", notes)

    assert result == Answer.model_validate(answer)
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/chat/completions"
    assert body["response_format"]["type"] == "json_schema"
    prompt = body["messages"][-1]["content"]
    assert "[1] Recorded Monday, 2026-09-21\nGym at seven." in prompt
    assert f"[2] Recorded Monday, 2026-09-28\n{'x' * MAX_NOTE_CHARS} [truncated]" in prompt
    assert prompt.endswith("Question: What bug?")
    assert f"today is {datetime.now(UTC):%A, %Y-%m-%d}" in body["messages"][0]["content"]


async def test_embedder_requests_configured_dimensions() -> None:
    vector = [0.5] * EMBEDDING_DIMENSIONS
    encoded = base64.b64encode(struct.pack(f"{len(vector)}f", *vector)).decode()
    client, requests = mock_openai(
        {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": encoded}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
    )

    result = await OpenAIEmbedder(KEY, "text-embedding-3-small", client).embed("hello")

    assert result == pytest.approx(vector)
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/embeddings"
    assert (body["model"], body["dimensions"]) == ("text-embedding-3-small", EMBEDDING_DIMENSIONS)


async def test_transcriber_uploads_audio_with_format_extension() -> None:
    client, requests = mock_openai({"text": "Fix the login bug."})

    result = await OpenAITranscriber(KEY, "transcribe-test", client).transcribe(b"RIFF", "webm")

    assert result == "Fix the login bug."
    request = requests[0]
    assert request.url.path == "/v1/audio/transcriptions"
    assert request.headers["content-type"].startswith("multipart/form-data")
    body = request.content
    assert b'name="model"\r\n\r\ntranscribe-test' in body
    assert b'filename="audio.webm"' in body
    assert b"RIFF" in body


@pytest.mark.parametrize("key", [None, ""])
def test_no_ai_without_api_key(key: str | None) -> None:
    ai = create_ai_services(Settings(openai_api_key=key))

    assert (ai.extractor, ai.embedder, ai.transcriber, ai.answerer) == (None,) * 4


async def test_ai_enabled_with_api_key() -> None:
    ai = create_ai_services(Settings(openai_api_key="sk-test"))

    assert isinstance(ai.extractor, OpenAIMetadataExtractor)
    assert isinstance(ai.embedder, OpenAIEmbedder)
    assert ai.embedder.model == "text-embedding-3-small"
    assert isinstance(ai.transcriber, OpenAITranscriber)
    assert ai.transcriber.model == "gpt-4o-mini-transcribe"
    assert isinstance(ai.answerer, OpenAIAnswerer)
    assert ai.http_client is not None
    await ai.aclose()
    assert ai.http_client.is_closed
