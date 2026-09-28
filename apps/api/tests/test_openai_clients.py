"""Run the real OpenAI client code against a mocked OpenAI HTTP API."""

import base64
import json
import struct

import httpx
import pytest
from pydantic import SecretStr

from app.ai.embeddings import OpenAIEmbedder
from app.ai.extraction import NoteMetadata, OpenAIMetadataExtractor
from app.ai.services import create_ai_services
from app.core.config import Settings
from app.notes.models import EMBEDDING_DIMENSIONS

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


@pytest.mark.parametrize("key", [None, ""])
def test_no_ai_without_api_key(key: str | None) -> None:
    ai = create_ai_services(Settings(openai_api_key=key))

    assert (ai.extractor, ai.embedder) == (None, None)


def test_ai_enabled_with_api_key() -> None:
    ai = create_ai_services(Settings(openai_api_key="sk-test"))

    assert isinstance(ai.extractor, OpenAIMetadataExtractor)
    assert isinstance(ai.embedder, OpenAIEmbedder)
