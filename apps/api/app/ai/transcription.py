from pathlib import PurePath
from typing import Protocol

import httpx
from openai import AsyncOpenAI
from pydantic import SecretStr

# OpenAI rejects larger uploads (26,214,400 bytes).
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# Formats OpenAI transcribes, by MIME type. OpenAI detects the format from the file
# name, so uploads are renamed with these extensions.
AUDIO_FORMATS = {
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/mp4": "m4a",  # Safari's MediaRecorder
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",  # Chrome's and Firefox's MediaRecorder
}
_EXTENSIONS = {*AUDIO_FORMATS.values(), "mp4", "mpeg", "mpga"}


def audio_format(content_type: str | None, filename: str | None) -> str | None:
    """The file extension to send to OpenAI, or None if the format is unsupported.

    The MIME type decides, since browsers name recorded blobs "blob". Tools like curl
    send `application/octet-stream` for audio, so the file extension is the fallback."""
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in AUDIO_FORMATS:
        return AUDIO_FORMATS[mime]
    extension = PurePath(filename or "").suffix.removeprefix(".").lower()
    return extension if extension in _EXTENSIONS else None


class Transcriber(Protocol):
    async def transcribe(self, audio: bytes, audio_format: str) -> str: ...


class OpenAITranscriber:
    def __init__(
        self, api_key: SecretStr, model: str, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            # Long recordings take a while; the request waits for the transcript.
            timeout=120,
            max_retries=1,
            http_client=http_client,
        )

    async def transcribe(self, audio: bytes, audio_format: str) -> str:
        result = await self._client.audio.transcriptions.create(
            model=self.model, file=(f"audio.{audio_format}", audio), response_format="json"
        )
        return result.text
