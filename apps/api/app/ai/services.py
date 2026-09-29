import logging
from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Depends, Request

from app.ai.answers import Answerer, OpenAIAnswerer
from app.ai.embeddings import Embedder, OpenAIEmbedder
from app.ai.extraction import MetadataExtractor, OpenAIMetadataExtractor
from app.ai.transcription import OpenAITranscriber, Transcriber
from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AIServices:
    extractor: MetadataExtractor | None = None
    embedder: Embedder | None = None
    transcriber: Transcriber | None = None
    answerer: Answerer | None = None
    # Shared by the OpenAI clients; closed on app shutdown.
    http_client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        """Whether new notes are enriched."""
        return self.extractor is not None or self.embedder is not None

    async def aclose(self) -> None:
        if self.http_client is not None:
            await self.http_client.aclose()


def create_ai_services(settings: Settings) -> AIServices:
    key = settings.openai_api_key
    if key is None or not key.get_secret_value():
        logger.warning(
            "OPENAI_API_KEY is not set; notes will be saved without AI metadata, "
            "and audio, search, and questions are unavailable."
        )
        return AIServices()
    http_client = httpx.AsyncClient()
    return AIServices(
        extractor=OpenAIMetadataExtractor(key, settings.openai_chat_model, http_client),
        embedder=OpenAIEmbedder(key, settings.openai_embedding_model, http_client),
        transcriber=OpenAITranscriber(key, settings.openai_transcription_model, http_client),
        answerer=OpenAIAnswerer(key, settings.openai_chat_model, http_client),
        http_client=http_client,
    )


def get_ai(request: Request) -> AIServices:
    return request.app.state.ai


AIDep = Annotated[AIServices, Depends(get_ai)]
