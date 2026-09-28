import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from app.ai.embeddings import Embedder, OpenAIEmbedder
from app.ai.extraction import MetadataExtractor, OpenAIMetadataExtractor
from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AIServices:
    extractor: MetadataExtractor | None = None
    embedder: Embedder | None = None


def create_ai_services(settings: Settings) -> AIServices:
    key = settings.openai_api_key
    if key is None or not key.get_secret_value():
        logger.warning("OPENAI_API_KEY is not set; notes will be saved without AI metadata.")
        return AIServices()
    return AIServices(
        extractor=OpenAIMetadataExtractor(key, settings.openai_chat_model),
        embedder=OpenAIEmbedder(key, settings.openai_embedding_model),
    )


def get_ai(request: Request) -> AIServices:
    return request.app.state.ai


AIDep = Annotated[AIServices, Depends(get_ai)]
