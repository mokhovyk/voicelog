from typing import Protocol

import httpx
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.notes.models import EMBEDDING_DIMENSIONS


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(
        self, api_key: SecretStr, model: str, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            dimensions=EMBEDDING_DIMENSIONS,
            timeout=30,
            max_retries=2,
            http_async_client=http_client,
        )

    async def embed(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)
