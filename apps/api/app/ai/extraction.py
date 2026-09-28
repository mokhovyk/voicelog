from typing import Literal, Protocol, get_args

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# The note categories. Notes reuse this so client-sent and extracted values agree.
Category = Literal["Work", "Personal", "Ideas", "Other"]
CATEGORIES: tuple[str, ...] = get_args(Category)


class ExtractedActionItem(BaseModel):
    text: str
    done: bool = False


class NoteMetadata(BaseModel):
    """Structured metadata extracted from a note transcript."""

    summary: str = Field(description="One or two sentences summarizing the note.")
    category: Category = Field(description="The single best-fitting category.")
    tags: list[str] = Field(description="Up to 5 short lowercase keyword tags.")
    action_items: list[ExtractedActionItem] = Field(
        description="Concrete tasks the speaker needs to do. Empty if there are none."
    )


class MetadataExtractor(Protocol):
    async def extract(self, transcript: str) -> NoteMetadata: ...


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You organize entries in a personal voice journal. Extract metadata from the "
            "transcript. Use only information in the transcript; do not invent tasks. "
            "Write the summary in the transcript's language. Mark action items as not done.",
        ),
        ("human", "{transcript}"),
    ]
)


class OpenAIMetadataExtractor:
    def __init__(
        self, api_key: SecretStr, model: str, http_client: httpx.AsyncClient | None = None
    ) -> None:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            timeout=30,
            max_retries=2,
            http_async_client=http_client,
        )
        self._chain = PROMPT | llm.with_structured_output(NoteMetadata)

    async def extract(self, transcript: str) -> NoteMetadata:
        result = await self._chain.ainvoke({"transcript": transcript})
        assert isinstance(result, NoteMetadata)
        return result
