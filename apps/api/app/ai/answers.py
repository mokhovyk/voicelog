from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

# Longer transcripts are cut, bounding the prompt size however many notes are sent.
MAX_NOTE_CHARS = 8000


@dataclass(frozen=True)
class SourceNote:
    created_at: datetime
    text: str


class Answer(BaseModel):
    """An answer to a question about the journal, with the notes it is based on."""

    answer: str = Field(description="The answer to the question, without note numbers.")
    note_numbers: list[int] = Field(
        description="Numbers of the notes the answer relies on. Empty if none do."
    )


class Answerer(Protocol):
    async def answer(self, question: str, notes: Sequence[SourceNote]) -> Answer: ...


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the user's personal voice journal. Use only the "
            "numbered notes provided; each shows the date it was recorded, and today is "
            "{today}. Resolve relative dates in a note, such as 'by Friday', against that "
            "note's date. If the notes do not answer the question, say so briefly instead "
            "of guessing. Answer concisely, in the language of the question, and speak to "
            "the user as 'you'.",
        ),
        ("human", "Notes:\n\n{notes}\n\nQuestion: {question}"),
    ]
)


def _format_date(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%A, %Y-%m-%d")


def format_notes(notes: Sequence[SourceNote]) -> str:
    """Number notes from 1, the numbers the model cites."""
    blocks = []
    for number, note in enumerate(notes, start=1):
        text = note.text
        if len(text) > MAX_NOTE_CHARS:
            text = text[:MAX_NOTE_CHARS] + " [truncated]"
        blocks.append(f"[{number}] Recorded {_format_date(note.created_at)}\n{text}")
    return "\n\n".join(blocks)


class OpenAIAnswerer:
    def __init__(
        self, api_key: SecretStr, model: str, http_client: httpx.AsyncClient | None = None
    ) -> None:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            timeout=60,
            max_retries=2,
            http_async_client=http_client,
        )
        self._chain = PROMPT | llm.with_structured_output(Answer)

    async def answer(self, question: str, notes: Sequence[SourceNote]) -> Answer:
        result = await self._chain.ainvoke(
            {
                "today": _format_date(datetime.now(UTC)),
                "notes": format_notes(notes),
                "question": question,
            }
        )
        assert isinstance(result, Answer)
        return result
