from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.notes.schemas import NoteFilters, NoteRead

MAX_QUERY_LENGTH = 2000


class SearchRequest(NoteFilters):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)
    ]
    limit: int = Field(10, ge=1, le=50)


class QueryRequest(SearchRequest):
    # How many of the most similar notes the answer is drawn from.
    limit: int = Field(5, ge=1, le=20)


class SearchHit(BaseModel):
    # Cosine similarity, 1 - cosine distance; higher is closer.
    score: float
    note: NoteRead


class SearchResults(BaseModel):
    items: list[SearchHit]


class QueryAnswer(BaseModel):
    # None when no notes matched, so there was nothing to answer from.
    answer: str | None
    # The notes the answer cites, most similar first.
    sources: list[SearchHit]


class VoiceQueryAnswer(QueryAnswer):
    # The transcribed question.
    query: str
