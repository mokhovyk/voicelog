from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.notes.schemas import NoteFilters, NoteRead

MAX_QUERY_LENGTH = 2000


class SearchRequest(NoteFilters):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)
    ]
    limit: int = Field(10, ge=1, le=50)


class SearchHit(BaseModel):
    # Cosine similarity, 1 - cosine distance; higher is closer.
    score: float
    note: NoteRead


class SearchResults(BaseModel):
    items: list[SearchHit]
