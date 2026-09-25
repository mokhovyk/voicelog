import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)]
Tag = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)]


class ActionItem(BaseModel):
    text: str
    done: bool = False


class NoteCreate(BaseModel):
    raw_transcript: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    category: Category | None = None
    tags: list[Tag] = []


class NoteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    raw_transcript: str
    summary: str | None
    category: str | None
    action_items: list[ActionItem]
    tags: list[str]
    created_at: datetime
    updated_at: datetime


class NoteList(BaseModel):
    items: list[NoteRead]
    total: int
    limit: int
    offset: int


class NoteListParams(BaseModel):
    category: Category | None = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
