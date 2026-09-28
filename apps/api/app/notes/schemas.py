import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

from app.ai.extraction import CATEGORIES
from app.ai.extraction import Category as _Category
from app.notes.models import NoteStatus

_CANONICAL = {category.lower(): category for category in CATEGORIES}


def _canonical_category(value: Any) -> Any:
    """Accept any casing, e.g. "work" -> "Work"."""
    return _CANONICAL.get(value.strip().lower(), value) if isinstance(value, str) else value


Category = Annotated[_Category, BeforeValidator(_canonical_category)]
Tag = Annotated[
    str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1, max_length=50)
]


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
    status: NoteStatus
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
    tag: Tag | None = None
    created_after: AwareDatetime | None = None
    created_before: AwareDatetime | None = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
