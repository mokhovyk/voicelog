import uuid
from datetime import datetime
from typing import Any, Literal, get_args

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.ai.embeddings import EMBEDDING_DIMENSIONS
from app.ai.extraction import CATEGORIES
from app.core.database import Base

# pending: not enriched yet (just created, or AI disabled). failed: an AI call failed.
# Both are picked up when notes are reprocessed.
NoteStatus = Literal["pending", "ready", "failed"]
NOTE_STATUSES: tuple[str, ...] = get_args(NoteStatus)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Note(Base):
    __tablename__ = "notes"
    __table_args__ = (
        CheckConstraint(_in("category", CATEGORIES), name="ck_notes_category"),
        CheckConstraint(_in("status", NOTE_STATUSES), name="ck_notes_status"),
        Index(
            "ix_notes_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_notes_tags", "tags", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    raw_transcript: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), server_default=text("'pending'"))
    summary: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(50), index=True)
    action_items: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb")
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String(50)), server_default=text("'{}'::varchar[]")
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    # Vectors from different models are not comparable; this finds notes to re-embed.
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
