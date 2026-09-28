import asyncio
import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.extraction import NoteMetadata
from app.ai.services import AIServices
from app.notes.models import Note
from app.notes.schemas import NoteCreate, NoteListParams

logger = logging.getLogger(__name__)

MAX_TAGS = 5
MAX_TAG_LENGTH = 50


async def create_note(session: AsyncSession, data: NoteCreate, ai: AIServices) -> Note:
    metadata, embedding = await _enrich(ai, data.raw_transcript)

    note = Note(
        raw_transcript=data.raw_transcript,
        category=data.category,
        tags=data.tags,
        embedding=embedding,
    )
    if metadata is not None:
        note.summary = metadata.summary
        note.action_items = [item.model_dump() for item in metadata.action_items]
        # Values the client sent explicitly win over extracted ones.
        note.category = data.category or metadata.category
        note.tags = data.tags or _clean_tags(metadata.tags)

    session.add(note)
    await session.commit()
    await session.refresh(note)
    return note


async def _enrich(
    ai: AIServices, transcript: str
) -> tuple[NoteMetadata | None, list[float] | None]:
    """Run extraction and embedding concurrently. A failure in either is logged and
    leaves that part empty, so the note is still saved (embedding IS NULL marks notes
    to reprocess later)."""

    async def extract() -> NoteMetadata | None:
        return await ai.extractor.extract(transcript) if ai.extractor else None

    async def embed() -> list[float] | None:
        return await ai.embedder.embed(transcript) if ai.embedder else None

    metadata, embedding = await asyncio.gather(extract(), embed(), return_exceptions=True)
    if isinstance(metadata, Exception):
        logger.exception("Metadata extraction failed", exc_info=metadata)
        metadata = None
    if isinstance(embedding, Exception):
        logger.exception("Embedding failed", exc_info=embedding)
        embedding = None
    return metadata, embedding


def _clean_tags(tags: list[str]) -> list[str]:
    cleaned = [tag.strip().lower()[:MAX_TAG_LENGTH] for tag in tags]
    return list(dict.fromkeys(tag for tag in cleaned if tag))[:MAX_TAGS]


async def list_notes(session: AsyncSession, params: NoteListParams) -> tuple[list[Note], int]:
    query = select(Note)
    if params.category is not None:
        query = query.where(Note.category == params.category)

    total = await session.scalar(select(func.count()).select_from(query.subquery()))
    notes = await session.scalars(
        query.order_by(Note.created_at.desc(), Note.id.desc())
        .limit(params.limit)
        .offset(params.offset)
    )
    return list(notes), total or 0


async def get_note(session: AsyncSession, note_id: uuid.UUID) -> Note | None:
    return await session.get(Note, note_id)
