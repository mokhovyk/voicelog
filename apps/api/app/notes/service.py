import asyncio
import logging
import uuid

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.extraction import NoteMetadata
from app.ai.services import AIServices
from app.notes.models import Note
from app.notes.schemas import NoteCreate, NoteFilters, NoteListParams

logger = logging.getLogger(__name__)

MAX_TAGS = 5
MAX_TAG_LENGTH = 50


async def create_note(session: AsyncSession, data: NoteCreate) -> Note:
    """Save the note as pending; enrich_note fills in AI metadata afterwards."""
    note = Note(raw_transcript=data.raw_transcript, category=data.category, tags=data.tags)
    session.add(note)
    await session.commit()
    await session.refresh(note)
    return note


async def enrich_note(
    sessionmaker: async_sessionmaker[AsyncSession],
    ai: AIServices,
    note_id: uuid.UUID,
    transcript: str,
) -> None:
    """Add AI metadata and the embedding to a saved note. Runs after the response is
    sent, so it opens its own session, and none is held open during the AI calls."""
    if not ai.enabled:
        return  # stays pending until it is reprocessed with AI configured

    metadata, embedding, failed = await _enrich(ai, transcript)

    async with sessionmaker() as session:
        note = await session.get(Note, note_id)
        if note is None:
            return
        if metadata is not None:
            note.summary = metadata.summary
            note.action_items = [item.model_dump() for item in metadata.action_items]
            # Values the client sent explicitly win over extracted ones.
            note.category = note.category or metadata.category
            note.tags = note.tags or _clean_tags(metadata.tags)
        if embedding is not None and ai.embedder is not None:
            note.embedding = embedding
            note.embedding_model = ai.embedder.model
        note.status = "failed" if failed else "ready"
        await session.commit()


async def _enrich(
    ai: AIServices, transcript: str
) -> tuple[NoteMetadata | None, list[float] | None, bool]:
    """Run extraction and embedding concurrently. A failure in either is logged and
    leaves that part empty; the returned flag reports whether anything failed."""

    async def extract() -> NoteMetadata | None:
        return await ai.extractor.extract(transcript) if ai.extractor else None

    async def embed() -> list[float] | None:
        return await ai.embedder.embed(transcript) if ai.embedder else None

    metadata, embedding = await asyncio.gather(extract(), embed(), return_exceptions=True)
    failed = False
    if isinstance(metadata, Exception):
        logger.exception("Metadata extraction failed", exc_info=metadata)
        metadata, failed = None, True
    if isinstance(embedding, Exception):
        logger.exception("Embedding failed", exc_info=embedding)
        embedding, failed = None, True
    return metadata, embedding, failed


def _clean_tags(tags: list[str]) -> list[str]:
    cleaned = [tag.strip().lower()[:MAX_TAG_LENGTH] for tag in tags]
    return list(dict.fromkeys(tag for tag in cleaned if tag))[:MAX_TAGS]


def apply_filters[T: tuple](query: Select[T], filters: NoteFilters) -> Select[T]:
    if filters.category is not None:
        query = query.where(Note.category == filters.category)
    if filters.tag is not None:
        query = query.where(Note.tags.contains([filters.tag]))  # @>, uses ix_notes_tags
    if filters.created_after is not None:
        query = query.where(Note.created_at >= filters.created_after)
    if filters.created_before is not None:
        query = query.where(Note.created_at < filters.created_before)
    return query


async def list_notes(session: AsyncSession, params: NoteListParams) -> tuple[list[Note], int]:
    query = apply_filters(select(Note), params)
    total = await session.scalar(select(func.count()).select_from(query.subquery()))
    notes = await session.scalars(
        query.order_by(Note.created_at.desc(), Note.id.desc())
        .limit(params.limit)
        .offset(params.offset)
    )
    return list(notes), total or 0


async def get_note(session: AsyncSession, note_id: uuid.UUID) -> Note | None:
    return await session.get(Note, note_id)
