import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.notes.models import Note
from app.notes.schemas import NoteCreate, NoteListParams


async def create_note(session: AsyncSession, data: NoteCreate) -> Note:
    note = Note(raw_transcript=data.raw_transcript, category=data.category, tags=data.tags)
    session.add(note)
    await session.commit()
    await session.refresh(note)
    return note


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
