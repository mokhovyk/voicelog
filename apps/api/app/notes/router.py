import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from app.ai.services import AIDep
from app.core.database import SessionDep, SessionmakerDep
from app.notes import service
from app.notes.schemas import NoteCreate, NoteList, NoteListParams, NoteRead

router = APIRouter(prefix="/notes", tags=["notes"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_note(
    data: NoteCreate,
    session: SessionDep,
    sessionmaker: SessionmakerDep,
    ai: AIDep,
    background: BackgroundTasks,
) -> NoteRead:
    """Returns the note as `pending`; AI metadata is added after the response."""
    note = await service.create_note(session, data)
    background.add_task(service.enrich_note, sessionmaker, ai, note.id, note.raw_transcript)
    return NoteRead.model_validate(note)


@router.get("")
async def list_notes(params: Annotated[NoteListParams, Query()], session: SessionDep) -> NoteList:
    notes, total = await service.list_notes(session, params)
    return NoteList(
        items=[NoteRead.model_validate(note) for note in notes],
        total=total,
        limit=params.limit,
        offset=params.offset,
    )


@router.get("/{note_id}")
async def get_note(note_id: uuid.UUID, session: SessionDep) -> NoteRead:
    note = await service.get_note(session, note_id)
    if note is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Note not found")
    return NoteRead.model_validate(note)
