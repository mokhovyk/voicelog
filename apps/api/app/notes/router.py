import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Query, UploadFile, status

from app.ai.services import AIDep
from app.audio import transcribe_upload
from app.core.database import SessionDep, SessionmakerDep
from app.notes import service
from app.notes.schemas import Category, NoteCreate, NoteList, NoteListParams, NoteRead, Tag

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


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_audio_note(
    file: UploadFile,
    session: SessionDep,
    sessionmaker: SessionmakerDep,
    ai: AIDep,
    background: BackgroundTasks,
    category: Annotated[Category | None, Form()] = None,
    tags: Annotated[list[Tag] | None, Form()] = None,
) -> NoteRead:
    """Transcribes the audio and saves the transcript as a text note, returned as
    `pending`. The audio itself is not stored."""
    transcript = await transcribe_upload(file, ai.transcriber)
    data = NoteCreate(raw_transcript=transcript, category=category, tags=tags or [])
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
