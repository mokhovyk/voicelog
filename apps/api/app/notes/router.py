import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Query, UploadFile, status

from app.ai.services import AIDep
from app.ai.transcription import MAX_AUDIO_BYTES, audio_format
from app.core.database import SessionDep, SessionmakerDep
from app.notes import service
from app.notes.schemas import Category, NoteCreate, NoteList, NoteListParams, NoteRead, Tag

logger = logging.getLogger(__name__)

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
    if ai.transcriber is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Transcription requires OPENAI_API_KEY"
        )
    fmt = audio_format(file.content_type, file.filename)
    if fmt is None:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Unsupported audio format")
    audio = await file.read(MAX_AUDIO_BYTES + 1)
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"Audio exceeds {MAX_AUDIO_BYTES} bytes"
        )
    if not audio:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Audio file is empty")

    try:
        transcript = (await ai.transcriber.transcribe(audio, fmt)).strip()
    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Transcription failed") from None
    if not transcript:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No speech recognized")

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
