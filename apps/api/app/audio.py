"""Audio uploads shared by audio notes and voice questions."""

import logging

from fastapi import HTTPException, UploadFile, status

from app.ai.transcription import MAX_AUDIO_BYTES, Transcriber, audio_format

logger = logging.getLogger(__name__)


async def transcribe_upload(file: UploadFile, transcriber: Transcriber | None) -> str:
    """The uploaded audio's transcript, stripped and non-empty, or an HTTP error: 503
    without a transcriber, 415, 413, or 422 for the file, 502 if transcription fails,
    and 422 if no speech was recognized."""
    if transcriber is None:
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
        transcript = (await transcriber.transcribe(audio, fmt)).strip()
    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Transcription failed") from None
    if not transcript:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No speech recognized")
    return transcript
