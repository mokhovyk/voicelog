import logging
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from pydantic import AwareDatetime, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.services import AIDep, AIServices
from app.audio import transcribe_upload
from app.core.database import SessionDep
from app.notes.models import Note
from app.notes.schemas import Category, NoteRead, Tag
from app.search import service
from app.search.schemas import (
    MAX_QUERY_LENGTH,
    QueryAnswer,
    QueryRequest,
    SearchHit,
    SearchRequest,
    SearchResults,
    VoiceQueryAnswer,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post("/search")
async def search(request: SearchRequest, session: SessionDep, ai: AIDep) -> SearchResults:
    """Semantic search over enriched notes, most similar first."""
    if ai.embedder is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Search requires OPENAI_API_KEY")
    try:
        hits = await service.search_notes(session, ai.embedder, request)
    except service.QueryEmbeddingError:
        logger.exception("Query embedding failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Query embedding failed") from None
    return SearchResults(items=_hits(hits))


@router.post("/query/text")
async def query_text(request: QueryRequest, session: SessionDep, ai: AIDep) -> QueryAnswer:
    """Answer a question from the most similar notes, citing the ones it used."""
    _require_query_services(ai)
    return await _answer(request, session, ai)


@router.post("/query/voice")
async def query_voice(
    file: UploadFile,
    session: SessionDep,
    ai: AIDep,
    category: Annotated[Category | None, Form()] = None,
    tag: Annotated[Tag | None, Form()] = None,
    created_after: Annotated[AwareDatetime | None, Form()] = None,
    created_before: Annotated[AwareDatetime | None, Form()] = None,
    limit: Annotated[int, Form(ge=1, le=20)] = 5,
) -> VoiceQueryAnswer:
    """Transcribe a spoken question, then answer it like `/query/text`. The audio is
    not stored."""
    _require_query_services(ai)
    transcript = await transcribe_upload(file, ai.transcriber)
    try:
        request = QueryRequest(
            query=transcript,
            category=category,
            tag=tag,
            created_after=created_after,
            created_before=created_before,
            limit=limit,
        )
    except ValidationError:
        # The form fields were already validated, so only the transcript can be invalid.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Question is longer than {MAX_QUERY_LENGTH} characters",
        ) from None
    answer = await _answer(request, session, ai)
    return VoiceQueryAnswer(query=request.query, answer=answer.answer, sources=answer.sources)


def _require_query_services(ai: AIServices) -> None:
    if ai.embedder is None or ai.answerer is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Queries require OPENAI_API_KEY")


async def _answer(request: QueryRequest, session: AsyncSession, ai: AIServices) -> QueryAnswer:
    try:
        answer, sources = await service.answer_query(session, ai.embedder, ai.answerer, request)
    except service.QueryEmbeddingError:
        logger.exception("Query embedding failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Query embedding failed") from None
    except service.AnswerGenerationError:
        logger.exception("Answer generation failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Answer generation failed") from None
    return QueryAnswer(answer=answer, sources=_hits(sources))


def _hits(hits: list[tuple[Note, float]]) -> list[SearchHit]:
    return [SearchHit(score=score, note=NoteRead.model_validate(note)) for note, score in hits]
