import logging

from fastapi import APIRouter, HTTPException, status

from app.ai.services import AIDep
from app.core.database import SessionDep
from app.notes.models import Note
from app.notes.schemas import NoteRead
from app.search import service
from app.search.schemas import QueryAnswer, QueryRequest, SearchHit, SearchRequest, SearchResults

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
    if ai.embedder is None or ai.answerer is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Queries require OPENAI_API_KEY")
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
