import logging

from fastapi import APIRouter, HTTPException, status

from app.ai.services import AIDep
from app.core.database import SessionDep
from app.notes.schemas import NoteRead
from app.search import service
from app.search.schemas import SearchHit, SearchRequest, SearchResults

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
    return SearchResults(
        items=[SearchHit(score=score, note=NoteRead.model_validate(note)) for note, score in hits]
    )
