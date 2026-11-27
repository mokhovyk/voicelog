import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent import service
from app.agent.schemas import AgentRequest, AgentResponse
from app.ai.services import AIDep, AIServices
from app.audio import transcribe_upload
from app.core.database import SessionDep, SessionmakerDep
from app.search import service as search
from app.search.schemas import MAX_QUERY_LENGTH

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/text")
async def agent_text(
    request: AgentRequest,
    session: SessionDep,
    sessionmaker: SessionmakerDep,
    ai: AIDep,
    background: BackgroundTasks,
) -> AgentResponse:
    """Carry out a typed request: save a note, answer from notes, list open tasks,
    mark one done, or reply directly."""
    _require_agent(ai)
    return await _handle(request.utterance, session, sessionmaker, ai, background)


@router.post("/voice")
async def agent_voice(
    file: UploadFile,
    session: SessionDep,
    sessionmaker: SessionmakerDep,
    ai: AIDep,
    background: BackgroundTasks,
) -> AgentResponse:
    """Transcribe a spoken request, then carry it out like `/agent/text`. The audio is
    not stored."""
    _require_agent(ai)
    transcript = await transcribe_upload(file, ai.transcriber)
    try:
        request = AgentRequest(utterance=transcript)
    except ValidationError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Request is longer than {MAX_QUERY_LENGTH} characters",
        ) from None
    return await _handle(request.utterance, session, sessionmaker, ai, background)


def _require_agent(ai: AIServices) -> None:
    if ai.interpreter is None or ai.embedder is None or ai.answerer is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The agent requires OPENAI_API_KEY"
        )


async def _handle(
    utterance: str,
    session: AsyncSession,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai: AIServices,
    background: BackgroundTasks,
) -> AgentResponse:
    try:
        return await service.handle(utterance, session, sessionmaker, ai, background)
    except service.InterpretationFailed:
        logger.exception("Agent interpretation failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "Interpreting the request failed"
        ) from None
    except search.QueryEmbeddingError:
        logger.exception("Query embedding failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Query embedding failed") from None
    except search.AnswerGenerationError:
        logger.exception("Answer generation failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Answer generation failed") from None
