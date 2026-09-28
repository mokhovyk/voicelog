import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text

from app.core.database import SessionDep

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


@router.get("/health")
async def health() -> HealthResponse:
    """Liveness check. Does not touch the database."""
    return HealthResponse(status="ok")


@router.get("/health/ready", responses={503: {"description": "Database unreachable"}})
async def ready(session: SessionDep) -> HealthResponse:
    """Readiness check: the database accepts queries."""
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("Readiness check failed: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable") from exc
    return HealthResponse(status="ok")
