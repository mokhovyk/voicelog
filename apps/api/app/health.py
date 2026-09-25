from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str


@router.get("/health")
async def health() -> HealthResponse:
    """Liveness check. Does not touch the database."""
    return HealthResponse(status="ok")
