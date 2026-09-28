from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embeddings import Embedder
from app.notes.models import Note
from app.notes.service import apply_filters
from app.search.schemas import SearchRequest


class QueryEmbeddingError(Exception):
    pass


async def search_notes(
    session: AsyncSession, embedder: Embedder, request: SearchRequest
) -> list[tuple[Note, float]]:
    """Notes closest to the query, most similar first, with their cosine similarity."""
    try:
        query_vector = await embedder.embed(request.query)
    except Exception as exc:
        raise QueryEmbeddingError from exc
    distance = Note.embedding.cosine_distance(query_vector)
    query = (
        select(Note, distance)
        # Vectors from different models are not comparable.
        .where(Note.embedding.is_not(None), Note.embedding_model == embedder.model)
        .order_by(distance)
        .limit(request.limit)
    )
    query = apply_filters(query, request)

    # HNSW filters after finding its candidates, so filtered searches could return fewer
    # than `limit` notes; iterative scans keep searching until enough rows match.
    await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    rows = (await session.execute(query)).all()
    # relaxed_order may return rows slightly out of order.
    return sorted(((note, 1 - dist) for note, dist in rows), key=lambda hit: -hit[1])
