from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.answers import Answerer, SourceNote
from app.ai.embeddings import Embedder
from app.notes.models import Note
from app.notes.service import apply_filters
from app.search.schemas import QueryRequest, SearchRequest


class QueryEmbeddingError(Exception):
    pass


class AnswerGenerationError(Exception):
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


async def answer_query(
    session: AsyncSession, embedder: Embedder, answerer: Answerer, request: QueryRequest
) -> tuple[str | None, list[tuple[Note, float]]]:
    """Answer the query from the most similar notes. Returns the answer, or None if no
    notes matched, and the hits it cites, most similar first."""
    hits = await search_notes(session, embedder, request)
    # End the read transaction so no connection is held during the LLM call. Loaded
    # notes stay readable.
    await session.close()
    if not hits:
        return None, []
    notes = [SourceNote(created_at=note.created_at, text=note.raw_transcript) for note, _ in hits]
    try:
        result = await answerer.answer(request.query, notes)
    except Exception as exc:
        raise AnswerGenerationError from exc
    # Numbers are 1-based; ignore any the model invented.
    cited = sorted({n for n in result.note_numbers if 1 <= n <= len(hits)})
    return result.answer, [hits[n - 1] for n in cited]
