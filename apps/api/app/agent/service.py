import re
from datetime import UTC, date, datetime, time, timedelta
from difflib import SequenceMatcher

from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.schemas import AgentResponse, OpenActionItem
from app.ai.agent import (
    TOOL_NAMES,
    AskNotes,
    CompleteActionItem,
    CreateNote,
    ListActionItems,
    Reply,
)
from app.ai.services import AIServices
from app.notes import service as notes
from app.notes.models import Note
from app.notes.schemas import NoteCreate, NoteFilters, NoteRead
from app.search import service as search
from app.search.schemas import MAX_QUERY_LENGTH, QueryRequest

# Open items are looked up in this many of the newest notes that have any.
MAX_NOTES_SCANNED = 200
# How many open items the spoken reply reads out; the response lists all of them.
MAX_SPOKEN_ITEMS = 5
# A task matches a description scoring at least this, unless another scores close.
MATCH_THRESHOLD = 0.5
AMBIGUITY_MARGIN = 0.1


class InterpretationFailed(Exception):
    pass


async def handle(
    utterance: str,
    session: AsyncSession,
    sessionmaker: async_sessionmaker[AsyncSession],
    ai: AIServices,
    background: BackgroundTasks,
    now: datetime | None = None,
) -> AgentResponse:
    """Let the model pick an action for the utterance and carry it out. Raises
    InterpretationFailed, or the search errors when answering a question fails."""
    assert ai.interpreter is not None
    try:
        command = await ai.interpreter.interpret(utterance, now or datetime.now(UTC))
    except Exception as exc:
        raise InterpretationFailed from exc

    response = AgentResponse(utterance=utterance, action=TOOL_NAMES[type(command)], reply="")
    match command:
        case CreateNote():
            text = command.text.strip() or utterance
            note = await notes.create_note(session, NoteCreate(raw_transcript=text))
            background.add_task(notes.enrich_note, sessionmaker, ai, note.id, text)
            response.note = NoteRead.model_validate(note)
            response.reply = "Saved to your journal."
        case AskNotes():
            await _ask(session, ai, command, response)
        case ListActionItems():
            items = await _open_items(session, _filters(command))
            response.action_items = items
            response.reply = _spoken_list(items)
        case CompleteActionItem():
            await _complete(session, command.item, response)
        case Reply():
            response.reply = command.message
    return response


async def _ask(
    session: AsyncSession, ai: AIServices, command: AskNotes, response: AgentResponse
) -> None:
    assert ai.embedder is not None and ai.answerer is not None
    filters = _filters(command)
    # The utterance stands in for an empty question; both fit the query limit.
    question = command.question.strip()[:MAX_QUERY_LENGTH] or response.utterance
    request = QueryRequest(query=question, **filters.model_dump())
    answer, hits = await search.answer_query(session, ai.embedder, ai.answerer, request)
    response.reply = answer or "I couldn't find any notes about that."
    response.sources = search.to_hits(hits)


def _filters(command: ListActionItems | AskNotes) -> NoteFilters:
    """Inclusive dates as the half-open UTC range NoteFilters uses."""
    start, end = command.start_date, command.end_date
    if start and end and start > end:
        start, end = end, start
    return NoteFilters(
        category=command.category,
        created_after=_midnight(start) if start else None,
        created_before=_midnight(end + timedelta(days=1)) if end else None,
    )


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time(), tzinfo=UTC)


async def _open_notes(session: AsyncSession, filters: NoteFilters) -> list[Note]:
    query = (
        select(Note)
        # JSONB containment: any element has done = false. Uses no index; the scan is
        # bounded by the limit.
        .where(Note.action_items.contains([{"done": False}]))
        .order_by(Note.created_at.desc(), Note.id.desc())
        .limit(MAX_NOTES_SCANNED)
    )
    return list(await session.scalars(notes.apply_filters(query, filters)))


def _items(note_list: list[Note]) -> list[OpenActionItem]:
    return [
        OpenActionItem(note_id=note.id, index=index, text=item["text"])
        for note in note_list
        for index, item in enumerate(note.action_items)
        if not item.get("done")
    ]


async def _open_items(session: AsyncSession, filters: NoteFilters) -> list[OpenActionItem]:
    return _items(await _open_notes(session, filters))


def _spoken_list(items: list[OpenActionItem]) -> str:
    if not items:
        return "You have no open tasks."
    spoken = "; ".join(item.text for item in items[:MAX_SPOKEN_ITEMS])
    more = len(items) - MAX_SPOKEN_ITEMS
    count = "1 open task" if len(items) == 1 else f"{len(items)} open tasks"
    return f"You have {count}: {spoken}" + (f"; and {more} more." if more > 0 else ".")


async def _complete(session: AsyncSession, description: str, response: AgentResponse) -> None:
    note_list = await _open_notes(session, NoteFilters())
    scored = sorted(
        ((match_score(description, item.text), item) for item in _items(note_list)),
        key=lambda pair: -pair[0],  # stable: the newest note wins a tie
    )
    matches = [(score, item) for score, item in scored if score >= MATCH_THRESHOLD]
    if not matches:
        response.reply = f"I couldn't find an open task matching “{description}”."
        return
    best_score, best = matches[0]
    rivals = [
        item
        for score, item in matches[1:]
        if best_score - score < AMBIGUITY_MARGIN and _norm(item.text) != _norm(best.text)
    ]
    if rivals:
        response.action_items = [best, *rivals]
        options = " or ".join(f"“{item.text}”" for item in response.action_items)
        response.reply = f"Which task did you mean: {options}?"
        return

    note = next(note for note in note_list if note.id == best.note_id)
    # Reassign so SQLAlchemy sees the JSONB change.
    note.action_items = [
        {**item, "done": True} if index == best.index else item
        for index, item in enumerate(note.action_items)
    ]
    await session.commit()
    await session.refresh(note)
    response.note = NoteRead.model_validate(note)
    response.reply = f"Marked “{best.text}” as done."


_WORD = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "a an and as at by done for finished i is it mark my of off on the tick to with".split()
)


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _words(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS]


def _same_word(a: str, b: str) -> bool:
    """Equal, or sharing a stem: "fixed" and "fix", "groceries" and "grocery"."""
    short, long = sorted((a, b), key=len)
    prefix = len(short) >= 3 and long.startswith(short)
    common = 0
    while common < len(short) and short[common] == long[common]:
        common += 1
    return a == b or prefix or common >= 5


def match_score(description: str, task: str) -> float:
    """How well a spoken description fits a task, from 0 to 1: the share of the
    description's words found in the task, or the character similarity if higher,
    which helps with phrasings that share few whole words."""
    words, task_words = _words(description), _words(task)
    overlap = (
        sum(any(_same_word(word, other) for other in task_words) for word in words) / len(words)
        if words
        else 0.0
    )
    return max(overlap, SequenceMatcher(None, _norm(description), _norm(task)).ratio())
