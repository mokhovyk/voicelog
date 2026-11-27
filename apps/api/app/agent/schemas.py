import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

from app.notes.schemas import NoteRead
from app.search.schemas import MAX_QUERY_LENGTH, SearchHit

AgentAction = Literal[
    "create_note", "ask_notes", "list_action_items", "complete_action_item", "reply"
]


class AgentRequest(BaseModel):
    utterance: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)
    ]


class OpenActionItem(BaseModel):
    note_id: uuid.UUID
    # Position in the note's action_items.
    index: int
    text: str


class AgentResponse(BaseModel):
    # What the user said: the request text, or the transcript for voice.
    utterance: str
    action: AgentAction
    # What to say back to the user.
    reply: str
    # create_note: the new note, still pending. complete_action_item: the updated note.
    note: NoteRead | None = None
    # ask_notes: the notes the answer cites, most similar first.
    sources: list[SearchHit] = []
    # list_action_items: the open items, newest note first. complete_action_item: the
    # candidates when the request matched several.
    action_items: list[OpenActionItem] = []
