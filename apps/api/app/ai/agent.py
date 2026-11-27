"""The voice agent's model step: turn what the user said into one journal action.

The model only chooses the action and its arguments; `app/agent/` carries it out.
Keeping the decision a plain value makes it testable, and evaluable, on its own."""

from datetime import UTC, date, datetime
from typing import Any, Protocol

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field, SecretStr, ValidationError

from app.ai.extraction import Category
from app.ai.llm import chat_model


class CreateNote(BaseModel):
    """Save a new journal note. Use for anything the user wants recorded or remembered:
    dictated thoughts, events of their day, ideas, and reminders or tasks to do later
    ("remind me to call mom", "I need to renew my passport")."""

    text: str = Field(
        description="The note in the user's words and language, without the command "
        "phrase such as 'note that' or 'remember that'."
    )


class _DateRange(BaseModel):
    start_date: date | None = Field(
        None,
        description="First day of the period the user names (inclusive), YYYY-MM-DD. "
        "Null if they name no period.",
    )
    end_date: date | None = Field(
        None,
        description="Last day of the period (inclusive), YYYY-MM-DD. Null if they name "
        "no period, or the period runs to today and start_date is set.",
    )
    category: Category | None = Field(
        None, description="Only if the user explicitly limits to a category."
    )


class AskNotes(_DateRange):
    """Answer a question about what the user recorded in past notes ("what did Sarah
    say about the release?", "remind me what I decided about the trip")."""

    question: str = Field(
        description="The question as a standalone search query, in the user's language."
    )


class ListActionItems(_DateRange):
    """List the user's open tasks and to-dos ("what do I have to do?", "any open
    tasks from last week?")."""


class CompleteActionItem(BaseModel):
    """Mark one of the user's open tasks as done ("I fixed the login bug", "tick off
    the groceries")."""

    item: str = Field(
        description="The task as the user describes it, in their language, e.g. 'fix "
        "the login bug'."
    )


class Reply(BaseModel):
    """Answer directly without using the journal: greetings, questions about what you
    can do, requests outside the journal (weather, timers, emails), or a request too
    unclear to act on, in which case ask what they meant."""

    message: str = Field(description="One or two short sentences, in the user's language.")


Command = CreateNote | AskNotes | ListActionItems | CompleteActionItem | Reply

TOOLS: dict[str, type[Command]] = {
    "create_note": CreateNote,
    "ask_notes": AskNotes,
    "list_action_items": ListActionItems,
    "complete_action_item": CompleteActionItem,
    "reply": Reply,
}
TOOL_NAMES = {model: name for name, model in TOOLS.items()}


def tool_definitions() -> list[dict[str, Any]]:
    """The tools in OpenAI's format, in strict mode so arguments match the schemas."""
    definitions = []
    for name, model in TOOLS.items():
        definition = convert_to_openai_tool(model, strict=True)
        definition["function"]["name"] = name
        definitions.append(definition)
    return definitions


SYSTEM_PROMPT = """\
You are the voice assistant of the user's personal voice journal. Each message is a \
transcript of something the user said. Choose exactly one tool for it.

Today is {today}. Weeks run Monday to Sunday. Resolve periods to dates: "yesterday" \
is one day; "this week" starts on this week's Monday; "last week" is the previous \
Monday to Sunday; "last month" is the previous calendar month; a month or weekday \
without a year means the most recent one that is not in the future.

Asking about the past ("what did I say about...", "remind me what...") is ask_notes; \
telling something to keep ("remind me to...", "note that...") is create_note. \
Transcripts can contain recognition errors; act on the most likely meaning. Keep \
arguments in the user's language."""


class Interpreter(Protocol):
    async def interpret(self, utterance: str, now: datetime) -> Command: ...


class InterpretationError(Exception):
    pass


def parse_tool_call(message: AIMessage) -> Command:
    """The command in the model's reply. A reply without a tool call is read as a
    direct answer; an unknown tool or invalid arguments raise InterpretationError."""
    if not message.tool_calls:
        text = message.text.strip()
        if not text:
            raise InterpretationError("The model returned neither a tool call nor text")
        return Reply(message=text)
    call = message.tool_calls[0]
    model = TOOLS.get(call["name"])
    if model is None:
        raise InterpretationError(f"Unknown tool {call['name']!r}")
    try:
        return model.model_validate(call["args"])
    except ValidationError as exc:
        raise InterpretationError(f"Invalid arguments for {call['name']}: {exc}") from exc


class OpenAIInterpreter:
    def __init__(
        self, api_key: SecretStr, model: str, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self.model = model
        self._llm = chat_model(api_key, model, http_client, timeout=30).bind_tools(
            tool_definitions(), tool_choice="required", parallel_tool_calls=False
        )

    async def interpret(self, utterance: str, now: datetime) -> Command:
        today = now.astimezone(UTC).strftime("%A, %Y-%m-%d")
        message = await self._llm.ainvoke(
            [SystemMessage(SYSTEM_PROMPT.format(today=today)), HumanMessage(utterance)]
        )
        assert isinstance(message, AIMessage)
        return parse_tool_call(message)
