from datetime import date, datetime

import pytest
from httpx import AsyncClient

from app.agent.service import match_score
from app.ai.agent import (
    AskNotes,
    Command,
    CompleteActionItem,
    CreateNote,
    ListActionItems,
    Reply,
)
from app.ai.answers import Answer
from app.ai.extraction import ExtractedActionItem, NoteMetadata
from app.ai.services import AIServices
from tests.conftest import FakeAnswerer, FakeTranscriber, KeywordEmbedder

pytestmark = pytest.mark.anyio

KEYWORDS = ["bug", "login", "sarah", "gym", "groceries"]


class FakeInterpreter:
    def __init__(self, result: Command | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, datetime]] = []

    async def interpret(self, utterance: str, now: datetime) -> Command:
        self.calls.append((utterance, now))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class TasksExtractor:
    """Extracts the action items listed after "tasks:", separated by commas."""

    async def extract(self, transcript: str) -> NoteMetadata:
        _, _, tasks = transcript.partition("tasks:")
        items = [ExtractedActionItem(text=t.strip()) for t in tasks.split(",") if t.strip()]
        return NoteMetadata(summary=transcript, category="Work", tags=[], action_items=items)


def agent_ai(command: Command | Exception, transcript: str = "") -> AIServices:
    return AIServices(
        extractor=TasksExtractor(),
        embedder=KeywordEmbedder(KEYWORDS),
        answerer=FakeAnswerer(Answer(answer="By Friday.", note_numbers=[1])),
        transcriber=FakeTranscriber(transcript),
        interpreter=FakeInterpreter(command),
    )


async def create(client: AsyncClient, transcript: str) -> dict:
    response = await client.post("/api/v1/notes", json={"raw_transcript": transcript})
    assert response.status_code == 201, response.text
    # Background enrichment has run by the time the test client returns.
    return (await client.get(f"/api/v1/notes/{response.json()['id']}")).json()


async def say(client: AsyncClient, utterance: str) -> dict:
    response = await client.post("/api/v1/agent/text", json={"utterance": utterance})
    assert response.status_code == 200, response.text
    return response.json()


class TestCreateNote:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(CreateNote(text="call mom on sunday"))

    async def test_saves_the_note_text_and_enriches_it(
        self, client: AsyncClient, ai: AIServices
    ) -> None:
        result = await say(client, "  remind me to call mom on sunday ")

        assert result["action"] == "create_note"
        assert result["utterance"] == "remind me to call mom on sunday"
        assert result["reply"] == "Saved to your journal."
        assert result["note"]["raw_transcript"] == "call mom on sunday"
        saved = (await client.get(f"/api/v1/notes/{result['note']['id']}")).json()
        assert saved["status"] == "ready"
        [(utterance, now)] = ai.interpreter.calls
        assert utterance == "remind me to call mom on sunday"
        assert now.tzinfo is not None


class TestAskNotes:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(
            AskNotes(
                question="login bug sarah",
                # Reversed on purpose: the range is normalized.
                start_date=date(2099, 1, 1),
                end_date=date(2000, 1, 1),
            )
        )

    async def test_answers_from_notes_in_the_date_range(
        self, client: AsyncClient, ai: AIServices
    ) -> None:
        bug = await create(client, "fix the login bug for sarah")

        result = await say(client, "what did sarah need")

        assert result["action"] == "ask_notes"
        assert result["reply"] == "By Friday."
        assert [source["note"]["id"] for source in result["sources"]] == [bug["id"]]
        [(question, _)] = ai.answerer.calls
        assert question == "login bug sarah"


class TestAskNotesOutsideRange:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(AskNotes(question="login bug", end_date=date(2000, 1, 1)))

    async def test_says_so_when_no_note_matches(self, client: AsyncClient, ai: AIServices) -> None:
        await create(client, "fix the login bug")

        result = await say(client, "what bug did I have in 1999")

        assert result["reply"] == "I couldn't find any notes about that."
        assert result["sources"] == []
        assert ai.answerer.calls == []


class TestAskNotesEmptyQuestion:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(AskNotes(question="  "))

    async def test_searches_with_the_utterance(self, client: AsyncClient, ai: AIServices) -> None:
        await create(client, "fix the login bug")

        await say(client, "login bug?")

        [(question, _)] = ai.answerer.calls
        assert question == "login bug?"


class TestListActionItems:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(ListActionItems())

    async def test_lists_open_items_newest_first(self, client: AsyncClient) -> None:
        old = await create(client, "work tasks: fix login bug, write tests")
        await create(client, "nothing to do today")
        new = await create(client, "home tasks: buy groceries")

        result = await say(client, "what do I have to do")

        assert result["action"] == "list_action_items"
        assert [(i["note_id"], i["index"], i["text"]) for i in result["action_items"]] == [
            (new["id"], 0, "buy groceries"),
            (old["id"], 0, "fix login bug"),
            (old["id"], 1, "write tests"),
        ]
        assert (
            result["reply"] == "You have 3 open tasks: buy groceries; fix login bug; write tests."
        )

    async def test_reads_out_at_most_five(self, client: AsyncClient) -> None:
        await create(client, "tasks: a1, a2, a3, a4, a5, a6, a7")

        result = await say(client, "what do I have to do")

        assert len(result["action_items"]) == 7
        assert result["reply"] == "You have 7 open tasks: a1; a2; a3; a4; a5; and 2 more."

    async def test_no_open_items(self, client: AsyncClient) -> None:
        await create(client, "nothing to do")

        result = await say(client, "what do I have to do")

        assert result == {**result, "action_items": [], "reply": "You have no open tasks."}


class TestListActionItemsInRange:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(ListActionItems(end_date=date(2000, 1, 1)))

    async def test_filters_by_date(self, client: AsyncClient) -> None:
        await create(client, "tasks: fix login bug")

        result = await say(client, "what did I have to do in 1999")

        assert result["action_items"] == []


class TestCompleteActionItem:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(CompleteActionItem(item="the login bug"))

    async def test_marks_the_matching_item_done(self, client: AsyncClient) -> None:
        note = await create(client, "tasks: buy groceries, fix login bug by friday")

        result = await say(client, "I fixed the login bug")

        assert result["action"] == "complete_action_item"
        assert result["reply"] == "Marked “fix login bug by friday” as done."
        expected = [
            {"text": "buy groceries", "done": False},
            {"text": "fix login bug by friday", "done": True},
        ]
        assert result["note"]["action_items"] == expected
        saved = (await client.get(f"/api/v1/notes/{note['id']}")).json()
        assert saved["action_items"] == expected

    async def test_asks_which_when_several_match(self, client: AsyncClient) -> None:
        await create(client, "tasks: fix login bug on web, fix login bug on ios")

        result = await say(client, "I fixed the login bug")

        assert result["reply"] == (
            "Which task did you mean: “fix login bug on web” or “fix login bug on ios”?"
        )
        assert len(result["action_items"]) == 2
        assert result["note"] is None

    async def test_duplicates_complete_the_newest(self, client: AsyncClient) -> None:
        await create(client, "tasks: fix login bug")
        new = await create(client, "tasks: fix login bug")

        result = await say(client, "I fixed the login bug")

        assert result["note"]["id"] == new["id"]

    async def test_says_so_when_nothing_matches(self, client: AsyncClient) -> None:
        await create(client, "tasks: buy groceries")

        result = await say(client, "I fixed the login bug")

        assert result["reply"] == "I couldn't find an open task matching “the login bug”."
        assert result["note"] is None


class TestReply:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(Reply(message="I can't check the weather."))

    async def test_replies_without_touching_notes(self, client: AsyncClient) -> None:
        result = await say(client, "what's the weather")

        assert result["action"] == "reply"
        assert result["reply"] == "I can't check the weather."
        assert (await client.get("/api/v1/notes")).json()["total"] == 0


class TestVoice:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(Reply(message="Hi!"), transcript=" hello there ")

    async def test_transcribes_then_acts(self, client: AsyncClient, ai: AIServices) -> None:
        response = await client.post(
            "/api/v1/agent/voice", files={"file": ("blob", b"audio", "audio/webm")}
        )

        assert response.status_code == 200, response.text
        assert response.json()["utterance"] == "hello there"
        assert ai.interpreter.calls[0][0] == "hello there"


class TestErrors:
    @pytest.fixture
    def ai(self) -> AIServices:
        return agent_ai(RuntimeError("OpenAI is down"))

    async def test_interpretation_failure_is_502(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/agent/text", json={"utterance": "hi"})

        assert response.status_code == 502

    @pytest.mark.parametrize("payload", [{}, {"utterance": "  "}, {"utterance": "x" * 2001}])
    async def test_rejects_invalid_input(self, client: AsyncClient, payload: dict) -> None:
        response = await client.post("/api/v1/agent/text", json=payload)

        assert response.status_code == 422


@pytest.mark.parametrize("path", ["/api/v1/agent/text", "/api/v1/agent/voice"])
async def test_unavailable_without_ai(client: AsyncClient, path: str) -> None:
    if path.endswith("text"):
        response = await client.post(path, json={"utterance": "hi"})
    else:
        response = await client.post(path, files={"file": ("a.webm", b"x", "audio/webm")})

    assert response.status_code == 503


@pytest.mark.parametrize(
    ("description", "task", "matches"),
    [
        ("the login bug", "Fix login bug by Friday", True),
        ("fixed the login bug", "fix the login bug", True),
        ("groceries", "Buy grocery items", True),
        ("call mom", "Fix login bug", False),
        ("купити продукти", "Купити продукти на вихідні", True),
        ("", "anything", False),
    ],
)
def test_match_score(description: str, task: str, matches: bool) -> None:
    assert (match_score(description, task) >= 0.5) is matches
