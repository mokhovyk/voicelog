"""The eval suites: their case formats, how a case runs against a model, and how its
output is scored. Datasets live in evals/datasets/<suite>.jsonl, one case per line."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.ai.agent import TOOL_NAMES, TOOLS, OpenAIInterpreter
from app.ai.answers import OpenAIAnswerer, SourceNote
from app.ai.extraction import Category, OpenAIMetadataExtractor
from evals import scoring
from evals.scoring import Phrase

DATASETS = Path(__file__).parent / "datasets"
# A Friday. Cases run as if it were noon UTC that day unless they set `today`.
DEFAULT_TODAY = date(2026, 11, 27)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Case(_Strict):
    id: str
    # A critical case may not regress when switching to a cheaper model.
    critical: bool = False
    today: date = DEFAULT_TODAY
    # Why the case exists; shown in reports next to failures.
    note: str = ""


class AgentExpect(_Strict):
    tool: str | list[str]
    # Arguments that must equal a value (dates as "YYYY-MM-DD", null for none) ...
    args: dict[str, Any] = {}
    # ... or one of several values.
    args_one_of: dict[str, list[Any]] = {}
    includes: dict[str, list[Phrase]] = {}
    excludes: dict[str, list[Phrase]] = {}

    def model_post_init(self, _: Any) -> None:
        tools = [self.tool] if isinstance(self.tool, str) else self.tool
        unknown = set(tools) - set(TOOLS)
        if unknown:
            raise ValueError(f"Unknown tools {sorted(unknown)}")
        fields = {name for tool in tools for name in TOOLS[tool].model_fields}
        named = {*self.args, *self.args_one_of, *self.includes, *self.excludes}
        if named - fields:
            raise ValueError(f"No such arguments: {sorted(named - fields)}")


class AgentCase(_Case):
    utterance: str
    expect: AgentExpect


class ExtractionExpect(_Strict):
    category: Category | list[Category]
    # Tasks that must be found, each matched by a different extracted item.
    action_items: list[Phrase] = []
    # Defaults to one more than the expected tasks, as models may split a task.
    max_action_items: int | None = None
    summary_script: str | None = Field(None, pattern="^(latin|cyrillic)$")
    summary_includes: list[Phrase] = []


class ExtractionCase(_Case):
    transcript: str
    expect: ExtractionExpect


class SourceNoteCase(_Strict):
    date: date
    text: str


class AnswerExpect(_Strict):
    # Note numbers (1-based) the answer must cite, and others it may cite.
    cites: list[int] = []
    may_cite: list[int] = []
    answer_includes: list[Phrase] = []
    answer_excludes: list[Phrase] = []
    answer_script: str | None = Field(None, pattern="^(latin|cyrillic)$")


class AnswerCase(_Case):
    question: str
    notes: list[SourceNoteCase]
    expect: AnswerExpect

    def model_post_init(self, _: Any) -> None:
        numbers = {*self.expect.cites, *self.expect.may_cite}
        if not numbers <= set(range(1, len(self.notes) + 1)):
            raise ValueError(f"{self.id}: cites a note that does not exist")


def _noon(day: date) -> datetime:
    return datetime.combine(day, time(12), tzinfo=UTC)


# Runs one case: (case) -> JSON-ready output.
Target = Callable[[Any], Awaitable[dict]]
# Builds a target for a model spec.
TargetFactory = Callable[[str, SecretStr, httpx.AsyncClient | None], Target]


def _agent_target(spec: str, key: SecretStr, http: httpx.AsyncClient | None) -> Target:
    interpreter = OpenAIInterpreter(key, spec, http)

    async def run(case: AgentCase) -> dict:
        command = await interpreter.interpret(case.utterance, _noon(case.today))
        return {"tool": TOOL_NAMES[type(command)], "args": command.model_dump(mode="json")}

    return run


def _extraction_target(spec: str, key: SecretStr, http: httpx.AsyncClient | None) -> Target:
    extractor = OpenAIMetadataExtractor(key, spec, http)

    async def run(case: ExtractionCase) -> dict:
        return (await extractor.extract(case.transcript)).model_dump(mode="json")

    return run


def _answer_target(spec: str, key: SecretStr, http: httpx.AsyncClient | None) -> Target:
    async def run(case: AnswerCase) -> dict:
        # One answerer per case, so each case has its own "today".
        answerer = OpenAIAnswerer(key, spec, http, clock=lambda: _noon(case.today))
        notes = [SourceNote(created_at=_noon(n.date), text=n.text) for n in case.notes]
        return (await answerer.answer(case.question, notes)).model_dump(mode="json")

    return run


@dataclass(frozen=True)
class Suite:
    name: str
    # The setting whose model this suite measures.
    setting: str
    case_type: type[_Case]
    target: TargetFactory
    score: Callable[[Any, dict], scoring.Score]
    # What "does the same thing" means for this suite: outputs with equal signatures
    # agree, so a candidate can be compared with the baseline on every case.
    signature: Callable[[dict], Any]

    def load(self, path: Path | None = None) -> list[Any]:
        path = path or DATASETS / f"{self.name}.jsonl"
        cases = []
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.strip() and not line.lstrip().startswith("//"):
                try:
                    cases.append(self.case_type.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"{path.name}:{number}: {exc}") from exc
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path.name}: duplicate case ids")
        return cases


SUITES = {
    suite.name: suite
    for suite in [
        Suite(
            name="agent",
            setting="OPENAI_AGENT_MODEL",
            case_type=AgentCase,
            target=_agent_target,
            score=lambda case, output: scoring.score_agent(case.expect, output),
            signature=lambda output: json.dumps(output, sort_keys=True),
        ),
        Suite(
            name="extraction",
            setting="OPENAI_CHAT_MODEL",
            case_type=ExtractionCase,
            target=_extraction_target,
            score=lambda case, output: scoring.score_extraction(case.expect, output),
            signature=lambda output: (output["category"], len(output["action_items"])),
        ),
        Suite(
            name="answers",
            setting="OPENAI_CHAT_MODEL",
            case_type=AnswerCase,
            target=_answer_target,
            score=lambda case, output: scoring.score_answer(case.expect, output),
            signature=lambda output: sorted(set(output["note_numbers"])),
        ),
    ]
}
