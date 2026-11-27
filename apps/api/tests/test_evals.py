"""The eval harness: datasets, scorers, and a full run against a mocked OpenAI API."""

import json

import httpx
import pytest
from pydantic import SecretStr

from evals import report, runner
from evals.scoring import has, score_agent, score_answer, score_extraction, script
from evals.suites import SUITES, AgentCase, AnswerCase, ExtractionCase

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("name", list(SUITES))
def test_datasets_are_valid(name: str) -> None:
    cases = SUITES[name].load()

    assert len(cases) >= 15
    assert any(case.critical for case in cases)


def test_dataset_rejects_unknown_arguments(tmp_path) -> None:
    path = tmp_path / "agent.jsonl"
    path.write_text(
        '{"id": "x", "utterance": "hi", "expect": {"tool": "reply", "args": {"text": "hi"}}}\n'
    )

    with pytest.raises(ValueError, match="No such arguments"):
        SUITES["agent"].load(path)


def test_phrases_and_scripts() -> None:
    assert has("You’re due  on FRIDAY", "you're due on friday")
    assert has("fix login bug", ["logout", "login"])
    assert not has("fix login bug", ["logout"])
    assert script("Купити продукти") == "cyrillic"
    assert script("Buy groceries") == "latin"


def agent_case(**expect) -> AgentCase:
    return AgentCase.model_validate({"id": "c", "utterance": "u", "expect": expect})


def test_agent_scoring() -> None:
    case = agent_case(
        tool="ask_notes",
        args={"start_date": "2026-11-16"},
        args_one_of={"end_date": [None, "2026-11-22"]},
        includes={"question": ["sarah"]},
    )
    good = {"start_date": "2026-11-16", "end_date": None, "question": "What did Sarah say"}

    assert score_agent(case.expect, {"tool": "ask_notes", "args": good}).passed
    wrong_date = {**good, "start_date": "2026-11-17"}
    assert score_agent(case.expect, {"tool": "ask_notes", "args": wrong_date}).failed() == [
        "start_date = '2026-11-16'"
    ]
    assert score_agent(case.expect, {"tool": "reply", "args": {}}).failed() == ["tool"]


def test_extraction_scoring_matches_each_task_once() -> None:
    case = ExtractionCase.model_validate(
        {
            "id": "c",
            "transcript": "t",
            "expect": {"category": "Work", "action_items": ["bug", "bug"], "max_action_items": 2},
        }
    )
    output = {
        "summary": "s",
        "category": "Work",
        "tags": [],
        "action_items": [{"text": "fix bug", "done": False}],
    }

    score = score_extraction(case.expect, output)

    assert score.failed() == ["finds task 'bug'"]  # the second one


def test_answer_scoring_checks_citations_both_ways() -> None:
    case = AnswerCase.model_validate(
        {
            "id": "c",
            "question": "q",
            "notes": [{"date": "2026-11-01", "text": "a"}, {"date": "2026-11-02", "text": "b"}],
            "expect": {"cites": [2], "answer_includes": ["madrid"]},
        }
    )

    assert score_answer(case.expect, {"answer": "In Madrid.", "note_numbers": [2]}).passed
    assert score_answer(case.expect, {"answer": "In Madrid.", "note_numbers": [1, 2]}).failed() == [
        "cites only [2]"
    ]
    assert score_answer(case.expect, {"answer": "Porto", "note_numbers": []}).failed() == [
        "cites [2]",
        "answer has 'madrid'",
    ]


def test_answer_case_rejects_citations_of_missing_notes() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        AnswerCase.model_validate(
            {"id": "c", "question": "q", "notes": [], "expect": {"cites": [1]}}
        )


# A simulated run: "gpt-5.4" and "gpt-5.4-mini" get everything right, "gpt-5.4-nano"
# misroutes the critical agent case. The mock plays each model.

USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 100,
    "total_tokens": 1100,
    "prompt_tokens_details": {"cached_tokens": 200},
    "completion_tokens_details": {"reasoning_tokens": 40},
}


def completion(message: dict) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": USAGE,
    }


def tool_call(name: str, arguments: dict) -> dict:
    call = {
        "id": "1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return {"role": "assistant", "content": None, "tool_calls": [call]}


def content(data: dict) -> dict:
    return {"role": "assistant", "content": json.dumps(data)}


def simulated_openai(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    model, prompt = body["model"], body["messages"][-1]["content"]
    if "tools" in body:  # agent
        if "call mom" in prompt:
            message = (
                tool_call(
                    "ask_notes",
                    {"question": "mom", "start_date": None, "end_date": None, "category": None},
                )
                if model == "gpt-5.4-nano"
                else tool_call("create_note", {"text": "call mom on Sunday"})
            )
        else:
            message = tool_call("reply", {"message": "I can't check the weather."})
    elif "Question:" in prompt:  # answers
        message = content({"answer": "In Madrid.", "note_numbers": [2]})
    else:  # extraction
        message = content(
            {
                "summary": "A run.",
                "category": "Personal",
                "tags": ["running"],
                "action_items": [],
            }
        )
    return httpx.Response(200, json=completion(message))


CASES = {
    "agent": [
        AgentCase.model_validate(
            {
                "id": "remind-me",
                "critical": True,
                "utterance": "Remind me to call mom on Sunday.",
                "expect": {"tool": "create_note", "includes": {"text": ["call mom"]}},
            }
        ),
        AgentCase.model_validate(
            {"id": "weather", "utterance": "Weather?", "expect": {"tool": "reply"}}
        ),
    ],
    "extraction": [
        ExtractionCase.model_validate(
            {
                "id": "run",
                "transcript": "Went for a run.",
                "expect": {"category": "Personal", "max_action_items": 0},
            }
        )
    ],
    "answers": [
        AnswerCase.model_validate(
            {
                "id": "offsite",
                "question": "Where is the offsite?",
                "notes": [
                    {"date": "2026-11-10", "text": "Offsite in Porto."},
                    {"date": "2026-11-20", "text": "Offsite moved to Madrid."},
                ],
                "expect": {"cites": [2], "answer_includes": ["madrid"]},
            }
        )
    ],
}
MODELS = ["gpt-5.4", "gpt-5.4-mini@low", "gpt-5.4-nano"]


async def simulate() -> list[runner.CaseResult]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(simulated_openai)) as http:
        return await runner.run(
            list(SUITES.values()),
            MODELS,
            SecretStr("sk-test"),
            cases=CASES,
            repeat=2,
            http_client=http,
        )


async def test_run_records_outcomes_tokens_and_latency() -> None:
    results = await simulate()

    assert len(results) == 4 * len(MODELS) * 2
    by_key = {(r.model, r.case_id, r.repeat): r for r in results}
    assert by_key["gpt-5.4", "remind-me", 0].passed
    nano = by_key["gpt-5.4-nano", "remind-me", 1]
    assert not nano.passed and nano.checks == {"tool": False}
    assert nano.output["tool"] == "ask_notes"
    assert all(r.error is None and r.latency_s >= 0 for r in results)
    assert nano.usage == runner.Usage(
        input_tokens=1000, cached_tokens=200, output_tokens=100, reasoning_tokens=40
    )


async def test_report_recommends_the_cheapest_model_that_does_the_same_job() -> None:
    results = await simulate()
    pricing = report.load_pricing()

    stats = report.summarize(results, pricing)
    # 800 uncached and 200 cached input tokens, 100 output, at gpt-5.4 prices.
    assert stats["agent", "gpt-5.4"].cost_per_case == pytest.approx(
        (800 * 2.5 + 200 * 0.25 + 100 * 15) / 1e6
    )
    nano = report.judge(stats["agent", "gpt-5.4"], stats["agent", "gpt-5.4-nano"], 0.03)
    assert not nano.acceptable
    assert "critical regressions: remind-me" in nano.reasons
    assert report.agreement(stats["agent", "gpt-5.4"], stats["agent", "gpt-5.4-nano"]) == 0.5

    picks = report.recommend(stats, "gpt-5.4", 0.03)
    # nano fails the agent but does the chat suites as well, and is cheapest there.
    assert picks["OPENAI_AGENT_MODEL"][0] == "gpt-5.4-mini@low"
    assert picks["OPENAI_CHAT_MODEL"][0] == "gpt-5.4-nano"

    text = report.render(results, pricing, "gpt-5.4", 0.03)
    assert "`OPENAI_AGENT_MODEL=gpt-5.4-mini@low`: same job as `gpt-5.4`, 70% cheaper." in text
    assert "- `gpt-5.4-nano` remind-me (critical): tool" in text


async def test_report_round_trips_through_saved_results() -> None:
    results = await simulate()

    saved = json.loads(json.dumps([r.to_json() for r in results]))

    assert [runner.CaseResult.from_json(r) for r in saved] == results


async def test_a_failed_call_is_a_failed_case() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(down)) as http:
        [result] = await runner.run(
            [SUITES["agent"]],
            ["gpt-5.4"],
            SecretStr("sk-test"),
            cases={"agent": CASES["agent"][:1]},
            http_client=http,
        )

    assert not result.passed
    assert result.checks == {"no error": False}
    assert "Error code: 400" in result.error
