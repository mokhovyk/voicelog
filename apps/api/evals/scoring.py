"""Deterministic scorers. Each returns named checks; a case passes when all pass.

Checks are plain code (string containment, exact dates, citation sets), so a score
means the same on every run and for every model, and needs no judge model."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# A phrase that must (or must not) appear: a string, or alternatives of which any one
# will do, e.g. ["login bug", "login issue"]. Matching ignores case.
Phrase = str | list[str]


@dataclass
class Score:
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def failed(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok]


_SPACE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text.replace("’", "'").lower()).strip()


def has(text: str, phrase: Phrase) -> bool:
    alternatives = [phrase] if isinstance(phrase, str) else phrase
    haystack = _norm(text)
    return any(_norm(alternative) in haystack for alternative in alternatives)


def script(text: str) -> str:
    """ "cyrillic" or "latin", whichever most of the letters are."""
    letters = [c for c in text if c.isalpha()]
    cyrillic = sum("Ѐ" <= c <= "ӿ" for c in letters)
    return "cyrillic" if letters and cyrillic * 2 >= len(letters) else "latin"


def _add_phrases(
    score: Score, label: str, text: str, includes: Sequence[Phrase], excludes: Sequence[Phrase]
) -> None:
    for phrase in includes:
        score.checks[f"{label} has {phrase!r}"] = has(text, phrase)
    for phrase in excludes:
        score.checks[f"{label} lacks {phrase!r}"] = not has(text, phrase)


def score_agent(expect: Any, output: dict) -> Score:
    """output: {"tool": name, "args": {...}} with dates as ISO strings."""
    score = Score()
    tools = [expect.tool] if isinstance(expect.tool, str) else expect.tool
    score.checks["tool"] = output["tool"] in tools
    if not score.checks["tool"]:
        return score  # argument checks are for the expected tool
    args = output["args"]
    for name, value in expect.args.items():
        score.checks[f"{name} = {value!r}"] = args.get(name) == value
    for name, values in expect.args_one_of.items():
        score.checks[f"{name} in {values!r}"] = args.get(name) in values
    for name, phrases in expect.includes.items():
        _add_phrases(score, name, str(args.get(name) or ""), phrases, [])
    for name, phrases in expect.excludes.items():
        _add_phrases(score, name, str(args.get(name) or ""), [], phrases)
    return score


def score_extraction(expect: Any, output: dict) -> Score:
    """output: NoteMetadata as a dict."""
    score = Score()
    categories = [expect.category] if isinstance(expect.category, str) else expect.category
    score.checks["category"] = output["category"] in categories

    items = [item["text"] for item in output["action_items"]]
    unmatched = list(items)
    for phrase in expect.action_items:
        found = next((item for item in unmatched if has(item, phrase)), None)
        score.checks[f"finds task {phrase!r}"] = found is not None
        if found is not None:
            unmatched.remove(found)  # one extracted item per expected task
    limit = (
        expect.max_action_items
        if expect.max_action_items is not None
        else len(expect.action_items) + 1
    )
    score.checks[f"at most {limit} tasks"] = len(items) <= limit
    score.checks["tasks not done"] = not any(item["done"] for item in output["action_items"])

    if expect.summary_script:
        score.checks[f"summary is {expect.summary_script}"] = (
            script(output["summary"]) == expect.summary_script
        )
    _add_phrases(score, "summary", output["summary"], expect.summary_includes, [])
    return score


def score_answer(expect: Any, output: dict) -> Score:
    """output: {"answer": str, "note_numbers": [int]}, numbers as the model gave them."""
    score = Score()
    cited = set(output["note_numbers"])
    required, allowed = set(expect.cites), set(expect.cites) | set(expect.may_cite)
    score.checks[f"cites {sorted(required)}"] = required <= cited
    score.checks[f"cites only {sorted(allowed)}"] = cited <= allowed
    _add_phrases(score, "answer", output["answer"], expect.answer_includes, expect.answer_excludes)
    if expect.answer_script:
        score.checks[f"answer is {expect.answer_script}"] = (
            script(output["answer"]) == expect.answer_script
        )
    return score
