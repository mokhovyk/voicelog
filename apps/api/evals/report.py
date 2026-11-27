"""Turn case results into a comparison: quality, agreement with the baseline, cost and
latency per model, and the cheapest model that does the same job."""

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.ai.llm import parse_model_spec
from evals.runner import CaseResult, Usage
from evals.suites import SUITES

PRICING_FILE = Path(__file__).parent / "pricing.json"


def load_pricing(path: Path = PRICING_FILE) -> dict[str, dict[str, float]]:
    return json.loads(path.read_text())["usd_per_million_tokens"]


def cost(model: str, usage: Usage, pricing: dict[str, dict[str, float]]) -> float | None:
    """USD for the usage, or None if the model is not in the pricing file."""
    price = pricing.get(parse_model_spec(model)[0])
    if price is None:
        return None
    uncached = usage.input_tokens - usage.cached_tokens
    return (
        uncached * price["input"]
        + usage.cached_tokens * price.get("cached_input", price["input"])
        + usage.output_tokens * price["output"]
    ) / 1_000_000


@dataclass
class Stats:
    suite: str
    model: str
    results: list[CaseResult] = field(default_factory=list)
    cost_per_case: float | None = None

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results)

    @property
    def critical(self) -> list[CaseResult]:
        return [r for r in self.results if r.critical]

    @property
    def critical_rate(self) -> float | None:
        critical = self.critical
        return sum(r.passed for r in critical) / len(critical) if critical else None

    def case_rates(self) -> dict[str, float]:
        runs: dict[str, list[bool]] = defaultdict(list)
        for r in self.results:
            runs[r.case_id].append(r.passed)
        return {case: sum(passes) / len(passes) for case, passes in runs.items()}

    @property
    def flaky(self) -> list[str]:
        return [case for case, rate in self.case_rates().items() if 0 < rate < 1]

    @property
    def errors(self) -> int:
        return sum(r.error is not None for r in self.results)

    def latency(self, quantile: float) -> float:
        values = sorted(r.latency_s for r in self.results)
        return values[min(len(values) - 1, int(quantile * len(values)))]

    def mean_tokens(self) -> tuple[float, float]:
        return (
            statistics.fmean(r.usage.input_tokens for r in self.results),
            statistics.fmean(r.usage.output_tokens for r in self.results),
        )


def summarize(
    results: list[CaseResult], pricing: dict[str, dict[str, float]]
) -> dict[tuple[str, str], Stats]:
    stats: dict[tuple[str, str], Stats] = {}
    for result in results:
        key = (result.suite, result.model)
        stats.setdefault(key, Stats(result.suite, result.model)).results.append(result)
    for entry in stats.values():
        costs = [cost(entry.model, r.usage, pricing) for r in entry.results]
        if all(c is not None for c in costs):
            entry.cost_per_case = statistics.fmean(costs)
    return stats


def agreement(baseline: Stats, candidate: Stats) -> float | None:
    """Share of the candidate's runs whose output has the same signature as the
    baseline's most common output for that case: how often it does the same thing."""
    signature = SUITES[baseline.suite].signature
    usual: dict[str, Counter] = defaultdict(Counter)
    for r in baseline.results:
        if r.output is not None:
            usual[r.case_id][json.dumps(signature(r.output), default=str)] += 1
    compared = [
        json.dumps(signature(r.output), default=str) == usual[r.case_id].most_common(1)[0][0]
        if r.output is not None
        else False
        for r in candidate.results
        if usual[r.case_id]
    ]
    return sum(compared) / len(compared) if compared else None


def regressions(baseline: Stats, candidate: Stats) -> list[str]:
    """Cases the baseline mostly passes and the candidate mostly fails."""
    base, cand = baseline.case_rates(), candidate.case_rates()
    return sorted(case for case in base if base[case] >= 0.5 and cand.get(case, 0) < 0.5)


@dataclass
class Verdict:
    model: str
    acceptable: bool
    reasons: list[str]


def judge(baseline: Stats, candidate: Stats, tolerance: float) -> Verdict:
    """A candidate does the same job if its pass rate is within `tolerance` of the
    baseline's and it regresses no critical case."""
    reasons = []
    if candidate.pass_rate < baseline.pass_rate - tolerance:
        reasons.append(
            f"pass rate {candidate.pass_rate:.0%} vs {baseline.pass_rate:.0%} "
            f"(tolerance {tolerance:.0%})"
        )
    critical = {r.case_id for r in baseline.critical}
    lost = [case for case in regressions(baseline, candidate) if case in critical]
    if lost:
        reasons.append(f"critical regressions: {', '.join(lost)}")
    if candidate.errors:
        reasons.append(f"{candidate.errors} failed calls")
    return Verdict(candidate.model, not reasons, reasons)


def recommend(
    stats: dict[tuple[str, str], Stats], baseline: str, tolerance: float
) -> dict[str, tuple[str, list[str]]]:
    """For each setting, the cheapest model acceptable on every suite that uses it,
    with notes. Falls back to the baseline."""
    by_setting: dict[str, list[str]] = defaultdict(list)
    for suite, _ in stats:
        if suite not in by_setting[SUITES[suite].setting]:
            by_setting[SUITES[suite].setting].append(suite)
    models = list(dict.fromkeys(model for _, model in stats))

    picks = {}
    for setting, suites in by_setting.items():
        options, notes = [], []
        for model in models:
            if model == baseline:
                continue
            verdicts = [judge(stats[s, baseline], stats[s, model], tolerance) for s in suites]
            costs = [stats[s, model].cost_per_case for s in suites]
            if all(v.acceptable for v in verdicts) and None not in costs:
                options.append((sum(costs), model))
            elif all(v.acceptable for v in verdicts):
                notes.append(f"{model} qualifies but has no price in pricing.json")
        base_cost = sum(stats[s, baseline].cost_per_case or 0 for s in suites)
        cheaper = sorted(option for option in options if option[0] < base_cost)
        picks[setting] = (cheaper[0][1] if cheaper else baseline, notes)
    return picks


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def _usd(value: float | None) -> str:
    return "–" if value is None else f"${value * 1000:.2f}"


def render(
    results: list[CaseResult],
    pricing: dict[str, dict[str, float]],
    baseline: str,
    tolerance: float,
) -> str:
    stats = summarize(results, pricing)
    suites = list(dict.fromkeys(suite for suite, _ in stats))
    models = list(dict.fromkeys(model for _, model in stats))
    repeats = max(r.repeat for r in results) + 1
    lines = [
        "# Eval report",
        "",
        f"Baseline `{baseline}`; tolerance {tolerance:.0%}; {repeats} run(s) per case.",
        "",
    ]
    for suite in suites:
        base = stats[suite, baseline]
        cases = len(base.case_rates())
        lines += [
            f"## {suite} ({cases} cases, model setting `{SUITES[suite].setting}`)",
            "",
            "| Model | Pass | Critical | Agrees w/ baseline | Flaky | p50 / p95 s "
            "| Tokens in / out | Cost / 1k calls | Verdict |",
            "| :-- | --: | --: | --: | --: | --: | --: | --: | :-- |",
        ]
        for model in models:
            entry = stats.get((suite, model))
            if entry is None:
                continue
            tokens_in, tokens_out = entry.mean_tokens()
            if model == baseline:
                agrees, verdict = "–", "baseline"
            else:
                agrees = _pct(agreement(base, entry))
                v = judge(base, entry, tolerance)
                verdict = "same job" if v.acceptable else "; ".join(v.reasons)
            lines.append(
                f"| `{model}` | {_pct(entry.pass_rate)} | {_pct(entry.critical_rate)} "
                f"| {agrees} | {len(entry.flaky)} "
                f"| {entry.latency(0.5):.1f} / {entry.latency(0.95):.1f} "
                f"| {tokens_in:.0f} / {tokens_out:.0f} | {_usd(entry.cost_per_case)} "
                f"| {verdict} |"
            )
        lines.append("")
        failures = _failures(stats, suite, models)
        if failures:
            lines += ["Failing cases (model: failed checks):", "", *failures, ""]

    lines += ["## Recommendation", ""]
    for setting, (model, notes) in recommend(stats, baseline, tolerance).items():
        if model == baseline:
            lines.append(f"- `{setting}={baseline}`: no cheaper model did the same job.")
        else:
            suites_for = [s for s in suites if SUITES[s].setting == setting]
            base_cost = sum(stats[s, baseline].cost_per_case or 0 for s in suites_for)
            new_cost = sum(stats[s, model].cost_per_case or 0 for s in suites_for)
            saving = f", {1 - new_cost / base_cost:.0%} cheaper" if base_cost else ""
            lines.append(f"- `{setting}={model}`: same job as `{baseline}`{saving}.")
        lines += [f"  - {note}" for note in notes]
    lines += [
        "",
        f"Costs use `{PRICING_FILE.name}`; check it against current OpenAI prices. With "
        "few cases one case moves the pass rate several points, so rerun with "
        "`--repeat 3` before acting on a close result.",
    ]
    return "\n".join(lines) + "\n"


def _failures(stats: dict[tuple[str, str], Stats], suite: str, models: list[str]) -> list[str]:
    lines = []
    for model in models:
        entry = stats.get((suite, model))
        if entry is None:
            continue
        failed: dict[str, Counter] = defaultdict(Counter)
        for r in entry.results:
            if not r.passed:
                failed[r.case_id].update(
                    [r.error] if r.error else [n for n, ok in r.checks.items() if not ok]
                )
        for case, checks in sorted(failed.items()):
            mark = (
                " (critical)" if any(r.critical for r in entry.results if r.case_id == case) else ""
            )
            detail = "; ".join(checks)
            lines.append(f"- `{model}` {case}{mark}: {detail}")
    return lines
