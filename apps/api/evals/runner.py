"""Run suites against models and record every outcome with its tokens and latency."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from langchain_core.callbacks import get_usage_metadata_callback
from pydantic import SecretStr

from evals.suites import Suite


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    # Part of output_tokens; billed as output.
    reasoning_tokens: int = 0

    def add(self, metadata: dict[str, Any]) -> None:
        self.input_tokens += metadata.get("input_tokens", 0)
        self.output_tokens += metadata.get("output_tokens", 0)
        self.cached_tokens += (metadata.get("input_token_details") or {}).get("cache_read", 0)
        self.reasoning_tokens += (metadata.get("output_token_details") or {}).get("reasoning", 0)


@dataclass
class CaseResult:
    suite: str
    model: str
    case_id: str
    repeat: int
    critical: bool
    passed: bool
    checks: dict[str, bool]
    latency_s: float
    usage: Usage = field(default_factory=Usage)
    output: dict | None = None
    error: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "CaseResult":
        return cls(**{**data, "usage": Usage(**data["usage"])})


async def _run_case(suite: Suite, model: str, target, case, repeat: int) -> CaseResult:
    usage = Usage()
    start = time.perf_counter()
    output, error = None, None
    try:
        # Collects token usage of every model call made in this task.
        with get_usage_metadata_callback() as callback:
            output = await target(case)
    except Exception as exc:  # a failed call is a failed case, not a failed run
        error = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - start
    for metadata in callback.usage_metadata.values():
        usage.add(metadata)

    if output is None:
        checks = {"no error": False}
    else:
        checks = suite.score(case, output).checks
    return CaseResult(
        suite=suite.name,
        model=model,
        case_id=case.id,
        repeat=repeat,
        critical=case.critical,
        passed=bool(checks) and all(checks.values()),
        checks=checks,
        latency_s=round(latency, 3),
        usage=usage,
        output=output,
        error=error,
    )


async def run(
    suites: list[Suite],
    models: list[str],
    api_key: SecretStr,
    *,
    cases: dict[str, list[Any]] | None = None,
    repeat: int = 1,
    concurrency: int = 8,
    http_client: httpx.AsyncClient | None = None,
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Every case of every suite, `repeat` times per model, at most `concurrency`
    requests at once. `cases` overrides the suites' datasets, by suite name."""
    cases = cases or {suite.name: suite.load() for suite in suites}
    semaphore = asyncio.Semaphore(concurrency)
    owns_client = http_client is None
    http_client = http_client or httpx.AsyncClient()

    async def limited(suite: Suite, model: str, target, case, n: int) -> CaseResult:
        async with semaphore:
            result = await _run_case(suite, model, target, case, n)
        if on_result:
            on_result(result)
        return result

    try:
        jobs = []
        for suite in suites:
            for model in models:
                target = suite.target(model, api_key, http_client)
                for case in cases[suite.name]:
                    for n in range(repeat):
                        jobs.append(limited(suite, model, target, case, n))
        return list(await asyncio.gather(*jobs))
    finally:
        if owns_client:
            await http_client.aclose()
