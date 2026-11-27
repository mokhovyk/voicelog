"""Run the evals: `uv run python -m evals --help`."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.ai.llm import parse_model_spec
from app.core.config import Settings
from evals import report, runner
from evals.runner import CaseResult
from evals.suites import SUITES

RESULTS = Path(__file__).parent / "results"
DEFAULT_MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Score the AI steps on fixed datasets and compare models.",
    )
    parser.add_argument(
        "--suite", action="append", choices=list(SUITES), help="default: all suites"
    )
    parser.add_argument(
        "--model",
        action="append",
        help=f"MODEL or MODEL@EFFORT; repeat to compare (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument("--baseline", help="the reference model (default: the first --model)")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case and model")
    parser.add_argument("--concurrency", type=int, default=8, help="requests at once")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="how far below the baseline's pass rate a candidate may be (default 0.03)",
    )
    parser.add_argument("--only", help="run only cases whose id contains this")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the datasets and stop; no API calls"
    )
    parser.add_argument(
        "--report", type=Path, help="re-render a saved results file instead of running"
    )
    args = parser.parse_args()

    suites = [SUITES[name] for name in args.suite or SUITES]
    models = args.model or DEFAULT_MODELS
    baseline = args.baseline or models[0]
    for spec in [*models, baseline]:
        parse_model_spec(spec)
    pricing = report.load_pricing()

    if args.report:
        saved = json.loads(args.report.read_text())
        results = [CaseResult.from_json(r) for r in saved["results"]]
        print(report.render(results, pricing, args.baseline or saved["baseline"], args.tolerance))
        return 0

    cases = {suite.name: suite.load() for suite in suites}
    if args.only:
        cases = {name: [c for c in found if args.only in c.id] for name, found in cases.items()}
    total = sum(len(found) for found in cases.values()) * len(models) * args.repeat
    for name, found in cases.items():
        critical = sum(case.critical for case in found)
        print(f"{name}: {len(found)} cases ({critical} critical)", file=sys.stderr)
    unpriced = [m for m in models if parse_model_spec(m)[0] not in pricing]
    if unpriced:
        print(f"No price in pricing.json for: {', '.join(unpriced)}", file=sys.stderr)
    if args.dry_run:
        print(f"Datasets are valid; a run would make {total} model calls.", file=sys.stderr)
        return 0

    key = Settings().openai_api_key
    if key is None or not key.get_secret_value():
        print("Set OPENAI_API_KEY in apps/api/.env to run, or use --dry-run.", file=sys.stderr)
        return 2
    if baseline not in models:
        models.insert(0, baseline)

    done = 0

    def progress(result: CaseResult) -> None:
        nonlocal done
        done += 1
        mark = "." if result.passed else ("E" if result.error else "F")
        print(mark, end="\n" if done % 80 == 0 else "", file=sys.stderr, flush=True)

    results = asyncio.run(
        runner.run(
            suites,
            models,
            key,
            cases=cases,
            repeat=args.repeat,
            concurrency=args.concurrency,
            on_result=progress,
        )
    )
    print(file=sys.stderr)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    saved_path = RESULTS / f"{stamp}.json"
    saved_path.write_text(
        json.dumps(
            {"baseline": baseline, "models": models, "results": [r.to_json() for r in results]},
            ensure_ascii=False,
            indent=1,
        )
    )
    text = report.render(results, pricing, baseline, args.tolerance)
    (RESULTS / f"{stamp}.md").write_text(text)
    print(text)
    print(f"Saved {saved_path} and {saved_path.with_suffix('.md')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
