# Evals

Fixed datasets that score the AI steps against real OpenAI models, so a model can be
swapped for a cheaper one only when it does the same job. Unit tests use fakes and
check the plumbing; evals check the model's output.

| Suite | Measures | Setting | Cases |
| :-- | :-- | :-- | --: |
| `agent` | Which action the voice agent picks, and its arguments: dates resolved from "last week", category, note text, task description | `OPENAI_AGENT_MODEL` | 40 |
| `extraction` | Note enrichment: category, action items found without inventing any, summary in the note's language | `OPENAI_CHAT_MODEL` | 18 |
| `answers` | Answers from notes: cites the right notes and only those, states the facts, abstains when the notes don't say, resists instructions inside notes | `OPENAI_CHAT_MODEL` | 17 |

Cases run with "today" pinned (Friday 2026-11-27 unless a case sets another), so date
expectations never drift. Transcription and embeddings are not evaluated here: cases
start from text, and the answers suite hands the model its notes directly.

## Running

Needs `OPENAI_API_KEY` in `apps/api/.env`. From `apps/api`:

```sh
uv run python -m evals --dry-run                 # validate datasets, count calls; no key needed
uv run python -m evals                           # all suites on gpt-5.4, gpt-5.4-mini, gpt-5.4-nano
uv run python -m evals --suite agent --model gpt-5.4 --model gpt-5.4-mini@low --repeat 3
uv run python -m evals --only ukrainian          # cases whose id contains "ukrainian"
uv run python -m evals --report results/<stamp>.json --tolerance 0.05   # re-judge a saved run
```

`MODEL@EFFORT` sets the reasoning effort (`none`, `minimal`, `low`, `medium`, `high`,
...), which often matters as much as the model for cost and latency. The same syntax
works in `OPENAI_CHAT_MODEL` and `OPENAI_AGENT_MODEL`.

A full run of the three default models is 225 calls, on the order of a dollar at the
prices in `pricing.json` (reasoning tokens make the exact figure vary; the report
shows it). Each run writes `results/<stamp>.json` (every output, check, token
count, and latency) and `results/<stamp>.md` (the report). `results/` is not committed.

## How a cheaper model is chosen

The first `--model` (or `--baseline`) is the reference. Per suite, a candidate does
**the same job** when:

1. its pass rate is at most `--tolerance` (default 3 points) below the baseline's,
2. it fails no **critical** case that the baseline passes (critical cases are the
   behaviours that must not break: out-of-scope requests do nothing, "remind me to"
   vs "remind me what", no invented tasks, answers in the user's language,
   injection in a note), and
3. none of its calls errored.

The recommendation for each setting is the cheapest candidate that does the same job
on every suite using that setting (both `extraction` and `answers` for
`OPENAI_CHAT_MODEL`). The report also shows **agreement**: how often the candidate
produced the same output as the baseline's usual one (same tool and arguments; same
category and task count; same citations). It is the most direct "does the same thing"
measure and catches differences the expectations don't cover. **Flaky** counts cases
that passed in some repeats but not others.

With 17–40 cases one case moves a pass rate by 2.5–6 points, so use `--repeat 3` for
the decision run, and read the failing cases listed under each table before switching.

## Adding cases

One JSON object per line in `datasets/<suite>.jsonl`; lines starting with `//` are
comments. The formats are the `*Case` models in `suites.py`, validated on load (an
unknown tool or argument name fails). Phrases match case-insensitively, and a list of
phrases means any one of them. Add a case whenever a real request goes wrong, mark it
`critical` if a regression there would be unacceptable, and add a `note` saying why
the case exists.
