# QuadCode

QuadCode turns a programming problem into one tested, measured implementation by having four
independently configured LLM agents write, review, test and improve code. Every claim it makes, in
its reports and its dashboard, comes from a logged record. If something could not be verified, it
says so.

The full design is in [SPEC.md](SPEC.md). The project rules, which take precedence over the spec
where they differ, are in [CLAUDE.md](CLAUDE.md). The milestone history, with every decision and
every real run, is in [PROGRESS.md](PROGRESS.md).

## How it works

Four agent slots, each with its own provider client, model, role instructions and prompts:
**Generator** (G1), **Reviewer** (C1), **Optimizer** (P1) and **Validator** (D1). One LLM never
plays all four.

| phase | what happens | API calls |
|---|---|---|
| 1 analysis | one call turns the problem into a structured spec (entry point, signature, constraints, edge cases) | 1 |
| 2 initial | four parallel, **blind** calls: each slot writes its own solution without seeing the others. The Validator also writes the test suite. Saved as immutable versions G1, C1, P1, D1 | 4 |
| 2.5 local test | every initial version runs the suite in the sandbox (pass/fail, runtime, peak memory) | 0 |
| 3 review | one call per slot: all four solutions plus the test table; strengths, weaknesses, bugs, suggestions | 4 |
| 4-6 rounds | V0 is the best initial version **by evidence**. Each round, every slot may propose unified diffs; each diff is applied and tested locally; failing or contested proposals get challenge calls. The orchestrator accepts on tests and measurements, **never by vote**. A round that accepts nothing stops the run early | 4+ per round |
| 7 benchmark | the initial versions and the final one are benchmarked under identical inputs (median/min/max of repeated runs) | 0 |
| 8 integrate | accepted changes become `final_solution.py`, with `# [role:M-xxx]` comments on the lines they changed | 0 |

Selection order, everywhere: correctness, then requirements, then tests passed, then time, then
memory. Security, readability and maintainability are reported as not measured. Every proposed
change becomes a modification record (`M-001`, ...), accepted or rejected, and none is ever deleted.
An agent that fails (API error, rate limit, timeout, invalid output) is marked UNAVAILABLE; missing
feedback never counts as approval.

## Current line-up and honest status

The four slots currently run on free tiers only, all on Groq (checked 2026-11-22). Groq's free tier
serves three chat models, so the Optimizer and the Validator share one model. They keep separate
clients, role instructions and prompts, and only the Validator writes tests.

| slot | model (first choice) | fallback | spec's intended default |
|---|---|---|---|
| Generator | `openai/gpt-oss-20b` | `openai/gpt-oss-120b` | Gemini |
| Reviewer | `qwen/qwen3.8-27b` | `openai/gpt-oss-20b` | Claude |
| Optimizer | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` | GPT |
| Validator | `openai/gpt-oss-120b` | `openai/gpt-oss-20b` | DeepSeek |

Tried and dropped, with the measured reason recorded in `config.yaml`: Google AI Studio (503 "high
demand" all day), NVIDIA NIM (timeouts on full-size requests), OpenRouter free models (429
"rate-limited upstream", or reasoning that used the whole token cap), DeepSeek (no free grant). With
paid keys the spec's intended models go into `config.yaml`; no code changes.

**Results so far.** These are quoted from PROGRESS.md. The run directories live under `runs/`, which
is git-ignored, so they are not in this repository.

- `real-two-sum-10` (two_sum, 2 rounds, budget 60): COMPLETED, 29 provider requests. The Validator
  wrote 10 tests; 2 of them were wrong and were set aside as disputed. Final version V2 passes every
  undisputed test.
- `real-bench-2` (merge_intervals, lis_length, top_k_frequent; full pipeline only; 52 provider
  attempts): hidden-test pass rate 66.7% for the full pipeline vs 100% for each of the Generator,
  Reviewer and Optimizer alone (66.7% for the Validator alone). Full pipeline vs each of those three:
  0 wins, 2 ties, 1 loss. **On these three problems the pipeline never beat the best single agent.**
  The loss (top_k_frequent) came from the Validator's test suite, not from the code: even after the
  repair retry its tests held Python expressions (`["same"] * 1000`) or arguments written as plain
  strings, so the run ended UNVERIFIED without a final solution.
- Three problems give very wide confidence intervals. These results say little about larger sets.

## Requirements

- Python 3.12 or newer (developed on 3.14), on Linux, macOS or Windows with WSL.
- Docker is optional. With it, generated code runs in a container with no network, a read-only
  filesystem, memory/CPU/pid caps and an empty environment. QuadCode never pulls images itself:

```bash
docker pull python:3.12-slim
```

  Without Docker (or without the image), code runs as a resource-limited subprocess with an empty
  environment, and every result and report carries a warning that isolation is weaker (no network
  or filesystem isolation).

## Setup

Clone the repository, `cd` into it, then:

```bash
python3 -m venv .venv && . .venv/bin/activate
```

```bash
pip install -e '.[dev]'
```

`.[dev]` adds pytest and Streamlit. Use `pip install -e '.[ui]'` for the dashboard without pytest,
or `pip install -e .` for the command line only.

API keys go in `.env`, which is git-ignored. The default line-up needs only `GROQ_API_KEY` (free,
no card, from console.groq.com). Mock mode needs no key at all.

```bash
cp .env.example .env
```

Check the configuration and the sandbox:

```bash
quadcode check-config
```

```bash
quadcode sandbox-info
```

`check-config` lists the four slots with their provider, model, limits and fallbacks, and says for
each provider whether its key is set (never the key itself). `sandbox-info` says whether Docker or
the subprocess fallback will run generated code, and why.

## Quick start: mock mode (no keys, no API calls)

Mock mode uses deterministic scripted agents that exercise every code path: a buggy initial
solution, a disagreement, an agent that is unavailable, a rejected modification and an early stop.
The scripted answers ignore the problem text. Sandbox results and timings are real.

```bash
quadcode solve samples/two_sum/problem.md --mock --rounds 3 --run-id demo
```

It takes under a minute (about 25 seconds with Docker; most of it is the deliberately slow C1
timing out on the performance test) and ends with `run demo: COMPLETED`. Expected: the Validator UNAVAILABLE in the review phase; V0 copied
from G1; M-001 (Reviewer) accepted into V1; M-002 (Optimizer) rejected on test T03 despite one
accept verdict; early stop after round 2; `runs/demo/final_solution.py` passes all 9 tests.

The report (`runs/demo/report.md` and `runs/demo/report.html`) is written at the end. Rebuild it from
the logs at any time, with no API calls:

```bash
quadcode report demo
```

See every stored version and modification record, and which agent each line of V1 came from:

```bash
quadcode versions runs/demo --blame V1
```

## Real runs on free tiers

List the model ids a provider offers (no completion call), then send one tiny call to every
configured model, fallbacks included, to check keys, model ids and limits:

```bash
quadcode models groq
```

```bash
quadcode smoke
```

Then a real run with a call budget:

```bash
quadcode solve samples/two_sum/problem.md --rounds 2 --budget-calls 60 --run-id real-demo
```

- `--budget-calls N` counts provider attempts, retries included (each one spends quota). Cache hits
  are free. When the budget runs out the run stops gracefully, still writes its report, and exits
  with code 4.
- Resume an interrupted or budget-stopped run. Completed calls are never repeated:

```bash
quadcode solve --resume real-demo --budget-calls 30
```

- Responses are cached on disk (`.quadcode_cache/`, keyed by model, prompt and parameters), so
  re-running the same problem reuses answers and logs them as cache hits. `--no-cache` turns that off.
- Rate limits: each provider block has a token bucket (requests/min and tokens/min). 429 and 5xx
  responses are retried with exponential backoff and jitter, honouring Retry-After. A slot whose
  model keeps failing moves to its next fallback, and the report names the model that actually
  answered each call.
- Exit codes: 0 completed (final version passes every test), 1 unverified, 2 failed or usage error,
  3 needs clarification, 4 aborted on budget.

## Configuration

`config.yaml` is the only place provider and model names come from. Reports, logs and the dashboard
show the provider/model that was actually called. Swapping a model is a config edit, never a code
change.

| block | what it holds |
|---|---|
| `providers` | one block per endpoint: `kind` (`openai_compat`, `gemini`, `anthropic`, `mock`), `base_url`, `api_key_env`, rpm/tpm limits, timeout, retries, backoff. Agents sharing a block share its limits |
| `agents` | `generator`, `reviewer`, `optimizer`, `validator`: provider, model, temperature, optional `request_options`, per-call-type `max_tokens`, and ordered `fallbacks` |
| `allow_duplicate_models` | two slots may share a model only if this is true |
| `max_tokens` | output caps per call type (analysis, initial, review, modify, challenge) |
| `sandbox` | backend (`auto`, `docker`, `subprocess`), Docker image, memory/CPU/pid caps, timeouts, repeats, seed |
| `selection` | runtime and memory tolerances: differences inside them are ties and decide nothing |
| `prompts` | how much failing-test output is sent back to agents |
| `pricing` | optional USD per million tokens, keyed `provider/model`; costs are labeled estimates |
| `evaluation` | benchmark problems directory, problem limit, conditions, bootstrap settings |
| `run` | default rounds, call budget, cache, runs directory, repair retries |

`openai_compat` serves OpenAI, Groq, OpenRouter, DeepSeek, GitHub Models and any other
OpenAI-compatible endpoint through `base_url`. To use paid models, add a provider block, point the
slot at it, put the key in `.env` under the name given in `api_key_env`, and run `quadcode smoke`.

## What a run writes

Everything lands in `runs/<run_id>/`. Versions and records are never overwritten.

| file | content |
|---|---|
| `problem.md`, `config.json` | the problem statement and a snapshot of the config (no keys) |
| `calls.jsonl` | every provider call: role, provider, model, prompt hash, tokens in/out, latency, cache hit, status |
| `events.jsonl` | every step of the run, in order; reports cite it as `events:<line>` |
| `state.json` | checkpoint for `--resume` |
| `versions/` | immutable versions G1, C1, P1, D1, V0, candidates (`V0+M-001`), V1, ... with per-line origins |
| `modifications.json` | every modification record, accepted and rejected |
| `test_suite.json`, `tests.json` | the Validator's suite and every sandbox suite run |
| `benchmark.json` | the final repeated benchmark, with its method |
| `summary.json` | status, stop reason, final version, unresolved issues, agent availability |
| `final_solution.py` | the integrated final version with `# [role:M-xxx]` comments |
| `report.md`, `report.html` | the ten-section report, built only from the files above |

## Benchmarks

`quadcode bench` runs benchmark problems under several conditions and scores each result on
**hidden tests the agents never see**. The 10 problems are in `benchmarks/problems/`
(see `benchmarks/README.md`).

| condition | where its code comes from | extra API calls |
|---|---|---|
| `single-generator` ... `single-validator` | each slot's blind initial solution from the full run | none |
| `oracle` | the best initial solution by hidden tests (an upper bound, not a system) | none |
| `select-only` | V0 of the full run | none |
| `full` | the full pipeline's final solution | one run |
| `no-review`, `no-test-feedback`, `no-optimizer` | ablation runs | one run each |

A mock bench (three problems have scripted scenarios: two_sum, merge_intervals, lis_length; every
run condition from `config.yaml`; about 5 minutes):

```bash
quadcode bench --mock --bench-id mock-bench
```

A small real bench with a budget shared by every run:

```bash
quadcode bench --problems merge_intervals,lis_length,top_k_frequent --conditions full --rounds 2 --budget-calls 120 --bench-id real-bench
```

Outputs in `runs/<bench_id>/`: `results.csv`, `summary.csv`, `comparisons.csv`, `summary.md` (mean
hidden pass rate with a 95% bootstrap CI, paired win/tie/loss per problem) and `charts/*.png`.
Rebuild them from the records with no API calls:

```bash
quadcode bench-report mock-bench
```

## Dashboard

A Streamlit view over saved runs and benches:

```bash
streamlit run quadcode/ui/app.py -- --runs-dir runs
```

Open the printed localhost URL. **Runs** shows a run's overview, the full report, the agent
communication timeline, calls and tokens, version diffs with line provenance, every modification
record and the measured benchmark next to the agents' theoretical complexity claims. **Benches**
shows the hidden-test results, CIs, win/tie/loss, charts and each pipeline run. **New run** starts
`solve`, `bench` or a resume as the same `quadcode` command line, shows the command first, and
follows its progress.

Viewing makes no API call and changes no file. Mock is preselected for new runs; a real run needs a
"spends API quota" checkbox and a call budget. `.streamlit/config.toml` keeps the server on
localhost and turns off Streamlit's usage statistics.

## Tests

```bash
pytest
```

About 640 tests, fully offline (real network access is blocked in tests), around 5 to 6 minutes. The
Docker sandbox tests skip themselves when Docker or the image is missing. Run one area:

```bash
pytest tests/test_orchestrator.py -q
```

## Reproducing a run

Section 10 of every report lists the exact commands and environment for that run. In short:

```bash
quadcode test-solution runs/demo/final_solution.py --tests runs/demo/test_suite.json --entry two_sum --benchmark
```

re-runs the stored suite against the final file in the sandbox, and `quadcode solve` with the same
problem and flags starts a new run. Mock runs give the same agent answers every time; real models
do not, unless the response cache replays them. Timings always vary between machines and runs,
which is why runtime only counts as a difference when the min/max ranges of repeated runs do not
overlap.

## Security model

- API keys are read from `.env` by the orchestrator process only. They are never logged (error
  text is redacted), never written to a run directory, and never passed to the sandbox.
- Generated code is untrusted and only runs in the sandbox. Expected test values never enter it;
  results are compared on the host. The runner keeps its own result channel, so prints from the
  solution cannot forge results.
- The dashboard listens on localhost only and starts runs only through the command line.

## Known limitations

- Free tiers give three distinct models for four slots; the spec's intended vendors need paid keys.
- The Validator's tests can be wrong. A test that every executed initial version fails is set
  aside as disputed: it no longer decides anything and is listed as unresolved. If every initial
  version shares one bug, a correct test that catches it is set aside the same way.
- If the Validator gives no usable test suite, the run ends UNVERIFIED with no final solution, even
  when other slots wrote correct code.
- A modification is accepted only if the candidate passes every undisputed test, so a partial fix
  to a failing V0 cannot be accepted.
- Millisecond-scale performance tests are noisy; a timing difference can occasionally decide a
  modification.
- The subprocess fallback has no network or filesystem isolation (stated in every result).
- Only Python solutions are executed; other languages are reported as not executed.
- See the "Open issues" of each milestone in PROGRESS.md for the full list.

## Project layout

| path | content |
|---|---|
| `quadcode/cli.py`, `cli_*.py` | the `quadcode` command and its subcommands |
| `quadcode/orchestrator.py`, `rounds.py`, `selection.py`, `agents.py`, `prompts.py` | the pipeline phases, rounds, evidence-based selection, agent calls and prompts |
| `quadcode/gateway.py`, `ratelimit.py`, `cache.py`, `providers/` | provider clients, rate limits, retries, fallbacks, cache, call log |
| `quadcode/sandbox.py`, `sandbox_runner.py`, `testing.py`, `benchmark.py` | sandboxed execution, test runs, measurements |
| `quadcode/versions.py`, `diffs.py`, `linemap.py`, `modifications.py`, `attribution.py` | versions, diff application, line provenance, modification records |
| `quadcode/report/` | the Markdown/HTML report |
| `quadcode/evaluation/` | the benchmark harness, statistics, CSV and charts |
| `quadcode/ui/` | the Streamlit dashboard |
| `benchmarks/problems/` | benchmark problems with hidden tests and reference solutions |
| `samples/two_sum/` | hand-written solutions, tests and diffs for trying the sandbox and diff tools |
| `tests/` | the pytest suite |

## Command reference

| command | what it does | API calls |
|---|---|---|
| `quadcode check-config` | validate config.yaml, show slots and which keys are set | none |
| `quadcode models <provider>` | list a provider's model ids | catalog request only |
| `quadcode smoke` | one tiny call per configured model | one per model |
| `quadcode solve <problem> [--mock]` | run the pipeline; `--resume`, `--budget-calls`, `--rounds`, `--ablation` | yes (none with `--mock`) |
| `quadcode report <run>` | rebuild a run's report | none |
| `quadcode versions <run_dir> [--blame LABEL]` | list versions and records, or the origin of every line | none |
| `quadcode sandbox-info` | which sandbox backend would run code | none |
| `quadcode test-solution <files> --tests <json> --entry <fn>` | run a test suite against solution files | none |
| `quadcode apply-diff <base> <diff>` | apply a unified diff, show where each hunk landed, optionally test and store it | none |
| `quadcode bench [--mock]` | run benchmark conditions, score on hidden tests | yes (none with `--mock`) |
| `quadcode bench-report <bench>` | rebuild a bench's CSV, Markdown and charts | none |
| `streamlit run quadcode/ui/app.py` | the dashboard | none for viewing |

Every command accepts `--config` and `--env-file` before the subcommand, for example
`quadcode --config other.yaml solve ...`.
