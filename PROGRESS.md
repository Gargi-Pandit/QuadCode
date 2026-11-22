# QuadCode progress log

Append-only. One entry per milestone.

---

## M1: Skeleton, schemas, config, mock provider, cache, rate limiter (2026-11-22)

**Approved plan (short):** create the package skeleton with pydantic schemas, a `config.yaml` that is the only source of provider/model names, a deterministic mock provider with scripted scenarios, an on-disk cache, a per-provider token-bucket limiter with full-jitter backoff, and an `LLMGateway` that logs every call to `calls.jsonl`. No real providers or network.

**Key decisions**
- Config splits `providers:` (kind, base_url, api_key_env, rpm/tpm, timeout, retries) from `agents:` (role -> provider, model, temperature). Agents sharing a provider block share its rate limit.
- Duplicate (provider, model) across agent slots is a config error unless `allow_duplicate_models: true`.
- API keys live in a `Secrets` object: never in repr, not picklable, `os.environ` untouched, and error text is redacted before logging.
- Call budget counts every provider attempt, retries included (they spend quota). Cache hits are free. Budget exhaustion is logged as a `budget_exhausted` call record and raises `BudgetExhausted`.
- Invalid JSON or schema mismatch → the call is UNAVAILABLE (`invalid_output` record). No hidden re-prompting. Only valid responses are cached, so `--resume` retries failures.
- Mock token counts are estimates (chars/4) and flagged `tokens_estimated: true` in every record.
- Timeouts are enforced by each provider client (`LLMRequest.timeout_s`), surfaced as `ProviderTimeout`.

**Deviations from plan/layout**
- Extra small modules: `config.py`, `jsonutil.py`, `runlog.py`, `gateway.py`, `providers/mock_scenarios.py`.
- `requires-python >=3.12` (uses PEP 695 generics); dev env is Python 3.14.4 in `.venv`.
- Real provider modules (`gemini.py`, `openai_compat.py`, `anthropic.py`) are deferred to M6; `build_client` raises `UnsupportedProviderError` for unregistered kinds.
- `quadcode check-config` is the only CLI command so far (no stub commands).

**Built**
- `quadcode/`: `schemas.py`, `config.py`, `jsonutil.py`, `runlog.py`, `ratelimit.py`, `cache.py`, `gateway.py`, `cli.py`, `providers/{base,mock,mock_scenarios,__init__}.py`.
- `two_sum` mock scenario: buggy P1 (sorted-index bug), O(n^2) C1 that should time out on the perf test, review disagreement, validator UNAVAILABLE in Phase 3, a transient 429 in round 1, an accepted-candidate diff (M-001) and a test-breaking diff (M-002) with split challenge verdicts, then empty round 2 for early stop.
- `config.yaml`, `.env.example`, `requirements.txt`, `pyproject.toml`, `.gitignore`.

**Tests:** `pytest`: 87 passed, 0 failed (all offline, fake clock).

**Open issues**
- V0 selection between G1 and D1 (both O(n) hash maps) needs a deterministic tie-break with a runtime tolerance in M4; measured runtimes alone could flip it.
- The M-001/M-002 diffs are written against G1; whether they apply cleanly gets verified with real diff application in M3.

---

## M2: Sandbox, test runner, benchmarking (2026-11-22)

**Approved plan (short):** execute untrusted Python solutions in a Docker sandbox (no network, read-only, capped) with a resource-limited subprocess fallback that carries a weaker-isolation warning; run test suites with inputs-only payloads and judge results on the host; take repeated perf_counter/tracemalloc measurements; expose `sandbox-info` and `test-solution`; hand-written two_sum samples including hostile ones. No LLM calls.

**Key decisions**
- Expected values never enter the sandbox. The runner returns normalized actual values (or a sha256 digest when the value is too large to transfer) and the host compares (`exact` / `unordered` / `float`; bool is never equal to int).
- `auto` uses Docker only if the daemon answers and the image is already local; it never pulls. Forced `docker` that is unusable raises instead of degrading.
- The runner writes one flushed JSON line per event on a private dup of stdout and points fd 1 at /dev/null, so solution prints (even `os.write(1, ...)`) cannot forge results. Unreported tests are marked from what happened: running test → TIMEOUT (wall clock) or ERROR (exit code), later tests → NOT_RUN with `measured=False`.
- Measurement: first call decides the outcome and counts as warmup; inputs deep-copied outside the timed region; `gc.collect()` before each call; runtime = median of timed calls; memory = tracemalloc peak above the copied inputs in one extra call (Python heap, labeled as such), plus process `ru_maxrss`.
- Generator inputs are seeded per test (`random.Random(f"{seed}:{test_id}")`), so every version gets identical inputs.

**Deviations from plan**
- rlimits are applied by the runner itself before it loads the solution, not via `preexec_fn` (unsafe with threads, which M4 will use). The CPU limit is a backstop set above the wall-clock limit plus start-up slack.
- Extra small modules: `compare.py`, `cli_sandbox.py`, `tables.py` (shared table formatter, also used by `check-config`).
- `test-solution` takes several files and has `--benchmark`, `--backend`, `--test-timeout` and `--json` flags.
- Bug found while building: an exception's traceback pinned the solution's frames, so a memory bomb's allocations survived into later runs. The runner now keeps only an exception summary.
- Found once the Docker tests ran: the container inherited the image's `ENV` lines (`GPG_KEY`, `PYTHON_SHA256`, ...) plus Docker's `HOSTNAME`/`HOME`, and `--env` can only add variables, not remove them. The host's `TEST_API_KEY` never reached the container. The runner is now launched through `env -i`, so the container sees exactly `PYTHONHASHSEED`, `LANG` and a fixed `PATH`. Requires `env` in the image (true for the official python images).

**Built**
- `quadcode/sandbox_runner.py`, `sandbox.py`, `compare.py`, `testing.py`, `benchmark.py`, `cli_sandbox.py`, `tables.py`; `SandboxConfig` in config; `SuiteRun`, `BenchmarkStats`, `BenchmarkReport` and extra `TestResult` fields in schemas.
- `samples/two_sum/`: tests.json, g1/c1/p1/d1 (identical to the mock scenario), and five hostile solutions with a README of expected outcomes.

**Tests:** `pytest`: 185 passed, 0 skipped, 0 failed (~21 s), including 7 Docker tests against `python:3.12-slim` (network blocked, read-only filesystem, exact environment, memory cap, container removed after a wall-clock timeout).

**Open issues**
- The subprocess fallback has no network or filesystem isolation (stated in every result). Code in the same process as the runner could still tamper with its own timings; the sandbox protects the host, not measurement integrity against deliberately malicious code.
- Only Python is executed; other languages are reported as not executed.
- The open M1 issue about the C1 performance test is resolved: C1 times out on T09 as intended. V0 tie-breaking stays open for M4.
