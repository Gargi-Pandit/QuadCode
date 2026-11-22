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

---

## M3: Version store, modification records, diff application, line mapping (2026-11-22)

**Approved plan (short):** a strict unified-diff applier that tolerates wrong header numbers but never fuzzy-matches code; per-line provenance so a line is credited to an agent only when a modification record inserted it; an immutable on-disk version store (G1/C1/P1/D1, V0, `<base>+M-xxx` candidates, V1..Vn); a modification ledger that keeps every record; `apply-diff` and `versions` commands; sample diffs. No LLM calls.

**Key decisions**
- Hunks are located by exact match of context plus removed lines. Header numbers are hints. Several matches are accepted only if one sits exactly at the stated line, otherwise the hunk is "ambiguous". The only relaxation is trailing whitespace, and it is flagged per hunk. A hunk with no context on a non-empty file is rejected.
- Records store the changed core of each hunk (context trimmed) as `hunks: [HunkChange]`; `original_code`/`replacement_code`/locations are the joined view. Locations always carry the version label (`V0:L8`, `V0:after L2`, `V1:removed, after L4`).
- A diff that does not apply still becomes a record: rejected, reason "diff did not apply: …", code taken from the diff's own -/+ lines, location "unresolved".
- Ledger rules: decided records never change; accept needs test status pass and a reason; reject/withdraw need a reason; verdicts are appended (revised opinions stay visible) and a proposer cannot review its own change; `place()` records the round version once.
- Versions: create-no-overwrite via hard link, read-only files, sha256 checked on load, identical re-save is a no-op (resume), different content raises. Derived versions are checked line by line against their base.
- Several accepted diffs written against the same base are composed in sequence; header lines are translated through the earlier changes, and a diff that no longer applies raises `ComposeConflict` naming it.

**Deviations from plan**
- `suite_status()` added to `testing.py` (one status per suite run); `cli_sandbox._sandbox_config` renamed `sandbox_config` for reuse.
- `versions` lists in lineage order instead of by timestamp (millisecond timestamps tie).
- Extra samples `pair_helpers.py` and `ambiguous_with_line.diff` (G1 has no repeated block to be ambiguous about).
- `apply-diff --run-dir` stores the base as an initial version, so candidates can sit on G1 (`G1+M-001`). The command never accepts or rejects a diff that applied.

**Built**
- `quadcode/diffs.py`, `linemap.py`, `versions.py`, `modifications.py`, `cli_versions.py`; `HunkChange`, `LineOrigin`, `VersionKind`, `VersionMeta`, `Decision.WITHDRAWN` and new `ModificationRecord` fields in schemas.
- `samples/two_sum/diffs/` with a README of expected outcomes.

**Tests:** `pytest`: 282 passed, 0 skipped, 0 failed (~24 s). Includes a randomized `make_diff`/`apply_diff` round trip and a sandbox run of the mock round-1 diffs: M-001 passes all 9 tests, M-002 fails exactly T03, as the scenario intends (resolves the M1 open issue).

**Open issues**
- The mock M-001/M-002 diffs are written against G1. If M4's V0 tie-break picks D1, they will not apply and will be recorded as rejected. M4 must settle the tie-break and keep the scenario consistent with it.
- `# [role:M-xxx]` comments in the final code are left for phase 8 in M4. The provenance needed for them is in place.

---

## M4: Orchestrator, phases 1-8 in mock mode (2026-11-22)

**Approved plan (short):** wire the existing parts into one orchestrator: one analysis call, four blind initial calls, local tests, one review call per agent, V0 chosen on evidence, optimization rounds with diffs applied and tested locally, challenge calls only for failing or contested proposals, decisions on evidence only, early stop, final benchmark, final solution with `# [role:M-xxx]` comments. Every step logged to `events.jsonl` and checkpointed in `state.json` for `--resume`; `--budget-calls` aborts gracefully. `quadcode solve ... --mock`.

**Key decisions**
- V0 and final choice use a filter cascade (`selection.py`): tests passed → runtime → memory → code size (stated as a size heuristic) → slot order (stated as an arbitrary tie-break). Security, readability and maintainability are reported as not measured.
- Acceptance needs evidence: the diff applies, the candidate passes every test, no measured regression against the base, and the code is not AST-identical to the base (comment- or format-only changes are rejected, so rounds cannot be padded). Challenge verdicts are recorded and logged but never decide.
- Challenge calls go to every non-proposer, only for failing proposals or contested ones (measured regression, or conflict with another passing proposal). An unavailable challenger is recorded as `unavailable`.
- Several passing proposals are composed in id order and tested together. On a conflict or a failing combination they are added one at a time, and the one that breaks is rejected.
- Modify prompts carry the current version once (calls are stateless, so the diff base has to be in the prompt), plus the Phase 3 review digest (round 1) or the previous round's accepted diff and rejected proposals (later rounds). Result tables sent to agents hold statuses and truncated errors, no timings, so prompts are deterministic and cacheable.
- `state.json` stores every valid agent output, so resume never repeats a completed call even with the cache off. The config fingerprint ignores the `run` block and `sandbox.backend`, so budget, cache and backend can change on resume.
- Exit codes: 0 completed, 1 unverified, 2 failed/usage error, 3 needs clarification, 4 aborted on budget.

**Deviations from plan**
- First real run picked D1 as V0: G1's three suite timings spanned 1.08-2.66 ms, so the 25% tolerance on a 3-run median was decided by noise (the 7-run benchmark shows G1 faster). Runtime comparisons are now noise-aware: a version counts as slower only if its median is beyond the tolerance AND its fastest run is slower than the other version's slowest run. With that, G1 and D1 tie on runtime and memory and G1 wins on size, so the scripted G1 diffs apply. Tests use a very large runtime tolerance so timing never decides there.
- Extra modules `runctx.py` (shared run services) and the rename of prompt helpers to `results_table`/`suite_listing` (pytest would collect `test_*` names).
- Mock challenge responses are keyed by `round-1:M-002`; the optimizer's M-001 challenge answer was dropped because M-001 is not contested. `MockProvider` now records requests (for the blindness test).
- Run directory also holds `problem.md`, `config.json` (config snapshot, no secrets), `test_suite.json`, `benchmark.json`, `summary.json`, `state.json`.

**Built**
- `quadcode/`: `selection.py`, `prompts.py`, `agents.py`, `runstate.py`, `runctx.py`, `rounds.py`, `attribution.py`, `orchestrator.py`, `cli_solve.py`; `selection:` and `prompts:` blocks in config; `RunStatus`, `RunSummary`, `SelectionEvidence`, `CriterionStep`, `AgentAvailability` in schemas.
- `samples/two_sum/problem.md`.
- Mock two_sum run: validator unavailable in Phase 3; reviewer/optimizer disagree on C1; V0 = G1; validator recovers from one 429 in round 1; M-001 accepted into V1; M-002 rejected on T03 despite one accept verdict; round 2 accepts nothing → early stop; `final_solution.py` carries `# [reviewer:M-001]` and passes all 9 tests.

**Tests:** `pytest`: 354 passed, 0 skipped, 0 failed (~105 s). New: selection, prompts (blindness, no vendor names, truncation), attribution (AST guard), state and agent pool, full two_sum run checked against its own logs, eight scenario variants (unappliable diff, comment-only change, compose conflict, memory regression, all agents unavailable, no test suite, clarification, analysis fallback), budget abort + resume at six budget points with the cache off, CLI.

**Open issues**
- The path where two proposals compose cleanly but fail together is implemented but not covered by a scenario (the conflict path is).
- Acceptance requires the candidate to pass every test. On an unverified V0 (nothing passes), a partial fix cannot be accepted, and the run ends unverified with the failing tests listed.
- Test expected values come from the validator agent and can be wrong. Tests failing on every initial version are flagged `suspect_tests` but kept. Hidden tests come with the evaluation harness.
- Agents cannot withdraw a proposal yet: every record is decided in the round that created it, so nothing stays pending to withdraw.
- No report yet (next milestone); `summary.json` and the CLI table are the only human-readable outputs.
