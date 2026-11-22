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

---

## M5: Report generator, Markdown and HTML (2026-11-22)

**Approved plan (short):** turn a run directory into `report.md` and `report.html`, built only from its records (events.jsonl, calls.jsonl, modifications.json, versions/, tests.json, test_suite.json, benchmark.json, summary.json, problem.md, config.json). Ten spec sections plus availability, token/cost and contribution tables. Parts of a run that never happened say "Not reached" and show no numbers. Optional `pricing:` block for cost estimates. `solve` writes the report at the end; `quadcode report <run>` rebuilds it with no API calls.

**Key decisions**
- One format-neutral document tree (`report/doc.py`) rendered by two small renderers, so Markdown and HTML always carry the same content. No new dependency.
- All agent text is treated as untrusted. HTML escapes everything and has no scripts, external resources or links other than in-page anchors. Markdown backslash-escapes inline markup and block markers at line starts, and code fences are longer than any backtick run in the code.
- `state.json` is not read: every fact the report needs is in the event log. Sections cite `events:<line>` and file names, and the output is deterministic (no generation timestamp), so regenerating a run's report gives identical bytes.
- Disagreement detection is mechanical only: split challenge verdicts, and verdicts that differ from the decision. Free-text review disagreement is shown side by side per target and the report says it is not detected by meaning.
- Runtime ratios between the initial and final versions count as a difference only when the min-max ranges of the timed runs do not overlap; otherwise the report says "not distinguishable".
- Final complexity is reported as the V0 source agent's claim, marked "not re-derived after M-xxx" when accepted changes exist.
- Prices are keyed `provider/model`, excluded from the resume fingerprint, and taken from `config.yaml` at report time. The cost is labeled an estimate, not a billed amount. Tokens of cache hits and budget stops are not counted.

**Deviations from plan**
- The report is a package `quadcode/report/` (data, doc, fmt, render_md, render_html, tables, sections_problem/log/rounds/results/eval) instead of a single `report.py`.
- Added a "Run overview" section before section 1 for status, warnings, agent slots, availability and the token/cost table. The contribution table sits in section 9.
- `quadcode report` takes prices from `--config`. If that config cannot be loaded it still writes the report, with cost "not configured".

**Built**
- `quadcode/report/` (11 modules), `quadcode/cli_report.py`; `ModelPrice`, `AppConfig.pricing` and `price_for()` in config; `pricing: {}` in `config.yaml`; `solve` writes the report after every outcome, including aborts and failures. A report error is printed but does not change the exit code.
- `samples/two_sum/README.md`: the full mock run and report commands with the expected outcome.

**Tests:** `pytest`: 401 passed, 0 skipped, 0 failed (~139 s), including the 7 Docker tests. 47 tests are new: renderer escaping, HTML well-formedness and the no-active-content check; the full two_sum report checked against its records (benchmark numbers equal benchmark.json, token totals equal calls.jsonl, every `events:N` citation exists, provider/model only from config.json, line credit from stored origins, byte-identical regeneration touching no other file); six budget-abort points plus no-suite, all-unavailable, clarification, failed analysis, hostile agent text and a damaged run; CLI; pricing.

**Open issues**
- `test_sandbox_docker.py::test_memory_cap_stops_bomb` failed once during the full suite and passes on its own. The memory bomb likely hits the per-call timeout before the cap under load. This is sandbox code from an earlier milestone and was not changed here.
- The report is regenerated in place (`report.md`/`report.html` are derived files). Versions and records are never touched.

---

## M6: Real providers on free tiers, smoke calls, one real end-to-end run (2026-11-22)

**Approved plan (short):** wire real HTTP clients (`openai_compat`, `gemini`, `anthropic`) with uniform error mapping, keys only from `.env`, every slot on a free tier with no spend. Add `quadcode models <provider>` (catalog listing, no completion call) and `quadcode smoke` (one tiny call per model, no retries, no cache). Then one real `solve` on `samples/two_sum` with `--budget-calls`, and check the report against calls.jsonl.

**Key decisions**
- httpx only, no vendor SDKs. One error mapping for all providers: 429 → rate limit (honouring Retry-After, retry-after-ms and Google's retryDelay); daily-quota 429s and waits over `max_retry_after_s` → permanent; 408/425/5xx/network → transient; timeouts → ProviderTimeout; a 200 with an error body is mapped by its code. Keys travel only in headers; every error text is redacted. An autouse test guard blocks real network access.
- Reasoning models: `request_options` (e.g. `reasoning_effort: low`), per-agent `max_tokens` overrides, and a clear error when reasoning used the whole cap and left no answer.
- **Fallback models per slot** (`agents.<role>.fallbacks`). A failed model hands the same call to the next model in the list. A provider failure moves the slot to its next model for the rest of the run; an invalid answer moves it for that call only. Every attempt is logged with its real provider/model and its position (`choice`), and reports name the model that actually answered. A cached answer from any listed model is reused before any request.
- **Repair retry** (`run.repair_attempts: 1`): an answer that fails JSON/schema validation goes back to the same model once, with the exact error and the location (`<<HERE>>`), the rejected answer (redacted), and the JSON rules models break most often (no `10**9`, no `None`, no trailing commas). A cut-off answer is asked to be shorter instead.
- **Output limit in every prompt** (`run.state_output_limit: true`): each request states its token cap and asks for a reply, reasoning included, that fits, preferring a complete JSON object over detail. Caps are sized to what the free tier can deliver, not to ideal answers.
- **Disputed tests are set aside**: a test that every executed initial implementation fails (at least two) is probably wrong. It no longer ranks versions, decides modifications or verifies the final version. It stays in test_suite.json/tests.json and the run lists it as unresolved. If every test is disputed, the run ends UNVERIFIED.
- JSON extraction never returns an object nested inside a larger malformed one (it used to return the `{}` from `seen = {}` inside a code string and hide the real error). `args: null`/`kwargs: null` in tests mean "none". Diffs wrapped in `*** Begin Patch`/`*** End Patch` lines are accepted.

**Deviations from plan**
- Final line-up is **Groq only**, by the user's decision after the alternatives failed (all checked 2026-11-22): Generator gpt-oss-20b; Reviewer qwen3.8-27b; Optimizer and Validator gpt-oss-120b; each slot falls back to another Groq model. Groq serves only three chat models, so `allow_duplicate_models: true`: the Optimizer and Validator share a model but keep separate clients, role instructions and prompts, and only the Validator writes tests. The user approved this explicitly.
- Tried and dropped, with the measured reason in config.yaml:
  - Google AI Studio (Gemma 4 31B/26B, Gemini 3.8 Flash): 503 "high demand" all day.
  - NVIDIA NIM (Kimi K3.1, GLM 5.3, Gemma 4 31B, Mistral Large 4): 37-278 s for one-line replies, 300 s timeouts on every full-size request.
  - OpenRouter free (Gemma, Ling 3.2, ERNIE 5.5): 429 "rate-limited upstream". Nemotron 3 Ultra / 3.5 Lightning answered, but usually spent the whole cap reasoning; Ultra also wrote Python inside JSON and a wrong test.
  - DeepSeek's API: 402, no free grant. No free DeepSeek exists anywhere we checked; Cerebras and SambaNova now need a card; ModelScope needs Chinese real-name verification.
- Ten real runs instead of one. Runs 1-9 each exposed a problem that is now fixed (provider outages, reasoning exhaustion, the JSON extraction bug, `10**9` in JSON, a lambda test generator, patch wrappers, a wrong test that proposals bent the code towards, Qwen's output limit).
- Groq allows qwen3.8-27b only 1,000 output tokens per minute and refuses a request asking for more, so the Reviewer is capped at 800 tokens for every call type.

**Built**
- `quadcode/providers/http.py`, `openai_compat.py`, `gemini.py`, `anthropic.py`; registry, `MissingAPIKeyError`, `build_clients` returning one client per listed model.
- `quadcode/cli_providers.py`: `models`, `smoke`. A missing-key preflight in `solve` and `smoke`.
- Config: `ModelChoice`/`AgentConfig.fallbacks`, `request_options`, `inline_system`, per-agent `max_tokens`, `max_retry_after_s`, `run.repair_attempts`, `run.state_output_limit`; the duplicate-model check also covers fallbacks (model ids compared without host prefix and `:free`).
- Gateway: ordered fallbacks, repair retry, output limit note; `CallRecord.choice`, `repair`, `finish_reason`, `output_excerpt`.
- Orchestrator: disputed tests set aside (`RunState.disputed_tests`). Report: which model answered each slot, fallbacks column, set-aside note; token totals are no longer marked "estimated" because of failed attempts that logged 0 tokens.
- `config.yaml` (Groq line-up and the record of what was tried), `.env.example` (only `GROQ_API_KEY` needed).

**Real run `real-two-sum-10`** (`quadcode solve samples/two_sum/problem.md --rounds 2 --budget-calls 60 --run-id real-two-sum-10`): **COMPLETED** in about 2 minutes. 29 requests: 26 ok, 2 Groq 429s on Qwen (retried), 1 Qwen answer cut off at 800 tokens and repaired. 42,921 provider-reported tokens. The Validator wrote 10 tests. T07 and T09 were set aside as disputed, and both are wrong by hand: T07 expects `[2,4]` for target 7, but 3+5=8; T09 expects 99998+99999 for target 199998. V0 = C1 (all four initial versions passed the 8 remaining tests; C1 is the smallest). Accepted M-001 (generator) and M-005 (validator), rejected M-002 to M-004 (conflicts with M-001). Final V2 passes every undisputed test. The report lists the real provider/model per slot and per call.

**Tests:** `pytest`: 515 passed, 0 failed (~200 s), fully offline. New in this milestone: HTTP error mapping, the three clients over MockTransport, `models`/`smoke` CLI, provider config, fallbacks (order, stickiness, cache reuse, budget, pinned choice, validation, report naming), repair retry, JSON extraction regressions, null test args, patch wrappers, disputed tests (set aside; whole suite disputed), the token-total label.

**Open issues**
- Only three distinct models for four slots (Groq's free catalog). The spec's intended Gemini, Claude, GPT and DeepSeek need paid keys; with them, only `config.yaml` changes.
- The Validator's tests can be wrong (2 of 10 in run 10). Setting them aside keeps wrong tests from deciding anything, but in run 10 it removed the only performance test, so no runtime/memory comparison decided anything.
- If every initial implementation shares the same bug, a correct test that catches it would be set aside as disputed. It still shows as unresolved in the report.
- Qwen on Groq: about 1,000 output tokens per minute, so back-to-back Reviewer calls wait on 429s.
- The Anthropic client is tested offline only. The Gemini client never got a successful live reply (Google was overloaded all day).
- A slot's fallback position is not kept across `--resume`. A repaired answer is cached under the repair prompt, so a rerun re-sends the original request.
- `test_sandbox_docker.py::test_memory_cap_stops_bomb` is still occasionally flaky under load (from M5).

---

## M7: Evaluation harness and charts (2026-11-22)

**Approved plan (short):** `quadcode bench` runs benchmark problems under several conditions and scores each on hidden tests the agents never see: four single-agent baselines, a best-of-4 oracle, select-only (V0), the full pipeline and three ablations (no-review, no-test-feedback, no-optimizer). Mean with 95% bootstrap CI, paired win/tie/loss vs the full pipeline, CSV + Markdown + matplotlib charts; `bench-report` rebuilds them with no API calls. Built and tested on mock; one small real run (3 problems, full only) after approval.

**Key decisions**
- Single-agent baselines are each slot's blind initial solution from the full run (zero extra calls); the summary states they shared one analysis call. The oracle picks the initial solution with most hidden tests passed and is labeled an upper bound, not a system.
- Ablations are an `Ablation` stored in `RunState` (kept on resume, shown in run_started, summary.json and the report). no-optimizer excludes the slot from every phase and logs it as `excluded`, never as unavailable. no-test-feedback removes result tables, failing-test text and test-quoting rejection reasons from prompts; decisions still use the tests. Normal-pipeline prompts are byte-identical to before, so cached answers still hit.
- Ablation runs reuse the full run's analysis and initial answers through the response cache (logged as cache hits); the harness checks the initial versions' sha256 and flags `controlled = false` if they differ.
- One call budget shared by every run of an invocation. On exhaustion the current run is saved for resume, nothing else starts, and outputs are still written; unfinished conditions are "missing" and excluded, a finished run without a solution scores 0.
- Hidden tests are executed once per distinct code (by sha256) with identical seeded inputs; results stay in `hidden/<problem>.json`. Bootstrap settings are stored in bench.json so rebuilt summaries never change. Outputs have no timestamps (PNG metadata stripped), so rebuilding gives identical bytes.

**Deviations from plan**
- Problem directories are named by plain id (`benchmarks/problems/two_sum`), ordered alphabetically, not `NN_slug`.
- Extra modules: `evaluation/records.py` (checkpoint and record models), `evaluation/summary.py` (aggregation); mock bench scenarios live in `providers/mock_bench_scenarios.py` with a small builder.
- `select-only` (V0 of the full run) added as a free condition. `evaluation:` config block excluded from the resume fingerprint.
- The merge_intervals optimizer proposal fails the touching test on buggy V0 (so it is challenged and rejected on tests), rather than being rejected as a no-op.

**Built**
- `quadcode/evaluation/` (problems, conditions, records, hidden, harness, stats, summary, outputs, charts), `quadcode/cli_bench.py` (`bench`, `bench-report`); `solve --ablation`.
- `benchmarks/problems/`: 10 problems, 12-13 hidden tests each (8 with generated performance tests), hand-written expected values cross-checked against hand-written reference solutions; `benchmarks/README.md`.
- Mock scenarios `merge_intervals` (full fixes a bug every baseline has) and `lis_length` (full ties the one correct baseline).

**Mock bench** (`quadcode bench --mock`, 3 problems, production timeouts, ~6 min): full 97.4% mean hidden pass rate [92.3%, 100%] vs singles 72-82% and oracle 89.7%. Full loses two_sum to single-validator (13/13 vs 12/13): the hidden "smallest i" test catches G1's overwrite of earlier indices, and D1 keeps the first index. Ablations tie the full pipeline everywhere, as expected for scripted answers.

**Real bench `real-bench-1`** (`quadcode bench --problems merge_intervals,lis_length,top_k_frequent --conditions full --rounds 2 --budget-calls 120 --bench-id real-bench-1`, Groq line-up, Docker sandbox): COMPLETED, 41 provider attempts. Hidden pass rate: full 33.3% (1 of 3 problems solved), G1/C1/P1 100% each, D1 66.7%, oracle 100%. The full pipeline loses 2 of 3 problems to every single agent except the validator (0 wins, 1 tie, 2 losses). Both losses come from the validator's test suite, not from the code:
- merge_intervals: the validator (gpt-oss-120b, then gpt-oss-20b) wrote Python expressions such as `[[1, 100]] * 100000` inside test JSON on all four attempts including repairs, so it gave no answer, there was no suite and the run ended UNVERIFIED without a final solution.
- top_k_frequent: the validator wrote every argument as a string (`"['apple', ...]"`, `"2"`), so all four correct solutions raised TypeError, the whole suite was set aside as disputed and the run ended UNVERIFIED without a final solution.
- lis_length: completed; the final version passes all 12 hidden tests, tying every single agent.
Every number in summary.md was checked against hidden/*.json and calls.jsonl.

**Fix after the real bench (validator test suites):**
- New `quadcode/testargs.py` checks the validator's tests against the specification's signature: a JSON string where the annotation names a list, dict, number or bool (e.g. `"['apple']"` for `list[str]`, `"2"` for `int`) is reported per test. Tests that expect an exception (wrong types on purpose) and generated inputs are skipped; unions, `Optional` and unparsable signatures are never flagged.
- The gateway takes an optional answer `check`. An answer that parses but fails it is logged as invalid output, goes through the usual repair retry with the exact problem, is never cached, and a cached answer that fails it is not reused. The orchestrator passes the check only for the validator's initial call.
- The test field description and the repair rules now forbid Python inside JSON values (`[x] * 1000`, comprehensions, `range(...)`) and stringified values. This changes the validator's initial prompt, so earlier cached validator answers are not reused.
- Not changed: if the validator still gives no usable suite, the run ends UNVERIFIED with no final solution, as before.

**Real re-run `real-bench-2`** (same command, `--bench-id real-bench-2`): COMPLETED, 52 provider attempts. The analysis and the three non-validator initial answers came from the cache (3/3 per problem), so only the validator's tests and later phases were new. Full: 66.7% hidden pass rate (2 of 3 solved, up from 1 of 3); G1/C1/P1 100%, D1 66.7%, oracle 100%. Full vs each of G1/C1/P1: 0 wins, 2 ties, 1 loss.
- merge_intervals: now COMPLETED; M-001 and M-005 accepted, M-002 to M-004 rejected, validator test T07 set aside as disputed; final passes 13/13 hidden tests (tie with the singles, which already passed).
- lis_length: COMPLETED, 12/12 (tie).
- top_k_frequent: still UNVERIFIED with no final solution. gpt-oss-120b wrote `["same"] * 1000`, then a list comprehension after the repair message forbade it; the gpt-oss-20b fallback was cut off at 5000 tokens, then wrote arguments as plain strings (`"hello"`, `"world"` for `words`/`k`), which the new check rejected. Both repairs were sent; the models did not follow them.
On these three problems the pipeline never beats the best single agent: three of the four single agents already solve all of them, and the pipeline's own losses all come from the validator's tests.

**Tests:** `pytest`: 579 passed, 0 failed (~300 s), including 7 for the argument check. New: ablation behaviour per mode and resume; every reference passes its hidden tests; problem validation; bootstrap and win/tie/loss on hand-computed data; end-to-end mock bench checked against hidden suite runs and calls.jsonl; hidden test inputs never reach a prompt or run file; budget abort and resume; byte-identical regeneration; CLI.

**Open issues**
- The pipeline delivers no solution when the validator's suite is missing or wholly disputed, even when other slots wrote correct code (2 of 3 real problems in real-bench-1, 1 of 3 in real-bench-2). Mistyped and expression-valued tests are now caught and repaired (see above); delivering an unverified V0 or letting another slot write tests when the validator still fails is not implemented.
- In the mock bench, two_sum's full run rejected M-001 for a "measured regression" of 1.19 ms -> 1.50 ms: noise on a millisecond perf test crossing the 25% tolerance (the min/max ranges did not overlap). Ablation runs accepted the same change, so the conditions diverged on timing noise, not on the ablation. Selection noise from M4 is not fully solved for very short tests.
- Mock ablations cannot show an effect (scripted answers ignore the prompt). Real ablations cost roughly one more full run each per problem.
- With 3-10 problems the CIs are wide; the summary says so.
- Groq's 429 error text includes the account's organization id, and it is logged verbatim in calls.jsonl (runs/ is git-ignored; API keys are still redacted).
