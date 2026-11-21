# QuadCode: project rules

Multi-agent code generation, review, testing and optimization system. Full original spec: `SPEC.md`. Read it when starting a new session. The rules below refine it for rate-limited free-tier APIs and take precedence where they differ.

## Workflow with me
- Plan mode first for any new milestone. Show the plan and wait for approval.
- Work in milestones M1-M9 (list below). After each one: run the tests, summarize in <=10 lines, then STOP and wait for my "continue".
- Ask me only when genuinely blocked. Otherwise pick a sensible default and state it.
- No placeholders, no TODO stubs, no fake implementations. Type hints on everything, small modules, pytest for all logic.
- Never wire real providers or spend real API calls before M6 and my approval. Everything before M6 runs on the mock provider.

## Non-negotiable rules
1. Four agents use four independently configured provider clients. Never one LLM simulating four.
2. Phase 2 is blind: agents never see each other's code. Save immutable versions G1, C1, P1, D1.
3. Never fabricate conversations, modifications, test results, timings or memory figures. Every report and UI line must trace to a record in the run log (JSONL). If code did not execute, say so.
4. If an agent fails (API error, rate limit, timeout), mark it UNAVAILABLE. Never treat missing feedback as approval. Never silently skip an agent.
5. No majority voting. Selection order: correctness > requirements > passing tests > time efficiency > memory > security > readability > maintainability.
6. Never overwrite old versions. Keep accepted AND rejected modification records.
7. Generated code is untrusted. It runs only in the sandbox, never in the host process.
8. If nothing can be verified correct, report the unresolved issues instead of claiming convergence.
9. If QuadCode does not beat a baseline on a metric, report that honestly. Never hardcode benchmark results.

## Agent naming (free-tier substitution)
- Agents are identified by ROLE: Generator, Reviewer, Optimizer, Validator. The spec's names (Gemini, Claude, GPT, DeepSeek) are the intended defaults, but I may substitute free models.
- Provider name and model for every agent come ONLY from `config.yaml`. Reports, UI and logs must display the actual provider and model used for each call. Never hardcode vendor names in output text or prompts.
- Version labels G1/C1/P1/D1 stay as labels for the four agent slots, but reports must show which real model sits behind each.

## Architecture
```
quadcode/
  config.yaml  .env.example  requirements.txt
  quadcode/
    providers/   base.py, gemini.py, openai_compat.py, anthropic.py, mock.py
    ratelimit.py cache.py schemas.py agents.py orchestrator.py
    versions.py  sandbox.py testing.py report.py cli.py ui/app.py
  benchmarks/  tests/  runs/<run_id>/
```
- `openai_compat.py` serves OpenAI, GitHub Models, OpenRouter, Groq, DeepSeek and others via `base_url`. Prefer it for every provider except Gemini and Anthropic.
- Models, limits, max_tokens per call type, rounds and timeouts all live in `config.yaml`. Swapping a model must never require code changes.
- Run layout: `runs/<run_id>/` holds calls.jsonl, events.jsonl, versions/, modifications.json, tests.json, report.md, report.html, final_solution.<ext>.
- API keys exist only in the orchestrator process, loaded from `.env`. Never log them, never pass them to the sandbox.

## Token and rate-limit strategy (implement all)
- Per-provider token bucket (requests/min and tokens/min), exponential backoff with jitter on 429/5xx, configurable timeouts.
- Disk cache keyed by sha256(model+prompt+params). Cache hits are logged as cache hits. Checkpoint/resume: `--resume run_id` never repeats completed calls.
- `--budget-calls N` aborts gracefully and still produces a report from existing data.
- max_tokens caps per call type. JSON-only outputs. Strip prose and markdown fences before parsing.
- After Phase 3, send diffs, not full files. Send only truncated failing-test output.
- Log every call to calls.jsonl: agent, role, provider, model, prompt hash, tokens in/out, latency, cache hit.
- Parallelize agent calls only when rate limits allow.
- Benchmark defaults must be small (about 10 problems) so free quotas survive. N is configurable.

## Workflow phases (minimum API calls)
1. ONE analysis call produces a structured spec. Ask me for clarification only if essential info is missing.
2. Four parallel blind calls. Each returns code, algorithm, data structures, complexity, pros, cons, assumptions. The Validator also returns the test suite.
2.5. Local, zero tokens: run the test suite against all four initial solutions in the sandbox. Record pass/fail, runtime, peak memory.
3. ONE call per agent: all four solutions (code only) plus the compact test table. Returns strengths, weaknesses, bugs, suggestions.
4+6. Per round (configurable 1-5, default 3): V0 is the best-passing initial solution chosen by evidence. ONE call per agent returns 0..N ModificationRecords as unified diffs. Apply and test locally. Challenge calls only for contested or failing proposals. The orchestrator accepts or rejects on test evidence and measurements, never by vote. EARLY STOP if a round accepts nothing. Never add changes to fill rounds.
7. Final repeated benchmark of all initial solutions and the final one under identical inputs. Report median/min/max and the method.
8. Integrate accepted modifications. Add concise `# [agent:M-xxx]` comments on important accepted changes.

ModificationRecord fields: id (M-001...), proposed_by, base_version, new_version, original_code, replacement_code, original_location, new_location, justification, expected_impact, disadvantages, reviewers + verdicts, test_status, decision (accepted/rejected + reason), timestamps. Keep per-version line mappings. Never claim an agent changed a line unless a record exists.

## Sandbox
- Preferred: Docker with `--network none`, read-only FS except a tmp dir, memory cap, CPU cap, wall-clock timeout, no env vars, no mounted credentials.
- Fallback without Docker: subprocess with RLIMIT_AS and RLIMIT_CPU, stripped environment, temp working dir, timeout. The report must carry a clear warning that isolation is weaker.
- Measure runtime with time.perf_counter over repeated runs, and peak memory with tracemalloc or resource.getrusage.

## Report (generated from logs only)
Ten sections: 1 Problem Analysis, 2 Four Initial Implementations (full code), 3 Initial Comparison, 4 Agent Communication Log (real messages, chronological, disagreements preserved), 5 Optimization Rounds, 6 Testing and Benchmarking (measured vs theoretical clearly labeled), 7 Modification History, 8 Final Implementation, 9 Final Technical Evaluation, 10 Reproduction Instructions. Also include a per-agent contribution table, an agent availability table, and a token/cost table from calls.jsonl.

## Evaluation harness
- Conditions: 4 single-agent baselines, best-of-4 oracle, full pipeline, ablations (no-review, no-testing-feedback, no-optimizer).
- Hidden tests MUST be separate from tests the agents saw.
- Report mean with 95% bootstrap CI, paired win/tie/loss per problem vs each single agent. Output CSV, Markdown table, matplotlib charts.

## UI (Streamlit, quadcode/ui/app.py)
- Thin view over existing run logs. No LLM calls except by invoking the same orchestrator the CLI uses. Must work fully in mock mode and load any saved run with zero API usage.
- Every number shown comes from logged data and is labeled measured or theoretical.

## Mock mode
`--mock` uses deterministic fake agents that exercise EVERY code path: a deliberately buggy solution, a disagreement, a rejected modification, an agent-unavailable case, early stop. Build and test everything in mock mode first.

## Milestones (stop after each, wait for "continue")
- M1 skeleton, schemas, config, mock provider, cache, rate limiter, tests
- M2 sandbox, test runner, benchmarking (no LLMs, hand-written solutions)
- M3 version store, modification records, diff application, line mapping
- M4 orchestrator phases 1-8 in mock mode, end-to-end, early stop, event logging
- M5 report generator (md + html)
- M6 real providers with ONE short smoke call each, then one real end-to-end run on a simple problem with `--budget-calls`
- M7 evaluation harness and charts
- M8 Streamlit dashboard
- M9 README (setup, .env, run, test, UI launch, reproduction)

## Commands
Fill in as they are created: `pytest`, `quadcode solve problem.md --rounds 3 --mock`, `quadcode bench`, `streamlit run quadcode/ui/app.py`.