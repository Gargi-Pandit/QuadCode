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
