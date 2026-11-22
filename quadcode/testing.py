"""Run a test suite against one code version in the sandbox and judge the results.

The sandbox receives inputs only. Expected values stay on the host and are compared
here, so solution code can neither read them nor influence the verdict. Tests the
sandbox never reported are marked from what actually happened (timeout, crash, not
run); nothing is assumed to have passed.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from quadcode.compare import Tolerance, compare_values, digest_matches, exception_matches
from quadcode.config import SandboxConfig
from quadcode.sandbox import RUNNER_FILE, WALL_SLACK_S, ExecResult, Sandbox, runner_source
from quadcode.sandbox_runner import (
    EVENT_DONE,
    EVENT_ENV,
    EVENT_LOAD_ERROR,
    EVENT_RESULT,
    EVENT_START,
    OUTCOME_INPUT_ERROR,
    OUTCOME_RAISED,
    OUTCOME_RETURNED,
    OUTCOME_TIMEOUT,
    canonical_json,
    normalize,
)
from quadcode.schemas import SandboxBackendKind, SuiteRun, TestCase, TestResult, TestStatus

PYTHON_LANGUAGES = frozenset({"python", "python3", "py"})


def is_python(language: str) -> bool:
    return language.strip().lower() in PYTHON_LANGUAGES


def measurement_method(backend: str, repeats: int, warmup: int, cfg: SandboxConfig, measure_memory: bool) -> str:
    parts = [
        f"{backend} sandbox",
        "the first call decides the test outcome and counts as warmup run 1" if warmup else
        "the first call decides the test outcome and is also timed",
        f"{warmup} warmup + {repeats} timed calls per test, each on a fresh deep copy of the inputs",
        "time.perf_counter_ns around the call only, gc.collect() before every call",
        "runtime_s = median of the timed calls",
        f"per-call limit {cfg.test_timeout_s}s, whole-process limit {cfg.wall_timeout_s}s",
        f"generator inputs seeded per test from seed {cfg.seed}",
    ]
    if measure_memory:
        parts.append("peak memory = tracemalloc peak above the copied inputs during one extra call "
                     "(Python heap only, not process RSS)")
    return "; ".join(parts) + "."


def build_payload(
    tests: Sequence[TestCase],
    entry_point: str,
    cfg: SandboxConfig,
    repeats: int,
    measure_memory: bool,
    rlimits: dict[str, int],
) -> dict[str, Any]:
    """Inputs-only payload. ``expected`` and ``expect_exception`` are deliberately absent."""
    return {
        "entry_point": entry_point,
        "seed": cfg.seed,
        "timeout_s": cfg.test_timeout_s,
        "repeats": repeats,
        "warmup": cfg.warmup,
        "measure_memory": measure_memory,
        "max_value_chars": max(1_000, cfg.max_output_bytes // 4),
        "rlimits": rlimits,
        "tests": [{"id": t.id, "args": t.args, "kwargs": t.kwargs, "generator": t.generator} for t in tests],
    }


def parse_protocol(stdout: str) -> list[dict[str, Any]]:
    """Protocol lines from the runner; anything that isn't a JSON event object is ignored."""
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("event"), str):
            events.append(obj)
    return events


def _store_actual(value: Any, max_chars: int) -> tuple[Any, bool]:
    text = canonical_json(value)
    if len(text) <= max_chars:
        return value, False
    return text[:max_chars] + f"... <truncated, {len(text)} chars total>", True


def _brief(value: Any, limit: int = 120) -> str:
    text = canonical_json(normalize(value)[0])
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _exc_text(exc: dict[str, Any]) -> str:
    message = exc.get("message") or ""
    return f"{exc.get('type', 'Exception')}: {message}" if message else str(exc.get("type", "Exception"))


def _judge(test: TestCase, line: dict[str, Any], version: str, cfg: SandboxConfig) -> TestResult:
    runtimes = [float(x) for x in line.get("runtimes_s") or []]
    common: dict[str, Any] = {
        "test_id": test.id,
        "version": version,
        "runtimes_s": runtimes,
        "runtime_s": statistics.median(runtimes) if runtimes else None,
        "peak_memory_bytes": line.get("peak_memory_bytes"),
        "measured": True,
    }
    outcome = line.get("outcome")
    if outcome == OUTCOME_TIMEOUT:
        run = line.get("timed_out_on_run")
        where = f" (on run {int(run) + 1})" if isinstance(run, int) else ""
        return TestResult(status=TestStatus.TIMEOUT,
                          error=f"call exceeded the {line.get('timeout_s', cfg.test_timeout_s)}s limit{where}", **common)
    if outcome == OUTCOME_INPUT_ERROR:
        exc = line.get("exception") or {}
        return TestResult(status=TestStatus.ERROR,
                          error=f"test input generator failed (test defect, not the solution): {_exc_text(exc)}",
                          **common)
    if outcome == OUTCOME_RAISED:
        exc = line.get("exception") or {}
        text = _exc_text(exc)
        if test.expect_exception is None:
            return TestResult(status=TestStatus.ERROR, actual=text, error=f"raised {text}", **common)
        if exception_matches(test.expect_exception, str(exc.get("type", "")), exc.get("mro") or []):
            return TestResult(status=TestStatus.PASS, actual=text, **common)
        return TestResult(status=TestStatus.FAIL, actual=text,
                          error=f"expected {test.expect_exception}, raised {text}", **common)
    if outcome != OUTCOME_RETURNED:
        return TestResult(status=TestStatus.ERROR, error=f"unrecognized runner outcome {outcome!r}", **common)

    non_json = bool(line.get("non_json"))
    if "value" in line:
        actual, truncated = _store_actual(line["value"], cfg.max_actual_chars)
        if test.expect_exception is not None:
            return TestResult(status=TestStatus.FAIL, actual=actual, truncated=truncated, non_json=non_json,
                              error=f"expected {test.expect_exception} but a value was returned", **common)
        ok = compare_values(line["value"], test.expected, test.compare,
                            Tolerance(cfg.float_rel_tol, cfg.float_abs_tol))
        error = None if ok else f"expected {_brief(test.expected)}, got {_brief(line['value'])}"
        return TestResult(status=TestStatus.PASS if ok else TestStatus.FAIL, actual=actual, truncated=truncated,
                          non_json=non_json, error=error, **common)

    # Value too large to transfer: only its digest came back.
    digest = str(line.get("value_digest", ""))
    actual = f"<{line.get('value_chars', '?')} chars, sha256 {digest[:16]}...>"
    if test.expect_exception is not None:
        return TestResult(status=TestStatus.FAIL, actual=actual, truncated=True, non_json=non_json,
                          error=f"expected {test.expect_exception} but a value was returned", **common)
    if test.compare != "exact":
        return TestResult(status=TestStatus.ERROR, actual=actual, truncated=True, non_json=non_json,
                          error=f"output too large to transfer for '{test.compare}' comparison", **common)
    ok = digest_matches(digest, test.expected)
    return TestResult(status=TestStatus.PASS if ok else TestStatus.FAIL, actual=actual, truncated=True,
                      non_json=non_json, error=None if ok else "actual digest != expected digest", **common)


def _abort_reason(res: ExecResult, wall_limit: float) -> str:
    if res.timed_out:
        return f"sandbox wall-clock limit ({wall_limit:.0f}s) reached"
    if res.output_truncated:
        return "sandbox output exceeded max_output_bytes and the process was killed"
    code = res.exit_code
    if code in (137, -9):
        return f"sandbox process was killed (exit {code}; SIGKILL, likely the memory limit)"
    if code in (152, -24):
        return f"sandbox process hit the CPU time limit (exit {code})"
    if code not in (0, None):
        return f"sandbox process exited with code {code}"
    return "runner finished without reporting this test"


def _missing(test: TestCase, version: str, was_running: bool, res: ExecResult, reason: str) -> TestResult:
    if not was_running:
        return TestResult(test_id=test.id, version=version, status=TestStatus.NOT_RUN,
                          error=f"not run: {reason}", measured=False)
    status = TestStatus.TIMEOUT if res.timed_out else TestStatus.ERROR
    return TestResult(test_id=test.id, version=version, status=status,
                      error=f"stopped during this test: {reason}", measured=True)


def not_executed(
    *,
    version: str,
    language: str,
    entry_point: str,
    tests: Sequence[TestCase],
    reason: str,
    cfg: SandboxConfig,
    repeats: int,
    backend: SandboxBackendKind = "none",
    isolation_warning: str | None = None,
) -> SuiteRun:
    """A suite record that states plainly that the code was not executed."""
    return SuiteRun(
        version=version, language=language, entry_point=entry_point, backend=backend,
        executed=False, not_executed_reason=reason, isolation_warning=isolation_warning,
        repeats=repeats, warmup=cfg.warmup, seed=cfg.seed,
        results=[TestResult(test_id=t.id, version=version, status=TestStatus.NOT_RUN,
                            error=f"not executed: {reason}", measured=False) for t in tests],
    )


def run_suite(
    *,
    code: str,
    entry_point: str,
    language: str,
    tests: Sequence[TestCase],
    sandbox: Sandbox | None,
    cfg: SandboxConfig,
    version: str,
    repeats: int | None = None,
    measure_memory: bool = True,
    unavailable_reason: str = "no sandbox backend available",
) -> SuiteRun:
    reps = repeats if repeats is not None else cfg.test_repeats
    if not is_python(language):
        return not_executed(version=version, language=language, entry_point=entry_point, tests=tests,
                            reason=f"no sandbox runner for language {language!r}", cfg=cfg, repeats=reps)
    if sandbox is None:
        return not_executed(version=version, language=language, entry_point=entry_point, tests=tests,
                            reason=unavailable_reason, cfg=cfg, repeats=reps)

    payload = build_payload(tests, entry_point, cfg, reps, measure_memory, sandbox.rlimits())
    files = {RUNNER_FILE: runner_source(), "solution.py": code, "payload.json": json.dumps(payload)}
    wall_limit = cfg.wall_timeout_s + WALL_SLACK_S
    res = sandbox.execute(files, timeout_s=wall_limit)
    events = parse_protocol(res.stdout)

    env = next((e for e in events if e["event"] == EVENT_ENV), None)
    if env is None:
        detail = res.stderr_tail.strip().splitlines()[-1] if res.stderr_tail.strip() else "no output"
        return not_executed(
            version=version, language=language, entry_point=entry_point, tests=tests, cfg=cfg, repeats=reps,
            reason=f"sandbox did not start the runner ({_abort_reason(res, wall_limit)}): {detail}",
            backend=sandbox.name, isolation_warning=sandbox.isolation_warning,
        ).model_copy(update={"exit_code": res.exit_code, "timed_out": res.timed_out,
                             "wall_time_s": res.wall_time_s, "stderr_tail": res.stderr_tail})

    load_error = next((e for e in events if e["event"] == EVENT_LOAD_ERROR), None)
    results_by_id = {e["test_id"]: e for e in events if e["event"] == EVENT_RESULT and "test_id" in e}
    started = [e["test_id"] for e in events if e["event"] == EVENT_START]
    running = started[-1] if started and started[-1] not in results_by_id else None
    done = next((e for e in events if e["event"] == EVENT_DONE), None)
    reason = _abort_reason(res, wall_limit)

    results: list[TestResult] = []
    for test in tests:
        if load_error is not None:
            exc = load_error.get("exception") or {}
            results.append(TestResult(test_id=test.id, version=version, status=TestStatus.ERROR,
                                      error=f"solution failed to load: {_exc_text(exc)}", measured=True))
        elif test.id in results_by_id:
            results.append(_judge(test, results_by_id[test.id], version, cfg))
        else:
            results.append(_missing(test, version, test.id == running, res, reason))

    return SuiteRun(
        version=version, language=language, entry_point=entry_point, backend=sandbox.name, executed=True,
        isolation_warning=sandbox.isolation_warning,
        sandbox_env={k: v for k, v in env.items() if k != "event"},
        method=measurement_method(sandbox.name, reps, cfg.warmup, cfg, measure_memory),
        repeats=reps, warmup=cfg.warmup, seed=cfg.seed, results=results,
        wall_time_s=res.wall_time_s, exit_code=res.exit_code, timed_out=res.timed_out,
        output_truncated=res.output_truncated, stderr_tail=res.stderr_tail,
        max_rss_kb=done.get("max_rss_kb") if done else None,
    )


_TESTS_ADAPTER = TypeAdapter(list[TestCase])


def suite_status(suite: SuiteRun) -> TestStatus:
    """One status for a whole suite run: PASS only if every test passed; otherwise the most
    severe status present (ERROR, then TIMEOUT, then FAIL); NOT_RUN if nothing executed."""
    if not suite.executed or not suite.results:
        return TestStatus.NOT_RUN
    if suite.all_passed:
        return TestStatus.PASS
    for status in (TestStatus.ERROR, TestStatus.TIMEOUT, TestStatus.FAIL):
        if suite.count(status):
            return status
    return TestStatus.NOT_RUN


def load_test_cases(path: str | Path) -> list[TestCase]:
    """Read a JSON file holding a list of test cases (or ``{"tests": [...]}``)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "tests" in raw:
        raw = raw["tests"]
    return _TESTS_ADAPTER.validate_python(raw)


def write_test_results(path: str | Path, runs: Sequence[SuiteRun]) -> Path:
    """Write suite runs as JSON (the run directory's ``tests.json``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([r.model_dump(mode="json") for r in runs], indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return p
