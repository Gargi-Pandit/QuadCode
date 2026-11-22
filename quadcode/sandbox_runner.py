"""Test runner executed INSIDE the sandbox. Standard library only, Python 3.12 compatible.

Invoked as ``python -I -S -B sandbox_runner.py <workdir>``. It reads ``payload.json``
(inputs only: expected outputs never enter the sandbox), loads ``solution.py`` and
calls the entry point once per test plus repeated timed runs. Every event is one
JSON line on a private copy of the original stdout, flushed immediately, so a crash
or kill still leaves every finished result readable. The host compares results.

The host process may import this module only for its pure helpers
(``normalize``, ``canonical_json``, ``value_digest``); it never runs solutions.
"""

from __future__ import annotations

import builtins
import copy
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import signal
import sys
import time
import traceback
from typing import Any, TextIO

EVENT_ENV = "env"
EVENT_LOAD_ERROR = "load_error"
EVENT_START = "start"
EVENT_RESULT = "result"
EVENT_DONE = "done"

OUTCOME_RETURNED = "returned"
OUTCOME_RAISED = "raised"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_INPUT_ERROR = "input_error"

_MAX_DEPTH = 64
_MAX_MESSAGE = 500
_MAX_CAPTURE = 4_000


# ------------------------------------------------------------------ pure helpers


def normalize(value: Any) -> tuple[Any, bool]:
    """Convert ``value`` to plain JSON data. Returns (normalized, had_non_json_parts).

    Tuples become lists, sets become sorted lists (repr-sorted if unorderable), dict
    keys that JSON would stringify are stringified, anything else becomes its repr.
    """
    flag = [False]
    return _norm(value, flag, 0), flag[0]


def _norm(value: Any, flag: list[bool], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        flag[0] = True
        return _safe_repr(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            flag[0] = True
            return repr(value)
        return value
    if isinstance(value, (list, tuple)):
        return [_norm(v, flag, depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        items = [_norm(v, flag, depth + 1) for v in value]
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=canonical_json)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str):
                key = k
            elif k is None or isinstance(k, (bool, int, float)):
                key = json.dumps(k)
            else:
                flag[0] = True
                key = _safe_repr(k)
            out[key] = _norm(v, flag, depth + 1)
        return out
    flag[0] = True
    return _safe_repr(value)


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)[:_MAX_MESSAGE]
    except Exception:  # noqa: BLE001 - repr of untrusted objects can raise anything
        return f"<unrepresentable {type(value).__name__}>"


def canonical_json(value: Any) -> str:
    """Deterministic JSON text of already-normalized data."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def value_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ execution


class _CallTimeout(BaseException):
    """Raised by SIGALRM. BaseException so ``except Exception`` in solutions can't swallow it."""


def _on_alarm(signum: int, frame: Any) -> None:
    raise _CallTimeout()


class _BoundedWriter(io.TextIOBase):
    """Stands in for sys.stdout while solution code runs; keeps only the first chars."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.kept = 0
        self.total = 0

    def write(self, s: str) -> int:
        self.total += len(s)
        if self.kept < self.limit:
            piece = s[: self.limit - self.kept]
            self.parts.append(piece)
            self.kept += len(piece)
        return len(s)

    def writable(self) -> bool:
        return True


class _Emitter:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def __call__(self, event: str, **data: Any) -> None:
        data["event"] = event
        self.stream.write(json.dumps(data, ensure_ascii=False) + "\n")
        self.stream.flush()


def _call_with_timeout(
    fn: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    timeout_s: float,
    before_call: Any = None,
) -> tuple[str, Any, int]:
    """Run ``fn`` once on fresh copies of the inputs.

    Returns (outcome, value or exception summary dict, elapsed_ns of the call itself). Copying the
    inputs and the garbage collection happen before the clock starts.
    """
    a = copy.deepcopy(args)
    kw = copy.deepcopy(kwargs)
    gc.collect()
    if before_call is not None:
        before_call()
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    t0 = time.perf_counter_ns()
    try:
        value = fn(*a, **kw)
        elapsed = time.perf_counter_ns() - t0
        signal.setitimer(signal.ITIMER_REAL, 0)
        return OUTCOME_RETURNED, value, elapsed
    except _CallTimeout:
        elapsed = time.perf_counter_ns() - t0
        return OUTCOME_TIMEOUT, None, elapsed
    except BaseException as exc:  # noqa: BLE001 - SystemExit/KeyboardInterrupt from solutions are results too
        elapsed = time.perf_counter_ns() - t0
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Keep only a summary: the traceback would pin the solution's frames (and
        # everything they allocated) in memory across later runs.
        return OUTCOME_RAISED, _exception_info(exc), elapsed
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _exception_info(exc: BaseException) -> dict[str, Any]:
    mro = [c.__name__ for c in type(exc).__mro__ if c not in (object,)]
    try:
        message = str(exc)[:_MAX_MESSAGE]
    except Exception:  # noqa: BLE001
        message = "<unprintable message>"
    return {"type": type(exc).__name__, "mro": mro, "message": message}


def _build_inputs(test: dict[str, Any], seed: int, timeout_s: float) -> tuple[list[Any], dict[str, Any]]:
    expr = test.get("generator")
    if not expr:
        return list(test.get("args") or []), dict(test.get("kwargs") or {})
    rng = random.Random(f"{seed}:{test['id']}")
    namespace = {"__builtins__": builtins, "rng": rng}
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        produced = eval(expr, namespace)  # noqa: S307 - test generators run inside the sandbox by design
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if not (isinstance(produced, (tuple, list)) and len(produced) == 2):
        raise ValueError("generator must evaluate to (args, kwargs)")
    args, kwargs = produced
    if not isinstance(args, (list, tuple)) or not isinstance(kwargs, dict):
        raise ValueError("generator must evaluate to (list_of_args, dict_of_kwargs)")
    return list(args), dict(kwargs)


def _measure_peak_memory(fn: Any, args: list[Any], kwargs: dict[str, Any], timeout_s: float) -> int | None:
    """Peak Python heap growth during one call (tracemalloc), above the copied inputs."""
    import tracemalloc

    baseline = [0]

    def mark_baseline() -> None:
        tracemalloc.reset_peak()
        baseline[0] = tracemalloc.get_traced_memory()[0]

    tracemalloc.start()
    try:
        try:
            outcome, _, _ = _call_with_timeout(fn, args, kwargs, timeout_s, before_call=mark_baseline)
        except _CallTimeout:
            outcome = OUTCOME_TIMEOUT
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    if outcome == OUTCOME_TIMEOUT:
        return None
    return max(0, int(peak - baseline[0]))


def _apply_rlimits(limits: dict[str, int]) -> None:
    try:
        import resource
    except ImportError:
        return
    names = {
        "as_bytes": resource.RLIMIT_AS,
        "cpu_s": resource.RLIMIT_CPU,
        "fsize_bytes": resource.RLIMIT_FSIZE,
        "nofile": resource.RLIMIT_NOFILE,
    }
    for key, value in limits.items():
        if key in names and value:
            resource.setrlimit(names[key], (int(value), int(value)))


def _max_rss_kb() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _run_test(
    fn: Any,
    test: dict[str, Any],
    payload: dict[str, Any],
    emit: _Emitter,
    capture: _BoundedWriter,
) -> None:
    timeout_s = float(payload["timeout_s"])
    repeats = int(payload["repeats"])
    warmup = int(payload["warmup"])
    max_value_chars = int(payload["max_value_chars"])
    tid = test["id"]
    emit(EVENT_START, test_id=tid)
    printed_before = capture.total
    try:
        args, kwargs = _build_inputs(test, int(payload["seed"]), timeout_s)
    except BaseException as exc:  # noqa: BLE001
        emit(EVENT_RESULT, test_id=tid, outcome=OUTCOME_INPUT_ERROR, exception=_exception_info(exc))
        return

    # The first call decides the outcome; it also counts as the first warmup run.
    first_outcome: str = OUTCOME_RETURNED
    first_value: Any = None
    timed_out_on_run: int | None = None
    runtimes_s: list[float] = []
    for i in range(warmup + repeats):
        try:
            outcome, value, elapsed_ns = _call_with_timeout(fn, args, kwargs, timeout_s)
        except _CallTimeout:  # alarm landed just outside the guarded call
            outcome, value, elapsed_ns = OUTCOME_TIMEOUT, None, 0
        if i == 0:
            first_outcome, first_value = outcome, value
        if outcome == OUTCOME_TIMEOUT:
            timed_out_on_run = i
            break
        if i >= warmup:
            runtimes_s.append(elapsed_ns / 1e9)

    result: dict[str, Any] = {"test_id": tid, "outcome": first_outcome, "runtimes_s": runtimes_s}
    if timed_out_on_run is not None:
        # Any run exceeding the per-call limit makes the test a timeout.
        result["outcome"] = OUTCOME_TIMEOUT
        result["timeout_s"] = timeout_s
        result["timed_out_on_run"] = timed_out_on_run
    elif first_outcome == OUTCOME_RAISED:
        result["exception"] = first_value
    else:
        normalized, non_json = normalize(first_value)
        text = canonical_json(normalized)
        result["non_json"] = non_json
        result["value_digest"] = value_digest(normalized)
        result["value_chars"] = len(text)
        if len(text) <= max_value_chars:
            result["value"] = normalized

    if payload.get("measure_memory") and result["outcome"] != OUTCOME_TIMEOUT:
        peak = _measure_peak_memory(fn, args, kwargs, timeout_s)
        if peak is not None:
            result["peak_memory_bytes"] = peak
    result["printed_chars"] = capture.total - printed_before
    emit(EVENT_RESULT, **result)


def main(argv: list[str]) -> int:
    workdir = argv[1] if len(argv) > 1 else "."
    sys.dont_write_bytecode = True
    # Keep a private handle on the real stdout for protocol lines, then point fd 1 at
    # /dev/null so prints from solution code (even os-level writes) cannot reach it.
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    capture = _BoundedWriter(_MAX_CAPTURE)
    sys.stdout = capture
    emit = _Emitter(proto)
    signal.signal(signal.SIGALRM, _on_alarm)

    with open(os.path.join(workdir, "payload.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    _apply_rlimits(payload.get("rlimits") or {})
    emit(
        EVENT_ENV,
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
    )

    timeout_s = float(payload["timeout_s"])
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        spec = importlib.util.spec_from_file_location("solution", os.path.join(workdir, "solution.py"))
        if spec is None or spec.loader is None:
            raise ImportError("cannot load solution.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, payload["entry_point"], None)
        if not callable(fn):
            raise AttributeError(f"solution does not define a callable '{payload['entry_point']}'")
    except _CallTimeout:
        emit(EVENT_LOAD_ERROR, exception={"type": "TimeoutError", "mro": ["TimeoutError"],
                                          "message": f"loading the solution took longer than {timeout_s}s"})
        return 0
    except BaseException as exc:  # noqa: BLE001
        info = _exception_info(exc)
        info["traceback"] = "".join(traceback.format_exception_only(type(exc), exc))[-_MAX_MESSAGE:]
        emit(EVENT_LOAD_ERROR, exception=info)
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    for test in payload["tests"]:
        _run_test(fn, test, payload, emit, capture)
    emit(EVENT_DONE, max_rss_kb=_max_rss_kb(), printed_chars=capture.total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
