"""Suite execution in the real subprocess sandbox, judged on the host."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pytest

from quadcode.config import SandboxConfig
from quadcode.providers import mock_scenarios
from quadcode.sandbox import ExecResult, SubprocessSandbox
from quadcode.schemas import SuiteRun, TestCase, TestStatus
from quadcode.testing import build_payload, load_test_cases, parse_protocol, run_suite, write_test_results

from .conftest import SAMPLES


def _run(code: str, tests: list[TestCase], sandbox: SubprocessSandbox, cfg: SandboxConfig,
         version: str = "V", **kw: Any) -> SuiteRun:
    return run_suite(code=code, entry_point="two_sum", language="python", tests=tests, sandbox=sandbox,
                     cfg=cfg, version=version, **kw)


def _status(run: SuiteRun) -> dict[str, TestStatus]:
    return {r.test_id: r.status for r in run.results}


def _case(id_: str = "X1", **kw: Any) -> TestCase:
    return TestCase.model_validate({"id": id_, "category": "normal", **kw})


# ---------------------------------------------------------------- the four mock solutions


@pytest.mark.parametrize("label", ["G1", "D1"])
def test_correct_solutions_pass_everything(label: str, two_sum_tests: list[TestCase],
                                           subprocess_sandbox: SubprocessSandbox, sandbox_cfg: SandboxConfig) -> None:
    code = getattr(mock_scenarios, f"{label}_CODE")
    run = _run(code, two_sum_tests, subprocess_sandbox, sandbox_cfg, version=label)
    assert run.executed and run.all_passed
    assert run.backend == "subprocess" and run.isolation_warning
    assert run.sandbox_env["python"]
    assert run.max_rss_kb and run.max_rss_kb > 0
    for r in run.results:
        assert r.measured and r.version == label
        assert len(r.runtimes_s) == sandbox_cfg.test_repeats
        assert r.runtime_s == statistics.median(r.runtimes_s)
        assert r.peak_memory_bytes is not None and r.peak_memory_bytes >= 0


def test_sorted_pointer_solution_fails_index_tests(two_sum_tests: list[TestCase],
                                                  subprocess_sandbox: SubprocessSandbox,
                                                  sandbox_cfg: SandboxConfig) -> None:
    run = _run(mock_scenarios.P1_CODE, two_sum_tests, subprocess_sandbox, sandbox_cfg, version="P1")
    failed = {k for k, v in _status(run).items() if v != TestStatus.PASS}
    assert failed == {"T02", "T06"}
    t02 = next(r for r in run.results if r.test_id == "T02")
    assert t02.actual == [0, 2]
    assert t02.error and "expected [1,2]" in t02.error


def test_quadratic_solution_times_out_on_performance_test(two_sum_tests: list[TestCase],
                                                         subprocess_sandbox: SubprocessSandbox,
                                                         sandbox_cfg: SandboxConfig) -> None:
    run = _run(mock_scenarios.C1_CODE, two_sum_tests, subprocess_sandbox, sandbox_cfg, version="C1")
    statuses = _status(run)
    assert statuses.pop("T09") == TestStatus.TIMEOUT
    assert all(s == TestStatus.PASS for s in statuses.values())
    t09 = next(r for r in run.results if r.test_id == "T09")
    assert t09.runtime_s is None and t09.runtimes_s == []
    assert t09.peak_memory_bytes is None


def test_sample_files_match_mock_scenario() -> None:
    for label in ("G1", "C1", "P1", "D1"):
        code = (SAMPLES / f"{label.lower()}.py").read_text(encoding="utf-8")
        assert code == getattr(mock_scenarios, f"{label}_CODE")
    on_disk = json.loads((SAMPLES / "tests.json").read_text(encoding="utf-8"))
    assert on_disk == mock_scenarios._TESTS


# ---------------------------------------------------------------- hostile samples


def _sample(name: str) -> str:
    return (SAMPLES / f"{name}.py").read_text(encoding="utf-8")


def test_infinite_loop_times_out(subprocess_sandbox: SubprocessSandbox, sandbox_cfg: SandboxConfig) -> None:
    cfg = sandbox_cfg.model_copy(update={"test_timeout_s": 0.3})
    tests = [_case("A", args=[[1], 1]), _case("B", args=[[2], 2])]
    run = _run(_sample("bad_infinite_loop"), tests, SubprocessSandbox(cfg), cfg)
    assert _status(run) == {"A": TestStatus.TIMEOUT, "B": TestStatus.TIMEOUT}
    assert "0.3s limit" in (run.results[0].error or "")


def test_memory_bomb_is_stopped(subprocess_sandbox: SubprocessSandbox, sandbox_cfg: SandboxConfig) -> None:
    cfg = sandbox_cfg.model_copy(update={"test_timeout_s": 5.0, "memory_mb": 256})
    run = _run(_sample("bad_memory_bomb"), [_case("A", args=[[1], 1]), _case("B", args=[[1], 1])],
               SubprocessSandbox(cfg), cfg)
    assert all(r.status == TestStatus.ERROR for r in run.results)
    assert all("MemoryError" in (r.error or "") for r in run.results)


def test_hard_exit_marks_running_test_and_skips_rest(two_sum_tests: list[TestCase],
                                                     subprocess_sandbox: SubprocessSandbox,
                                                     sandbox_cfg: SandboxConfig) -> None:
    run = _run(_sample("bad_exit"), two_sum_tests, subprocess_sandbox, sandbox_cfg)
    assert run.results[0].status == TestStatus.ERROR
    assert "exited with code 3" in (run.results[0].error or "")
    assert run.results[0].measured
    assert all(r.status == TestStatus.NOT_RUN and not r.measured for r in run.results[1:])
    assert run.exit_code == 3


def test_api_keys_do_not_reach_solution(two_sum_tests: list[TestCase], subprocess_sandbox: SubprocessSandbox,
                                        sandbox_cfg: SandboxConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-fake")
    run = _run(_sample("bad_env_probe"), two_sum_tests, subprocess_sandbox, sandbox_cfg)
    assert run.all_passed, [(r.test_id, r.actual) for r in run.results]


def test_solution_output_cannot_forge_results(two_sum_tests: list[TestCase],
                                              subprocess_sandbox: SubprocessSandbox,
                                              sandbox_cfg: SandboxConfig) -> None:
    run = _run(_sample("bad_prints"), two_sum_tests, subprocess_sandbox, sandbox_cfg)
    assert run.all_passed
    assert run.results[0].actual == [0, 1]


# ---------------------------------------------------------------- load failures and wall clock


@pytest.mark.parametrize(
    ("code", "needle"),
    [
        ("def two_sum(:\n", "SyntaxError"),
        ("def other():\n    return 1\n", "does not define a callable 'two_sum'"),
        ("raise RuntimeError('boom at import')\n", "RuntimeError: boom at import"),
        ("while True:\n    pass\n", "loading the solution took longer"),
    ],
)
def test_load_failures_error_every_test(code: str, needle: str, subprocess_sandbox: SubprocessSandbox,
                                        sandbox_cfg: SandboxConfig) -> None:
    cfg = sandbox_cfg.model_copy(update={"test_timeout_s": 0.5})
    run = _run(code, [_case("A"), _case("B")], SubprocessSandbox(cfg), cfg)
    assert run.executed
    for r in run.results:
        assert r.status == TestStatus.ERROR
        assert "failed to load" in (r.error or "") and needle in (r.error or "")


def test_wall_clock_limit_marks_running_and_unrun_tests(sandbox_cfg: SandboxConfig) -> None:
    cfg = sandbox_cfg.model_copy(update={"test_timeout_s": 60.0, "wall_timeout_s": 1.0})
    code = "def two_sum(nums, target):\n    while nums:\n        pass\n    return []\n"
    tests = [_case("A", args=[[], 0], expected=[]), _case("B", args=[[1], 0]), _case("C", args=[[1], 0])]
    run = _run(code, tests, SubprocessSandbox(cfg), cfg)
    assert run.timed_out
    assert _status(run) == {"A": TestStatus.PASS, "B": TestStatus.TIMEOUT, "C": TestStatus.NOT_RUN}
    assert "wall-clock" in (run.results[1].error or "")


# ---------------------------------------------------------------- not executed


def test_non_python_is_not_executed(two_sum_tests: list[TestCase], subprocess_sandbox: SubprocessSandbox,
                                    sandbox_cfg: SandboxConfig) -> None:
    run = run_suite(code="int main(){}", entry_point="two_sum", language="C++", tests=two_sum_tests,
                    sandbox=subprocess_sandbox, cfg=sandbox_cfg, version="X")
    assert not run.executed and run.backend == "none"
    assert "C++" in (run.not_executed_reason or "")
    assert all(r.status == TestStatus.NOT_RUN and not r.measured and r.runtime_s is None for r in run.results)


def test_missing_sandbox_is_not_executed(two_sum_tests: list[TestCase], sandbox_cfg: SandboxConfig) -> None:
    run = run_suite(code=mock_scenarios.G1_CODE, entry_point="two_sum", language="python", tests=two_sum_tests,
                    sandbox=None, cfg=sandbox_cfg, version="G1", unavailable_reason="docker forced but unusable")
    assert not run.executed
    assert run.not_executed_reason == "docker forced but unusable"
    assert not run.all_passed


class _SilentSandbox:
    """A backend whose process dies before the runner says anything (e.g. bad image)."""

    name: Literal["docker"] = "docker"
    isolation_warning: str | None = None

    def rlimits(self) -> dict[str, int]:
        return {}

    def execute(self, files: Mapping[str, str], timeout_s: float) -> ExecResult:
        return ExecResult(exit_code=125, timed_out=False, stdout="", stderr_tail="docker: image not found\n",
                          wall_time_s=0.1, output_truncated=False)


def test_runner_that_never_started_is_not_executed(sandbox_cfg: SandboxConfig) -> None:
    run = run_suite(code=mock_scenarios.G1_CODE, entry_point="two_sum", language="python", tests=[_case()],
                    sandbox=_SilentSandbox(), cfg=sandbox_cfg, version="G1")
    assert not run.executed and run.backend == "docker"
    assert "image not found" in (run.not_executed_reason or "")
    assert "code 125" in (run.not_executed_reason or "")
    assert run.results[0].status == TestStatus.NOT_RUN


# ---------------------------------------------------------------- judging


IDENTITY = "def two_sum(*args, **kwargs):\n    return args[0] if args else kwargs\n"


@pytest.mark.parametrize(
    ("case", "status"),
    [
        (dict(args=[(0, 1)], expected=[0, 1]), TestStatus.PASS),
        (dict(args=[[3, 1, 2]], expected=[1, 2, 3], compare="unordered"), TestStatus.PASS),
        (dict(args=[[3, 1, 2]], expected=[1, 2, 3]), TestStatus.FAIL),
        (dict(args=[[0.1 + 0.2]], expected=[0.3], compare="float"), TestStatus.PASS),
        (dict(args=[[0.1 + 0.2]], expected=[0.3]), TestStatus.FAIL),
        (dict(args=[True], expected=1), TestStatus.FAIL),
        (dict(kwargs={"a": 1}, expected={"a": 1}), TestStatus.PASS),
        (dict(args=[[1]], expected=[1], expect_exception="ValueError"), TestStatus.FAIL),
    ],
)
def test_comparison_modes(case: dict[str, Any], status: TestStatus, subprocess_sandbox: SubprocessSandbox,
                          sandbox_cfg: SandboxConfig) -> None:
    run = _run(IDENTITY, [_case(**case)], subprocess_sandbox, sandbox_cfg)
    assert run.results[0].status == status, run.results[0]


def test_exception_expectations(subprocess_sandbox: SubprocessSandbox, sandbox_cfg: SandboxConfig) -> None:
    code = ("def two_sum(kind):\n"
            "    if kind == 'key':\n        raise KeyError('k')\n"
            "    if kind == 'exit':\n        raise SystemExit(4)\n"
            "    raise ValueError('bad value')\n")
    tests = [
        _case("base", args=["key"], expect_exception="LookupError"),
        _case("wrong", args=["value"], expect_exception="TypeError"),
        _case("unexpected", args=["value"], expected=1),
        _case("sysexit", args=["exit"], expect_exception="SystemExit"),
    ]
    run = _run(code, tests, subprocess_sandbox, sandbox_cfg)
    assert _status(run) == {"base": TestStatus.PASS, "wrong": TestStatus.FAIL,
                            "unexpected": TestStatus.ERROR, "sysexit": TestStatus.PASS}
    wrong = run.results[1]
    assert "expected TypeError, raised ValueError: bad value" == wrong.error


def test_non_json_return_is_flagged(subprocess_sandbox: SubprocessSandbox, sandbox_cfg: SandboxConfig) -> None:
    code = "class Pair:\n    pass\n\ndef two_sum():\n    return Pair()\n"
    run = _run(code, [_case(expected=[0, 1])], subprocess_sandbox, sandbox_cfg)
    r = run.results[0]
    assert r.status == TestStatus.FAIL and r.non_json
    assert "Pair object" in str(r.actual)


def test_large_output_uses_digest(subprocess_sandbox: SubprocessSandbox) -> None:
    cfg = SandboxConfig(backend="subprocess", max_output_bytes=10_000, test_timeout_s=2, test_repeats=1)
    code = "def two_sum(n):\n    return list(range(n))\n"
    tests = [
        _case("exact_ok", args=[3000], expected=list(range(3000))),
        _case("exact_bad", args=[3000], expected=list(range(1, 3001))),
        _case("unordered", args=[3000], expected=list(range(3000)), compare="unordered"),
    ]
    run = _run(code, tests, SubprocessSandbox(cfg), cfg)
    assert _status(run) == {"exact_ok": TestStatus.PASS, "exact_bad": TestStatus.FAIL,
                            "unordered": TestStatus.ERROR}
    assert all(r.truncated for r in run.results)
    assert "sha256" in str(run.results[0].actual)


def test_long_actual_is_truncated_for_storage(subprocess_sandbox: SubprocessSandbox) -> None:
    cfg = SandboxConfig(backend="subprocess", max_actual_chars=50, test_repeats=1)
    code = "def two_sum(n):\n    return list(range(n))\n"
    run = _run(code, [_case(args=[100], expected=list(range(100)))], SubprocessSandbox(cfg), cfg)
    r = run.results[0]
    assert r.status == TestStatus.PASS and r.truncated
    assert "truncated" in str(r.actual)


def test_generator_inputs_are_seeded_and_identical(subprocess_sandbox: SubprocessSandbox,
                                                   sandbox_cfg: SandboxConfig) -> None:
    code = "def two_sum(values):\n    return values\n"
    tests = [_case("G", generator="([[rng.randint(0, 10**9) for _ in range(5)]], {})", expected=None)]
    first = _run(code, tests, subprocess_sandbox, sandbox_cfg).results[0].actual
    second = _run(code, tests, subprocess_sandbox, sandbox_cfg).results[0].actual
    assert first == second and isinstance(first, list) and len(first) == 5
    other_seed = sandbox_cfg.model_copy(update={"seed": 1})
    assert _run(code, tests, SubprocessSandbox(other_seed), other_seed).results[0].actual != first


def test_broken_generator_is_reported_as_test_defect(subprocess_sandbox: SubprocessSandbox,
                                                     sandbox_cfg: SandboxConfig) -> None:
    tests = [_case("G1", generator="1/0"), _case("G2", generator="[1, 2, 3]")]
    run = _run(IDENTITY, tests, subprocess_sandbox, sandbox_cfg)
    for r in run.results:
        assert r.status == TestStatus.ERROR
        assert "test input generator failed" in (r.error or "")


def test_inputs_are_copied_between_runs(subprocess_sandbox: SubprocessSandbox, sandbox_cfg: SandboxConfig) -> None:
    code = "def two_sum(items):\n    items.append(len(items))\n    return items\n"
    run = _run(code, [_case(args=[[0]], expected=[0, 1])], subprocess_sandbox, sandbox_cfg)
    assert run.results[0].status == TestStatus.PASS


# ---------------------------------------------------------------- payload and files


def test_payload_never_contains_expected_values(two_sum_tests: list[TestCase], sandbox_cfg: SandboxConfig) -> None:
    payload = build_payload(two_sum_tests, "two_sum", sandbox_cfg, 3, True, {})
    for t in payload["tests"]:
        assert set(t) == {"id", "args", "kwargs", "generator"}
    assert "expect" not in json.dumps({k: v for k, v in payload.items() if k != "tests"})


def test_parse_protocol_skips_garbage() -> None:
    text = 'noise\n{"event": "start", "test_id": "A"}\n[1, 2]\n{"no_event": 1}\n{"event": "done"'
    assert parse_protocol(text) == [{"event": "start", "test_id": "A"}]


def test_results_round_trip(tmp_path: Path, two_sum_tests: list[TestCase], subprocess_sandbox: SubprocessSandbox,
                            sandbox_cfg: SandboxConfig) -> None:
    run = _run(mock_scenarios.G1_CODE, two_sum_tests[:2], subprocess_sandbox, sandbox_cfg, version="G1")
    path = write_test_results(tmp_path / "run" / "tests.json", [run])
    loaded = [SuiteRun.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]
    assert loaded == [run]


def test_load_test_cases_accepts_wrapped_form(tmp_path: Path) -> None:
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"tests": [{"id": "A", "category": "normal", "args": [1], "expected": 1}]}))
    assert [t.id for t in load_test_cases(p)] == ["A"]
