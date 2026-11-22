"""The benchmark set: every problem loads, and its hand-written expected values agree with an independent
hand-written reference solution executed in the sandbox."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quadcode.config import SandboxConfig
from quadcode.evaluation.problems import (
    HIDDEN_FILE,
    REFERENCE_FILE,
    ProblemError,
    load_problem,
    load_problems,
)
from quadcode.sandbox import SubprocessSandbox
from quadcode.schemas import TestCategory, TestStatus
from quadcode.testing import run_suite

ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "problems"
PROBLEMS = load_problems(ROOT)
CFG = SandboxConfig(backend="subprocess", test_timeout_s=5.0, wall_timeout_s=120.0, test_repeats=1, warmup=0)


def test_ten_problems_with_hidden_and_performance_tests() -> None:
    assert len(PROBLEMS) == 10
    for p in PROBLEMS:
        assert len(p.hidden_tests) >= 10, p.id
        assert all(t.hidden for t in p.hidden_tests)
    with_perf = [p.id for p in PROBLEMS if any(t.category is TestCategory.PERFORMANCE for t in p.hidden_tests)]
    assert len(with_perf) >= 8


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_reference_passes_every_hidden_test(problem) -> None:  # type: ignore[no-untyped-def]
    code = (problem.directory / REFERENCE_FILE).read_text(encoding="utf-8")
    suite = run_suite(code=code, entry_point=problem.entry_point, language="python",
                      tests=list(problem.hidden_tests), sandbox=SubprocessSandbox(CFG), cfg=CFG,
                      version="reference")
    assert suite.executed, suite.not_executed_reason
    bad = {r.test_id: (r.status.value, r.error, r.actual) for r in suite.results if r.status is not TestStatus.PASS}
    assert not bad


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_statement_reveals_no_hidden_test(problem) -> None:  # type: ignore[no-untyped-def]
    for t in problem.hidden_tests:
        if t.generator:
            assert t.generator not in problem.statement
        if t.args and len(json.dumps(t.args)) > 12:
            assert json.dumps(t.args) not in problem.statement


def test_mock_mode_keeps_only_problems_with_a_scenario() -> None:
    assert [p.id for p in load_problems(ROOT, mock=True)] == ["lis_length", "merge_intervals", "two_sum"]
    with pytest.raises(ProblemError, match="no mock scenario"):
        load_problems(ROOT, names=["edit_distance"], mock=True)


def test_selection_by_name_and_limit() -> None:
    assert [p.id for p in load_problems(ROOT, names=["two_sum", "lis_length"])] == ["two_sum", "lis_length"]
    assert len(load_problems(ROOT, limit=3)) == 3
    with pytest.raises(ProblemError, match="unknown problems"):
        load_problems(ROOT, names=["nope"])


def _copy(tmp: Path, name: str = "demo") -> Path:
    src = ROOT / "two_sum"
    dst = tmp / name
    dst.mkdir()
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    return dst


def test_entry_point_must_be_named_in_statement(tmp_path: Path) -> None:
    d = _copy(tmp_path)
    (d / "problem.md").write_text("Write a function.", encoding="utf-8")
    with pytest.raises(ProblemError, match="never names the entry point"):
        load_problem(d)


def test_hidden_flag_and_unique_ids_required(tmp_path: Path) -> None:
    d = _copy(tmp_path)
    tests = json.loads((d / HIDDEN_FILE).read_text(encoding="utf-8"))
    tests[0]["hidden"] = False
    (d / HIDDEN_FILE).write_text(json.dumps(tests), encoding="utf-8")
    with pytest.raises(ProblemError, match="not marked hidden"):
        load_problem(d)
    tests[0]["hidden"] = True
    tests[1]["id"] = tests[0]["id"]
    (d / HIDDEN_FILE).write_text(json.dumps(tests), encoding="utf-8")
    with pytest.raises(ProblemError, match="duplicate hidden test ids"):
        load_problem(d)
