"""End-to-end mock bench, checked against its own records."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from quadcode.evaluation.conditions import Condition
from quadcode.evaluation.outputs import write_outputs
from quadcode.providers.mock_scenarios import get_scenario
from quadcode.evaluation.records import BenchStatus, PairStatus, load_bench, load_problem_results
from quadcode.runlog import RunLog
from quadcode.schemas import CallType, Role, TestStatus

from .bench_helpers import Recorder, make_harness

PROBLEMS = ["lis_length", "merge_intervals", "two_sum"]
SINGLES = [Condition.SINGLE_GENERATOR, Condition.SINGLE_REVIEWER, Condition.SINGLE_OPTIMIZER,
           Condition.SINGLE_VALIDATOR]


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Recorder]:
    root = tmp_path_factory.mktemp("bench")
    rec = Recorder()
    harness = make_harness(root, "b1", PROBLEMS, [Condition.FULL, Condition.NO_OPTIMIZER], recorder=rec)
    harness.run()
    write_outputs(harness.bench_dir)
    return harness.bench_dir, rec


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_run_finished(bench: tuple[Path, Recorder]) -> None:
    bench_dir, _ = bench
    state = load_bench(bench_dir)
    assert state.status is BenchStatus.COMPLETED
    assert {p.status for p in state.pairs.values()} == {PairStatus.FINISHED}
    assert sorted(state.pairs) == sorted(f"{p}--{c}" for p in PROBLEMS for c in ("full", "no-optimizer"))


def _squash(text: str) -> str:
    return "".join(text.split())


def test_hidden_tests_never_reach_an_agent(bench: tuple[Path, Recorder]) -> None:
    """No hidden test input reaches a prompt or a run file unless the scripted agent wrote it itself."""
    bench_dir, rec = bench
    prompts = [_squash(p) for p in rec.prompts()]
    assert prompts
    for pid in PROBLEMS:
        scripted = _squash(" ".join(get_scenario(pid).responses.values()))
        tests = json.loads((Path(__file__).resolve().parent.parent / "benchmarks" / "problems" / pid /
                            "hidden_tests.json").read_text(encoding="utf-8"))
        files = [_squash(f.read_text(encoding="utf-8")) for run in (bench_dir / "runs").glob(f"{pid}--*")
                 for f in run.rglob("*") if f.is_file() and f.suffix in {".json", ".jsonl", ".md", ".py"}]
        checked = 0
        for t in tests:
            needle = _squash(t["generator"] if t.get("generator") else json.dumps(t["args"]))
            if len(needle) < 12 or needle in scripted:
                continue
            checked += 1
            assert not any(needle in p for p in prompts), (pid, t["id"])
            assert not any(needle in f for f in files), (pid, t["id"])
        assert checked >= 3, pid


def test_merge_intervals_full_beats_every_single_agent(bench: tuple[Path, Recorder]) -> None:
    bench_dir, _ = bench
    res = load_problem_results(bench_dir, "merge_intervals")
    assert res is not None
    full = res.record(Condition.FULL)
    assert full is not None and full.solved and full.version == "final_solution.py"
    for c in SINGLES:
        single = res.record(c)
        assert single is not None and single.status == "scored" and single.passed < full.passed


def test_lis_full_ties_the_reviewer_and_select_only_is_c1(bench: tuple[Path, Recorder]) -> None:
    bench_dir, _ = bench
    res = load_problem_results(bench_dir, "lis_length")
    assert res is not None
    full, reviewer, select = (res.record(c) for c in (Condition.FULL, Condition.SINGLE_REVIEWER,
                                                        Condition.SELECT_ONLY))
    assert full is not None and reviewer is not None and select is not None
    assert full.passed == reviewer.passed == full.total
    assert select.source_version == "C1" and select.code_sha256 == reviewer.code_sha256


def test_oracle_is_the_best_single_by_hidden_tests(bench: tuple[Path, Recorder]) -> None:
    bench_dir, _ = bench
    for pid in PROBLEMS:
        res = load_problem_results(bench_dir, pid)
        assert res is not None
        singles = [res.record(c) for c in SINGLES]
        best = max(singles, key=lambda r: r.passed if r else -1)
        oracle = res.record(Condition.ORACLE)
        assert oracle is not None and best is not None
        assert oracle.passed == best.passed and oracle.source_version == best.version


def test_scores_trace_to_hidden_suite_runs(bench: tuple[Path, Recorder]) -> None:
    bench_dir, _ = bench
    for pid in PROBLEMS:
        res = load_problem_results(bench_dir, pid)
        assert res is not None
        for r in res.records:
            assert r.status == "scored"
            suite = res.suites[r.code_sha256 or ""]
            assert sum(1 for t in suite.results if t.status is TestStatus.PASS) == r.passed
            assert len(suite.results) == r.total
            for tid, median in r.perf_median_s.items():
                assert next(t.runtime_s for t in suite.results if t.test_id == tid) == median


def test_costs_trace_to_calls_jsonl(bench: tuple[Path, Recorder]) -> None:
    bench_dir, _ = bench
    for pid in PROBLEMS:
        res = load_problem_results(bench_dir, pid)
        assert res is not None
        calls = RunLog(bench_dir / "runs" / f"{pid}--full").read_calls()
        single = res.record(Condition.SINGLE_GENERATOR)
        assert single is not None
        expected = [c for c in calls if not c.cache_hit and (c.call_type is CallType.ANALYSIS or (
            c.call_type is CallType.INITIAL and c.role is Role.GENERATOR))]
        assert single.cost.provider_attempts == len(expected)
        full = res.record(Condition.FULL)
        assert full is not None and full.cost.provider_attempts == sum(1 for c in calls if not c.cache_hit)


def test_ablation_reused_the_full_runs_initial_answers(bench: tuple[Path, Recorder]) -> None:
    bench_dir, _ = bench
    for pid in PROBLEMS:
        res = load_problem_results(bench_dir, pid)
        assert res is not None
        abl = res.record(Condition.NO_OPTIMIZER)
        assert abl is not None and abl.controlled is True
        calls = RunLog(bench_dir / "runs" / f"{pid}--no-optimizer").read_calls()
        early = [c for c in calls if c.call_type in (CallType.ANALYSIS, CallType.INITIAL)]
        assert early and all(c.cache_hit for c in early)
        assert not [c for c in calls if c.role is Role.OPTIMIZER]


def test_csv_and_summary_agree_with_records(bench: tuple[Path, Recorder]) -> None:
    bench_dir, _ = bench
    rows = _rows(bench_dir / "results.csv")
    assert len(rows) == len(PROBLEMS) * 8  # 4 singles, oracle, select-only, full, no-optimizer
    for row in rows:
        res = load_problem_results(bench_dir, row["problem"])
        assert res is not None
        r = res.record(Condition(row["condition"]))
        assert r is not None and int(row["passed"]) == r.passed and int(row["total"]) == r.total
    summary = {r["condition"]: r for r in _rows(bench_dir / "summary.csv")}
    for cond, srow in summary.items():
        rates = [int(r["passed"]) / int(r["total"]) for r in rows if r["condition"] == cond]
        assert float(srow["pass_rate_mean"]) == pytest.approx(sum(rates) / len(rates), abs=1e-6)
        assert float(srow["pass_rate_low"]) <= float(srow["pass_rate_mean"]) <= float(srow["pass_rate_high"])
    by = {(r["problem"], r["condition"]): int(r["passed"]) for r in rows}
    for comp in _rows(bench_dir / "comparisons.csv"):
        other = comp["full_vs"]
        wins = [p for p in PROBLEMS if by[(p, "full")] > by[(p, other)]]
        losses = [p for p in PROBLEMS if by[(p, "full")] < by[(p, other)]]
        assert comp["win_problems"].split() == wins and comp["loss_problems"].split() == losses
    md = (bench_dir / "summary.md").read_text(encoding="utf-8")
    assert "Mock mode" in md and "| full vs single-generator |" in md
    for name in ("pass_rate", "win_tie_loss"):
        assert (bench_dir / "charts" / f"{name}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_outputs_regenerate_identically(bench: tuple[Path, Recorder]) -> None:
    bench_dir, _ = bench
    first = write_outputs(bench_dir)
    before = {p: p.read_bytes() for p in first.paths()}
    other = {p: p.stat().st_mtime_ns for p in (bench_dir / "runs").rglob("*") if p.is_file()}
    second = write_outputs(bench_dir)
    assert {p: p.read_bytes() for p in second.paths()} == before
    assert {p: p.stat().st_mtime_ns for p in (bench_dir / "runs").rglob("*") if p.is_file()} == other


def test_budget_abort_then_resume(tmp_path: Path) -> None:
    first = make_harness(tmp_path, "b2", ["two_sum"], [Condition.FULL, Condition.NO_REVIEW], budget=5)
    state = first.run()
    write_outputs(first.bench_dir)
    assert state.status is BenchStatus.ABORTED_BUDGET and "--resume b2" in state.stop_reason
    assert state.pair("two_sum", Condition.FULL).status is PairStatus.ABORTED
    assert state.pair("two_sum", Condition.NO_REVIEW).status is PairStatus.PENDING
    assert first.budget.used == 5
    res = load_problem_results(first.bench_dir, "two_sum")
    assert res is not None and {r.status for r in res.records} == {"missing"}
    md = (first.bench_dir / "summary.md").read_text(encoding="utf-8")
    assert "ABORTED (call budget)" in md and "excluded from the statistics" in md

    second = make_harness(tmp_path, "b2", ["two_sum"], [], resume=True)
    state = second.run()
    assert state.status is BenchStatus.COMPLETED
    full_dir = second.bench_dir / "runs" / "two_sum--full"
    started = [json.loads(line) for line in (full_dir / "events.jsonl").read_text().splitlines()
               if '"run_started"' in line]
    assert [e["payload"]["resumed"] for e in started] == [False, True]
    ok = [(c.role, c.call_type, c.tag) for c in RunLog(full_dir).read_calls() if c.status.value == "ok"
          and not c.cache_hit]
    assert len(ok) == len(set(ok))  # no completed call was made twice
    res = load_problem_results(second.bench_dir, "two_sum")
    assert res is not None and {r.status for r in res.records} == {"scored"}
