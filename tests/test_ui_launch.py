"""The dashboard launcher: commands it builds, what it refuses, progress, and one real mock launch."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from quadcode.ui import launch
from quadcode.ui.launch import BenchRequest, CommonOptions, LaunchError, ResumeRequest, SolveRequest

from .orch_helpers import TEST_CONFIG

ROOT = Path(__file__).resolve().parent.parent
PROBLEM = ROOT / "samples" / "two_sum" / "problem.md"
PROBLEMS_DIR = ROOT / "benchmarks" / "problems"


@pytest.fixture
def opts(tmp_path: Path) -> CommonOptions:
    return CommonOptions(config=tmp_path / "config.yaml", env_file=tmp_path / "none.env", runs_dir=tmp_path / "runs")


def test_mock_solve_command(opts: CommonOptions) -> None:
    plan = launch.plan_solve(SolveRequest(problem=PROBLEM, rounds=2, scenario="two_sum", run_id="r1"), opts)
    assert plan.argv[:3] == [sys.executable, "-m", "quadcode"]
    tail = plan.argv[3:]
    assert tail[:4] == ["--config", str(opts.config), "--env-file", str(opts.env_file)]
    assert tail[4:] == ["solve", str(PROBLEM), "--rounds", "2", "--run-id", "r1", "--mock", "--scenario", "two_sum",
                        "--runs-dir", str(opts.runs_dir)]
    assert plan.command.startswith("quadcode --config ") and plan.run_dir == opts.runs_dir / "r1"


def test_generated_ids_are_valid(opts: CommonOptions) -> None:
    plan = launch.plan_solve(SolveRequest(problem=PROBLEM, rounds=1), opts)
    assert re.fullmatch(r"\d{8}-\d{6}-problem-[0-9a-f]{4}", plan.id)  # slug from samples/two_sum/problem.md
    assert plan.argv[plan.argv.index("--run-id") + 1] == plan.id


def test_real_runs_need_confirmation_and_budget(opts: CommonOptions) -> None:
    with pytest.raises(LaunchError, match="confirm"):
        launch.plan_solve(SolveRequest(problem=PROBLEM, rounds=1, mock=False, budget_calls=10), opts)
    with pytest.raises(LaunchError, match="budget"):
        launch.plan_solve(SolveRequest(problem=PROBLEM, rounds=1, mock=False, real_confirmed=True), opts)
    with pytest.raises(LaunchError, match="confirm"):
        launch.plan_bench(BenchRequest(problems=["two_sum"], mock=False, budget_calls=5), opts)
    plan = launch.plan_solve(SolveRequest(problem=PROBLEM, rounds=1, mock=False, real_confirmed=True,
                                          budget_calls=12, run_id="real"), opts)
    assert "--mock" not in plan.argv and plan.argv[-2:] == ["--budget-calls", "12"]


@pytest.mark.parametrize("req, message", [
    (SolveRequest(problem=PROBLEM, rounds=0), "rounds"),
    (SolveRequest(problem=PROBLEM, rounds=6), "rounds"),
    (SolveRequest(problem=Path("nope.md"), rounds=1), "no such problem"),
    (SolveRequest(problem=PROBLEM, rounds=1, run_id="../escape"), "not a valid run id"),
    (SolveRequest(problem=PROBLEM, rounds=1, budget_calls=0), "at least 1"),
    (SolveRequest(problem=PROBLEM, rounds=1, ablation="no_tests"), "unknown ablation"),
])
def test_invalid_solve_requests(opts: CommonOptions, req: SolveRequest, message: str) -> None:
    with pytest.raises(LaunchError, match=message):
        launch.plan_solve(req, opts)


def test_existing_run_is_not_overwritten(opts: CommonOptions) -> None:
    (opts.runs_dir / "taken").mkdir(parents=True)
    with pytest.raises(LaunchError, match="already exists"):
        launch.plan_solve(SolveRequest(problem=PROBLEM, rounds=1, run_id="taken"), opts)


def test_bench_and_resume_commands(opts: CommonOptions) -> None:
    with pytest.raises(LaunchError, match="full"):
        launch.plan_bench(BenchRequest(problems=["two_sum"], conditions=["no-review"]), opts)
    with pytest.raises(LaunchError, match="at least one problem"):
        launch.plan_bench(BenchRequest(problems=[]), opts)
    plan = launch.plan_bench(BenchRequest(problems=["two_sum", "lis_length"], conditions=["full", "no-review"],
                                          rounds=1, bench_id="b1", budget_calls=30), opts)
    assert plan.argv[7:] == ["bench", "--problems", "two_sum,lis_length", "--conditions", "full,no-review",
                             "--rounds", "1", "--bench-id", "b1", "--mock", "--runs-dir", str(opts.runs_dir),
                             "--budget-calls", "30"]
    with pytest.raises(LaunchError, match="no run directory"):
        launch.plan_resume(ResumeRequest(kind="solve", id="gone", mock=True), opts)
    (opts.runs_dir / "r1").mkdir(parents=True)
    resume = launch.plan_resume(ResumeRequest(kind="solve", id="r1", mock=True, scenario="lis_length"), opts)
    assert resume.argv[7:] == ["solve", "--resume", "r1", "--mock", "--scenario", "lis_length",
                               "--runs-dir", str(opts.runs_dir)]
    with pytest.raises(LaunchError, match="confirm"):
        launch.plan_resume(ResumeRequest(kind="bench", id="r1", mock=False, budget_calls=3), opts)


def test_mock_scenario_of_reads_the_config_snapshot(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"providers": {"mock": {"kind": "mock",
                                                                             "scenario": "merge_intervals"}}}))
    assert launch.mock_scenario_of(tmp_path) == "merge_intervals"
    assert launch.mock_scenario_of(tmp_path / "missing") is None


def test_problem_pickers() -> None:
    assert PROBLEM in launch.problem_files(ROOT / "samples", ROOT / "nowhere")
    every = launch.bench_problem_ids(PROBLEMS_DIR, mock=False)
    mock = launch.bench_problem_ids(PROBLEMS_DIR, mock=True)
    assert len(every) == 10 and set(mock) < set(every) and {"merge_intervals", "lis_length"} <= set(mock)
    assert launch.bench_problem_ids(ROOT / "nowhere", mock=False) == []


def test_progress_tolerates_a_half_written_log(tmp_path: Path) -> None:
    events = [{"ts": "t1", "phase": "1-analysis", "kind": "phase_started"},
              {"ts": "t2", "phase": "1-analysis", "kind": "phase_completed"},
              {"ts": "t3", "phase": "2-initial", "kind": "phase_started"}]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events) + '{"ts": "t4", "ki')
    calls = [{"status": "ok", "cache_hit": False}, {"status": "provider_error", "cache_hit": False},
             {"status": "ok", "cache_hit": True}, {"status": "budget_exhausted", "cache_hit": False}]
    (tmp_path / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    p = launch.solve_progress(tmp_path)
    assert p.phases_completed == ["1-analysis"] and p.current_phase == "2-initial"
    assert (p.provider_attempts, p.cache_hits, p.events, p.last_event) == (2, 1, 3, "phase_started")
    assert p.finished_status is None and p.fraction == pytest.approx(1 / 7)


def test_bench_progress_counts_finished_runs(tmp_path: Path) -> None:
    pairs = {"a--full": {"run_id": "a--full", "status": "finished"}, "b--full": {"run_id": "b--full",
                                                                               "status": "pending"}}
    (tmp_path / "bench.json").write_text(json.dumps({"status": "running", "pairs": pairs}))
    (tmp_path / "runs" / "a--full").mkdir(parents=True)
    (tmp_path / "runs" / "a--full" / "calls.jsonl").write_text(json.dumps({"status": "ok", "cache_hit": False}))
    p = launch.bench_progress(tmp_path)
    assert (p.pairs_done, p.pairs_total, p.provider_attempts, p.finished_status) == (1, 2, 1, None)
    assert p.fraction == 0.5


def test_stop_interrupts_the_child(opts: CommonOptions) -> None:
    plan = launch.Plan("solve", "sleeper", [sys.executable, "-c", "import time; time.sleep(60)"],
                       opts.runs_dir / "sleeper")
    item = launch.start(plan, opts)
    assert item.running
    launch.stop(item)
    item.process.wait(timeout=10)
    assert item.returncode not in (None, 0)
    assert item.log_tail().startswith("$ ")


def test_mock_solve_launch_runs_the_cli_to_completion(tmp_path: Path) -> None:
    data = copy.deepcopy(TEST_CONFIG)
    data["run"] = {"rounds": 1, "cache_dir": str(tmp_path / "cache"), "runs_dir": str(tmp_path / "runs")}
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(data))
    opts = CommonOptions(config=cfg, env_file=tmp_path / "none.env", runs_dir=tmp_path / "runs")
    plan = launch.plan_solve(SolveRequest(problem=PROBLEM, rounds=1, run_id="ui-e2e"), opts)
    item = launch.start(plan, opts, cwd=ROOT)
    try:
        item.process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        item.process.kill()
        raise
    assert item.returncode == 0, item.log_tail()
    summary = json.loads((plan.run_dir / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["run_id"] == "ui-e2e"
    p = launch.progress(plan)
    assert p.finished_status == "completed" and p.fraction == 1.0
    log = item.log_path.read_text()
    assert log.startswith(f"$ {plan.command}") and "run ui-e2e: COMPLETED" in log
    assert (plan.run_dir / "calls.jsonl").is_file()  # the child process made the (mock) agent calls
