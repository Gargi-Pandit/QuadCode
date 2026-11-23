"""Start ``solve`` and ``bench`` runs from the dashboard through the regular command line.

The dashboard never calls a provider itself. It builds the same argv a user would type, runs it as
``python -m quadcode ...`` in a child process, and follows the run through the files the run
writes. The child loads ``.env`` exactly as the CLI does; the dashboard process never reads keys.
Real (non-mock) launches need an explicit confirmation and a call budget.
"""

from __future__ import annotations

import json
import os
import re
import secrets as pysecrets
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal

from quadcode.evaluation.problems import PROBLEM_FILE, ProblemError, available_ids, load_problem
from quadcode.evaluation.records import BENCH_FILE, RUNS_DIR
from quadcode.orchestrator import CONFIG_SNAPSHOT, SUMMARY_FILE
from quadcode.runlog import CALLS_FILE, EVENTS_FILE
from quadcode.schemas import Ablation, Phase

LOG_DIR = ".ui-logs"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_ATTEMPT_STATUSES = {"ok", "provider_error", "invalid_output"}


class LaunchError(ValueError):
    """A launch request that must not be started; the message says why."""


@dataclass(frozen=True)
class CommonOptions:
    config: Path = Path("config.yaml")
    env_file: Path = Path(".env")
    runs_dir: Path = Path("runs")
    backend: Literal["auto", "docker", "subprocess"] | None = None
    no_cache: bool = False


@dataclass(frozen=True)
class SolveRequest:
    problem: Path
    rounds: int
    mock: bool = True
    scenario: str | None = None
    budget_calls: int | None = None
    run_id: str = ""
    ablation: str = Ablation.NONE.value
    real_confirmed: bool = False


@dataclass(frozen=True)
class BenchRequest:
    problems: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=lambda: ["full"])
    rounds: int = 3
    mock: bool = True
    budget_calls: int | None = None
    bench_id: str = ""
    real_confirmed: bool = False


@dataclass(frozen=True)
class ResumeRequest:
    kind: Literal["solve", "bench"]
    id: str
    mock: bool
    scenario: str | None = None
    budget_calls: int | None = None
    real_confirmed: bool = False


@dataclass(frozen=True)
class Plan:
    """A validated launch: the argv to run and the directory the run will write."""

    kind: Literal["solve", "bench"]
    id: str
    argv: list[str]
    run_dir: Path

    @property
    def command(self) -> str:
        return shlex.join(["quadcode", *self.argv[3:]])  # drop "<python> -m quadcode"


def new_id(prefix: str, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or prefix
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{slug}-{pysecrets.token_hex(2)}"


def _check_id(value: str, runs_dir: Path, must_exist: bool) -> None:
    if not _ID.match(value):
        raise LaunchError(f"'{value}' is not a valid run id (letters, digits, '.', '_', '-'; no path separators)")
    exists = (runs_dir / value).exists()
    if must_exist and not exists:
        raise LaunchError(f"no run directory {runs_dir / value}")
    if not must_exist and exists:
        raise LaunchError(f"{runs_dir / value} already exists; pick another id or resume it")


def _check_spend(mock: bool, confirmed: bool, budget: int | None) -> None:
    if budget is not None and budget < 1:
        raise LaunchError("the call budget must be at least 1")
    if mock:
        return
    if not confirmed:
        raise LaunchError("real providers spend API quota: confirm it explicitly to start a real run")
    if budget is None:
        raise LaunchError("a real run needs a call budget (--budget-calls)")


def _check_rounds(rounds: int) -> None:
    if not 1 <= rounds <= 5:
        raise LaunchError("rounds must be between 1 and 5")


def _base(opts: CommonOptions) -> list[str]:
    return [sys.executable, "-m", "quadcode", "--config", str(opts.config), "--env-file", str(opts.env_file)]


def _tail(opts: CommonOptions, budget: int | None) -> list[str]:
    out = ["--runs-dir", str(opts.runs_dir)]
    if budget is not None:
        out += ["--budget-calls", str(budget)]
    if opts.backend:
        out += ["--backend", opts.backend]
    if opts.no_cache:
        out.append("--no-cache")
    return out


def plan_solve(req: SolveRequest, opts: CommonOptions) -> Plan:
    if not req.problem.is_file():
        raise LaunchError(f"no such problem file: {req.problem}")
    _check_rounds(req.rounds)
    _check_spend(req.mock, req.real_confirmed, req.budget_calls)
    if req.ablation not in {a.value for a in Ablation}:
        raise LaunchError(f"unknown ablation '{req.ablation}'")
    run_id = req.run_id.strip() or new_id("problem", req.problem.stem)
    _check_id(run_id, opts.runs_dir, must_exist=False)
    argv = [*_base(opts), "solve", str(req.problem), "--rounds", str(req.rounds), "--run-id", run_id]
    if req.mock:
        argv.append("--mock")
        if req.scenario:
            argv += ["--scenario", req.scenario]
    if req.ablation != Ablation.NONE.value:
        argv += ["--ablation", req.ablation]
    return Plan("solve", run_id, argv + _tail(opts, req.budget_calls), opts.runs_dir / run_id)


def plan_bench(req: BenchRequest, opts: CommonOptions) -> Plan:
    _check_rounds(req.rounds)
    _check_spend(req.mock, req.real_confirmed, req.budget_calls)
    if "full" not in req.conditions:
        raise LaunchError("conditions must include full (the baselines come from it)")
    if not req.problems:
        raise LaunchError("choose at least one problem")
    bench_id = req.bench_id.strip() or new_id("bench", "bench")
    _check_id(bench_id, opts.runs_dir, must_exist=False)
    argv = [*_base(opts), "bench", "--problems", ",".join(req.problems), "--conditions", ",".join(req.conditions),
            "--rounds", str(req.rounds), "--bench-id", bench_id]
    if req.mock:
        argv.append("--mock")
    return Plan("bench", bench_id, argv + _tail(opts, req.budget_calls), opts.runs_dir / bench_id)


def plan_resume(req: ResumeRequest, opts: CommonOptions) -> Plan:
    _check_id(req.id, opts.runs_dir, must_exist=True)
    _check_spend(req.mock, req.real_confirmed, req.budget_calls)
    argv = [*_base(opts), req.kind, "--resume", req.id]
    if req.mock:
        argv.append("--mock")
        if req.kind == "solve" and req.scenario:
            argv += ["--scenario", req.scenario]
    return Plan(req.kind, req.id, argv + _tail(opts, req.budget_calls), opts.runs_dir / req.id)


def problem_files(*roots: Path) -> list[Path]:
    """Problem statements found as ``<root>/<name>/problem.md``, for the problem picker."""
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found += sorted(p for p in root.glob(f"*/{PROBLEM_FILE}") if p.is_file())
    return found


def bench_problem_ids(root: Path, mock: bool) -> list[str]:
    """Benchmark problem ids; in mock mode only those with a scripted scenario can run."""
    try:
        ids = available_ids(root)
    except ProblemError:
        return []
    if not mock:
        return ids
    out: list[str] = []
    for pid in ids:
        try:
            if load_problem(root / pid).meta.mock_scenario:
                out.append(pid)
        except (ProblemError, ValueError, OSError):
            continue  # a broken problem directory is reported by `quadcode bench`, not hidden here
    return out


def mock_scenario_of(run_dir: Path) -> str | None:
    """The mock scenario a solve run was started with, from its config snapshot (for resuming it)."""
    try:
        config = json.loads((run_dir / CONFIG_SNAPSHOT).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    scenarios = {p.get("scenario") for p in config.get("providers", {}).values() if p.get("kind") == "mock"}
    scenarios.discard(None)
    return scenarios.pop() if len(scenarios) == 1 else None


# ---------------------------------------------------------------- processes


@dataclass
class Launched:
    plan: Plan
    process: subprocess.Popen[bytes]
    log_path: Path
    started_at: str

    @property
    def returncode(self) -> int | None:
        return self.process.poll()

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    def log_tail(self, lines: int = 40) -> str:
        if not self.log_path.is_file():
            return ""
        return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def start(plan: Plan, opts: CommonOptions, cwd: Path | None = None) -> Launched:
    """Run the plan's argv in a child process, its output going to ``<runs_dir>/.ui-logs/<id>.log``."""
    log_dir = opts.runs_dir / LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{plan.id}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log: IO[bytes]
    with open(log_path, "ab") as log:
        log.write(f"$ {plan.command}\n".encode())
        log.flush()
        # Own session: stopping the dashboard does not kill the run; it can be followed again after a restart.
        proc = subprocess.Popen(plan.argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=cwd, env=env, start_new_session=True)
    return Launched(plan, proc, log_path, datetime.now(UTC).isoformat(timespec="seconds"))


def stop(launched: Launched) -> None:
    """Interrupt the child like Ctrl-C. Completed steps are checkpointed, so ``--resume`` continues it."""
    if launched.running:
        launched.process.send_signal(signal.SIGINT)


# ---------------------------------------------------------------- progress


@dataclass(frozen=True)
class Progress:
    phases_completed: list[str]
    current_phase: str | None
    rounds_completed: int
    events: int
    provider_attempts: int
    cache_hits: int
    last_event: str | None
    last_event_ts: str | None
    finished_status: str | None  # from summary.json / bench.json once written
    pairs_done: int | None = None  # benches only: finished (problem, condition) runs
    pairs_total: int | None = None

    @property
    def fraction(self) -> float:
        """Share of the pipeline phases completed (benches: share of finished runs)."""
        if self.pairs_total:
            return (self.pairs_done or 0) / self.pairs_total
        return len(self.phases_completed) / len(Phase)


def _jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    out: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # the writer may be in the middle of a line
        if isinstance(row, dict):
            out.append(row)
    return out


def _calls(run_dirs: list[Path]) -> tuple[int, int]:
    attempts = hits = 0
    for d in run_dirs:
        for c in _jsonl(d / CALLS_FILE):
            if c.get("cache_hit"):
                hits += 1
            elif c.get("status") in _ATTEMPT_STATUSES:
                attempts += 1
    return attempts, hits


def solve_progress(run_dir: Path) -> Progress:
    events = _jsonl(run_dir / EVENTS_FILE)
    done = list(dict.fromkeys(str(e.get("phase")) for e in events if e.get("kind") == "phase_completed"))
    started = [str(e.get("phase")) for e in events if e.get("kind") == "phase_started"]
    current = next((p for p in reversed(started) if p not in done), None)
    attempts, hits = _calls([run_dir])
    status = None
    if (run_dir / SUMMARY_FILE).is_file():
        try:
            status = str(json.loads((run_dir / SUMMARY_FILE).read_text(encoding="utf-8")).get("status"))
        except (OSError, json.JSONDecodeError):
            status = None
    last = events[-1] if events else None
    return Progress(phases_completed=done, current_phase=current,
                    rounds_completed=sum(1 for e in events if e.get("kind") == "round_completed"),
                    events=len(events), provider_attempts=attempts, cache_hits=hits,
                    last_event=str(last.get("kind")) if last else None,
                    last_event_ts=str(last.get("ts")) if last else None, finished_status=status)


def bench_progress(bench_dir: Path) -> Progress:
    try:
        state = json.loads((bench_dir / BENCH_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {}
    pairs = list(state.get("pairs", {}).values())
    events = _jsonl(bench_dir / EVENTS_FILE)
    runs = [bench_dir / RUNS_DIR / str(p.get("run_id")) for p in pairs]
    attempts, hits = _calls(runs)
    status = state.get("status")
    last = events[-1] if events else None
    return Progress(phases_completed=[], current_phase=None, rounds_completed=0, events=len(events),
                    provider_attempts=attempts, cache_hits=hits,
                    last_event=str(last.get("kind")) if last else None,
                    last_event_ts=str(last.get("ts")) if last else None,
                    finished_status=None if status in (None, "running") else str(status),
                    pairs_done=sum(1 for p in pairs if p.get("status") == "finished"), pairs_total=len(pairs))


def progress(plan: Plan) -> Progress:
    return bench_progress(plan.run_dir) if plan.kind == "bench" else solve_progress(plan.run_dir)
