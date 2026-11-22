"""Run benchmark problems under each pipeline condition, then score everything on hidden tests.

One call budget is shared by every run of an invocation. When it runs out the current run is stopped
(the orchestrator saves it for ``--resume``), no further run starts, and the summary is still written
from whatever finished. Resuming a bench never repeats a finished run and resumes stopped ones.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quadcode.cli_solve import mock_config
from quadcode.config import AppConfig
from quadcode.evaluation.conditions import RUN_ABLATION, Condition
from quadcode.evaluation.hidden import HiddenScorer
from quadcode.evaluation.problems import BenchProblem
from quadcode.evaluation.records import (
    BENCH_EVENTS,
    RUNS_DIR,
    BenchState,
    BenchStatus,
    PairState,
    PairStatus,
    ProblemResults,
    hidden_path,
    load_problem_results,
    save_bench,
    write_atomic,
)
from quadcode.gateway import CallBudget, LLMGateway
from quadcode.orchestrator import Orchestrator
from quadcode.report import write_report
from quadcode.runstate import ResumeError, config_fingerprint
from quadcode.sandbox import Sandbox
from quadcode.schemas import RunStatus, utc_now

GatewayFactory = Callable[[AppConfig, Path, CallBudget], LLMGateway]


class BenchError(RuntimeError):
    """The bench cannot start or resume as asked."""


def problem_config(base: AppConfig, problem: BenchProblem, mock: bool) -> AppConfig:
    """Config for one problem's runs: unchanged for real providers, the problem's scenario in mock mode."""
    if not mock:
        return base
    if problem.meta.mock_scenario is None:
        raise BenchError(f"problem {problem.id} has no mock scenario")
    return mock_config(base, problem.meta.mock_scenario)


def new_state(bench_id: str, problems: list[BenchProblem], conditions: list[Condition], rounds: int, mock: bool,
              base: AppConfig) -> BenchState:
    state = BenchState(bench_id=bench_id, mock=mock, rounds=rounds, problems=[p.id for p in problems],
                       problem_sha256={p.id: p.statement_sha256 for p in problems},
                       hidden_sha256={p.id: p.hidden_sha256 for p in problems},
                       hidden_tests={p.id: len(p.hidden_tests) for p in problems},
                       run_conditions=conditions, config_sha256=config_fingerprint(base),
                       bootstrap_samples=base.evaluation.bootstrap_samples, confidence=base.evaluation.confidence,
                       seed=base.evaluation.seed)
    for p in problems:
        for cond in conditions:
            key = BenchState.key(p.id, cond)
            state.pairs = {**state.pairs, key: PairState(problem=p.id, condition=cond, run_id=key)}
    return state


def check_resumable(state: BenchState, problems: list[BenchProblem], base: AppConfig) -> None:
    for p in problems:
        if state.problem_sha256.get(p.id) != p.statement_sha256:
            raise BenchError(f"the statement of {p.id} changed since this bench started")
    if state.config_sha256 != config_fingerprint(base):
        raise BenchError("config.yaml (providers, agents, token caps, sandbox, selection or prompts) changed since "
                         "this bench started")


@dataclass
class BenchHarness:
    base_config: AppConfig
    bench_dir: Path
    state: BenchState
    problems: list[BenchProblem]
    sandbox: Sandbox
    make_gateway: GatewayFactory
    budget: CallBudget
    say: Callable[[str], None] = print
    results: dict[str, ProblemResults] = field(default_factory=dict)

    @property
    def runs_root(self) -> Path:
        return self.bench_dir / RUNS_DIR

    def event(self, kind: str, **payload: Any) -> None:
        line = json.dumps({"ts": utc_now(), "kind": kind, **payload}, ensure_ascii=False, default=str)
        with (self.bench_dir / BENCH_EVENTS).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def save(self) -> None:
        save_bench(self.bench_dir, self.state)

    # ------------------------------------------------------------ runs

    def _run_pair(self, problem: BenchProblem, pair: PairState) -> None:
        cfg = problem_config(self.base_config, problem, self.state.mock)
        run_dir = self.runs_root / pair.run_id
        resumed = (run_dir / "state.json").is_file()
        self.event("run_started", problem=problem.id, condition=pair.condition.value, run_id=pair.run_id,
                   resumed=resumed, budget_used=self.budget.used)
        gateway = self.make_gateway(cfg, run_dir, self.budget)
        try:
            if resumed:
                orch = Orchestrator.resume(cfg, gateway, self.sandbox, run_dir)
            else:
                orch = Orchestrator.create(cfg, gateway, self.sandbox, run_dir, pair.run_id, problem.statement,
                                           self.state.rounds, ablation=RUN_ABLATION[pair.condition])
            summary = orch.run()
        except (ResumeError, FileExistsError, OSError, ValueError) as exc:
            pair.status = PairStatus.ERROR
            pair.error = f"{type(exc).__name__}: {exc}"
            self.event("run_error", problem=problem.id, condition=pair.condition.value, error=pair.error)
            self.say(f"  {pair.run_id}: ERROR {pair.error}")
            return
        finally:
            gateway.close()
        try:
            write_report(run_dir, pricing=cfg)
        except Exception as exc:  # noqa: BLE001 - the run's records are complete; the report can be rebuilt
            self.event("report_error", run_id=pair.run_id, error=f"{type(exc).__name__}: {exc}")
        pair.run_status = summary.status
        pair.error = None
        pair.status = PairStatus.ABORTED if summary.status is RunStatus.ABORTED_BUDGET else PairStatus.FINISHED
        self.event("run_finished", problem=problem.id, condition=pair.condition.value, run_id=pair.run_id,
                   run_status=summary.status.value, budget_used=self.budget.used)
        self.say(f"  {pair.run_id}: {summary.status.value}")

    def run(self) -> BenchState:
        state = self.state
        state.status = BenchStatus.RUNNING
        state.stop_reason = ""
        self.save()
        self.event("bench_started", bench_id=state.bench_id, problems=state.problems,
                   conditions=[c.value for c in state.run_conditions], mock=state.mock,
                   budget_calls=self.budget.limit)
        stopped = False
        for problem in self.problems:
            if stopped:
                break
            self.say(f"{problem.id}:")
            for cond in state.run_conditions:
                pair = state.pair(problem.id, cond)
                if pair.status is PairStatus.FINISHED:
                    continue
                self._run_pair(problem, pair)
                self.save()
                if pair.status is PairStatus.ABORTED:
                    stopped = True
                    state.status = BenchStatus.ABORTED_BUDGET
                    state.stop_reason = (f"call budget of {self.budget.limit} exhausted during {pair.run_id}; "
                                         f"resume with --resume {state.bench_id}")
                    break
        if not stopped:
            unfinished = [p.run_id for p in state.pairs.values() if p.status is not PairStatus.FINISHED]
            state.status = BenchStatus.COMPLETED
            state.stop_reason = ("every run finished" if not unfinished else
                                 f"finished with {len(unfinished)} run(s) that could not start: {', '.join(unfinished)}")
        self.save()
        self.score()
        self.event("bench_finished", status=state.status.value, reason=state.stop_reason,
                   budget_used=self.budget.used)
        return state

    # ------------------------------------------------------------ hidden tests

    def score(self) -> dict[str, ProblemResults]:
        """Score every problem on its hidden tests; already scored conditions are kept as they are."""
        for problem in self.problems:
            previous = load_problem_results(self.bench_dir, problem.id)
            scorer = HiddenScorer(problem, self.sandbox, self.base_config, previous)
            res = scorer.score(self.state, self.runs_root)
            write_atomic(hidden_path(self.bench_dir, problem.id), res.model_dump_json(indent=2) + "\n")
            self.results[problem.id] = res
            self.event("problem_scored", problem=problem.id,
                       records={r.condition.value: [r.status, r.passed, r.total] for r in res.records})
        return self.results
