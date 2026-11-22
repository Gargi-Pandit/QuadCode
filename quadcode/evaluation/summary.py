"""Aggregate scored records into per-condition statistics and paired comparisons against the full pipeline.

Everything is computed from bench.json, hidden/<problem>.json and the runs' calls.jsonl, so a summary can
be rebuilt at any time without running anything.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from quadcode.evaluation.conditions import RUN_ABLATION, Condition, all_conditions
from quadcode.evaluation.records import (
    RUNS_DIR,
    BenchState,
    CallCost,
    ProblemResults,
    ResultRecord,
    load_problem_results,
)
from quadcode.evaluation.stats import Interval, WinTieLoss, bootstrap_geomean, bootstrap_mean, paired
from quadcode.runlog import RunLog


@dataclass
class ConditionSummary:
    condition: Condition
    counted: int
    missing: int
    pass_rate: Interval | None
    solved: Interval | None
    runtime_vs_full: Interval | None
    cost: CallCost
    uncontrolled: list[str] = field(default_factory=list)


@dataclass
class BenchSummary:
    state: BenchState
    conditions: list[Condition]
    results: dict[str, ProblemResults]
    by_condition: dict[Condition, dict[str, ResultRecord]]
    summaries: dict[Condition, ConditionSummary]
    comparisons: dict[Condition, WinTieLoss]
    answered: list[tuple[str, str, int]]  # (role, provider/model, ok calls) from every run's calls.jsonl
    methods: list[str]
    isolation_warnings: list[str]

    def record(self, condition: Condition, problem: str) -> ResultRecord | None:
        return self.by_condition.get(condition, {}).get(problem)


def _sum_cost(records: list[ResultRecord]) -> CallCost:
    total = CallCost()
    for r in records:
        total.provider_attempts += r.cost.provider_attempts
        total.ok += r.cost.ok
        total.cache_hits += r.cost.cache_hits
        total.tokens_in += r.cost.tokens_in
        total.tokens_out += r.cost.tokens_out
        total.tokens_estimated = total.tokens_estimated or r.cost.tokens_estimated
    return total


def load_results(bench_dir: Path, state: BenchState) -> dict[str, ProblemResults]:
    out: dict[str, ProblemResults] = {}
    for pid in state.problems:
        res = load_problem_results(bench_dir, pid)
        if res is not None:
            out[pid] = res
    return out


def _answered(bench_dir: Path, state: BenchState) -> list[tuple[str, str, int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for pair in sorted(state.pairs.values(), key=lambda p: p.run_id):
        run_dir = bench_dir / RUNS_DIR / pair.run_id
        if not (run_dir / "calls.jsonl").is_file():
            continue
        for c in RunLog(run_dir).read_calls():
            if c.status.value == "ok":
                counts[(c.role.value, f"{c.provider}/{c.model}")] += 1
    return [(role, model, n) for (role, model), n in sorted(counts.items())]


def summarize(bench_dir: Path, state: BenchState, results: dict[str, ProblemResults] | None = None) -> BenchSummary:
    results = results if results is not None else load_results(bench_dir, state)
    conditions = all_conditions(state.run_conditions)
    by_condition: dict[Condition, dict[str, ResultRecord]] = {c: {} for c in conditions}
    for pid in state.problems:
        res = results.get(pid)
        for c in conditions:
            rec = res.record(c) if res is not None else None
            if rec is None:
                rec = ResultRecord(problem=pid, condition=c, status="missing", reason="not scored yet")
            by_condition[c][pid] = rec

    def seed(*parts: str) -> str:
        return ":".join([str(state.seed), *parts])

    full = {p: r for p, r in by_condition[Condition.FULL].items() if r.counted}
    summaries: dict[Condition, ConditionSummary] = {}
    comparisons: dict[Condition, WinTieLoss] = {}
    for c in conditions:
        recs = by_condition[c]
        counted = [r for r in recs.values() if r.counted]
        rates = [r.pass_rate for r in counted if r.pass_rate is not None]
        solved = [1.0 if r.solved else 0.0 for r in counted]
        ratios = []
        if c is not Condition.FULL:
            for pid, r in recs.items():
                f = full.get(pid)
                a, b = r.perf_total_s, f.perf_total_s if f else None
                if a is not None and b is not None and a > 0 and b > 0:
                    ratios.append(a / b)
        summaries[c] = ConditionSummary(
            condition=c, counted=len(counted), missing=len(recs) - len(counted),
            pass_rate=bootstrap_mean(rates, state.bootstrap_samples, state.confidence, seed("pass", c.value)),
            solved=bootstrap_mean(solved, state.bootstrap_samples, state.confidence, seed("solved", c.value)),
            runtime_vs_full=bootstrap_geomean(ratios, state.bootstrap_samples, state.confidence,
                                              seed("runtime", c.value)),
            cost=_sum_cost(counted),
            uncontrolled=sorted(p for p, r in recs.items() if r.controlled is False))
        if c is not Condition.FULL:
            comparisons[c] = paired({p: r.pass_rate for p, r in full.items() if r.pass_rate is not None},
                                    {p: r.pass_rate for p, r in recs.items() if r.pass_rate is not None})
    suites = [s for res in results.values() for s in res.suites.values()]
    methods = sorted({s.method for s in suites if s.method})
    warnings = sorted({s.isolation_warning for s in suites if s.isolation_warning})
    return BenchSummary(state=state, conditions=conditions, results=results, by_condition=by_condition,
                        summaries=summaries, comparisons=comparisons, answered=_answered(bench_dir, state),
                        methods=methods, isolation_warnings=warnings)


def is_run_condition(c: Condition) -> bool:
    return c in RUN_ABLATION
