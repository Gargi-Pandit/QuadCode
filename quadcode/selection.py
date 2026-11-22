"""Evidence-based choice between code versions. No votes and no agent opinions enter here.

The selection order is: tests passed (correctness, requirements) > runtime > memory. Security,
readability and maintainability are not measured and are reported as such. Remaining ties are
broken by code size and then by slot order, and both are stated as tie-breaks, not as quality.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quadcode.config import SelectionConfig
from quadcode.schemas import CriterionStep, SelectionEvidence, SuiteRun, TestStatus
from quadcode.testing import is_python
from quadcode.versions import sort_key

NOT_MEASURED = ["security", "readability", "maintainability"]


def passed_count(suite: SuiteRun) -> int:
    return suite.count(TestStatus.PASS) if suite.executed else 0


def code_size(code: str) -> int:
    """Non-blank lines that are not pure comment lines."""
    return sum(1 for line in code.splitlines() if line.strip() and not line.strip().startswith("#"))


def _median_runtimes(suite: SuiteRun, test_ids: Sequence[str]) -> dict[str, float]:
    return {r.test_id: r.runtime_s for r in suite.results
            if r.test_id in test_ids and r.status is TestStatus.PASS and r.measured and r.runtime_s is not None}


def _runtime_ranges(suite: SuiteRun, test_ids: Sequence[str]) -> dict[str, tuple[float, float]]:
    """(fastest, slowest) timed run per passed test; the spread of the measurement."""
    out: dict[str, tuple[float, float]] = {}
    for r in suite.results:
        if r.test_id in test_ids and r.status is TestStatus.PASS and r.measured and r.runtime_s is not None:
            runs = r.runtimes_s or [r.runtime_s]
            out[r.test_id] = (min(runs), max(runs))
    return out


def _slower_beyond_noise(value: float, value_min: float, best: float, best_max: float, rel: float) -> bool:
    """A runtime is worse only if its median exceeds the tolerance AND even its fastest run is
    slower than the other version's slowest run (the measured ranges do not overlap)."""
    return value > best * (1 + rel) and value_min > best_max


def _peak_memory(suite: SuiteRun, test_ids: Sequence[str]) -> dict[str, int]:
    return {r.test_id: r.peak_memory_bytes for r in suite.results
            if r.test_id in test_ids and r.status is TestStatus.PASS and r.measured
            and r.peak_memory_bytes is not None}


def _common(per_version: Mapping[str, Mapping[str, float | int]], perf_ids: Sequence[str]) -> list[str]:
    return [t for t in perf_ids if all(t in values for values in per_version.values())]


def _within(value: float, best: float, rel: float, floor: float = 0.0) -> bool:
    return value <= best * (1 + rel) or value - best <= floor


def _runtime_step(kept: list[str], suites: Mapping[str, SuiteRun], perf_ids: Sequence[str],
                  cfg: SelectionConfig) -> CriterionStep:
    runtimes = {v: _median_runtimes(suites[v], perf_ids) for v in kept}
    common = _common(runtimes, perf_ids)
    if not common:
        return CriterionStep(criterion="runtime", measured=False, kept=kept,
                             note="no performance test with a measured runtime passed by every remaining version")
    totals = {v: sum(runtimes[v][t] for t in common) for v in kept}
    ranges = {v: _runtime_ranges(suites[v], common) for v in kept}
    lo = {v: sum(ranges[v][t][0] for t in common) for v in kept}
    hi = {v: sum(ranges[v][t][1] for t in common) for v in kept}
    fastest = min(kept, key=lambda v: totals[v])
    keep = [v for v in kept
            if not _slower_beyond_noise(totals[v], lo[v], totals[fastest], hi[fastest], cfg.runtime_tolerance)]
    return CriterionStep(
        criterion="runtime", measured=True, values=dict(totals), kept=keep,
        dropped=[v for v in kept if v not in keep],
        note=f"sum of median runtimes (s) over {', '.join(common)}; a version is dropped only if it is more than "
             f"{cfg.runtime_tolerance:.0%} slower than the fastest and its fastest run is slower than the fastest "
             "version's slowest run; otherwise the difference is treated as measurement noise")


def _memory_step(kept: list[str], suites: Mapping[str, SuiteRun], perf_ids: Sequence[str],
                 cfg: SelectionConfig) -> CriterionStep:
    peaks = {v: _peak_memory(suites[v], perf_ids) for v in kept}
    common = _common(peaks, perf_ids)
    if not common:
        return CriterionStep(criterion="memory", measured=False, kept=kept,
                             note="no performance test with a measured peak memory passed by every remaining "
                                  "version")
    worst = {v: max(peaks[v][t] for t in common) for v in kept}
    best = min(worst.values())
    keep = [v for v in kept if _within(worst[v], best, cfg.memory_tolerance, cfg.memory_floor_bytes)]
    return CriterionStep(
        criterion="memory", measured=True, values=dict(worst), kept=keep,
        dropped=[v for v in kept if v not in keep],
        note=f"largest Python-heap peak (bytes) over {', '.join(common)}; within {cfg.memory_tolerance:.0%} or "
             f"{cfg.memory_floor_bytes} bytes of the smallest counts as a tie")


def rank_versions(suites: Mapping[str, SuiteRun], codes: Mapping[str, str], perf_ids: Sequence[str],
                  cfg: SelectionConfig, purpose: str) -> SelectionEvidence:
    """Pick one version by a filter cascade over measured evidence. Every step is recorded."""
    candidates = sorted(suites, key=sort_key)
    if not candidates:
        return SelectionEvidence(purpose=purpose, candidates=[], winner=None, not_measured=NOT_MEASURED,
                                 verified=False)
    steps: list[CriterionStep] = []

    passes = {v: passed_count(suites[v]) for v in candidates}
    most = max(passes.values())
    kept = [v for v in candidates if passes[v] == most]
    total = {v: len(suites[v].results) for v in candidates}
    steps.append(CriterionStep(
        criterion="tests_passed", measured=True, values=dict(passes), kept=kept,
        dropped=[v for v in candidates if v not in kept],
        note=f"tests passed out of {max(total.values(), default=0)} (correctness and requirements as covered "
             "by the suite)"))

    for step_fn in (_runtime_step, _memory_step):
        if len(kept) > 1:
            step = step_fn(kept, suites, perf_ids, cfg)
            steps.append(step)
            kept = step.kept

    if len(kept) > 1:
        sizes = {v: code_size(codes[v]) for v in kept}
        smallest = min(sizes.values())
        keep = [v for v in kept if sizes[v] == smallest]
        steps.append(CriterionStep(criterion="code_size", measured=True, values=dict(sizes), kept=keep,
                                   dropped=[v for v in kept if v not in keep],
                                   note="tie-break: fewer non-blank, non-comment lines (a size heuristic, "
                                        "not a readability measurement)"))
        kept = keep

    if len(kept) > 1:
        steps.append(CriterionStep(criterion="slot_order", measured=False, kept=kept[:1], dropped=kept[1:],
                                   note="arbitrary deterministic tie-break by slot order; the evidence does not "
                                        "separate these versions"))
        kept = kept[:1]

    winner = kept[0]
    return SelectionEvidence(purpose=purpose, candidates=candidates, winner=winner, steps=steps,
                             not_measured=NOT_MEASURED, verified=suites[winner].all_passed)


@dataclass(frozen=True)
class BaseComparison:
    """A candidate measured against its base on the performance tests both passed."""

    runtime_base_s: float | None
    runtime_candidate_s: float | None
    memory_base_bytes: int | None
    memory_candidate_bytes: int | None
    runtime_regression: bool
    memory_regression: bool
    runtime_improvement: bool
    memory_improvement: bool
    tests: tuple[str, ...]

    @property
    def regression(self) -> bool:
        return self.runtime_regression or self.memory_regression

    def describe(self) -> str:
        if not self.tests:
            return "no performance measurement to compare"
        parts: list[str] = []
        if self.runtime_base_s is not None and self.runtime_candidate_s is not None:
            parts.append(f"median runtime {self.runtime_base_s:.6f}s -> {self.runtime_candidate_s:.6f}s"
                         + (" (regression)" if self.runtime_regression else
                            " (improvement)" if self.runtime_improvement else " (within noise)"))
        if self.memory_base_bytes is not None and self.memory_candidate_bytes is not None:
            parts.append(f"peak memory {self.memory_base_bytes} -> {self.memory_candidate_bytes} bytes"
                         + (" (regression)" if self.memory_regression else
                            " (improvement)" if self.memory_improvement else " (within noise)"))
        return f"measured on {', '.join(self.tests)}: " + "; ".join(parts)


def compare_to_base(base: SuiteRun, candidate: SuiteRun, perf_ids: Sequence[str],
                    cfg: SelectionConfig) -> BaseComparison:
    rt = {"b": _median_runtimes(base, perf_ids), "c": _median_runtimes(candidate, perf_ids)}
    mem = {"b": _peak_memory(base, perf_ids), "c": _peak_memory(candidate, perf_ids)}
    rt_common = _common(rt, perf_ids)
    mem_common = _common(mem, perf_ids)
    rb = sum(rt["b"][t] for t in rt_common) if rt_common else None
    rc = sum(rt["c"][t] for t in rt_common) if rt_common else None
    mb = max(mem["b"][t] for t in mem_common) if mem_common else None
    mc = max(mem["c"][t] for t in mem_common) if mem_common else None
    rt_reg = rt_imp = False
    if rb is not None and rc is not None:
        rng = {"b": _runtime_ranges(base, rt_common), "c": _runtime_ranges(candidate, rt_common)}
        lo = {k: sum(rng[k][t][0] for t in rt_common) for k in rng}
        hi = {k: sum(rng[k][t][1] for t in rt_common) for k in rng}
        rt_reg = _slower_beyond_noise(rc, lo["c"], rb, hi["b"], cfg.runtime_tolerance)
        rt_imp = _slower_beyond_noise(rb, lo["b"], rc, hi["c"], cfg.runtime_tolerance)
    mem_reg = (mb is not None and mc is not None
               and not _within(mc, mb, cfg.memory_tolerance, cfg.memory_floor_bytes))
    mem_imp = (mb is not None and mc is not None
               and not _within(mb, mc, cfg.memory_tolerance, cfg.memory_floor_bytes))
    tests = tuple(dict.fromkeys([*rt_common, *mem_common]))
    return BaseComparison(rb, rc, mb, mc, rt_reg, mem_reg, rt_imp, mem_imp, tests)


def is_noop(base_code: str, new_code: str, language: str) -> bool:
    """True when the change cannot alter behaviour: identical Python AST (only comments or
    formatting changed), or identical text up to whitespace for other languages."""
    if is_python(language):
        try:
            return ast.dump(ast.parse(base_code)) == ast.dump(ast.parse(new_code))
        except SyntaxError:
            return False
    return base_code.split() == new_code.split()
