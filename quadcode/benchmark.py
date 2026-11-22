"""Repeated benchmarking of several code versions under identical inputs.

Every version runs in its own sandbox process with the same payload, seed, limits
and repeat count. Statistics are computed only from timed runs that actually
happened; a test that timed out or never ran gets no runtime statistics.
"""

from __future__ import annotations

import statistics
from collections.abc import Collection, Mapping, Sequence

from quadcode.config import SandboxConfig
from quadcode.sandbox import Sandbox
from quadcode.schemas import BenchmarkReport, BenchmarkStats, SuiteRun, TestCase, TestCategory, TestStatus
from quadcode.testing import measurement_method, run_suite


def summarize(version: str, test_id: str, result_status: TestStatus, runtimes: Sequence[float],
              peak_memory_bytes: int | None) -> BenchmarkStats | None:
    """Summary of the timed runs, or None when there were none."""
    if not runtimes:
        return None
    return BenchmarkStats(
        version=version,
        test_id=test_id,
        status=result_status,
        repeats=len(runtimes),
        runtimes_s=list(runtimes),
        median_s=statistics.median(runtimes),
        min_s=min(runtimes),
        max_s=max(runtimes),
        mean_s=statistics.fmean(runtimes),
        stdev_s=statistics.stdev(runtimes) if len(runtimes) > 1 else None,
        peak_memory_bytes=peak_memory_bytes,
        measured=True,
    )


def stats_for_suite(suite: SuiteRun) -> list[BenchmarkStats]:
    out: list[BenchmarkStats] = []
    for r in suite.results:
        s = summarize(suite.version, r.test_id, r.status, r.runtimes_s, r.peak_memory_bytes)
        if s is not None:
            out.append(s)
    return out


def benchmark_versions(
    versions: Mapping[str, str],
    *,
    entry_point: str,
    language: str,
    tests: Sequence[TestCase],
    sandbox: Sandbox,
    cfg: SandboxConfig,
    categories: Collection[TestCategory] | None = None,
    repeats: int | None = None,
) -> BenchmarkReport:
    """Benchmark each version (label -> code) on the same tests, in the given order."""
    selected = [t for t in tests if categories is None or t.category in categories]
    if not selected:
        raise ValueError("no tests selected for benchmarking")
    reps = repeats if repeats is not None else cfg.benchmark_repeats
    suites: list[SuiteRun] = []
    stats: list[BenchmarkStats] = []
    for label, code in versions.items():
        suite = run_suite(code=code, entry_point=entry_point, language=language, tests=selected,
                          sandbox=sandbox, cfg=cfg, version=label, repeats=reps)
        suites.append(suite)
        stats.extend(stats_for_suite(suite))
    method = (measurement_method(sandbox.name, reps, cfg.warmup, cfg, measure_memory=True)
              + " Versions run one after another, each in a fresh sandbox process with an identical payload.")
    return BenchmarkReport(
        backend=sandbox.name,
        isolation_warning=sandbox.isolation_warning,
        method=method,
        repeats=reps,
        warmup=cfg.warmup,
        seed=cfg.seed,
        order=list(versions),
        suites=suites,
        stats=stats,
    )
