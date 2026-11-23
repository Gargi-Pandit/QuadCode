"""Dashboard view models of one bench, from its records (bench.json, hidden/*.json, sub-run calls).

The numbers are the ones ``quadcode bench-report`` writes: ``summarize`` computes them from the
stored hidden-test runs with the bootstrap settings saved in bench.json, so the dashboard and the
CSV files always agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quadcode.evaluation.charts import CHARTS_DIR
from quadcode.evaluation.conditions import DESCRIPTION
from quadcode.evaluation.records import RUNS_DIR, load_bench
from quadcode.evaluation.stats import Interval
from quadcode.evaluation.summary import BenchSummary, summarize

Row = dict[str, Any]

CHART_TITLES = {"pass_rate": "Hidden-test pass rate per condition (measured)",
                "win_tie_loss": "Full pipeline vs each condition, per problem (measured)",
                "runtime_vs_full": "Runtime relative to the full pipeline (measured)"}


def load_summary(bench_dir: Path) -> BenchSummary:
    return summarize(bench_dir, load_bench(bench_dir))


def _pct(iv: Interval | None) -> str:
    return "-" if iv is None else f"{iv.mean:.1%} [{iv.low:.1%}, {iv.high:.1%}]"


def condition_rows(s: BenchSummary) -> list[Row]:
    """Mean hidden pass rate and solved share with their bootstrap CI, per condition."""
    level = f"{s.state.confidence:.0%} CI"
    rows: list[Row] = []
    for c in s.conditions:
        x = s.summaries[c]
        ratio = "-" if x.runtime_vs_full is None else (
            f"{x.runtime_vs_full.mean:.3g}x [{x.runtime_vs_full.low:.3g}, {x.runtime_vs_full.high:.3g}]")
        rows.append({"condition": c.value, f"hidden pass rate, mean [{level}] (measured)": _pct(x.pass_rate),
                     f"solved [{level}] (measured)": _pct(x.solved),
                     "runtime vs full, geomean (measured)": ratio,
                     "problems scored": x.counted, "missing": x.missing,
                     "provider attempts": x.cost.provider_attempts, "cache hits": x.cost.cache_hits,
                     "tokens in": f"{x.cost.tokens_in}{' (estimated)' if x.cost.tokens_estimated else ''}",
                     "tokens out": f"{x.cost.tokens_out}{' (estimated)' if x.cost.tokens_estimated else ''}",
                     "not controlled": ", ".join(x.uncontrolled) or "-",
                     "what it is": DESCRIPTION[c]})
    return rows


def comparison_rows(s: BenchSummary) -> list[Row]:
    """Paired per-problem comparison of the full pipeline against every other condition."""
    rows: list[Row] = []
    for c, w in s.comparisons.items():
        rows.append({"full vs": c.value, "full wins": len(w.wins), "ties": len(w.ties), "full loses": len(w.losses),
                     "problems compared": w.n, "won on": ", ".join(w.wins) or "-",
                     "lost on": ", ".join(w.losses) or "-"})
    return rows


def result_rows(s: BenchSummary) -> list[Row]:
    """One row per problem and condition: what was scored and how it did on the hidden tests."""
    rows: list[Row] = []
    for pid in s.state.problems:
        for c in s.conditions:
            r = s.by_condition[c][pid]
            rows.append({"problem": pid, "condition": c.value, "status": r.status,
                         "run": r.run_id or "-", "run status": r.run_status.value if r.run_status else "-",
                         "version": r.version or "-", "picked": r.source_version or "-",
                         "hidden passed (measured)": f"{r.passed}/{r.total}" if r.counted else "-",
                         "solved": "-" if r.solved is None else ("yes" if r.solved else "no"),
                         "perf total s (measured)": "-" if r.perf_total_s is None else f"{r.perf_total_s:.6g}",
                         "reason": r.reason or "-", "notes": "; ".join(r.notes) or "-"})
    return rows


def answered_rows(s: BenchSummary) -> list[Row]:
    return [{"role": role, "provider/model": model, "ok calls": n} for role, model, n in s.answered]


def chart_paths(bench_dir: Path) -> dict[str, Path]:
    """The chart PNGs ``bench``/``bench-report`` wrote, if any (a chart with no data is not written)."""
    root = bench_dir / CHARTS_DIR
    return {name: root / f"{name}.png" for name in CHART_TITLES if (root / f"{name}.png").is_file()}


def sub_run_dir(bench_dir: Path, run_id: str) -> Path:
    return bench_dir / RUNS_DIR / run_id
