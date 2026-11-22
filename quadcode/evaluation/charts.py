"""Matplotlib charts of a bench summary (static PNG, light surface).

Colors follow the reference palette: one blue series for estimates with intervals, and blue / neutral
gray / orange for wins / ties / losses. Text stays in ink colors; every chart has a title naming its
measure and a note saying the values are measured on hidden tests.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from quadcode.evaluation.conditions import Condition  # noqa: E402
from quadcode.evaluation.summary import BenchSummary  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
NEUTRAL = "#b9b8b2"
CHARTS_DIR = "charts"
PNG_META = {"Software": None}  # no version string: identical bytes when regenerated


def _style(fig: Figure, ax: Axes, title: str, note: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    fig.text(0.01, 0.01, note, color=INK_2, fontsize=8, ha="left", va="bottom")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _labels(conditions: list[Condition]) -> list[str]:
    return [c.value for c in conditions]


def pass_rate_chart(summary: BenchSummary, path: Path) -> Path | None:
    rows = [(c, s.pass_rate) for c, s in summary.summaries.items() if s.pass_rate is not None]
    if not rows:
        return None
    rows.reverse()  # first condition at the top
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.6))
    _style(fig, ax, f"Hidden-test pass rate per condition (mean, {summary.state.confidence:.0%} bootstrap CI)",
           f"Measured on hidden tests; n = problems scored per condition. Source: hidden/*.json, "
           f"bench {summary.state.bench_id}.")
    for y, (c, iv) in enumerate(rows):
        assert iv is not None
        ax.plot([iv.low, iv.high], [y, y], color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot([iv.mean], [y], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(1.02, y, f"{iv.mean:.0%}  (n={iv.n})", va="center", ha="left", color=INK, fontsize=9,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)), _labels([c for c, _ in rows]))
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.subplots_adjust(left=0.22, right=0.83, top=0.86, bottom=0.18)
    fig.savefig(path, dpi=144, facecolor=SURFACE, metadata=PNG_META)
    plt.close(fig)
    return path


def win_tie_loss_chart(summary: BenchSummary, path: Path) -> Path | None:
    rows = [(c, w) for c, w in summary.comparisons.items() if w.n > 0]
    if not rows:
        return None
    rows.reverse()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.9))
    _style(fig, ax, "Full pipeline vs each condition, per problem (hidden-test pass rate)",
           "Win = the full pipeline passes more hidden tests on that problem. Only problems scored for both. "
           "Measured.")
    parts = (("wins", BLUE, "full better"), ("ties", NEUTRAL, "tie"), ("losses", ORANGE, "full worse"))
    for y, (_, w) in enumerate(rows):
        left = 0
        for attr, color, _ in parts:
            k = len(getattr(w, attr))
            if k:
                ax.barh(y, k, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=2)
                ax.text(left + k / 2, y, str(k), ha="center", va="center", fontsize=9,
                        color=SURFACE if color != NEUTRAL else INK)
            left += k
    ax.set_yticks(range(len(rows)), [f"vs {c.value}" for c, _ in rows])
    ax.set_xlim(0, max(w.n for _, w in rows))
    ax.set_xlabel("problems", color=INK_2, fontsize=9)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    handles = [matplotlib.patches.Patch(color=color, label=label) for _, color, label in parts]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK_2, handlelength=1.2)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.86, bottom=0.27)
    fig.savefig(path, dpi=144, facecolor=SURFACE, metadata=PNG_META)
    plt.close(fig)
    return path


def runtime_chart(summary: BenchSummary, path: Path) -> Path | None:
    rows = [(c, s.runtime_vs_full) for c, s in summary.summaries.items()
            if c is not Condition.FULL and s.runtime_vs_full is not None]
    if not rows:
        return None
    rows.reverse()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.6))
    _style(fig, ax, f"Hidden perf-test runtime vs the full pipeline (geometric mean, "
                    f"{summary.state.confidence:.0%} CI)",
           "Ratio > 1: slower than the full pipeline. Only problems where both pass every hidden performance "
           "test. Measured (median of timed runs).")
    lo = min(iv.low for _, iv in rows if iv is not None)
    hi = max(iv.high for _, iv in rows if iv is not None)
    for y, (_, iv) in enumerate(rows):
        assert iv is not None
        ax.plot([iv.low, iv.high], [y, y], color=BLUE, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot([iv.mean], [y], "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(1.02, y, f"{iv.mean:.2f}x  (n={iv.n})", va="center", ha="left", color=INK, fontsize=9,
                transform=ax.get_yaxis_transform())
    ax.axvline(1.0, color=INK_2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xscale("log")
    left, right = min(lo, 1.0) / 1.25, max(hi, 1.0) * 1.25
    ax.set_xlim(left, right)
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0, 100.0)
             if left <= t <= right]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(range(len(rows)), _labels([c for c, _ in rows]))
    fig.subplots_adjust(left=0.22, right=0.83, top=0.86, bottom=0.18)
    fig.savefig(path, dpi=144, facecolor=SURFACE, metadata=PNG_META)
    plt.close(fig)
    return path


def write_charts(summary: BenchSummary, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}
    for name, fn in (("pass_rate", pass_rate_chart), ("win_tie_loss", win_tie_loss_chart),
                     ("runtime_vs_full", runtime_chart)):
        target = out_dir / f"{name}.png"
        result = fn(summary, target)
        if result is not None:
            made[name] = result
        else:
            target.unlink(missing_ok=True)
    return made
