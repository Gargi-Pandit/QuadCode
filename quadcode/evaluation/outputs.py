"""Write a bench's results: results.csv, summary.csv, comparisons.csv, summary.md and charts/*.png.

All files are derived from the bench's records and contain no generation timestamp, so rebuilding them
from the same records gives identical bytes.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

from quadcode.evaluation.charts import CHARTS_DIR, write_charts
from quadcode.evaluation.conditions import DESCRIPTION, Condition
from quadcode.evaluation.records import BenchStatus, ResultRecord, load_bench, write_atomic
from quadcode.evaluation.stats import Interval
from quadcode.evaluation.summary import BenchSummary, summarize

RESULTS_CSV = "results.csv"
SUMMARY_CSV = "summary.csv"
COMPARISONS_CSV = "comparisons.csv"
SUMMARY_MD = "summary.md"


@dataclass(frozen=True)
class BenchOutputs:
    results_csv: Path
    summary_csv: Path
    comparisons_csv: Path
    summary_md: Path
    charts: dict[str, Path]

    def paths(self) -> list[Path]:
        return [self.results_csv, self.summary_csv, self.comparisons_csv, self.summary_md, *self.charts.values()]


def _f(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _csv(header: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def results_csv(s: BenchSummary) -> str:
    header = ["problem", "condition", "status", "reason", "run_id", "run_status", "version", "source_version",
              "code_sha256", "executed", "passed", "total", "pass_rate", "solved", "perf_total_s",
              "perf_median_s", "controlled", "provider_attempts", "cache_hits", "tokens_in", "tokens_out",
              "tokens_estimated", "notes"]
    rows: list[list[str]] = []
    for pid in s.state.problems:
        for c in s.conditions:
            r = s.by_condition[c][pid]
            rows.append([pid, c.value, r.status, r.reason, r.run_id or "", r.run_status.value if r.run_status else "",
                         r.version or "", r.source_version or "", r.code_sha256 or "", str(r.executed).lower(),
                         str(r.passed) if r.counted else "", str(r.total) if r.counted else "",
                         _f(r.pass_rate), "" if r.solved is None else str(r.solved).lower(),
                         _f(r.perf_total_s, 9), json.dumps(r.perf_median_s, sort_keys=True) if r.perf_median_s else "",
                         "" if r.controlled is None else str(r.controlled).lower(),
                         str(r.cost.provider_attempts), str(r.cost.cache_hits), str(r.cost.tokens_in),
                         str(r.cost.tokens_out), str(r.cost.tokens_estimated).lower(), " | ".join(r.notes)])
    return _csv(header, rows)


def _iv(iv: Interval | None, digits: int = 6) -> list[str]:
    return ["", "", "", "0"] if iv is None else [_f(iv.mean, digits), _f(iv.low, digits), _f(iv.high, digits),
                                                str(iv.n)]


def summary_csv(s: BenchSummary) -> str:
    header = ["condition", "problems_scored", "problems_missing", "pass_rate_mean", "pass_rate_low",
              "pass_rate_high", "pass_rate_n", "solved_mean", "solved_low", "solved_high", "solved_n",
              "runtime_ratio_vs_full", "runtime_ratio_low", "runtime_ratio_high", "runtime_ratio_n",
              "provider_attempts", "cache_hits", "tokens_in", "tokens_out", "uncontrolled_problems", "description"]
    rows = []
    for c in s.conditions:
        x = s.summaries[c]
        rows.append([c.value, str(x.counted), str(x.missing), *_iv(x.pass_rate), *_iv(x.solved),
                     *_iv(x.runtime_vs_full), str(x.cost.provider_attempts), str(x.cost.cache_hits),
                     str(x.cost.tokens_in), str(x.cost.tokens_out), " ".join(x.uncontrolled), DESCRIPTION[c]])
    return _csv(header, rows)


def comparisons_csv(s: BenchSummary) -> str:
    header = ["full_vs", "wins", "ties", "losses", "problems_compared", "win_problems", "tie_problems",
              "loss_problems"]
    rows = [[c.value, str(len(w.wins)), str(len(w.ties)), str(len(w.losses)), str(w.n), " ".join(w.wins),
             " ".join(w.ties), " ".join(w.losses)] for c, w in s.comparisons.items()]
    return _csv(header, rows)


# ---------------------------------------------------------------- markdown


def _cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(_cell(h) for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(iv: Interval | None) -> str:
    return "-" if iv is None else f"{iv.mean:.1%} [{iv.low:.1%}, {iv.high:.1%}]"


def _ratio(iv: Interval | None) -> str:
    return "-" if iv is None else f"{iv.mean:.2f}x [{iv.low:.2f}x, {iv.high:.2f}x] (n={iv.n})"


def _cellrec(r: ResultRecord) -> str:
    if r.status == "missing":
        return "missing"
    if r.status == "no_solution":
        return f"0/{r.total} (no solution)"
    mark = " ✓" if r.solved else ""
    src = f" ({r.source_version})" if r.source_version and r.condition in (Condition.ORACLE, Condition.SELECT_ONLY) \
        else ""
    return f"{r.passed}/{r.total}{mark}{src}"


def summary_md(s: BenchSummary, charts: dict[str, Path]) -> str:
    st = s.state
    conf = f"{st.confidence:.0%}"
    out: list[str] = [f"# QuadCode benchmark {st.bench_id}", ""]
    status = {BenchStatus.COMPLETED: "COMPLETED", BenchStatus.ABORTED_BUDGET: "ABORTED (call budget)",
              BenchStatus.RUNNING: "RUNNING (interrupted)"}[st.status]
    out += [f"**Status:** {status}. {st.stop_reason}", ""]
    if st.mock:
        out += ["> **Mock mode.** Agent answers are scripted per problem and do not come from a language model. "
                "Hidden tests and timings are real executions in the sandbox, but these numbers say nothing about "
                "real model quality. In mock mode the ablations exercise code paths only: scripted answers do not "
                "react to what an ablation removes.", ""]
    for w in s.isolation_warnings:
        out += [f"> **Sandbox:** {w}", ""]
    missing = sum(x.missing for x in s.summaries.values())
    if missing:
        out += [f"> {missing} problem/condition results are missing (runs not finished) and are excluded from "
                "the statistics; see results.csv.", ""]

    out += ["## Setup", "",
            f"- Problems ({len(st.problems)}): {', '.join(st.problems)}",
            f"- Optimization rounds per run: {st.rounds}",
            f"- Pipeline runs per problem: {', '.join(c.value for c in st.run_conditions)}",
            "- Hidden tests per problem: " + ", ".join(f"{pid} {st.hidden_tests.get(pid, '-')}"
                                                       for pid in st.problems),
            ""]
    out += ["Conditions:", ""]
    out += [f"- `{c.value}`: {DESCRIPTION[c]}" for c in s.conditions]
    out += [""]
    if s.answered:
        out += ["Models that answered (successful calls in the runs' calls.jsonl):", "",
                _table(["role", "provider/model", "ok calls"], [[r, m, str(n)] for r, m, n in s.answered]), ""]

    out += [f"## Hidden-test results ({conf} percentile bootstrap over problems, {st.bootstrap_samples} resamples, "
            f"seed {st.seed})", ""]
    rows = []
    for c in s.conditions:
        x = s.summaries[c]
        rows.append([c.value, _pct(x.pass_rate), _pct(x.solved), str(x.counted), str(x.missing),
                     str(x.cost.provider_attempts), str(x.cost.cache_hits),
                     f"{x.cost.tokens_in + x.cost.tokens_out}" + (" (estimated)" if x.cost.tokens_estimated else "")])
    out += [_table(["condition", "mean pass rate [CI]", "solved rate [CI]", "problems", "missing",
                    "provider attempts", "cache hits", "tokens"], rows), ""]
    out += ["Measured: every rate comes from executing the condition's code against the hidden tests. "
            "Calls and tokens of derived conditions (single-*, oracle, select-only) are the subset of the full "
            "run's calls they needed (the analysis call plus the initial calls), so they overlap with `full`.", ""]

    out += ["## Full pipeline vs each condition (paired, per problem)", ""]
    rows = [[f"full vs {c.value}", str(len(w.wins)), str(len(w.ties)), str(len(w.losses)), str(w.n)]
            for c, w in s.comparisons.items()]
    out += [_table(["comparison", "full wins", "ties", "full loses", "problems compared"], rows), ""]
    beaten = [c.value for c, w in s.comparisons.items() if w.n and len(w.losses) > len(w.wins)]
    not_better = [c.value for c, w in s.comparisons.items() if w.n and len(w.wins) <= len(w.losses)]
    if not_better:
        out += [f"The full pipeline does not win more problems than it loses against: {', '.join(not_better)}."
                + (f" It loses more often than it wins against: {', '.join(beaten)}." if beaten else ""), ""]

    out += ["## Runtime on hidden performance tests, relative to the full pipeline", ""]
    rows = [[c.value, _ratio(s.summaries[c].runtime_vs_full)] for c in s.conditions if c is not Condition.FULL]
    out += [_table(["condition", f"geometric mean ratio [{conf} CI]"], rows), "",
            "Measured. Ratio of summed median runtimes over a problem's hidden performance tests, only on problems "
            "where both versions pass every hidden performance test. Above 1 means slower than the full pipeline.",
            ""]

    out += ["## Per problem (hidden tests passed / total; ✓ = all passed)", ""]
    rows = [[pid, *[_cellrec(s.by_condition[c][pid]) for c in s.conditions]] for pid in st.problems]
    out += [_table(["problem", *[c.value for c in s.conditions]], rows), ""]

    notes = [(r.problem, r.condition.value, n) for c in s.conditions for r in s.by_condition[c].values()
             for n in ([r.reason] if r.reason else []) + r.notes]
    if notes:
        out += ["Notes from the records:", ""]
        out += [f"- {pid} / {cond}: {n}" for pid, cond, n in notes]
        out += [""]

    if charts:
        out += ["## Charts", ""]
        out += [f"![{name}]({CHARTS_DIR}/{p.name})" for name, p in charts.items()]
        out += [""]

    out += ["## Method and caveats", "",
            "- Hidden tests live only in the benchmark directory. They are read after a run has finished and are "
            "never part of any prompt; the agents only see the problem statement.",
            "- Single-agent baselines are each slot's blind initial solution from the full run. All four saw the "
            "same specification from the run's one analysis call, so they are not fully independent of the "
            "pipeline.",
            "- The oracle picks the initial solution with the most hidden tests passed. It uses the hidden tests, "
            "so it is an upper bound on choosing one initial solution, not something the pipeline could do.",
            "- Ablation runs reuse the full run's analysis and initial answers through the response cache (logged "
            "as cache hits). `controlled = false` in results.csv marks an ablation whose initial solutions differ "
            "from the full run's.",
            f"- {len(st.problems)} problems: confidence intervals are wide and small differences are not evidence "
            "of a real effect. Ties count as ties; nothing is rounded into a win.",
            "- Runs that did not finish are excluded, not counted as failures. A run that finished without a "
            "solution scores 0.",
            ]
    out += [f"- Measurement: {m}" for m in s.methods]
    out += ["", "Sources: bench.json, hidden/<problem>.json, runs/<run>/calls.jsonl; per-run reports in "
            "runs/<run>/report.md.", ""]
    return "\n".join(out)


def write_outputs(bench_dir: Path) -> BenchOutputs:
    """(Re)build every output file of a bench from its records. Runs nothing and calls no model."""
    state = load_bench(bench_dir)
    s = summarize(bench_dir, state)
    charts = write_charts(s, bench_dir / CHARTS_DIR)
    paths = {name: bench_dir / name for name in (RESULTS_CSV, SUMMARY_CSV, COMPARISONS_CSV, SUMMARY_MD)}
    write_atomic(paths[RESULTS_CSV], results_csv(s))
    write_atomic(paths[SUMMARY_CSV], summary_csv(s))
    write_atomic(paths[COMPARISONS_CSV], comparisons_csv(s))
    write_atomic(paths[SUMMARY_MD], summary_md(s, charts))
    return BenchOutputs(paths[RESULTS_CSV], paths[SUMMARY_CSV], paths[COMPARISONS_CSV], paths[SUMMARY_MD], charts)


