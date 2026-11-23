"""Dashboard page for one bench. Reads its records; the only write is "Rebuild bench outputs"."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from quadcode.config import AppConfig
from quadcode.evaluation.outputs import write_outputs
from quadcode.ui import bench_views
from quadcode.ui.catalog import bench_runs
from quadcode.ui.page_run import load_run_cached, show_run
from quadcode.ui.render import safe_inline, safe_text, show_rows


def show_bench(bench_dir: Path, pricing: AppConfig | None) -> None:
    try:
        s = bench_views.load_summary(bench_dir)
    except (OSError, ValueError) as exc:
        st.error(safe_text(f"Cannot read bench {bench_dir.name}: {type(exc).__name__}: {exc}"))
        return
    state = s.state
    st.title(safe_inline(f"Bench {state.bench_id}"), anchor=False)
    st.caption(safe_inline(f"{state.status.value.upper()}: {state.stop_reason or 'no stop reason recorded'}"))
    if state.mock:
        st.info("Mock bench: agent answers are scripted. Hidden tests and timings are real sandbox executions.")
    c = st.columns(3)
    c[0].metric("Problems", len(state.problems))
    c[1].metric("Pipeline runs per problem", len(state.run_conditions))
    c[2].metric("Rounds per run", state.rounds)
    st.markdown(safe_text(f"Bootstrap: {state.bootstrap_samples} resamples, {state.confidence:.0%} CI, seed "
                          f"{state.seed}. With few problems the intervals are wide."))
    for w in s.isolation_warnings:
        st.warning(safe_text(w))

    st.subheader("Hidden-test results per condition", anchor=False)
    show_rows(bench_views.condition_rows(s))
    st.subheader("Full pipeline vs each condition (paired, per problem)", anchor=False)
    show_rows(bench_views.comparison_rows(s), empty="No comparisons: the full pipeline has no scored problem.")
    charts = bench_views.chart_paths(bench_dir)
    for name, path in charts.items():
        st.image(str(path), caption=bench_views.CHART_TITLES[name])
    st.subheader("Every problem and condition", anchor=False)
    show_rows(bench_views.result_rows(s))
    st.subheader("Models that answered (from every run's calls.jsonl)", anchor=False)
    show_rows(bench_views.answered_rows(s), empty="No call answered.")
    if s.methods:
        st.caption(safe_inline("Measurement method: " + " | ".join(s.methods)))
    if st.button("Rebuild CSV, Markdown and charts (no API calls)", key=f"bench-{state.bench_id}-rebuild"):
        out = write_outputs(bench_dir)
        st.success(f"Wrote {len(out.paths())} files in {bench_dir}")

    subs = bench_runs(bench_dir)
    if subs:
        st.divider()
        pick = st.selectbox("Open a pipeline run of this bench", ["-", *[e.id for e in subs]],
                            key=f"bench-{state.bench_id}-sub")
        if pick != "-":
            show_run(load_run_cached(bench_views.sub_run_dir(bench_dir, pick)), pricing, key=f"sub-{pick}")
