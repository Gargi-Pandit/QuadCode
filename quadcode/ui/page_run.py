"""Dashboard page for one solve run. Reads the run's records; the only write is "Rebuild report"."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from quadcode.config import AppConfig
from quadcode.report import REPORT_HTML, REPORT_MD, build_document, write_report
from quadcode.report.data import RunRecords, load_run
from quadcode.report.tables import availability_blocks, final_label, token_blocks
from quadcode.ui import views
from quadcode.ui.render import safe_inline, safe_text, show_blocks, show_rows, split_sections

def _signature(run_dir: Path) -> tuple[tuple[str, int, int], ...]:
    """Size and modification time of every file in the run, so a changed run is reloaded."""
    return tuple(sorted((str(p.relative_to(run_dir)), s.st_size, s.st_mtime_ns)
                        for p in run_dir.rglob("*") if p.is_file() and (s := p.stat())))


@st.cache_resource(max_entries=16, show_spinner=False)
def _load(run_dir: str, signature: tuple[tuple[str, int, int], ...]) -> RunRecords:
    return load_run(Path(run_dir))


def load_run_cached(run_dir: Path) -> RunRecords:
    """``load_run``, reused across reruns until a file in the run changes. Read-only."""
    return _load(str(run_dir), _signature(run_dir))


TABS = ["Overview", "Report", "Communication timeline", "Calls & tokens", "Versions & diffs", "Modifications",
        "Tests & benchmark"]


def show_run(rec: RunRecords, pricing: AppConfig | None, key: str) -> None:
    """The whole run view. ``key`` keeps widget ids apart when several runs are shown in one session."""
    ov = views.overview(rec)
    st.title(safe_inline(f"Run {ov.run_id}"), anchor=False)
    if ov.problem_title:
        st.caption(safe_inline(ov.problem_title))
    if ov.mock:
        st.info("Mock run: agent answers are scripted and do not depend on the problem text. Sandbox results and "
                "timings are real measurements.")
    tabs = st.tabs(TABS)
    with tabs[0]:
        _overview(rec, ov, pricing, key)
    with tabs[1]:
        _report(rec, pricing, key)
    with tabs[2]:
        _timeline(rec, key)
    with tabs[3]:
        _calls(rec, pricing)
    with tabs[4]:
        _versions(rec, key)
    with tabs[5]:
        _modifications(rec, key)
    with tabs[6]:
        _tests(rec, key)


def _overview(rec: RunRecords, ov: views.Overview, pricing: AppConfig | None, key: str) -> None:
    c = st.columns(4)
    c[0].metric("Status", ov.status.upper())
    c[1].metric("Rounds run", ov.rounds)
    c[2].metric("Reference", ov.reference or "-")
    verified = {True: "verified", False: "NOT verified", None: "-"}[ov.final_verified]
    c[3].metric("Final", f"{ov.final_version or '-'} ({verified})" if ov.final_version else "-")
    st.markdown(f"**Stop reason:** {safe_inline(ov.stop_reason or '-')}")
    st.markdown(f"**Mode:** {safe_inline(ov.ablation)}")
    if ov.accepted or ov.rejected:
        st.markdown(f"**Accepted:** {safe_inline(', '.join(ov.accepted) or 'none')} · "
                    f"**Rejected:** {safe_inline(', '.join(ov.rejected) or 'none')}")
    if ov.isolation_warning:
        st.warning(safe_text(ov.isolation_warning))
    for issue in ov.unresolved:
        st.warning(safe_text(f"Unresolved: {issue}"))
    for err in ov.load_errors:
        st.error(safe_text(f"Record problem: {err}"))
    st.subheader("Agent slots", anchor=False)
    show_rows(views.slot_rows(rec))
    st.subheader("How V0 was chosen", anchor=False)
    show_rows(views.selection_rows(rec), empty="No reference version was selected.")
    st.subheader("Agent availability", anchor=False)
    show_blocks(availability_blocks(rec))
    if rec.final_code is not None:
        st.subheader(f"Final solution ({rec.final_file})", anchor=False)
        st.code(rec.final_code, language="python" if (rec.final_file or "").endswith(".py") else None)
    if st.button("Rebuild report.md and report.html (no API calls)", key=f"{key}-rebuild"):
        paths = write_report(rec.run_dir, pricing=pricing)
        st.success(f"Wrote {', '.join(str(p) for p in (paths.markdown, paths.html) if p)}")


def _report(rec: RunRecords, pricing: AppConfig | None, key: str) -> None:
    parts = split_sections(build_document(rec, pricing))
    titles = [t for t, _ in parts]
    choice = st.selectbox("Section", ["All sections", *titles], key=f"{key}-section")
    for title, blocks in parts:
        if choice in ("All sections", title):
            show_blocks(blocks)
    cols = st.columns(2)
    for col, name, mime in ((cols[0], REPORT_MD, "text/markdown"), (cols[1], REPORT_HTML, "text/html")):
        path = rec.run_dir / name
        if path.is_file():
            col.download_button(f"Download {name}", path.read_bytes(), file_name=f"{rec.run_id}-{name}", mime=mime,
                                key=f"{key}-dl-{name}")


def _timeline(rec: RunRecords, key: str) -> None:
    phases, senders, kinds = views.timeline_filters(rec)
    c = st.columns(3)
    ph = c[0].multiselect("Phase", phases, key=f"{key}-tl-phase")
    who = c[1].multiselect("From", senders, key=f"{key}-tl-from")
    kd = c[2].multiselect("Event", kinds, key=f"{key}-tl-kind")
    rows = views.timeline_rows(rec, set(ph), set(who), set(kd))
    st.caption(f"{len(rows)} of {len(rec.events)} logged events (events.jsonl). Only messages that were really "
               "exchanged are shown.")
    show_rows(rows, empty="No events match.")
    if rec.events:
        lines = [r["record"] for r in rows] or [e.ref for e in rec.events]
        pick = st.selectbox("Show the full logged record", lines, key=f"{key}-tl-pick")
        payload = views.event_payload(rec, int(str(pick).split(":")[1]))
        if payload is not None:
            st.json(payload, expanded=2)


def _calls(rec: RunRecords, pricing: AppConfig | None) -> None:
    show_blocks(token_blocks(rec, pricing))
    st.subheader("Every call (calls.jsonl)", anchor=False)
    st.caption("Tokens are provider-reported unless 'tokens estimated' is true. Latency is measured wall time.")
    show_rows(views.call_rows(rec), empty="calls.jsonl holds no calls.")


def _versions(rec: RunRecords, key: str) -> None:
    show_rows(views.version_rows(rec), empty="No stored versions.")
    labels = list(rec.codes)
    if not labels:
        return
    st.subheader("Compare two versions", anchor=False)
    c = st.columns(2)
    final = final_label(rec)
    default_new = labels.index(final) if final in labels else len(labels) - 1
    old = c[0].selectbox("From", labels, index=0, key=f"{key}-v-old")
    new = c[1].selectbox("To", labels, index=default_new, key=f"{key}-v-new")
    diff = views.version_diff(rec, old, new)
    if diff:
        st.code(diff, language="diff")
    else:
        st.caption("The two versions are identical.")
    st.subheader("Line provenance", anchor=False)
    label = st.selectbox("Version", labels, index=default_new, key=f"{key}-v-prov")
    st.caption("A line is credited to a modification only when a recorded modification inserted it.")
    show_rows(views.provenance_rows(rec, label))


def _modifications(rec: RunRecords, key: str) -> None:
    rows = views.modification_rows(rec)
    st.caption("Accepted and rejected records alike. Decisions are made on test evidence and measurements, never "
               "by vote; challenge verdicts are shown but do not decide.")
    show_rows(rows, empty="No modification was proposed.")
    if not rec.modifications:
        return
    pick = st.selectbox("Record", [r.id for r in rec.modifications], key=f"{key}-mod-pick")
    r = rec.record(pick)
    if r is None:
        return
    st.markdown(f"**{safe_inline(r.id)}** by {safe_inline(r.proposed_by.value)} "
                f"({safe_inline(rec.agent_identity(r.proposed_by))}), round {r.round_index}, "
                f"against {safe_inline(r.base_version)}: **{safe_inline(r.decision.value)}**")
    st.markdown(f"**Decision reason:** {safe_text(r.decision_reason or '-')}")
    st.markdown(f"**Justification:** {safe_text(r.justification or '-')}")
    st.markdown(f"**Expected impact (claimed):** {safe_text(r.expected_impact or '-')}")
    st.markdown(f"**Disadvantages (claimed):** {safe_text(r.disadvantages or '-')}")
    st.markdown(f"**Location:** {safe_inline(r.original_location)} -> {safe_inline(r.new_location or '-')}")
    if r.apply_error:
        st.warning(safe_text(f"Diff did not apply: {r.apply_error}"))
    show_rows([{"reviewer": v.role.value, "verdict": v.verdict.value, "comment": v.comment} for v in r.reviewers],
              empty="No challenge verdicts (the proposal was not contested or failing).")
    st.code(r.diff, language="diff")


def _tests(rec: RunRecords, key: str) -> None:
    st.subheader("Initial solutions on the suite (Phase 2.5)", anchor=False)
    show_rows(views.initial_suite_rows(rec), empty="No initial version was tested.")
    st.subheader("Claimed complexity (theoretical, not measured)", anchor=False)
    show_rows(views.theoretical_rows(rec), empty="No initial solution was recorded.")
    st.subheader("Final benchmark (measured)", anchor=False)
    if rec.benchmark is None:
        st.caption("Not reached: there is no benchmark.json.")
        return
    st.caption(safe_inline(f"Method: {rec.benchmark.method}"))
    if rec.benchmark.isolation_warning:
        st.warning(safe_text(rec.benchmark.isolation_warning))
    tests = views.benchmark_tests(rec)
    if tests:
        test = st.selectbox("Test", tests, key=f"{key}-bench-test")
        chart = views.runtime_chart_rows(rec, test)
        if chart:
            st.bar_chart(chart, x="version", y="median s (measured)", x_label="version",
                         y_label="median runtime, s (measured)")
            st.caption("Median of the timed runs in benchmark.json; versions that did not pass this test are left "
                       "out. Min/max per version are in the table below.")
        else:
            st.caption("No version passed this test, so no runtime is shown for it.")
    show_rows(views.benchmark_rows(rec))
