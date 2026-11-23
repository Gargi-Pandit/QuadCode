"""Dashboard page that starts solve and bench runs through the regular command line and follows them.

Mock is preselected. A real run needs the quota checkbox and a call budget, and the exact command
is shown before anything starts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from quadcode.config import AppConfig
from quadcode.evaluation.conditions import RUN_ABLATION
from quadcode.providers.mock_scenarios import SCENARIOS
from quadcode.schemas import Ablation
from quadcode.ui import launch
from quadcode.ui.catalog import CatalogEntry
from quadcode.ui.render import safe_inline, safe_text

REAL_WARNING = ("Real providers spend API quota. The run uses the keys in .env through the same command line as "
                "`quadcode solve` / `quadcode bench`, stops after the call budget, and can be resumed.")


@dataclass(frozen=True)
class LaunchOptions:
    common: launch.CommonOptions
    problems_dir: str


@st.cache_resource
def _registry() -> dict[str, launch.Launched]:
    """Runs started from this dashboard process (kept across reruns and browser sessions)."""
    return {}


def _mode(key: str) -> tuple[bool, bool, int | None]:
    """(mock, real confirmed, budget) from the shared mode widgets."""
    real = st.radio("Agents", ["mock (no API calls)", "real providers"], horizontal=True, key=f"{key}-mode")
    mock = real.startswith("mock")
    budget_in = st.number_input("Call budget (provider attempts; 0 = no limit)", min_value=0, value=0 if mock else 40,
                                step=1, key=f"{key}-budget-{mock}")
    budget = int(budget_in) or None
    confirmed = False
    if not mock:
        st.warning(REAL_WARNING)
        confirmed = st.checkbox("I understand this run spends real API quota", key=f"{key}-confirm")
    return mock, confirmed, budget


def _start(plan: launch.Plan, common: launch.CommonOptions, id_key: str) -> None:
    """Button callback: start exactly the plan whose command was shown, then offer a fresh id."""
    _registry()[plan.id] = launch.start(plan, common)
    st.session_state.pop(id_key, None)
    st.toast(f"Started {plan.kind} {plan.id}")


def _submit(make_plan: Callable[[], launch.Plan], common: launch.CommonOptions, key: str,
            id_key: str | None = None) -> None:
    try:
        plan = make_plan()
    except launch.LaunchError as exc:
        st.error(safe_text(str(exc)))
        return
    st.code(plan.command, language="bash")
    st.button("Start", type="primary", key=f"{key}-start", on_click=_start, args=(plan, common, id_key or ""))


def _id_input(label: str, key: str, prefix: str) -> str:
    if key not in st.session_state:
        st.session_state[key] = launch.new_id(prefix, f"ui-{prefix}")
    return str(st.text_input(label, key=key))


def _solve_form(opts: LaunchOptions) -> None:
    files = launch.problem_files(Path("samples"), Path(opts.problems_dir))
    options = [str(p) for p in files] + ["other file..."]
    pick = st.selectbox("Problem statement", options, key="solve-problem")
    path = Path(st.text_input("Path to a problem file", key="solve-path")) if pick == "other file..." else Path(pick)
    c = st.columns(3)
    rounds = c[0].slider("Rounds", 1, 5, 3, key="solve-rounds")
    ablation = c[1].selectbox("Pipeline", [a.value for a in Ablation], key="solve-ablation")
    with c[2]:
        run_id = _id_input("Run id", "solve-id", "solve")
    mock, confirmed, budget = _mode("solve")
    scenario = None
    if mock:
        scenario = st.selectbox("Mock scenario (scripted answers; they ignore the problem text)", sorted(SCENARIOS),
                                index=sorted(SCENARIOS).index("two_sum"), key="solve-scenario")
    req = launch.SolveRequest(problem=path, rounds=rounds, mock=mock, scenario=scenario, budget_calls=budget,
                              run_id=run_id, ablation=ablation, real_confirmed=confirmed)
    _submit(lambda: launch.plan_solve(req, opts.common), opts.common, "solve", "solve-id")


def _bench_form(opts: LaunchOptions) -> None:
    mock, confirmed, budget = _mode("bench")
    ids = launch.bench_problem_ids(Path(opts.problems_dir), mock)
    problems = st.multiselect("Problems", ids, default=ids[:3], key=f"bench-problems-{mock}")
    conditions = st.multiselect("Pipeline runs per problem (single agents, oracle and select-only come from full)",
                                [c.value for c in RUN_ABLATION], default=["full"], key="bench-conditions")
    c = st.columns(2)
    rounds = c[0].slider("Rounds", 1, 5, 2, key="bench-rounds")
    with c[1]:
        bench_id = _id_input("Bench id", "bench-id", "bench")
    req = launch.BenchRequest(problems=problems, conditions=conditions, rounds=rounds, mock=mock,
                              budget_calls=budget, bench_id=bench_id, real_confirmed=confirmed)
    _submit(lambda: launch.plan_bench(req, opts.common), opts.common, "bench", "bench-id")


def _resume_form(opts: LaunchOptions, entries: list[CatalogEntry]) -> None:
    resumable = [e for e in entries if (e.kind == "solve" and e.status in ("unfinished", "aborted_budget"))
                 or (e.kind == "bench" and e.status in ("running", "aborted_budget"))]
    running = {k for k, v in _registry().items() if v.running}
    resumable = [e for e in resumable if e.id not in running]
    if not resumable:
        st.caption("No unfinished or budget-aborted run to resume.")
        return
    entry = st.selectbox("Run", resumable, format_func=lambda e: e.label, key="resume-pick")
    if entry.mock is None:
        st.error("Cannot tell whether this run used mock agents (no config snapshot), so it is not resumed here.")
        return
    mock = bool(entry.mock)
    st.caption("Mock run: resumed with mock agents." if mock else "Real run: resuming spends API quota.")
    confirmed = mock or st.checkbox("I understand this run spends real API quota", key="resume-confirm")
    budget_in = st.number_input("Call budget (0 = no limit)", min_value=0, value=0 if mock else 40, step=1,
                                key="resume-budget")
    req = launch.ResumeRequest(kind="bench" if entry.kind == "bench" else "solve", id=entry.id, mock=mock,
                               scenario=launch.mock_scenario_of(entry.path) if mock else None,
                               budget_calls=int(budget_in) or None, real_confirmed=bool(confirmed))
    _submit(lambda: launch.plan_resume(req, opts.common), opts.common, "resume")


def _follow(item: launch.Launched) -> None:
    @st.fragment(run_every=2.0)
    def body() -> None:
        p = launch.progress(item.plan)
        code = item.returncode
        state = "running" if code is None else f"exited with code {code}"
        st.markdown(f"**{safe_inline(item.plan.kind)} {safe_inline(item.plan.id)}**: {safe_inline(state)}"
                    + (f"; status {safe_inline(p.finished_status)}" if p.finished_status else ""))
        st.progress(min(p.fraction, 1.0))
        if p.pairs_total is not None:
            st.caption(f"{p.pairs_done} of {p.pairs_total} pipeline runs finished")
        else:
            st.caption(safe_inline(f"phases done: {len(p.phases_completed)}; current: {p.current_phase or '-'}; "
                                   f"rounds completed: {p.rounds_completed}"))
        st.caption(safe_inline(f"{p.events} events; {p.provider_attempts} provider attempts; {p.cache_hits} cache "
                               f"hits; last event: {p.last_event or '-'} at {p.last_event_ts or '-'}"))
        with st.expander("Command output"):
            st.code(item.log_tail() or "(no output yet)", language=None)
        if code is None and st.button("Stop (resumable)", key=f"stop-{item.plan.id}"):
            launch.stop(item)

    body()


def show_launcher(common: launch.CommonOptions, config: AppConfig | None, entries: list[CatalogEntry]) -> None:
    st.title("New run", anchor=False)
    st.caption("Runs start as `python -m quadcode ...` child processes: the same orchestrator and checks as the "
               "command line. Output goes to runs/.ui-logs/<id>.log.")
    opts = LaunchOptions(common, config.evaluation.problems_dir if config else "benchmarks/problems")
    kind = st.segmented_control("What to start", ["Solve a problem", "Benchmark", "Resume"], default="Solve a problem",
                                key="launch-kind")
    if kind == "Solve a problem":
        _solve_form(opts)
    elif kind == "Benchmark":
        _bench_form(opts)
    elif kind == "Resume":
        _resume_form(opts, entries)
    started = list(_registry().values())
    if started:
        st.divider()
        st.subheader("Started from this dashboard", anchor=False)
        for item in reversed(started):
            _follow(item)
