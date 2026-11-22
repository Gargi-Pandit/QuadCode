"""Report overview and sections 1-3: problem analysis, initial implementations, initial comparison."""

from __future__ import annotations

from quadcode.config import AppConfig
from quadcode.report.data import RunRecords
from quadcode.report.doc import Block, Code, Heading, Note, Para, bullets, source, table
from quadcode.report.fmt import MEASURED, THEORETICAL, listing, not_reached, num, one_line, stop_text
from quadcode.report.tables import availability_blocks, token_blocks
from quadcode.schemas import ROLE_VERSION_LABEL, Ablation, Phase, Role, SuiteRun, TestStatus


def suite_text(suite: SuiteRun | None) -> str:
    """'7/9 passed (T03 fail, T05 error)' for a suite run, or why it was not executed."""
    if suite is None:
        return "not tested"
    if not suite.executed:
        return f"not executed: {suite.not_executed_reason}"
    failing = [f"{r.test_id} {r.status.value}" for r in suite.results if r.status is not TestStatus.PASS]
    text = f"{suite.count(TestStatus.PASS)}/{len(suite.results)} passed"
    return text + (f" ({', '.join(failing)})" if failing else "")


# ---------------------------------------------------------------- overview


def overview(rec: RunRecords, pricing: AppConfig | None) -> list[Block]:
    blocks: list[Block] = [Heading(2, "Run overview")]
    s = rec.summary
    rows = [["run id", rec.run_id],
            ["status", s.status.value.upper() if s else "unknown (no summary.json)"],
            ["problem", s.problem_title or "-" if s else "-"],
            ["stop", stop_text(rec)],
            ["rounds", f"{s.rounds_run} run of {s.rounds_requested} requested" if s else "-"],
            ["final version", f"{s.final_version} ({'passes every test' if s.final_verified else 'NOT verified'})"
             if s and s.final_version else "none"],
            ["sandbox", s.sandbox_backend or "-" if s else "-"],
            ["pipeline", rec.ablation().description]]
    blocks.append(table(["item", "value"], rows))
    if rec.ablation() is not Ablation.NONE:
        blocks.append(Note(f"This run is an evaluation ablation ({rec.ablation().value}): "
                           f"{rec.ablation().description}. It is not the full QuadCode pipeline.", "warning"))
    if rec.is_mock():
        blocks.append(Note("Mock mode: every agent slot used the scripted mock provider. Agent answers are "
                           "fixed by the mock scenario and do not come from a language model. Test results "
                           "and measurements are still real executions in the sandbox.", "warning"))
    if s and s.isolation_warning:
        blocks.append(Note(s.isolation_warning, "warning"))
    if s and s.unresolved:
        blocks.append(Note("Unresolved issues (no convergence is claimed):\n" + "\n".join(s.unresolved), "warning"))
    for err in rec.load_errors:
        blocks.append(Note(f"Run records could not be fully read: {err}", "warning"))
    if rec.config is not None:
        blocks.append(table(["role", "slot", "provider", "kind", "model", "fallbacks, in order"],
                            [[r["role"], r["label"], r["provider"], r["kind"], r["model"], r["fallbacks"]]
                             for r in rec.config.describe_agents()],
                            caption="Agent slots (from the run's config snapshot)"))
    blocks.append(Para("Labels: values marked 'measured' come from executing code in the sandbox. Values "
                       "marked 'theoretical' are complexity claims made by an agent and were not verified."))
    blocks.append(source("summary.json", "config.json"))
    blocks.append(Heading(3, "Agent availability"))
    blocks.extend(availability_blocks(rec))
    blocks.append(Heading(3, "Calls, tokens and cost"))
    blocks.extend(token_blocks(rec, pricing))
    return blocks


# ---------------------------------------------------------------- 1


def section_problem(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "1. Problem analysis")]
    found = rec.spec()
    failed = [m for m in rec.messages("analysis") if not m.ok]
    for m in failed:
        blocks.append(Note(f"Analysis call by {m.who} UNAVAILABLE: {m.error}", "warning"))
    if found is None:
        if not failed:
            blocks.extend(not_reached(rec, "the analysis call"))
    else:
        spec, msg = found
        blocks.append(Para(f"One analysis call, answered by {msg.who}."))
        blocks.append(table(["field", "value"], [
            ["title", spec.title], ["summary", spec.summary], ["language", spec.language],
            ["entry point", spec.entry_point], ["signature", spec.signature],
            ["expected input", spec.inputs], ["expected output", spec.outputs]]))
        for title, items in [("Requirements", spec.functional_requirements), ("Constraints", spec.constraints),
                             ("Performance constraints", spec.performance_constraints),
                             ("Edge cases", spec.edge_cases), ("Dependencies", spec.dependencies),
                             ("Important assumptions", spec.assumptions)]:
            blocks.append(Heading(4, title))
            blocks.append(bullets(items))
        if spec.clarification_needed:
            blocks.append(Note("The analysis asked for clarification, so the run stopped:\n"
                               + "\n".join(spec.clarification_questions or ["(no questions given)"]), "warning"))
        blocks.append(source(msg.logged.ref))
    if rec.problem is not None:
        blocks.append(Code(rec.problem, "markdown", "Original problem statement (problem.md)"))
    blocks.append(source("problem.md", "events.jsonl (agent_output, call analysis)"))
    return blocks


# ---------------------------------------------------------------- 2


def section_initial(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "2. Four independent initial implementations")]
    if not rec.phase_started(Phase.INITIAL):
        return blocks + not_reached(rec, "the blind initial implementations (Phase 2)")
    blocks.append(Para("Each agent wrote its solution blind, from the problem and the specification only. "
                       "The code below is the stored immutable version, shown in full."))
    for role, label in ROLE_VERSION_LABEL.items():
        blocks.append(Heading(3, f"{label}: {rec.who(role)}"))
        for m in rec.messages("initial"):
            if m.role is role and not m.ok:
                blocks.append(Note(f"UNAVAILABLE: {m.error}", "warning"))
        found = rec.initial(role)
        if found is None:
            if not any(m.role is role for m in rec.messages("initial")):
                blocks.append(Note("No record of this agent's initial call: the phase was interrupted before it "
                                   "finished.", "warning"))
            continue
        sol, msg = found
        rows = [["algorithm", sol.algorithm], ["data structures", listing(sol.data_structures)],
                [f"time complexity ({THEORETICAL})", sol.time_complexity],
                [f"space complexity ({THEORETICAL})", sol.space_complexity],
                ["pros (agent's own)", listing(sol.pros)], ["cons (agent's own)", listing(sol.cons)],
                ["assumptions", listing(sol.assumptions)], ["language", sol.language]]
        if role is Role.VALIDATOR:
            rows.append(["test cases returned", str(len(sol.tests))])
        blocks.append(table(["field", "value"], rows))
        code = rec.codes.get(label, sol.code)
        blocks.append(Code(code, sol.language, f"{label} (versions/{label}{_ext(rec, label)})"))
        blocks.append(source(msg.logged.ref, f"versions/{label}"))
    return blocks


def _ext(rec: RunRecords, label: str) -> str:
    meta = rec.versions.get(label)
    return meta.ext if meta else ""


# ---------------------------------------------------------------- 3


def _review_findings(rec: RunRecords, label: str) -> tuple[str, str]:
    weaknesses: list[str] = []
    bugs: list[str] = []
    for role in Role:
        found = rec.review(role)
        if found is None:
            continue
        for item in found[0].items:
            if item.target == label:
                weaknesses.extend(f"{role.value}: {w}" for w in item.weaknesses)
                bugs.extend(f"{role.value}: {b}" for b in item.bugs)
    return "\n".join(weaknesses) or "-", "\n".join(bugs) or "-"


def section_comparison(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "3. Initial comparison")]
    if not any(rec.initial(r) for r in Role):
        return blocks + not_reached(rec, "a comparison of initial implementations (none was produced)")
    rows: list[list[str]] = []
    for role, label in ROLE_VERSION_LABEL.items():
        found = rec.initial(role)
        weak, bugs = _review_findings(rec, label)
        if found is None:
            status = "EXCLUDED (ablation)" if rec.excluded(role) else "UNAVAILABLE"
            rows.append([label, rec.who(role), status, "-", "-", "-", "-", weak, bugs, "-"])
            continue
        sol = found[0]
        rows.append([label, rec.who(role), sol.algorithm, sol.time_complexity, sol.space_complexity,
                     "\n".join(sol.pros) or "-", "\n".join(sol.cons) or "-", weak, bugs,
                     suite_text(rec.initial_suite(label))])
    blocks.append(table(["slot", "agent", "algorithm", f"time ({THEORETICAL})", f"space ({THEORETICAL})",
                         "advantages (own)", "weaknesses (own)", "weaknesses found by reviewers",
                         "bugs found by reviewers", f"local tests ({MEASURED})"], rows))
    unavailable = [m for m in rec.messages("review") if not m.ok]
    for m in unavailable:
        blocks.append(Note(f"Review by {m.who} UNAVAILABLE: {m.error}. Its missing review is not counted as "
                           "approval of any version.", "warning"))
    blocks.append(source("events.jsonl (initial and review outputs)", "tests.json (Phase 2.5 suite runs)"))
    blocks.extend(_reference_blocks(rec))
    return blocks


def _reference_blocks(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(3, "Reference version V0, chosen on evidence")]
    found = rec.reference()
    if found is None:
        return blocks + not_reached(rec, "the choice of the reference version V0")
    ev, logged = found
    blocks.append(Para(f"V0 is an exact copy of {ev.winner}. Versions were filtered criterion by criterion in "
                       "priority order; no vote was taken. "
                       + ("The winner passed every test." if ev.verified else
                          "The winner did NOT pass every test, so it is not verified.")))
    rows = [[st.criterion, MEASURED if st.measured else "not measured",
             ", ".join(f"{k}={num(v)}" for k, v in st.values.items()) or "-", ", ".join(st.kept),
             ", ".join(st.dropped) or "-", one_line(st.note)] for st in ev.steps]
    blocks.append(table(["criterion", "basis", "values", "kept", "dropped", "rule"], rows))
    if ev.not_measured:
        blocks.append(Para("Not measured, so not used to decide: " + ", ".join(ev.not_measured) + "."))
    blocks.append(source(logged.ref))
    return blocks

