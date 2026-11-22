"""Sections 9 and 10: final technical evaluation and reproduction instructions."""

from __future__ import annotations

from quadcode.report.data import RunRecords
from quadcode.report.doc import Block, Code, Heading, Para, bullets, source, table
from quadcode.report.fmt import MEASURED, THEORETICAL, not_reached, num, seconds
from quadcode.report.sections_results import theoretical_time
from quadcode.report.tables import contribution_table, final_label
from quadcode.schemas import ROLE_VERSION_LABEL, Decision, Role, TestCategory, TestStatus


def _why_blocks(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(3, "Why this implementation")]
    final = final_label(rec)
    ref = rec.reference()
    if ref is None or final is None:
        return blocks + not_reached(rec, "a final implementation")
    ev = ref[0]
    chain = "; ".join(f"{st.criterion}: kept {', '.join(st.kept)}"
                      + (f", dropped {', '.join(st.dropped)}" if st.dropped else "") for st in ev.steps)
    lines = [f"V0 is a copy of {ev.winner}, chosen by the evidence cascade ({chain})."]
    for e in rec.of_kind("round_completed"):
        accepted = e.payload.get("accepted") or []
        idx = e.payload.get("round")
        lines.append(f"Round {idx}: accepted {', '.join(accepted)} into {e.payload.get('version')}." if accepted
                     else f"Round {idx}: nothing accepted.")
    verified = rec.summary is not None and rec.summary.final_verified
    lines.append(f"The final implementation is {final}; it "
                 + ("passes every test of the suite." if verified else "is NOT verified by the suite."))
    blocks.append(bullets(lines))
    return blocks


def _rejected_blocks(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(3, "Rejected suggestions and versions")]
    ref = rec.reference()
    if ref is not None:
        rows: list[list[str]] = []
        for st in ref[0].steps:
            for label in st.dropped:
                rows.append([label, st.criterion, ", ".join(f"{k}={num(v)}" for k, v in st.values.items()),
                             st.note])
        if rows:
            blocks.append(table(["initial version", "dropped at", "values", "rule"], rows,
                                caption="Initial versions not chosen as V0"))
    rejected = [r for r in rec.modifications if r.decision is Decision.REJECTED]
    if rejected:
        blocks.append(table(["modification", "agent", "proposal", "why rejected", "challenge verdicts"],
                            [[r.id, rec.who(r.proposed_by), r.justification, r.decision_reason,
                              "\n".join(f"{v.role.value}: {v.verdict.value}" for v in r.reviewers) or "-"]
                             for r in rejected], caption="Rejected modifications"))
    elif rec.modifications:
        blocks.append(Para("No modification was rejected."))
    return blocks


def _improvement_blocks(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(3, f"Improvement over the initial implementations ({MEASURED})")]
    final = final_label(rec)
    bench = rec.benchmark
    if bench is None or final is None or final not in bench.order:
        return blocks + not_reached(rec, "the final benchmark, so no measured comparison exists")
    stats = {(s.version, s.test_id): s for s in bench.stats}
    passed = {s.version: s.count(TestStatus.PASS) for s in bench.suites}
    total = len(rec.suite or [])
    perf = [t.id for t in rec.suite or [] if t.category is TestCategory.PERFORMANCE]
    rows: list[list[str]] = []
    for label in bench.order:
        if label == final:
            continue
        row = [label, f"{passed.get(label, 0)}/{total}", f"{passed.get(final, 0)}/{total}"]
        ratios: list[str] = []
        for test_id in perf:
            a, b = stats.get((label, test_id)), stats.get((final, test_id))
            if a is None or b is None or b.median_s <= 0:
                ratios.append(f"{test_id}: not comparable (not measured for both)")
                continue
            if b.max_s < a.min_s:
                verdict = "final faster: the timing ranges do not overlap"
            elif b.min_s > a.max_s:
                verdict = "final slower: the timing ranges do not overlap"
            else:
                verdict = "not distinguishable: the timing ranges overlap"
            ratios.append(f"{test_id}: {label} {seconds(a.median_s)} s vs {final} {seconds(b.median_s)} s "
                          f"= {a.median_s / b.median_s:.2f}x ({verdict})")
        row.append("\n".join(ratios) or "no performance tests")
        rows.append(row)
    blocks.append(table(["initial version", "its tests passed", f"{final} tests passed",
                         "median runtime ratio initial / final (>1 means the final version is faster)"], rows))
    blocks.append(Para("Ratios are computed from the medians in benchmark.json. A difference counts only when the "
                       "min-max ranges of the timed runs do not overlap; otherwise it is reported as not "
                       "distinguishable from run-to-run variation."))
    return blocks


def _limitations(rec: RunRecords) -> list[str]:
    items: list[str] = []
    s = rec.summary
    if s is not None:
        items.extend(f"Unresolved: {u}" for u in s.unresolved)
    for role in Role:
        missed = [m.tag for m in rec.messages() if m.role is role and not m.ok]
        if missed:
            items.append(f"{rec.who(role)} was unavailable for: {', '.join(missed)}.")
    ref = rec.reference()
    if ref is not None and ref[0].not_measured:
        items.append("Not measured and not used in any decision: " + ", ".join(ref[0].not_measured) + ".")
    if s is not None and s.isolation_warning:
        items.append(s.isolation_warning)
    for e in rec.of_kind("suspect_tests"):
        items.append(f"Suspect tests (failed on every initial version): {', '.join(e.payload.get('test_ids') or [])}.")
    unapplied = [r.id for r in rec.modifications if r.apply_error]
    if unapplied:
        items.append(f"Proposals whose diff did not apply and were never executed: {', '.join(unapplied)}.")
    items.append("Test expected values come from the validator agent; hidden tests are not part of a solve run.")
    items.append("Peak memory is the Python heap measured with tracemalloc, not total process memory.")
    if rec.is_mock():
        items.append("Mock mode: agent answers are scripted, so this run says nothing about real model quality.")
    items.extend(f"Records not fully readable: {err}" for err in rec.load_errors)
    return items


def section_evaluation(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "9. Final technical evaluation")]
    blocks.extend(_why_blocks(rec))
    blocks.append(Heading(3, "What each agent contributed"))
    blocks.append(contribution_table(rec))
    blocks.append(Para("Counts come from the logged messages and modifications.json. No contribution is added "
                       "to make the work look evenly shared."))
    blocks.extend(_rejected_blocks(rec))
    blocks.extend(_improvement_blocks(rec))
    final = final_label(rec)
    ref = rec.reference()
    if final is not None and ref is not None and ref[0].winner is not None:
        winner = ref[0].winner
        space = next((found[0].space_complexity for role, slot in ROLE_VERSION_LABEL.items()
                      if slot == winner and (found := rec.initial(role)) is not None), "-")
        rows = [[f"time ({THEORETICAL})", theoretical_time(rec, final)],
                [f"space ({THEORETICAL})", f"{space}; claimed for {winner}"]]
        rows.extend([f"claimed impact of {r.id}", r.expected_impact] for r in rec.modifications
                    if r.decision is Decision.ACCEPTED)
        blocks.append(Heading(3, "Final complexity"))
        blocks.append(table(["item", "value"], rows))
    blocks.append(Heading(3, "Remaining limitations"))
    blocks.append(bullets(_limitations(rec)))
    blocks.append(source("events.jsonl", "modifications.json", "benchmark.json", "summary.json"))
    return blocks


# ---------------------------------------------------------------- 10


def section_reproduction(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "10. Reproduction instructions")]
    cfg = rec.config
    runs_dir = cfg.run.runs_dir if cfg else "runs"
    run = f"{runs_dir}/{rec.run_id}"
    found = rec.spec()
    entry = found[0].entry_point if found else "<entry_point>"
    deps = found[0].dependencies if found else []
    env_rows: list[list[str]] = []
    bench = rec.benchmark
    if bench is not None and bench.suites:
        env = bench.suites[0].sandbox_env
        env_rows += [["sandbox backend", bench.backend], ["sandbox Python", str(env.get("python", "-"))],
                     ["sandbox platform", str(env.get("platform", "-"))]]
    if cfg is not None:
        sb = cfg.sandbox
        env_rows += [["docker image (docker backend)", sb.docker_image], ["memory cap", f"{sb.memory_mb} MB"],
                     ["per-call timeout", f"{sb.test_timeout_s} s"], ["benchmark repeats", str(sb.benchmark_repeats)]]
    if env_rows:
        blocks.append(table(["item", "value"], env_rows, caption="Environment of this run"))
    blocks.append(Para("Dependencies of the solution: " + (", ".join(deps) if deps else
                                                           "none listed by the analysis") + ". "
                       "QuadCode itself needs Python 3.12+ and the packages in requirements.txt; Docker is optional."))
    rounds = rec.summary.rounds_requested if rec.summary else 3
    mock = ""
    if rec.is_mock() and cfg is not None:
        scenarios = sorted({cfg.provider_for(r).scenario or "" for r in Role})
        mock = " --mock" + (f" --scenario {scenarios[0]}" if len(scenarios) == 1 and scenarios[0] else "")
    final = rec.final_file or "final_solution.py"
    commands = [
        "python -m venv .venv && . .venv/bin/activate",
        "pip install -e .",
        f"quadcode solve {run}/problem.md --rounds {rounds}{mock}",
        f"quadcode test-solution {run}/{final} --tests {run}/test_suite.json --entry {entry} --benchmark",
        f"quadcode versions {run} --blame {final_label(rec) or 'V0'}",
        f"quadcode report {rec.run_id}",
    ]
    if rec.summary is not None and rec.summary.status.value in ("aborted_budget", "running"):
        commands.append(f"quadcode solve --resume {rec.run_id}")
    blocks.append(Code("\n".join(commands), "bash", "Commands"))
    blocks.append(bullets([
        "Line 3 starts a new run of the same problem in a new run directory.",
        "Line 4 runs the stored test suite against the final file in the sandbox.",
        "Line 5 shows the origin of every line of the final version.",
        "Line 6 rebuilds this report from the logs without any API call.",
        ("Mock runs give the same agent answers every time; timings still vary between machines and runs."
         if rec.is_mock() else
         "Real model answers vary between calls; identical prompts are served from the response cache "
         "(.quadcode_cache) if it is kept."),
        f"If the run was written with --runs-dir, replace {runs_dir}/ with that directory.",
    ]))
    blocks.append(source("config.json", "benchmark.json"))
    return blocks
