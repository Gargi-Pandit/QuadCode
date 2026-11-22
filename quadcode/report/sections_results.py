"""Sections 6 and 8: testing and benchmarking, and the final implementation."""

from __future__ import annotations

from quadcode.report.data import RunRecords
from quadcode.report.doc import Block, Code, Heading, Note, Para, source, table
from quadcode.report.fmt import MEASURED, THEORETICAL, clip, not_reached, num, seconds, value_text
from quadcode.report.sections_problem import suite_text
from quadcode.report.tables import final_label
from quadcode.schemas import (
    ROLE_VERSION_LABEL,
    BenchmarkStats,
    Decision,
    Phase,
    TestCase,
    TestCategory,
    TestStatus,
)

VALUE_LIMIT = 160


def test_input(t: TestCase) -> str:
    if t.generator:
        return clip(f"generated in the sandbox: {t.generator}", VALUE_LIMIT, "test_suite.json")
    text = value_text(t.args, VALUE_LIMIT, "test_suite.json")
    if t.kwargs:
        text += " " + value_text(t.kwargs, VALUE_LIMIT, "test_suite.json")
    return text


def test_expected(t: TestCase) -> str:
    if t.expect_exception:
        return f"raises {t.expect_exception}"
    return value_text(t.expected, VALUE_LIMIT, "test_suite.json")


def version_role(rec: RunRecords, label: str) -> str:
    for role, slot in ROLE_VERSION_LABEL.items():
        if slot == label:
            return f"initial, {role.value}"
    return "final" if label == final_label(rec) else "derived"


def theoretical_time(rec: RunRecords, label: str) -> str:
    """The time complexity an agent claimed for a version, saying whose claim it is."""
    for role, slot in ROLE_VERSION_LABEL.items():
        if slot == label:
            found = rec.initial(role)
            return f"{found[0].time_complexity} ({role.value})" if found else "-"
    ref = rec.reference()
    if ref is None or ref[0].winner is None:
        return "-"
    source_label = ref[0].winner
    claim = theoretical_time(rec, source_label)
    changed = [r.id for r in rec.modifications if r.new_version is not None and r.decision is Decision.ACCEPTED]
    suffix = f"; claimed for {source_label}, not re-derived after {', '.join(changed)}" if changed else \
        f"; claimed for {source_label}"
    return claim + suffix


# ---------------------------------------------------------------- 6


def _suite_blocks(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(3, "Test suite")]
    if rec.suite is None:
        if rec.phase_completed(Phase.INITIAL):
            return blocks + [Note("No test suite: the validator produced none, so nothing could be verified.",
                                  "warning")]
        return blocks + not_reached(rec, "the test suite (it comes from the validator's initial answer)")
    rows = [[t.id, t.category.value, t.description or "-", test_input(t), test_expected(t), t.compare]
            for t in rec.suite]
    blocks.append(table(["test", "category", "description", "input", "expected output", "comparison"], rows))
    saved = rec.last("test_suite_saved")
    blocks.append(Para("The suite was written by the validator agent. Expected values are that agent's claims; "
                       "inputs only are sent into the sandbox and results are compared on the host."))
    for e in rec.of_kind("suspect_tests"):
        blocks.append(Note(f"Suspect tests {', '.join(e.payload.get('test_ids') or [])}: "
                           f"{e.payload.get('note')}", "warning"))
    blocks.append(source("test_suite.json", saved.ref if saved else ""))
    return blocks


def _stats_index(rec: RunRecords) -> dict[tuple[str, str], BenchmarkStats]:
    assert rec.benchmark is not None
    return {(s.version, s.test_id): s for s in rec.benchmark.stats}


def _benchmark_blocks(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(3, "Final benchmark")]
    bench = rec.benchmark
    if bench is None:
        return blocks + not_reached(rec, "the final benchmark (Phase 7)")
    ev = rec.last("final_benchmark")
    blocks.append(table(["item", "value"], [
        ["backend", bench.backend], ["timed runs per test", str(bench.repeats)], ["warmup runs", str(bench.warmup)],
        ["seed", str(bench.seed)], ["execution order", " -> ".join(bench.order)], ["started", bench.started_at]]))
    blocks.append(Para(f"Method: {bench.method}"))
    if bench.isolation_warning:
        blocks.append(Note(bench.isolation_warning, "warning"))
    blocks.append(table(["version", "role", f"tests passed ({MEASURED})", "sandbox Python"],
                        [[s.version, version_role(rec, s.version), suite_text(s),
                          str(s.sandbox_env.get("python", "-"))] for s in bench.suites],
                        caption="Pass/fail per version"))

    stats = _stats_index(rec)
    expected = {t.id: test_expected(t) for t in rec.suite or []}
    for suite in bench.suites:
        rows: list[list[str]] = []
        for r in suite.results:
            st = stats.get((suite.version, r.test_id))
            actual = (value_text(r.actual, VALUE_LIMIT, "benchmark.json") if r.status is not TestStatus.NOT_RUN
                      and r.actual is not None else "-")
            rows.append([r.test_id, r.status.value.upper(), actual, expected.get(r.test_id, "-"),
                         clip(r.error or "-", VALUE_LIMIT, "benchmark.json"),
                         seconds(st.median_s) if st else "not measured", seconds(st.min_s) if st else "-",
                         seconds(st.max_s) if st else "-",
                         num(r.peak_memory_bytes) if r.peak_memory_bytes is not None else "not measured"])
        if not suite.executed:
            blocks.append(Note(f"{suite.version} was not executed: {suite.not_executed_reason}", "warning"))
        blocks.append(table(["test", "status", "actual output", "expected output", "error", f"median s ({MEASURED})",
                             f"min s ({MEASURED})", f"max s ({MEASURED})", f"peak heap bytes ({MEASURED})"], rows,
                            caption=f"{suite.version} ({version_role(rec, suite.version)})"))
    blocks.extend(_comparison_blocks(rec))
    blocks.append(source("benchmark.json", ev.ref if ev else ""))
    return blocks


def _comparison_blocks(rec: RunRecords) -> list[Block]:
    assert rec.benchmark is not None
    perf = [t.id for t in rec.suite or [] if t.category is TestCategory.PERFORMANCE]
    if not perf:
        return [Para("The suite has no performance tests, so there is no performance comparison.")]
    stats = _stats_index(rec)
    rows: list[list[str]] = []
    for test_id in perf:
        for label in rec.benchmark.order:
            st = stats.get((label, test_id))
            suite = rec.benchmark_suite(label)
            result = next((r for r in suite.results if r.test_id == test_id), None) if suite else None
            status = result.status.value.upper() if result else "-"
            rows.append([test_id, label, status, seconds(st.median_s) if st else "not measured",
                         seconds(st.min_s) if st else "-", seconds(st.max_s) if st else "-",
                         num(st.peak_memory_bytes) if st and st.peak_memory_bytes is not None else "-",
                         theoretical_time(rec, label)])
    return [table(["test", "version", "status", f"median s ({MEASURED})", f"min s ({MEASURED})",
                   f"max s ({MEASURED})", f"peak heap bytes ({MEASURED})", f"time complexity ({THEORETICAL})"],
                  rows, caption="Initial vs final on the performance tests, identical inputs")]


def section_testing(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "6. Testing and benchmarking")]
    blocks.extend(_suite_blocks(rec))
    blocks.extend(_benchmark_blocks(rec))
    if rec.suite_runs:
        blocks.append(Para(f"tests.json also holds all {len(rec.suite_runs)} suite runs made during the run "
                           "(initial versions, candidates, composition trials and the final file). The timings "
                           "above come only from the final benchmark."))
    return blocks


# ---------------------------------------------------------------- 8


def section_final(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "8. Final optimized implementation")]
    ev = rec.last("final_written")
    if ev is None or rec.final_code is None:
        return blocks + not_reached(rec, "the final integration (Phase 8), so there is no final implementation")
    p = ev.payload
    check = next((e for e in reversed(rec.of_kind("suite_run")) if e.phase is Phase.INTEGRATE), None)
    check_text = (f"{check.payload.get('passed')}/{check.payload.get('total')} tests passed ({MEASURED})"
                  if check is not None and check.payload.get("executed") else "not executed")
    blocks.append(Para(f"{rec.final_file} is version {p.get('version')} with attribution comments added. "
                       f"The annotated file itself was run against the suite: {check_text}."))
    if not p.get("verified"):
        unresolved = rec.summary.unresolved if rec.summary else []
        blocks.append(Note("The final file is NOT verified. Unresolved:\n" + ("\n".join(unresolved) or "-"),
                           "warning"))
    annotations = p.get("annotations") or []
    if annotations:
        blocks.append(table(["modification", "agent", "comment line", "not placed because"],
                            [[a.get("id"), a.get("role"), str(a.get("line") or "-"), a.get("skipped_reason") or "-"]
                             for a in annotations], caption="Attribution comments"))
    else:
        blocks.append(Para("No modification was accepted, so the final code carries no attribution comments."))
    found = rec.spec()
    blocks.append(Code(rec.final_code, found[0].language if found else "", rec.final_file or ""))
    blocks.append(source(rec.final_file or "", ev.ref, check.ref if check else ""))
    return blocks
