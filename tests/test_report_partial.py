"""Reports of runs that stopped early, failed or hold hostile agent output."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from quadcode.providers.mock import WILDCARD, FailureSpec
from quadcode.report import build_document, write_report
from quadcode.report.data import load_run
from quadcode.report.render_html import render_html
from quadcode.report.render_md import render_markdown
from quadcode.schemas import CallType, Role, RunStatus

from .orch_helpers import build, scenario_with
from .test_report_run import section

G, R, V = Role.GENERATOR, Role.REVIEWER, Role.VALIDATOR


def _report(tmp_path: Path, **kwargs: object) -> tuple[RunStatus, str, str]:
    built = build(tmp_path, **kwargs)  # type: ignore[arg-type]
    summary = built.orch.run()
    paths = write_report(built.run_dir)
    assert paths.markdown is not None and paths.html is not None
    return summary.status, paths.markdown.read_text(), paths.html.read_text()


@pytest.mark.parametrize("budget", [1, 3, 9, 14, 18, 22])
def test_budget_aborted_run_reports_what_exists(tmp_path: Path, budget: int) -> None:
    status, md, html = _report(tmp_path, budget=budget)
    assert status is RunStatus.ABORTED_BUDGET
    assert "| status | ABORTED_BUDGET |" in md and "call budget of" in md
    assert "Not reached: the final integration (Phase 8)" in section(md, "8. Final optimized implementation")
    testing = section(md, "6. Testing and benchmarking")
    assert "Not reached: the final benchmark (Phase 7)" in testing
    assert "(measured)" not in testing.split("### Final benchmark")[1]  # no measured numbers without a benchmark
    assert "quadcode solve --resume run" in md
    assert "<script" not in html


def test_budget_abort_inside_phase_two_says_which_calls_have_no_record(tmp_path: Path) -> None:
    _, md, _ = _report(tmp_path, budget=1)
    initial = section(md, "2. Four independent initial implementations")
    assert initial.count("No record of this agent's initial call") == 4
    assert "Not reached: the optimization rounds" in section(md, "5. Iterative optimization process")


def test_no_test_suite_is_unverified_and_says_so(tmp_path: Path) -> None:
    scen = scenario_with(failures={(V, CallType.INITIAL, WILDCARD): FailureSpec("permanent")})
    status, md, _ = _report(tmp_path, scenario=scen)
    assert status is RunStatus.UNVERIFIED
    assert "Unresolved issues (no convergence is claimed)" in md
    assert "no test suite; nothing was executed" in md
    assert "No test suite: the validator produced none" in section(md, "6. Testing and benchmarking")
    assert "| final version | none |" in md
    assert "passes every test" not in md
    initial = section(md, "2. Four independent initial implementations")
    assert "### D1: validator (mock/mock-validator-v1)" in initial and "UNAVAILABLE: " in initial


def test_every_agent_unavailable_in_a_round(tmp_path: Path) -> None:
    scen = scenario_with(failures={(r, CallType.MODIFY, "round-1"): FailureSpec("permanent") for r in Role})
    _, md, _ = _report(tmp_path, scenario=scen)
    rounds = section(md, "5. Iterative optimization process")
    assert rounds.count("| UNAVAILABLE |") == 4
    assert "every agent was unavailable in round 1; stopped without evidence of convergence" in rounds
    assert "No agent proposed a modification" in section(md, "7. Complete modification history")


def test_clarification_stops_with_the_questions(tmp_path: Path) -> None:
    spec = json.loads(scenario_with().responses[(G, CallType.ANALYSIS, WILDCARD)])
    spec.update(clarification_needed=True, clarification_questions=["Which index base?"])
    scen = scenario_with({(r, CallType.ANALYSIS, WILDCARD): json.dumps(spec) for r in Role})
    status, md, _ = _report(tmp_path, scenario=scen)
    assert status is RunStatus.NEEDS_CLARIFICATION
    assert "The analysis asked for clarification, so the run stopped:" in section(md, "1. Problem analysis")
    assert "Which index base?" in md
    assert "Not reached: the blind initial implementations" in md


def test_failed_analysis_lists_every_unavailable_slot(tmp_path: Path) -> None:
    scen = scenario_with(failures={(r, CallType.ANALYSIS, WILDCARD): FailureSpec("permanent") for r in Role})
    status, md, _ = _report(tmp_path, scenario=scen)
    assert status is RunStatus.FAILED
    assert section(md, "1. Problem analysis").count("UNAVAILABLE") == 4


def test_hostile_agent_text_cannot_inject_markup(tmp_path: Path) -> None:
    hostile = {"items": [{"target": "G1", "strengths": ["<script>alert(1)</script>"],
                          "weaknesses": ["a | b"], "bugs": ["```\n</code></pre><img src=x onerror=alert(2)>"],
                          "suggestions": ["# heading\n- list"], "justification": "[x](javascript:alert(3))"}]}
    scen = scenario_with({(R, CallType.REVIEW, WILDCARD): json.dumps(hostile)})
    _, md, html = _report(tmp_path, scenario=scen, rounds=1)
    assert "<script" not in html and "<img" not in html and "javascript:alert(3)</a>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert not re.search(r"(?<!\\)<(script|img|/code|/pre)", md)  # every '<' of agent text is escaped
    assert not re.search(r"(?<!\\)\]\(javascript", md) and "\\[x\\](javascript:alert(3))" in md
    assert "\\<script\\>alert(1)\\</script\\>" in md
    assert "a \\| b" in md  # a pipe cannot split the table cell


def test_missing_summary_and_truncated_log_are_reported(tmp_path: Path) -> None:
    built = build(tmp_path / "src", budget=9)
    built.orch.run()
    damaged = tmp_path / "damaged"
    shutil.copytree(built.run_dir, damaged)
    (damaged / "summary.json").unlink()
    with (damaged / "events.jsonl").open("a") as fh:
        fh.write('{"kind": "phase_star')
    rec = load_run(damaged)
    assert rec.summary is None and any("incomplete line skipped" in e for e in rec.load_errors)
    doc = build_document(rec)
    md = render_markdown(doc)
    assert "unknown (no summary.json)" in md
    assert "Run records could not be fully read: events.jsonl" in md
    assert render_html(doc).startswith("<!doctype html>")
