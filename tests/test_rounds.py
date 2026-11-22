"""Variants of the two_sum scenario that drive the remaining orchestrator paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quadcode.modifications import ModificationLedger
from quadcode.providers.mock import WILDCARD, FailureSpec
from quadcode.providers.mock_scenarios import M001_DIFF
from quadcode.runstate import load_state
from quadcode.schemas import CallType, Decision, Role, RunStatus, TestStatus, Verdict
from quadcode.versions import VersionStore

from .orch_helpers import build, calls, events, scenario_with

G, R, O, V = Role.GENERATOR, Role.REVIEWER, Role.OPTIMIZER, Role.VALIDATOR


def _proposals(*diffs: str) -> str:
    return json.dumps({"proposals": [{"diff": d, "justification": "j", "expected_impact": "e"} for d in diffs]})


EMPTY = _proposals()

STALE_DIFF = '''--- a/solution.py
+++ b/solution.py
@@ -2,2 +2,2 @@
     """A docstring that is not in the code."""
-    seen: dict[int, int] = {}
+    seen = dict()
'''

COMMENT_ONLY_DIFF = '''--- a/solution.py
+++ b/solution.py
@@ -2,3 +2,4 @@
     """Return indices [i, j] (i < j) with nums[i] + nums[j] == target, or [] if none."""
+    # map each value to the index where it was last seen
     seen: dict[int, int] = {}
     for i, value in enumerate(nums):
'''

NONE_CHECK_DIFF = '''--- a/solution.py
+++ b/solution.py
@@ -1,5 +1,7 @@
 def two_sum(nums: list[int], target: int) -> list[int]:
     """Return indices [i, j] (i < j) with nums[i] + nums[j] == target, or [] if none."""
+    if nums is None:
+        raise TypeError("nums must not be None")
     seen: dict[int, int] = {}
     for i, value in enumerate(nums):
         complement = target - value
'''

MEMORY_HOG_DIFF = '''--- a/solution.py
+++ b/solution.py
@@ -3,3 +3,4 @@
     seen: dict[int, int] = {}
+    scratch = [0] * (len(nums) * 50)
     for i, value in enumerate(nums):
         complement = target - value
'''


def _round1(**by_role: str) -> dict[tuple[Role, CallType, str], str]:
    out = {(r, CallType.MODIFY, "round-1"): EMPTY for r in Role}
    out.update({(Role(k), CallType.MODIFY, "round-1"): v for k, v in by_role.items()})
    return out


def _run(tmp_path: Path, **kwargs: Any) -> tuple[Any, Path]:
    built = build(tmp_path, **kwargs)
    return built.orch.run(), built.run_dir


def test_unappliable_and_comment_only_proposals_are_rejected(tmp_path: Path) -> None:
    scen = scenario_with(_round1(generator=_proposals(STALE_DIFF, COMMENT_ONLY_DIFF)))
    summary, run_dir = _run(tmp_path, scenario=scen)
    ledger = ModificationLedger(run_dir)
    stale, comment = ledger.get("M-001"), ledger.get("M-002")
    assert stale.decision is Decision.REJECTED
    assert stale.decision_reason.startswith("diff did not apply") and stale.candidate_version is None
    assert stale.original_location == "unresolved" and stale.test_status is TestStatus.NOT_RUN
    assert comment.decision is Decision.REJECTED and comment.test_status is TestStatus.PASS
    assert "no executable change" in comment.decision_reason
    assert comment.reviewers == []  # neither failing nor contested: no challenge calls
    assert not calls(run_dir, lambda c: c["call_type"] == "challenge")
    assert summary.stop_reason == "round 1 accepted no modification (early stop)"
    assert summary.final_version == "V0" and summary.status is RunStatus.COMPLETED
    assert (run_dir / "final_solution.py").read_text() == VersionStore(run_dir).load("G1").code


def test_conflicting_proposals_keep_the_first_and_challenge_the_other(tmp_path: Path) -> None:
    scen = scenario_with(_round1(generator=_proposals(NONE_CHECK_DIFF), reviewer=_proposals(M001_DIFF)))
    summary, run_dir = _run(tmp_path, scenario=scen)
    ledger = ModificationLedger(run_dir)
    first, second = ledger.get("M-001"), ledger.get("M-002")
    assert first.proposed_by is G and first.decision is Decision.ACCEPTED and first.new_version == "V1"
    assert second.proposed_by is R and second.decision is Decision.REJECTED
    assert second.test_status is TestStatus.PASS
    assert second.decision_reason.startswith("conflicts with M-001")
    # Challengers: generator and validator answer, the optimizer has no scripted answer.
    verdicts = {v.role: v.verdict for v in second.reviewers}
    assert verdicts == {G: Verdict.ACCEPT, O: Verdict.UNAVAILABLE, V: Verdict.REJECT}
    assert summary.accepted == ["M-001"] and summary.final_version == "V1"
    assert "# [generator:M-001]" in (run_dir / "final_solution.py").read_text()


def test_measured_memory_regression_is_rejected(tmp_path: Path) -> None:
    scen = scenario_with(_round1(generator=_proposals(MEMORY_HOG_DIFF)))
    summary, run_dir = _run(tmp_path, scenario=scen)
    rec = ModificationLedger(run_dir).get("M-001")
    assert rec.test_status is TestStatus.PASS and rec.decision is Decision.REJECTED
    assert rec.decision_reason.startswith("measured regression against V0")
    assert "peak memory" in rec.decision_reason and "(regression)" in rec.decision_reason
    assert {v.role: v.verdict for v in rec.reviewers} == {R: Verdict.UNAVAILABLE, O: Verdict.UNAVAILABLE,
                                                         V: Verdict.UNAVAILABLE}
    [tested] = events(run_dir, "candidate_tested")
    assert tested.payload["regression"] is True
    assert summary.final_version == "V0"


def test_all_agents_unavailable_is_not_convergence(tmp_path: Path) -> None:
    scen = scenario_with(failures={(r, CallType.MODIFY, "round-1"): FailureSpec("permanent") for r in Role})
    summary, run_dir = _run(tmp_path, scenario=scen)
    assert summary.stop_reason == "every agent was unavailable in round 1; stopped without evidence of convergence"
    state = load_state(run_dir)
    assert state.rounds_state[0].unavailable == list(Role)
    assert len([e for e in events(run_dir, "agent_unavailable") if e.payload["tag"] == "round-1"]) == 4
    assert summary.final_version == "V0"


def test_no_test_suite_stops_unverified(tmp_path: Path) -> None:
    scen = scenario_with(failures={(V, CallType.INITIAL, WILDCARD): FailureSpec("permanent")})
    summary, run_dir = _run(tmp_path, scenario=scen)
    assert summary.status is RunStatus.UNVERIFIED
    assert summary.stop_reason.startswith("no test suite")
    assert summary.final_version is None and summary.final_file is None
    assert not calls(run_dir, lambda c: c["call_type"] in ("review", "modify"))
    assert not (run_dir / "final_solution.py").exists()
    assert [m.label for m in VersionStore(run_dir).list()] == ["G1", "C1", "P1"]


def test_clarification_stops_after_one_analysis_call(tmp_path: Path) -> None:
    spec = json.loads(scenario_with().responses[(G, CallType.ANALYSIS, WILDCARD)])
    spec.update(clarification_needed=True, clarification_questions=["Which index base?"])
    scen = scenario_with({(r, CallType.ANALYSIS, WILDCARD): json.dumps(spec) for r in Role})
    summary, run_dir = _run(tmp_path, scenario=scen)
    assert summary.status is RunStatus.NEEDS_CLARIFICATION
    assert summary.unresolved == ["Which index base?"]
    assert len(calls(run_dir)) == 1
    [ev] = events(run_dir, "clarification_needed")
    assert ev.payload["questions"] == ["Which index base?"]


def test_analysis_falls_back_to_the_next_slot(tmp_path: Path) -> None:
    scen = scenario_with(failures={(G, CallType.ANALYSIS, WILDCARD): FailureSpec("permanent")})
    built = build(tmp_path, scenario=scen, rounds=1)
    summary = built.orch.run()
    state = load_state(built.run_dir)
    assert state.analysis_role is R
    [unavail] = [e for e in events(built.run_dir, "agent_unavailable") if e.payload["tag"] == "analysis"]
    assert unavail.role is G
    assert summary.status is RunStatus.COMPLETED and summary.rounds_requested == 1


def test_analysis_unavailable_everywhere_fails(tmp_path: Path) -> None:
    scen = scenario_with(failures={(r, CallType.ANALYSIS, WILDCARD): FailureSpec("permanent") for r in Role})
    summary, run_dir = _run(tmp_path, scenario=scen)
    assert summary.status is RunStatus.FAILED
    assert len(calls(run_dir)) == 4 and not (run_dir / "versions" / "G1.py").exists()
