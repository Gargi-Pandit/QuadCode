"""Full two_sum mock run: every scripted path, checked against the run's own logs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from quadcode.modifications import ModificationLedger
from quadcode.providers.mock_scenarios import C1_CODE, D1_CODE, G1_CODE, P1_CODE
from quadcode.runstate import load_state
from quadcode.schemas import CallType, Decision, Role, RunStatus, RunSummary, TestStatus, Verdict
from quadcode.versions import VersionStore

from .orch_helpers import Built, build, calls, events

VENDORS = re.compile(r"gemini|claude|gpt|deepseek|openai|anthropic", re.IGNORECASE)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Built, RunSummary]:
    built = build(tmp_path_factory.mktemp("orch"))
    return built, built.orch.run()


def test_summary(run: tuple[Built, RunSummary]) -> None:
    built, summary = run
    assert summary.status is RunStatus.COMPLETED
    assert summary.reference_source == "G1"
    assert summary.final_version == "V1" and summary.final_verified
    assert summary.accepted == ["M-001"] and summary.rejected == ["M-002"]
    assert summary.rounds_run == 2 and summary.rounds_requested == 3
    assert summary.stop_reason == "round 2 accepted no modification (early stop)"
    assert summary.isolation_warning  # subprocess backend says so
    on_disk = RunSummary.model_validate_json((built.run_dir / "summary.json").read_text())
    assert on_disk.status is RunStatus.COMPLETED


def test_run_directory_layout(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    for name in ("calls.jsonl", "events.jsonl", "modifications.json", "tests.json", "test_suite.json",
                 "benchmark.json", "final_solution.py", "summary.json", "state.json", "problem.md", "config.json"):
        assert (built.run_dir / name).is_file(), name
    labels = [m.label for m in VersionStore(built.run_dir).list()]
    assert labels == ["G1", "C1", "P1", "D1", "V0", "V0+M-001", "V0+M-002", "V1"]


def test_initial_versions_are_the_agents_code(run: tuple[Built, RunSummary]) -> None:
    store = VersionStore(run[0].run_dir)
    for label, code in (("G1", G1_CODE), ("C1", C1_CODE), ("P1", P1_CODE), ("D1", D1_CODE)):
        assert store.load(label).code == code


def test_phase2_is_blind_and_identical(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    # the output limit note at the end depends on each slot's token cap; the task itself must be identical
    prompts = {r: next(q.prompt for q in built.provider(r).requests
                       if q.call_type is CallType.INITIAL).split("\n\nOUTPUT LIMIT:")[0]
               for r in Role}
    for prompt in prompts.values():
        for code in (G1_CODE, C1_CODE, P1_CODE, D1_CODE):
            assert code.splitlines()[1] not in prompt
    assert prompts[Role.GENERATOR] == prompts[Role.REVIEWER] == prompts[Role.OPTIMIZER]
    assert prompts[Role.VALIDATOR].startswith(prompts[Role.GENERATOR])
    assert '"tests"' in prompts[Role.VALIDATOR] and '"tests"' not in prompts[Role.GENERATOR]


def test_no_vendor_names_in_any_prompt(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    for role in Role:
        for req in built.provider(role).requests:
            assert not VENDORS.search(req.system + req.prompt), (role, req.call_type, req.tag)


def test_local_tests_and_selection_evidence(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    state = load_state(built.run_dir)
    suites = state.initial_suites
    assert suites["G1"].all_passed and suites["D1"].all_passed
    assert {r.test_id: r.status for r in suites["C1"].results}["T09"] is TestStatus.TIMEOUT
    assert {r.test_id for r in suites["P1"].results if r.status is TestStatus.FAIL} == {"T02", "T06"}
    ev = state.reference
    assert ev is not None and ev.winner == "G1" and ev.verified
    assert [s.criterion for s in ev.steps] == ["tests_passed", "runtime", "memory", "code_size"]
    assert ev.steps[0].dropped == ["C1", "P1"]
    assert ev.steps[-1].values == {"G1": 9, "D1": 10}
    [selected] = events(built.run_dir, "reference_selected")
    assert selected.payload["source"] == "G1"


def test_validator_unavailable_in_review_is_recorded(run: tuple[Built, RunSummary]) -> None:
    built, summary = run
    state = load_state(built.run_dir)
    assert Role.VALIDATOR not in state.reviews and len(state.reviews) == 3
    [unavail] = [e for e in events(built.run_dir, "agent_unavailable") if e.payload["tag"] == "review"]
    assert unavail.role is Role.VALIDATOR and "RateLimitError" in unavail.payload["error"]
    review_calls = calls(built.run_dir, lambda c: c["role"] == "validator" and c["call_type"] == "review")
    assert len(review_calls) == 4 and all(c["status"] == "provider_error" for c in review_calls)
    assert any(a.role is Role.VALIDATOR and a.phase == "review" and a.status.value == "unavailable"
               for a in summary.availability)


def test_review_disagreement_is_preserved(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    reviews = {e.role: e.payload["data"] for e in events(built.run_dir, "agent_output")
               if e.payload["tag"] == "review"}
    c1_by_reviewer = next(i for i in reviews[Role.REVIEWER]["items"] if i["target"] == "C1")
    c1_by_optimizer = next(i for i in reviews[Role.OPTIMIZER]["items"] if i["target"] == "C1")
    assert "acceptable" in c1_by_reviewer["justification"]
    assert "Disagree" in c1_by_optimizer["justification"]
    review_prompt = next(q.prompt for q in built.provider(Role.GENERATOR).requests if q.call_type is CallType.REVIEW)
    for code in (G1_CODE, C1_CODE, P1_CODE, D1_CODE):
        assert code.splitlines()[1] in review_prompt
    assert "C1 | 8/9 | T09 timeout" in review_prompt
    assert "runtime" not in review_prompt.split("SANDBOX TEST RESULTS")[1].split("Evaluate")[0]


def test_round_one_records(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    ledger = ModificationLedger(built.run_dir)
    m1, m2 = ledger.get("M-001"), ledger.get("M-002")
    assert m1.proposed_by is Role.REVIEWER and m1.decision is Decision.ACCEPTED
    assert m1.test_status is TestStatus.PASS and m1.reviewers == []
    assert m1.new_version == "V1" and m1.new_location == "V1:L3-L4"
    assert m1.original_location == "V0:after L2"
    assert m2.proposed_by is Role.OPTIMIZER and m2.decision is Decision.REJECTED
    assert m2.test_status is TestStatus.FAIL and "T03" in m2.decision_reason
    verdicts = {v.role: v.verdict for v in m2.reviewers}
    assert verdicts == {Role.GENERATOR: Verdict.ACCEPT, Role.REVIEWER: Verdict.REJECT,
                        Role.VALIDATOR: Verdict.REJECT}
    # Only the failing proposal was challenged.
    tags = {c["tag"] for c in calls(built.run_dir, lambda c: c["call_type"] == "challenge")}
    assert tags == {"round-1:M-002"}


def test_round_prompts_carry_the_right_context(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    reqs = [q for q in built.provider(Role.REVIEWER).requests if q.call_type is CallType.MODIFY]
    r1 = next(q.prompt for q in reqs if q.tag == "round-1")
    r2 = next(q.prompt for q in reqs if q.tag == "round-2")
    assert "Validate input type explicitly" in r1 and G1_CODE.splitlines()[2] in r1
    assert "+    if not isinstance(nums, list):" in r2
    assert "M-002 by optimizer" in r2 and "T03" in r2
    assert D1_CODE.splitlines()[2] not in r2  # earlier versions are not resent


def test_transient_429_in_round_one_recovers(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    rows = calls(built.run_dir, lambda c: c["role"] == "validator" and c["tag"] == "round-1")
    assert [c["status"] for c in rows] == ["provider_error", "ok"]
    assert built.clock.sleeps  # backoff happened on the fake clock


def test_early_stop_event(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    [stop] = events(built.run_dir, "early_stop")
    assert stop.payload["round"] == 2
    assert not calls(built.run_dir, lambda c: c["tag"] == "round-3")


def test_final_benchmark_and_solution(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    bench = json.loads((built.run_dir / "benchmark.json").read_text())
    assert bench["order"] == ["G1", "C1", "P1", "D1", "V1"]
    assert bench["repeats"] == 2
    final = (built.run_dir / "final_solution.py").read_text()
    assert "    # [reviewer:M-001]\n    if not isinstance(nums, list):" in final
    [written] = events(built.run_dir, "final_written")
    assert written.payload["verified"] and written.payload["annotations"][0]["line"] == 3


def test_every_agent_message_traces_to_a_call(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    rows = calls(built.run_dir)
    for e in events(built.run_dir):
        if e.kind not in ("agent_output", "agent_unavailable"):
            continue
        assert e.role is not None
        matching = [c for c in rows if c["role"] == e.role.value and c["call_type"] == e.payload["call_type"]
                    and c["tag"] == e.payload["tag"]]
        assert matching, (e.kind, e.role, e.payload["tag"])
        ok = [c for c in matching if c["status"] == "ok"]
        assert (len(ok) == 1) if e.kind == "agent_output" else not ok
        assert all(c["provider"] == e.payload["provider"] and c["model"] == e.payload["model"] for c in matching)


def test_tests_json_holds_every_suite_run(run: tuple[Built, RunSummary]) -> None:
    built, _ = run
    versions = [s["version"] for s in json.loads((built.run_dir / "tests.json").read_text())]
    assert versions == ["G1", "C1", "P1", "D1", "V0", "V0+M-001", "V0+M-002", "V1", "final_solution"]


def test_create_refuses_existing_run(run: tuple[Built, RunSummary], tmp_path: Path) -> None:
    built, _ = run
    with pytest.raises(FileExistsError):
        build(built.run_dir.parent)
