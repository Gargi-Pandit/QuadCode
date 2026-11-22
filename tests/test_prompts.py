from __future__ import annotations

import re

from quadcode.config import PromptConfig
from quadcode.jsonutil import parse_model
from quadcode.prompts import (
    analysis_prompt,
    challenge_prompt,
    initial_prompt,
    later_round_context,
    modify_prompt,
    results_table,
    review_digest,
    review_prompt,
    suite_listing,
    system_prompt,
    truncate,
)
from quadcode.providers.mock_scenarios import G1_CODE, M002_DIFF, get_scenario
from quadcode.schemas import (
    CallType,
    ModificationRecord,
    ProblemSpec,
    Review,
    Role,
    SuiteRun,
    TestCase,
    TestResult,
    TestStatus,
)

VENDORS = re.compile(r"gemini|claude|gpt|deepseek|openai|anthropic", re.IGNORECASE)
SPEC = parse_model(get_scenario("two_sum").responses[(Role.GENERATOR, CallType.ANALYSIS, "*")], ProblemSpec)
CFG = PromptConfig(max_error_chars=30, max_failing_tests=2)


def _suite(label: str, statuses: dict[str, tuple[TestStatus, str | None]], executed: bool = True) -> SuiteRun:
    return SuiteRun(version=label, language="python", entry_point="two_sum", backend="subprocess",
                    executed=executed, not_executed_reason=None if executed else "no runner for 'cobol'",
                    repeats=1, warmup=0,
                    results=[TestResult(test_id=t, version=label, status=s, error=e, measured=True, runtime_s=0.123)
                             for t, (s, e) in statuses.items()])


def test_system_prompts_are_role_based_and_vendor_free() -> None:
    texts = {r: system_prompt(r) for r in Role}
    assert len(set(texts.values())) == 4
    for role, text in texts.items():
        assert role.value.capitalize() in text
        assert "JSON object" in text
        assert not VENDORS.search(text)


def test_initial_prompt_identical_except_validator_tests() -> None:
    base = initial_prompt("Problem text", SPEC, with_tests=False)
    with_tests = initial_prompt("Problem text", SPEC, with_tests=True)
    assert with_tests.startswith(base) and '"tests"' in with_tests and '"tests"' not in base
    assert "clarification" not in base  # spec is passed without the clarification fields
    assert "two_sum" in base and "Problem text" in base


def test_analysis_prompt_lists_every_spec_field() -> None:
    text = analysis_prompt("p")
    for field in ProblemSpec.model_fields:
        assert f'"{field}"' in text


def test_truncate() -> None:
    assert truncate("a  b\n c", 10) == "a b c"
    assert truncate("x" * 50, 10) == "xxxxxxx..."


def test_results_table_has_no_timings_and_truncates() -> None:
    suites = {
        "G1": _suite("G1", {"T1": (TestStatus.PASS, None)}),
        "P1": _suite("P1", {f"T{i}": (TestStatus.FAIL, "expected [0, 1] but the function returned " + "x" * 40)
                            for i in range(4)}),
        "Z9": _suite("Z9", {"T1": (TestStatus.NOT_RUN, None)}, executed=False),
    }
    table = results_table(suites, CFG)
    lines = table.splitlines()
    assert lines[1] == "G1 | 1/1 | -"
    assert lines[2].startswith("P1 | 0/4 | T0 fail: ") and lines[2].endswith("; ... 2 more")
    assert "x" * 31 not in table
    assert lines[3].startswith("Z9 | not executed |")
    assert "0.123" not in table


def test_suite_listing() -> None:
    tests = [TestCase(id="T1", category="normal", args=[[1, 2], 3], expected=[0, 1]),
             TestCase(id="T2", category="adversarial", args=[None, 1], expect_exception="TypeError"),
             TestCase(id="T3", category="performance", generator="([list(range(9))], {})", expected=[])]
    text = suite_listing(tests, CFG)
    assert "T1 [normal]" in text and "-> [0, 1]" in text
    assert "raises TypeError" in text and "list(range(9))" in text


def test_review_prompt_marks_unavailable_slots() -> None:
    text = review_prompt(SPEC, "C1", {"G1": G1_CODE, "C1": None}, "table")
    assert "Your own implementation is C1" in text
    assert G1_CODE.splitlines()[2] in text
    assert "### C1\nUNAVAILABLE" in text


def test_review_digest_filters_by_target() -> None:
    reviews = {Role.REVIEWER: Review.model_validate({"items": [
        {"target": "G1", "bugs": ["b1"], "suggestions": ["s1"]}, {"target": "D1", "bugs": ["other"]}]})}
    digest = review_digest(reviews, "G1")
    assert "- reviewer (bug): b1" in digest and "- reviewer (suggestion): s1" in digest
    assert "other" not in digest
    assert "no bugs" in review_digest({}, "G1")


def test_modify_and_later_round_context() -> None:
    rec = ModificationRecord(id="M-002", proposed_by=Role.OPTIMIZER, round_index=1, base_version="V0",
                             candidate_version="V0+M-002", diff=M002_DIFF, original_code="a", replacement_code="b",
                             original_location="V0:L8", justification="skip self", expected_impact="fewer",
                             decision="rejected", decision_reason="T03 fail: expected [0,1], got []")
    ctx = later_round_context("V0", "--- a\n+++ b\n", [rec], CFG)
    assert "M-002 by optimizer" in ctx and "T03 fail" in ctx
    text = modify_prompt(SPEC, 2, "V1", G1_CODE, [], "V1 | 9/9 | -", ctx, CFG)
    assert "Optimization round 2" in text and "### V1 (solution.py)" in text and ctx in text
    challenge = challenge_prompt(SPEC, rec, "fails T03", "table")
    assert "optimizer proposed modification M-002 to V0" in challenge and M002_DIFF.strip() in challenge
    assert '"verdict": accept | reject | needs_changes' in challenge
