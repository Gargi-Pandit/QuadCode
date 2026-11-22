from __future__ import annotations

import pytest
from pydantic import ValidationError

from quadcode.schemas import (
    ROLE_VERSION_LABEL,
    CallRecord,
    CallStatus,
    CallType,
    Decision,
    InitialSolution,
    ModificationRecord,
    ReviewerVerdict,
    Role,
    TestCase,
    TestResult,
    TestStatus,
    Verdict,
)


def _mod(**over: object) -> ModificationRecord:
    data: dict[str, object] = {
        "id": "M-001", "proposed_by": Role.OPTIMIZER, "round_index": 1, "base_version": "V0",
        "diff": "--- a\n+++ b\n", "original_code": "x = 1", "replacement_code": "x = 2",
        "original_location": "solution.py:1-1", "justification": "j", "expected_impact": "e",
    }
    data.update(over)
    return ModificationRecord.model_validate(data)


def test_version_labels() -> None:
    assert [ROLE_VERSION_LABEL[r] for r in Role] == ["G1", "C1", "P1", "D1"]


def test_modification_record_roundtrip_and_frozen() -> None:
    rec = _mod(reviewers=[ReviewerVerdict(role=Role.REVIEWER, verdict=Verdict.REJECT, comment="breaks T03")])
    again = ModificationRecord.model_validate_json(rec.model_dump_json())
    assert again == rec
    with pytest.raises(ValidationError):
        rec.decision = Decision.ACCEPTED  # type: ignore[misc]
    decided = rec.model_copy(update={"decision": Decision.REJECTED, "decision_reason": "T03 fails"})
    assert rec.decision is Decision.PENDING and decided.decision is Decision.REJECTED


@pytest.mark.parametrize("bad_id", ["M-1", "X-001", "M001", ""])
def test_modification_id_pattern(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        _mod(id=bad_id)


def test_modification_id_allows_more_digits() -> None:
    assert _mod(id="M-1234").id == "M-1234"


def test_initial_solution_ignores_extra_keys_and_parses_tests() -> None:
    sol = InitialSolution.model_validate({
        "language": "python", "code": "def f(): pass", "algorithm": "a",
        "time_complexity": "O(1)", "space_complexity": "O(1)", "chatter": "ignored",
        "tests": [{"id": "T1", "category": "boundary", "args": [[]], "expected": []}],
    })
    assert sol.tests[0].category.value == "boundary"
    assert sol.tests[0].hidden is False


def test_test_case_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        TestCase.model_validate({"id": "T", "category": "vibes"})


def test_test_result_requires_measured_flag() -> None:
    with pytest.raises(ValidationError):
        TestResult.model_validate({"test_id": "T1", "version": "G1", "status": "pass"})
    r = TestResult(test_id="T1", version="G1", status=TestStatus.NOT_RUN, measured=False)
    assert r.runtime_s is None


def test_call_record_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        CallRecord(role=Role.GENERATOR, provider="p", model="m", call_type=CallType.INITIAL, attempt=1,
                   prompt_hash="h", cache_key="k", tokens_in=-1, tokens_out=0, tokens_estimated=False,
                   latency_s=0.0, cache_hit=False, status=CallStatus.OK)
