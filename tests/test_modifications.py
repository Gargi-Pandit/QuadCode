from __future__ import annotations

import json
from pathlib import Path

import pytest

from quadcode.modifications import (
    UNRESOLVED,
    LedgerError,
    ModificationLedger,
    build_record,
)
from quadcode.providers.mock_scenarios import G1_CODE, M001_DIFF, M002_DIFF
from quadcode.schemas import (
    Decision,
    ModificationProposal,
    ModificationRecord,
    ReviewerVerdict,
    Role,
    TestStatus,
    Verdict,
)

STALE = "@@ -3,2 +3,2 @@\n     lookup: dict[int, int] = {}\n-    x = 1\n+    x = 2\n"


def _proposal(diff: str) -> ModificationProposal:
    return ModificationProposal(diff=diff, justification="because", expected_impact="faster",
                                disadvantages="none")


def _build(mod_id: str, diff: str, role: Role = Role.REVIEWER) -> ModificationRecord:
    return build_record(_proposal(diff), modification_id=mod_id, proposed_by=role, round_index=1,
                        base_label="V0", base_code=G1_CODE).record


def test_build_record_fills_code_and_locations() -> None:
    built = build_record(_proposal(M001_DIFF), modification_id="M-001", proposed_by=Role.REVIEWER,
                         round_index=1, base_label="V0", base_code=G1_CODE)
    rec = built.record
    assert built.result is not None
    assert rec.candidate_version == "V0+M-001" and rec.decision is Decision.PENDING
    assert rec.original_code == "" and rec.replacement_code.startswith("    if not isinstance(nums, list):")
    assert rec.original_location == "V0:after L2" and rec.new_location == "V0+M-001:L3-L4"
    assert len(rec.hunks) == 1 and rec.hunks[0].offset == 0 and rec.apply_error is None
    assert rec.justification == "because" and rec.disadvantages == "none"


def test_build_record_replacement() -> None:
    rec = _build("M-002", M002_DIFF, Role.OPTIMIZER)
    assert rec.original_code == "        seen[value] = i"
    assert rec.replacement_code == "        if complement != value:\n            seen[value] = i"
    assert rec.original_location == "V0:L8" and rec.new_location == "V0+M-002:L8-L9"


def test_multi_hunk_record_joins_hunks() -> None:
    diff = (Path(__file__).resolve().parents[1] / "samples/two_sum/diffs/two_hunks.diff").read_text()
    rec = _build("M-003", diff)
    assert len(rec.hunks) == 2
    assert rec.original_location == "V0:after L2; V0:L8"
    assert rec.new_location == "V0+M-003:L3-L4; V0+M-003:L10"
    assert rec.original_code == "\n...\n        seen[value] = i"


def test_unappliable_diff_is_kept_as_rejected() -> None:
    built = build_record(_proposal(STALE), modification_id="M-004", proposed_by=Role.VALIDATOR,
                         round_index=2, base_label="V1", base_code=G1_CODE)
    rec = built.record
    assert built.result is None
    assert rec.decision is Decision.REJECTED and rec.test_status is TestStatus.NOT_RUN
    assert rec.decision_reason.startswith("diff did not apply: hunk 1: context does not match")
    assert rec.apply_error and rec.original_location == UNRESOLVED and rec.candidate_version is None
    assert rec.original_code == "    x = 1" and rec.replacement_code == "    x = 2"  # as the diff stated them
    assert rec.decided_at is not None


def test_unparseable_diff_is_kept_as_rejected() -> None:
    rec = _build("M-005", "I would change the loop to use a set.")
    assert rec.decision is Decision.REJECTED and "no hunks" in rec.decision_reason
    assert rec.original_code == "" and rec.replacement_code == ""


def test_ledger_ids_persist_and_resume(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    assert ledger.next_id() == "M-001"
    ledger.add(_build(ledger.next_id(), M001_DIFF))
    ledger.add(_build(ledger.next_id(), M002_DIFF, Role.OPTIMIZER))
    resumed = ModificationLedger(tmp_path)
    assert [r.id for r in resumed.all()] == ["M-001", "M-002"]
    assert resumed.next_id() == "M-003"
    data = json.loads((tmp_path / "modifications.json").read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == ["M-001", "M-002"]
    assert not list(tmp_path.glob(".tmp-*"))


def test_ledger_add_is_idempotent_but_never_replaces(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    rec = ledger.add(_build("M-001", M001_DIFF))
    assert ledger.add(_build("M-001", M001_DIFF)) == rec
    with pytest.raises(LedgerError, match="different content"):
        ledger.add(_build("M-001", M002_DIFF))


def test_ledger_rejects_new_records_that_are_already_accepted(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    rec = _build("M-001", M001_DIFF).model_copy(update={"decision": Decision.ACCEPTED})
    with pytest.raises(LedgerError, match="pending or rejected"):
        ledger.add(rec)


def test_accept_requires_passing_tests(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    ledger.add(_build("M-001", M001_DIFF))
    with pytest.raises(LedgerError, match="test status not_run"):
        ledger.accept("M-001", "looks good")
    ledger.set_test_status("M-001", TestStatus.FAIL)
    with pytest.raises(LedgerError, match="test status fail"):
        ledger.accept("M-001", "looks good")
    ledger.set_test_status("M-001", TestStatus.PASS)
    with pytest.raises(LedgerError, match="needs a reason"):
        ledger.accept("M-001", "  ")
    rec = ledger.accept("M-001", "all 9 tests pass on V0+M-001")
    assert rec.decision is Decision.ACCEPTED and rec.decided_at is not None


def test_decided_records_are_frozen(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    ledger.add(_build("M-002", M002_DIFF, Role.OPTIMIZER))
    ledger.set_test_status("M-002", TestStatus.FAIL)
    ledger.reject("M-002", "T03 fails")
    for action in (lambda: ledger.accept("M-002", "x"), lambda: ledger.reject("M-002", "again"),
                   lambda: ledger.withdraw("M-002", "x"), lambda: ledger.set_test_status("M-002", TestStatus.PASS),
                   lambda: ledger.add_verdict("M-002", ReviewerVerdict(role=Role.GENERATOR, verdict=Verdict.ACCEPT))):
        with pytest.raises(LedgerError, match="already rejected"):
            action()
    assert ModificationLedger(tmp_path).get("M-002").decision_reason == "T03 fails"


def test_reject_and_withdraw_need_reasons(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    ledger.add(_build("M-001", M001_DIFF))
    with pytest.raises(LedgerError, match="needs a reason"):
        ledger.reject("M-001", "")
    rec = ledger.withdraw("M-001", "proposer withdrew it after review")
    assert rec.decision is Decision.WITHDRAWN


def test_verdicts_are_appended_and_proposer_cannot_review(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    ledger.add(_build("M-002", M002_DIFF, Role.OPTIMIZER))
    ledger.add_verdict("M-002", ReviewerVerdict(role=Role.GENERATOR, verdict=Verdict.ACCEPT, comment="fine"))
    ledger.add_verdict("M-002", ReviewerVerdict(role=Role.GENERATOR, verdict=Verdict.REJECT, comment="revised"))
    rec = ledger.get("M-002")
    assert [(v.role, v.verdict) for v in rec.reviewers] == [(Role.GENERATOR, Verdict.ACCEPT),
                                                            (Role.GENERATOR, Verdict.REJECT)]
    with pytest.raises(LedgerError, match="own modification"):
        ledger.add_verdict("M-002", ReviewerVerdict(role=Role.OPTIMIZER, verdict=Verdict.ACCEPT))


def test_test_status_needs_a_candidate(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    rec = _build("M-001", M001_DIFF).model_copy(update={"candidate_version": None})
    ledger.add(rec)
    with pytest.raises(LedgerError, match="no candidate"):
        ledger.set_test_status("M-001", TestStatus.PASS)


def test_place_only_accepted_and_only_once(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    ledger.add(_build("M-001", M001_DIFF))
    with pytest.raises(LedgerError, match="only accepted"):
        ledger.place("M-001", "V1", "V1:L3-L4")
    ledger.set_test_status("M-001", TestStatus.PASS)
    ledger.accept("M-001", "tests pass")
    rec = ledger.place("M-001", "V1", "V1:L3-L4")
    assert rec.new_version == "V1" and rec.new_location == "V1:L3-L4"
    assert rec.hunks[0].new_location == "V0+M-001:L3-L4"  # candidate location is kept
    with pytest.raises(LedgerError, match="already integrated"):
        ledger.place("M-001", "V2", "V2:L3")


def test_unknown_id_and_by_round(tmp_path: Path) -> None:
    ledger = ModificationLedger(tmp_path)
    with pytest.raises(LedgerError, match="unknown"):
        ledger.get("M-001")
    ledger.add(_build("M-001", M001_DIFF))
    assert [r.id for r in ledger.by_round(1)] == ["M-001"] and ledger.by_round(2) == []


def test_duplicate_ids_in_file_are_an_error(tmp_path: Path) -> None:
    rec = _build("M-001", M001_DIFF).model_dump(mode="json")
    (tmp_path / "modifications.json").write_text(json.dumps([rec, rec]), encoding="utf-8")
    with pytest.raises(LedgerError, match="twice"):
        ModificationLedger(tmp_path)
