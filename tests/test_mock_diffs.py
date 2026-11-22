"""The mock scenario's round-1 diffs, applied for real and tested in the sandbox."""

from __future__ import annotations

from pathlib import Path

from quadcode.config import SandboxConfig
from quadcode.linemap import compose
from quadcode.modifications import ModificationLedger, build_record
from quadcode.providers.mock_scenarios import G1_CODE, M001_DIFF, M002_DIFF, get_scenario
from quadcode.sandbox import SubprocessSandbox
from quadcode.schemas import CallType, ModificationProposal, ModifyOutput, Role, TestCase, TestStatus
from quadcode.testing import run_suite, suite_status
from quadcode.versions import VersionStore

from .conftest import SAMPLES


def _proposals(role: Role) -> list[ModificationProposal]:
    text = get_scenario("two_sum").responses[(role, CallType.MODIFY, "round-1")]
    return ModifyOutput.model_validate_json(text).proposals


def test_sample_diffs_match_the_mock_scenario() -> None:
    assert (SAMPLES / "diffs" / "m001_validation.diff").read_text(encoding="utf-8") == M001_DIFF
    assert (SAMPLES / "diffs" / "m002_skip_self.diff").read_text(encoding="utf-8") == M002_DIFF
    assert (SAMPLES / "g1.py").read_text(encoding="utf-8") == G1_CODE


def test_mock_round_one_candidates_behave_as_scripted(
        tmp_path: Path, subprocess_sandbox: SubprocessSandbox, sandbox_cfg: SandboxConfig,
        two_sum_tests: list[TestCase]) -> None:
    store = VersionStore(tmp_path)
    ledger = ModificationLedger(tmp_path)
    store.save_initial("G1", G1_CODE, "python")
    v0 = store.save_reference("G1")

    statuses: dict[str, dict[str, TestStatus]] = {}
    for role in (Role.REVIEWER, Role.OPTIMIZER):
        (proposal,) = _proposals(role)
        built = build_record(proposal, modification_id=ledger.next_id(), proposed_by=role, round_index=1,
                             base_label="V0", base_code=v0.code)
        assert built.result is not None, built.record.apply_error
        rec = ledger.add(built.record)
        cand = store.save_candidate("V0", rec.id, role, built.result)
        suite = run_suite(code=cand.code, entry_point="two_sum", language="python", tests=two_sum_tests,
                          sandbox=subprocess_sandbox, cfg=sandbox_cfg, version=cand.label, repeats=1)
        ledger.set_test_status(rec.id, suite_status(suite))
        statuses[rec.id] = {r.test_id: r.status for r in suite.results}

    assert set(statuses["M-001"].values()) == {TestStatus.PASS}
    assert statuses["M-002"]["T03"] is TestStatus.FAIL
    assert {t for t, s in statuses["M-002"].items() if s is not TestStatus.PASS} == {"T03"}
    assert ledger.get("M-001").test_status is TestStatus.PASS
    assert ledger.get("M-002").test_status is TestStatus.FAIL

    ledger.accept("M-001", "all tests pass on V0+M-001")
    ledger.reject("M-002", "T03 fails on V0+M-002")
    composed = compose(v0.code, v0.meta.line_origins, "V1", [("M-001", Role.REVIEWER, M001_DIFF)])
    v1 = store.save_round("V1", "V0", composed)
    ledger.place("M-001", "V1", composed.location_of("M-001", "V1"))
    assert v1.code == store.load("V0+M-001").code
    assert ledger.get("M-001").new_location == "V1:L3-L4"
    blame = {n: o for n, _, o in store.blame("V1")}
    assert {n for n, o in blame.items() if o.modification_id} == {3, 4}
    assert all(o.role is Role.GENERATOR for n, o in blame.items() if n not in (3, 4))
