"""Modification records: building them from agent proposals and keeping the ledger.

The ledger (``runs/<run_id>/modifications.json``) holds every record ever created,
accepted, rejected and withdrawn alike. It is rewritten atomically on each change and never
loses a record. Once a record is decided its decision cannot change; the only later update
allowed is placing an accepted record in the round version it was integrated into.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from quadcode.diffs import ApplyResult, DiffError, apply_diff, parse_diff
from quadcode.linemap import new_location, old_location
from quadcode.schemas import (
    Decision,
    HunkChange,
    ModificationProposal,
    ModificationRecord,
    ReviewerVerdict,
    Role,
    TestStatus,
    utc_now,
)
from quadcode.versions import candidate_label

MODIFICATIONS_FILE = "modifications.json"
UNRESOLVED = "unresolved"
_HUNK_SEPARATOR = "\n...\n"
_ID_RE = re.compile(r"^M-(\d{3,})$")


class LedgerError(ValueError):
    pass


def format_id(n: int) -> str:
    return f"M-{n:03d}"


@dataclass(frozen=True)
class BuiltModification:
    record: ModificationRecord
    result: ApplyResult | None
    """None when the diff did not apply; the record is then already rejected."""


def _hunk_changes(base_label: str, cand_label: str, result: ApplyResult) -> list[HunkChange]:
    return [
        HunkChange(
            index=h.index,
            old_start=h.core_old_start, old_count=len(h.core_old_lines),
            new_start=h.core_new_start, new_count=len(h.core_new_lines),
            original_code="\n".join(h.core_old_lines),
            replacement_code="\n".join(h.core_new_lines),
            original_location=old_location(base_label, h),
            new_location=new_location(cand_label, h),
            offset=h.offset,
            whitespace_relaxed=h.whitespace_relaxed,
        )
        for h in result.hunks
    ]


def _stated_code(diff: str) -> tuple[str, str]:
    """Removed and added lines as the diff states them, for a diff that did not apply."""
    try:
        parsed = parse_diff(diff)
    except DiffError:
        return "", ""
    removed = _HUNK_SEPARATOR.join("\n".join(h.removed) for h in parsed.hunks)
    added = _HUNK_SEPARATOR.join("\n".join(h.added) for h in parsed.hunks)
    return removed, added


def build_record(proposal: ModificationProposal, *, modification_id: str, proposed_by: Role,
                 round_index: int, base_label: str, base_code: str) -> BuiltModification:
    """Apply a proposal's diff to its base and describe exactly what it changes.

    A diff that does not apply still produces a record, rejected with the reason, so that no
    proposal disappears from the history.
    """
    common = dict(id=modification_id, proposed_by=proposed_by, round_index=round_index,
                  base_version=base_label, diff=proposal.diff, justification=proposal.justification,
                  expected_impact=proposal.expected_impact, disadvantages=proposal.disadvantages)
    try:
        result = apply_diff(base_code, proposal.diff)
    except DiffError as exc:
        removed, added = _stated_code(proposal.diff)
        record = ModificationRecord(
            **common, original_code=removed, replacement_code=added, original_location=UNRESOLVED,
            apply_error=str(exc), test_status=TestStatus.NOT_RUN, decision=Decision.REJECTED,
            decision_reason=f"diff did not apply: {exc}", decided_at=utc_now())
        return BuiltModification(record, None)
    cand = candidate_label(base_label, modification_id)
    hunks = _hunk_changes(base_label, cand, result)
    record = ModificationRecord(
        **common, candidate_version=cand, hunks=hunks,
        original_code=_HUNK_SEPARATOR.join(h.original_code for h in hunks),
        replacement_code=_HUNK_SEPARATOR.join(h.replacement_code for h in hunks),
        original_location="; ".join(h.original_location for h in hunks),
        new_location="; ".join(h.new_location for h in hunks))
    return BuiltModification(record, result)


class ModificationLedger:
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / MODIFICATIONS_FILE
        run_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ModificationRecord] = {}
        if self.path.is_file():
            for raw in json.loads(self.path.read_text(encoding="utf-8")):
                rec = ModificationRecord.model_validate(raw)
                if rec.id in self._records:
                    raise LedgerError(f"{self.path} lists {rec.id} twice")
                self._records[rec.id] = rec

    # ------------------------------------------------------------ reading

    def get(self, modification_id: str) -> ModificationRecord:
        try:
            return self._records[modification_id]
        except KeyError:
            raise LedgerError(f"unknown modification {modification_id}") from None

    def all(self) -> list[ModificationRecord]:
        return sorted(self._records.values(), key=lambda r: int(r.id[2:]))

    def by_round(self, round_index: int) -> list[ModificationRecord]:
        return [r for r in self.all() if r.round_index == round_index]

    def next_id(self) -> str:
        numbers = [int(m.group(1)) for r in self._records if (m := _ID_RE.match(r))]
        return format_id(max(numbers, default=0) + 1)

    # ------------------------------------------------------------ writing

    def _save(self) -> None:
        data = [r.model_dump(mode="json") for r in self.all()]
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def add(self, record: ModificationRecord) -> ModificationRecord:
        """Add a new record. On resume, adding an identical record again is a no-op."""
        existing = self._records.get(record.id)
        if existing is not None:
            if existing.model_dump(exclude={"created_at", "decided_at"}) == record.model_dump(
                    exclude={"created_at", "decided_at"}):
                return existing
            raise LedgerError(f"{record.id} already exists with different content")
        if record.decision is Decision.ACCEPTED or record.decision is Decision.WITHDRAWN:
            raise LedgerError(f"{record.id}: a new record can only be pending or rejected (unappliable diff)")
        if record.decision is Decision.REJECTED and not record.decision_reason:
            raise LedgerError(f"{record.id}: a rejection needs a reason")
        self._records[record.id] = record
        self._save()
        return record

    def _update(self, modification_id: str, check: Callable[[ModificationRecord], None],
                **changes: object) -> ModificationRecord:
        current = self.get(modification_id)
        check(current)
        updated = ModificationRecord.model_validate({**current.model_dump(), **changes})
        self._records[modification_id] = updated
        self._save()
        return updated

    @staticmethod
    def _require_pending(rec: ModificationRecord) -> None:
        if rec.decision is not Decision.PENDING:
            raise LedgerError(f"{rec.id} is already {rec.decision.value}; decided records do not change")

    def add_verdict(self, modification_id: str, verdict: ReviewerVerdict) -> ModificationRecord:
        """Append a reviewer's verdict. Earlier verdicts are kept, so revised opinions stay visible."""
        def check(rec: ModificationRecord) -> None:
            self._require_pending(rec)
            if verdict.role is rec.proposed_by:
                raise LedgerError(f"{rec.id}: the proposer cannot review its own modification")
        rec = self.get(modification_id)
        return self._update(modification_id, check, reviewers=[*rec.reviewers, verdict])

    def set_test_status(self, modification_id: str, status: TestStatus) -> ModificationRecord:
        def check(rec: ModificationRecord) -> None:
            self._require_pending(rec)
            if rec.candidate_version is None:
                raise LedgerError(f"{rec.id} has no candidate version to test")
        return self._update(modification_id, check, test_status=status)

    def accept(self, modification_id: str, reason: str) -> ModificationRecord:
        """Accept on evidence: the candidate must have passed its tests."""
        def check(rec: ModificationRecord) -> None:
            self._require_pending(rec)
            if rec.candidate_version is None or rec.test_status is not TestStatus.PASS:
                raise LedgerError(f"{rec.id} cannot be accepted: its candidate has test status "
                                  f"{rec.test_status.value}, acceptance needs pass")
            if not reason.strip():
                raise LedgerError(f"{rec.id}: acceptance needs a reason")
        return self._update(modification_id, check, decision=Decision.ACCEPTED, decision_reason=reason,
                            decided_at=utc_now())

    def reject(self, modification_id: str, reason: str) -> ModificationRecord:
        return self._close(modification_id, Decision.REJECTED, reason)

    def withdraw(self, modification_id: str, reason: str) -> ModificationRecord:
        """The proposing agent took the suggestion back. The record stays in the history."""
        return self._close(modification_id, Decision.WITHDRAWN, reason)

    def _close(self, modification_id: str, decision: Decision, reason: str) -> ModificationRecord:
        def check(rec: ModificationRecord) -> None:
            self._require_pending(rec)
            if not reason.strip():
                raise LedgerError(f"{rec.id}: {decision.value} needs a reason")
        return self._update(modification_id, check, decision=decision, decision_reason=reason,
                            decided_at=utc_now())

    def place(self, modification_id: str, new_version: str, location: str) -> ModificationRecord:
        """Record the round version an accepted modification was integrated into, and where."""
        def check(rec: ModificationRecord) -> None:
            if rec.decision is not Decision.ACCEPTED:
                raise LedgerError(f"{rec.id} is {rec.decision.value}; only accepted modifications are integrated")
            if rec.new_version is not None:
                raise LedgerError(f"{rec.id} was already integrated into {rec.new_version}")
        return self._update(modification_id, check, new_version=new_version, new_location=location)
