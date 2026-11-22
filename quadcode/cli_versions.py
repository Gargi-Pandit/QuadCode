"""CLI commands for diffs and stored versions: ``apply-diff`` and ``versions``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from quadcode.cli_sandbox import print_suite, sandbox_config
from quadcode.config import ConfigError
from quadcode.diffs import ApplyResult, DiffError, apply_diff
from quadcode.linemap import new_location, old_location
from quadcode.modifications import LedgerError, ModificationLedger, build_record
from quadcode.sandbox import SandboxUnavailable, select_sandbox
from quadcode.schemas import ROLE_VERSION_LABEL, Decision, ModificationProposal, Role, SuiteRun, TestCase
from quadcode.tables import format_table
from quadcode.testing import load_test_cases, run_suite, suite_status
from quadcode.versions import VersionError, VersionStore

PATCHED = "patched"


def _print_hunks(base_label: str, new_label: str, result: ApplyResult) -> None:
    rows = []
    for h in result.hunks:
        offset = "-" if h.offset is None else f"{h.offset:+d}"
        rows.append([str(h.index + 1), str(h.stated_old_start or "-"), str(h.old_start), offset,
                     old_location(base_label, h), new_location(new_label, h),
                     "yes" if h.whitespace_relaxed else "no"])
    print(format_table(["hunk", "stated_line", "applied_at", "offset", "original", "new", "ws_relaxed"], rows))


def _run_tests(args: argparse.Namespace, code: str, label: str) -> SuiteRun:
    cfg = sandbox_config(args)
    tests: list[TestCase] = load_test_cases(args.tests)
    sandbox, reason = select_sandbox(cfg)
    print(f"\nsandbox: {sandbox.name} ({reason})\n")
    suite = run_suite(code=code, entry_point=args.entry, language=args.language, tests=tests,
                      sandbox=sandbox, cfg=cfg, version=label, repeats=args.repeats)
    print_suite(suite, label, tests)
    return suite


def _base_label(args: argparse.Namespace) -> str:
    label = args.base_label or Path(args.base).stem.upper()
    if label not in ROLE_VERSION_LABEL.values():
        raise VersionError(f"--base-label must be G1, C1, P1 or D1 when storing to a run directory, got {label!r}")
    return label


def _cmd_apply_diff(args: argparse.Namespace) -> int:
    if args.tests and not args.entry:
        print("--entry is required with --tests", file=sys.stderr)
        return 2
    if args.run_dir and not args.proposed_by:
        print("--proposed-by is required with --run-dir", file=sys.stderr)
        return 2
    try:
        base_code = Path(args.base).read_text(encoding="utf-8")
        diff_text = Path(args.diff).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read input: {exc}", file=sys.stderr)
        return 2

    try:
        if args.run_dir:
            return _apply_and_store(args, base_code, diff_text)
        return _apply_only(args, base_code, diff_text)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except VersionError as exc:
        print(f"version store: {exc}", file=sys.stderr)
        return 2
    except LedgerError as exc:
        print(f"modification ledger: {exc}", file=sys.stderr)
        return 2
    except SandboxUnavailable as exc:
        print(f"sandbox unavailable: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, ValidationError) as exc:
        print(f"cannot read tests: {exc}", file=sys.stderr)
        return 2


def _apply_only(args: argparse.Namespace, base_code: str, diff_text: str) -> int:
    base_label = args.base_label or Path(args.base).stem
    try:
        result = apply_diff(base_code, diff_text)
    except DiffError as exc:
        print(f"diff did not apply: {exc}")
        return 1
    print(f"diff applied to {base_label} ({len(result.hunks)} hunk(s)); locations: "
          f"'{base_label}' = base, '{PATCHED}' = result\n")
    _print_hunks(base_label, PATCHED, result)
    print(f"\n--- {PATCHED} ---\n{result.text}", end="" if result.text.endswith("\n") else "\n")
    ok = True
    payload: dict[str, Any] = {"applied": True, "text": result.text,
                               "hunks": [asdict(h) for h in result.hunks]}
    if args.tests:
        suite = _run_tests(args, result.text, PATCHED)
        ok = suite.all_passed
        payload["suite"] = suite.model_dump(mode="json")
    _write_json(args, payload)
    return 0 if ok else 1


def _apply_and_store(args: argparse.Namespace, base_code: str, diff_text: str) -> int:
    base_label = _base_label(args)
    run_dir = Path(args.run_dir)
    store = VersionStore(run_dir)
    ledger = ModificationLedger(run_dir)
    base = store.save_initial(base_label, base_code, args.language, note=f"imported from {args.base}")
    role = Role(args.proposed_by)
    proposal = ModificationProposal(diff=diff_text, justification=args.justification,
                                    expected_impact=args.expected_impact)
    built = build_record(proposal, modification_id=ledger.next_id(), proposed_by=role,
                         round_index=args.round, base_label=base.label, base_code=base.code)
    record = ledger.add(built.record)
    print(f"stored {base.label} in {store.root}")
    if built.result is None:
        print(f"{record.id}: diff did not apply: {record.apply_error}")
        print(f"{record.id} recorded as {record.decision.value} in {ledger.path}")
        _write_json(args, record.model_dump(mode="json"))
        return 1
    cand = store.save_candidate(base.label, record.id, role, built.result, note=f"diff from {args.diff}")
    print(f"{record.id} by {role.value}: stored candidate {cand.label}\n")
    _print_hunks(base.label, cand.label, built.result)
    ok = True
    if args.tests:
        suite = _run_tests(args, cand.code, cand.label)
        ok = suite.all_passed
        record = ledger.set_test_status(record.id, suite_status(suite))
    print(f"{record.id}: test status {record.test_status.value}, decision {record.decision.value} "
          f"(this command never accepts or rejects a working diff)")
    print(f"ledger: {ledger.path}")
    _write_json(args, record.model_dump(mode="json"))
    return 0 if ok else 1


def _write_json(args: argparse.Namespace, payload: Any) -> None:
    if not args.json:
        return
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _cmd_versions(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not (run_dir / "versions").is_dir():
        print(f"no versions directory in {run_dir}", file=sys.stderr)
        return 2
    store = VersionStore(run_dir)
    try:
        if args.blame:
            return _print_blame(store, args.blame)
        metas = store.list()
    except VersionError as exc:
        print(f"version store: {exc}", file=sys.stderr)
        return 2
    rows = [[m.label, m.kind.value, ",".join(m.parents) or "-", m.created_by.value if m.created_by else "-",
             ",".join(m.modification_ids) or "-", str(m.line_count), m.sha256[:12], m.created_at]
            for m in metas]
    print(format_table(["version", "kind", "parents", "by", "modifications", "lines", "sha256", "created"], rows))
    ledger = ModificationLedger(run_dir)
    records = ledger.all()
    if records:
        print()
        mrows = [[r.id, r.proposed_by.value, r.base_version, r.candidate_version or "-", r.new_version or "-",
                  r.original_location, r.test_status.value, r.decision.value, _clip(r.decision_reason)]
                 for r in records]
        print(format_table(["id", "by", "base", "candidate", "integrated", "original", "tests", "decision",
                            "reason"], mrows))
        pending = sum(1 for r in records if r.decision is Decision.PENDING)
        print(f"\n{len(records)} modification record(s), {pending} pending")
    return 0


def _clip(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _print_blame(store: VersionStore, label: str) -> int:
    lineage = " -> ".join(store.lineage(label))
    print(f"{label}  (lineage: {lineage})")
    rows = []
    for n, text, origin in store.blame(label):
        if origin.modification_id:
            who = f"{origin.modification_id} {origin.role.value if origin.role else '?'}"
        else:
            who = f"{origin.version}:L{origin.line} {origin.role.value if origin.role else '-'}"
        rows.append([str(n), who, text])
    print(format_table(["line", "origin", "code"], rows))
    return 0


def add_version_commands(sub: Any) -> None:
    ad = sub.add_parser("apply-diff", help="apply a unified diff to a file, show where each hunk landed, "
                                           "optionally test the result and store it as a candidate version")
    ad.add_argument("base", help="file the diff was written against")
    ad.add_argument("diff", help="unified diff file")
    ad.add_argument("--base-label", default=None, help="version label of the base (default: file stem; "
                    "with --run-dir it must be G1, C1, P1 or D1)")
    ad.add_argument("--tests", default=None, help="JSON test suite to run against the patched code")
    ad.add_argument("--entry", default=None, help="function to call (required with --tests)")
    ad.add_argument("--language", default="python")
    ad.add_argument("--repeats", type=int, default=None, help="timed runs per test")
    ad.add_argument("--test-timeout", type=float, default=None, help="per-call limit in seconds")
    ad.add_argument("--backend", choices=["auto", "docker", "subprocess"], help="override sandbox.backend")
    ad.add_argument("--run-dir", default=None, help="store the base, the candidate and a modification record here")
    ad.add_argument("--proposed-by", choices=[r.value for r in Role], default=None,
                    help="agent slot credited with the diff (required with --run-dir)")
    ad.add_argument("--round", type=int, default=1, help="round index for the modification record")
    ad.add_argument("--justification", default="supplied on the command line")
    ad.add_argument("--expected-impact", default="not stated")
    ad.add_argument("--json", default=None, help="write the result as JSON to this path")
    ad.set_defaults(func=_cmd_apply_diff)

    vs = sub.add_parser("versions", help="list the stored versions and modification records of a run")
    vs.add_argument("run_dir", help="run directory, e.g. runs/<run_id>")
    vs.add_argument("--blame", default=None, metavar="LABEL", help="show the origin of every line of a version")
    vs.set_defaults(func=_cmd_versions)
