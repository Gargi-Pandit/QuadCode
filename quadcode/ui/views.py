"""Dashboard view models of one run: plain rows built only from the run's records.

Every function here reads a :class:`RunRecords` (loaded read-only by ``quadcode.report.data``) and
returns lists of dicts that Streamlit shows as tables. Column names say whether a number was
measured in the sandbox, is an agent's theoretical claim, or was reported by a provider. No
function computes a number that is not in the records, except sums and counts of logged values.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any

from quadcode.report.data import LoggedEvent, RunRecords, message_of
from quadcode.report.fmt import MEASURED, THEORETICAL, one_line
from quadcode.report.tables import final_label
from quadcode.schemas import ROLE_VERSION_LABEL, Role, TestStatus

Row = dict[str, Any]

SUMMARY_LIMIT = 200


# ---------------------------------------------------------------- overview


@dataclass(frozen=True)
class Overview:
    run_id: str
    status: str
    stop_reason: str
    problem_title: str | None
    rounds: str
    reference: str | None
    final_version: str | None
    final_file: str | None
    final_verified: bool | None
    accepted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    ablation: str = "full pipeline"
    mock: bool = False
    isolation_warning: str | None = None
    load_errors: list[str] = field(default_factory=list)


def overview(rec: RunRecords) -> Overview:
    s = rec.summary
    ref = rec.reference()
    reference = f"V0 = copy of {ref[0].winner}" if ref and ref[0].winner else None
    if s is None:
        started = rec.last("run_started")
        rounds = f"? of {started.payload.get('rounds')}" if started else "-"
        return Overview(run_id=rec.run_id, status="unfinished",
                        stop_reason="no summary.json: the run is still running, was stopped or crashed",
                        problem_title=spec_title(rec), rounds=rounds, reference=reference, final_version=None,
                        final_file=rec.final_file, final_verified=None, ablation=rec.ablation().description,
                        mock=rec.is_mock(), load_errors=list(rec.load_errors))
    return Overview(run_id=s.run_id, status=s.status.value, stop_reason=s.stop_reason,
                    problem_title=s.problem_title or spec_title(rec), rounds=f"{s.rounds_run} of {s.rounds_requested}",
                    reference=reference, final_version=s.final_version, final_file=s.final_file,
                    final_verified=s.final_verified, accepted=list(s.accepted), rejected=list(s.rejected),
                    unresolved=list(s.unresolved), ablation=s.ablation.description, mock=rec.is_mock(),
                    isolation_warning=s.isolation_warning, load_errors=list(rec.load_errors))


def spec_title(rec: RunRecords) -> str | None:
    found = rec.spec()
    return found[0].title if found else None


def slot_rows(rec: RunRecords) -> list[Row]:
    """The four agent slots with the provider/model that actually answered, from the run log."""
    rows: list[Row] = []
    for role in Role:
        mine = [m for m in rec.messages() if m.role is role]
        rows.append({"role": role.value, "slot": ROLE_VERSION_LABEL[role],
                     "provider/model": rec.agent_identity(role),
                     "answered": "excluded (ablation)" if rec.excluded(role) else sum(1 for m in mine if m.ok),
                     "unavailable": "-" if rec.excluded(role) else sum(1 for m in mine if not m.ok)})
    return rows


def selection_rows(rec: RunRecords) -> list[Row]:
    """How V0 was chosen: each criterion of the filter cascade, as logged in reference_selected."""
    found = rec.reference()
    if found is None:
        return []
    evidence, ev = found
    return [{"criterion": step.criterion, "basis": MEASURED if step.measured else "not measured",
             "kept": ", ".join(step.kept), "dropped": ", ".join(step.dropped) or "-",
             "values": ", ".join(f"{k}={v}" for k, v in step.values.items()) or "-", "note": step.note,
             "record": ev.ref} for step in evidence.steps]


# ---------------------------------------------------------------- timeline


def _payload_text(e: LoggedEvent) -> str:
    p = e.payload
    m = message_of(e)
    if m is not None:
        if not m.ok:
            return f"UNAVAILABLE ({m.provider}/{m.model}): {m.error}"
        return f"{m.call_type} answer from {m.provider}/{m.model}"
    match e.kind:
        case "suite_run":
            if not p.get("executed"):
                return f"{p.get('version')}: not executed ({p.get('not_executed_reason')})"
            failing = ", ".join(sorted(p.get("failing") or {})) or "none"
            return f"{p.get('version')}: {p.get('passed')}/{p.get('total')} tests passed; failing: {failing}"
        case "reference_selected":
            return f"V0 = {p.get('evidence', {}).get('winner')}"
        case "proposal_recorded":
            where = p.get("candidate") if not p.get("apply_error") else f"diff did not apply: {p.get('apply_error')}"
            return f"{p.get('id')} against {p.get('base')}: {where}"
        case "candidate_tested":
            return f"{p.get('id')}: tests {p.get('test_status')}; {p.get('comparison') or p.get('failing') or ''}"
        case "challenge_requested":
            return f"{p.get('id')}: asked {', '.join(p.get('challengers') or [])}: {p.get('reason')}"
        case "challenge_verdict":
            return f"{p.get('id')}: {p.get('verdict')}: {p.get('comment', '')}"
        case "modification_decided":
            return f"{p.get('id')} {p.get('decision')}: {p.get('reason')}"
        case "round_started":
            return f"round {p.get('round')} on {p.get('base')}"
        case "round_completed":
            return (f"round {p.get('round')}: accepted {', '.join(p.get('accepted') or []) or 'none'}, "
                    f"rejected {', '.join(p.get('rejected') or []) or 'none'}")
        case "run_finished":
            return f"{p.get('status')}: {p.get('stop_reason')}"
    keys = ("reason", "file", "version", "note")
    parts = [f"{k}={p[k]}" for k in keys if p.get(k) is not None]
    return "; ".join(parts) if parts else json.dumps(p, sort_keys=True, ensure_ascii=False)


def timeline_rows(rec: RunRecords, phases: set[str] | None = None, roles: set[str] | None = None,
                  kinds: set[str] | None = None) -> list[Row]:
    """One row per logged event, in log order, optionally filtered. ``roles`` may contain ``orchestrator``."""
    rows: list[Row] = []
    for e in rec.events:
        who = e.role.value if e.role is not None else "orchestrator"
        if phases and e.phase.value not in phases or roles and who not in roles or kinds and e.kind not in kinds:
            continue
        rows.append({"record": e.ref, "time": e.ts, "phase": e.phase.value, "from": rec.who(e.role),
                     "event": e.kind, "summary": one_line(_payload_text(e), SUMMARY_LIMIT)})
    return rows


def event_payload(rec: RunRecords, line: int) -> dict[str, Any] | None:
    """The full logged payload of the event on ``events.jsonl`` line ``line``."""
    e = next((e for e in rec.events if e.line == line), None)
    return None if e is None else e.event.model_dump(mode="json")


def timeline_filters(rec: RunRecords) -> tuple[list[str], list[str], list[str]]:
    """The phases, senders and event kinds that occur in the log, for filter widgets."""
    phases = list(dict.fromkeys(e.phase.value for e in rec.events))
    senders = list(dict.fromkeys(e.role.value if e.role else "orchestrator" for e in rec.events))
    kinds = sorted({e.kind for e in rec.events})
    return phases, senders, kinds


# ---------------------------------------------------------------- calls


def call_rows(rec: RunRecords) -> list[Row]:
    """Every line of calls.jsonl. Tokens are provider-reported unless ``tokens estimated`` is set."""
    return [{"time": c.ts, "role": c.role.value, "provider": c.provider, "model": c.model,
             "call": c.call_type.value, "tag": c.tag, "attempt": c.attempt, "status": c.status.value,
             "cache hit": c.cache_hit, "tokens in": c.tokens_in, "tokens out": c.tokens_out,
             "tokens estimated": c.tokens_estimated, f"latency s ({MEASURED})": round(c.latency_s, 6),
             "model choice": "first" if c.choice == 0 else f"fallback {c.choice}",
             "repair": c.repair, "finish": c.finish_reason or "-", "error": one_line(c.error, SUMMARY_LIMIT) or "-"}
            for c in rec.calls]


# ---------------------------------------------------------------- modifications


def modification_rows(rec: RunRecords) -> list[Row]:
    """Every modification record, accepted and rejected alike."""
    return [{"id": r.id, "round": r.round_index, "proposed by": f"{r.proposed_by.value} "
             f"({rec.agent_identity(r.proposed_by)})", "base": r.base_version,
             "candidate": r.candidate_version or "-", "integrated into": r.new_version or "-",
             f"tests ({MEASURED})": r.test_status.value, "decision": r.decision.value,
             "verdicts": ", ".join(f"{v.role.value}:{v.verdict.value}" for v in r.reviewers) or "-",
             "reason": one_line(r.decision_reason, SUMMARY_LIMIT)} for r in rec.modifications]


# ---------------------------------------------------------------- versions


def version_rows(rec: RunRecords) -> list[Row]:
    final = final_label(rec)
    rows: list[Row] = []
    for label, meta in rec.versions.items():
        rows.append({"version": label, "kind": meta.kind.value + (" (final)" if label == final else ""),
                     "parents": ", ".join(meta.parents) or "-",
                     "created by": meta.created_by.value if meta.created_by else "-",
                     "modifications": ", ".join(meta.modification_ids) or "-", "lines": meta.line_count,
                     "sha256": meta.sha256[:12], "note": meta.note or "-"})
    return rows


def version_diff(rec: RunRecords, old: str, new: str, context: int = 3) -> str:
    """Unified diff between two stored versions ("" if identical)."""
    a = rec.codes[old].splitlines(keepends=True)
    b = rec.codes[new].splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile=old, tofile=new, n=context))


def provenance_rows(rec: RunRecords, label: str) -> list[Row]:
    """Each line of a stored version with where it first appeared and who it is credited to.

    A line is credited to a modification only when a record inserted it (stored line origins)."""
    meta = rec.versions[label]
    lines = rec.codes[label].splitlines()
    rows: list[Row] = []
    for n, (text, origin) in enumerate(zip(lines, meta.line_origins, strict=False), start=1):
        rows.append({"line": n, "code": text, "first appeared": f"{origin.version}:L{origin.line}",
                     "agent": origin.role.value if origin.role else "-",
                     "modification": origin.modification_id or "-"})
    return rows


# ---------------------------------------------------------------- tests and benchmark


def initial_suite_rows(rec: RunRecords) -> list[Row]:
    """Phase 2.5: the suite run of each initial version, as stored in tests.json."""
    rows: list[Row] = []
    for label in ROLE_VERSION_LABEL.values():
        s = rec.initial_suite(label)
        if s is None:
            continue
        failing = [f"{r.test_id} {r.status.value}" for r in s.results if r.status is not TestStatus.PASS]
        rows.append({"version": label, "executed": s.executed, "backend": s.backend,
                     f"passed ({MEASURED})": s.count(TestStatus.PASS), "total": len(s.results),
                     "not passed": ", ".join(failing) or "-",
                     "not executed because": s.not_executed_reason or "-"})
    return rows


def benchmark_rows(rec: RunRecords) -> list[Row]:
    """Phase 7: per version and test, the measured repeated-run statistics from benchmark.json."""
    if rec.benchmark is None:
        return []
    return [{"version": st.version, "test": st.test_id, "status": st.status.value, "runs": st.repeats,
             f"median s ({MEASURED})": st.median_s, f"min s ({MEASURED})": st.min_s,
             f"max s ({MEASURED})": st.max_s,
             f"peak heap bytes ({MEASURED})": st.peak_memory_bytes if st.peak_memory_bytes is not None else "-"}
            for st in rec.benchmark.stats if st.measured]


def benchmark_tests(rec: RunRecords) -> list[str]:
    """Test ids with measured statistics, performance tests first."""
    if rec.benchmark is None:
        return []
    perf = {t.id for t in rec.suite or [] if t.category.value == "performance"}
    ids = list(dict.fromkeys(st.test_id for st in rec.benchmark.stats if st.measured))
    return sorted(ids, key=lambda t: (t not in perf, ids.index(t)))


def runtime_chart_rows(rec: RunRecords, test_id: str) -> list[Row]:
    """Measured median runtime of one test for each version that passed it, in execution order."""
    if rec.benchmark is None:
        return []
    return [{"version": st.version, "median s (measured)": st.median_s}
            for st in rec.benchmark.stats if st.test_id == test_id and st.measured
            and st.status is TestStatus.PASS]


def theoretical_rows(rec: RunRecords) -> list[Row]:
    """The complexity each agent claimed for its initial version. Never measured."""
    rows: list[Row] = []
    for role, label in ROLE_VERSION_LABEL.items():
        found = rec.initial(role)
        if found is None:
            continue
        sol, _ = found
        rows.append({"version": label, "agent": f"{role.value} ({rec.agent_identity(role)})",
                     f"time ({THEORETICAL})": sol.time_complexity, f"space ({THEORETICAL})": sol.space_complexity,
                     "algorithm": one_line(sol.algorithm, SUMMARY_LIMIT)})
    return rows
