"""Sections 5 and 7: the optimization rounds and the complete modification history."""

from __future__ import annotations

from quadcode.linemap import format_lines
from quadcode.report.data import LoggedEvent, RunRecords
from quadcode.report.doc import Block, Code, Heading, Note, Para, source, table
from quadcode.report.fmt import MEASURED, listing, not_reached, one_line
from quadcode.report.tables import final_label
from quadcode.schemas import Decision, ModificationRecord, Phase


def _language(rec: RunRecords) -> str:
    found = rec.spec()
    return found[0].language if found else ""


def _round_events(rec: RunRecords, index: int) -> list[LoggedEvent]:
    """Events of round ``index``: from its first round_started up to the next round's start."""
    starts = [e for e in rec.of_kind("round_started") if e.payload.get("round") == index]
    if not starts:
        return []
    first = starts[0].line
    later = [e.line for e in rec.of_kind("round_started") if e.payload.get("round", 0) > index]
    end = min(later) if later else None
    return [e for e in rec.events if e.line >= first and (end is None or e.line < end)
            and e.phase is Phase.ROUNDS]


def _record_blocks(rec: RunRecords, r: ModificationRecord, events: list[LoggedEvent]) -> list[Block]:
    blocks: list[Block] = [Heading(4, f"{r.id} by {rec.who(r.proposed_by)}")]
    tested = next((e for e in events if e.kind == "candidate_tested" and e.payload.get("id") == r.id), None)
    rows = [["base version", r.base_version], ["candidate version", r.candidate_version or "none (not applied)"],
            ["original location", r.original_location], ["new location", r.new_location or "-"],
            ["technical reasoning", r.justification], ["expected impact (agent's claim)", r.expected_impact],
            ["disadvantages (agent's own)", r.disadvantages or "-"],
            [f"test status ({MEASURED})", r.test_status.value]]
    if tested is not None:
        p = tested.payload
        if p.get("failing"):
            rows.append([f"failing tests ({MEASURED})", str(p["failing"])])
        rows.append([f"comparison with {r.base_version} ({MEASURED})", str(p.get("comparison"))])
        if p.get("noop"):
            rows.append(["executable change", "none: the code is AST-identical to the base"])
    rows.append(["peer review (challenge verdicts)",
                 "\n".join(f"{v.role.value}: {v.verdict.value}: {v.comment}" for v in r.reviewers)
                 or "none requested (not failing and not contested)"])
    rows.append(["decision", f"{r.decision.value}: {r.decision_reason}"])
    rows.append(["resulting version", r.new_version or "-"])
    blocks.append(table(["field", "value"], rows))
    if r.apply_error:
        blocks.append(Note(f"The diff did not apply to {r.base_version}: {r.apply_error}. Nothing was executed "
                           "for this proposal.", "warning"))
    lang = _language(rec)
    if r.original_code:
        blocks.append(Code(r.original_code, lang, f"Original code ({r.original_location})"))
    elif r.replacement_code:
        blocks.append(Para(f"Original code: none; this is a pure insertion ({r.original_location})."))
    if r.replacement_code:
        blocks.append(Code(r.replacement_code, lang, f"Proposed replacement ({r.new_location or 'unresolved'})"))
    elif r.original_code:
        blocks.append(Para("Proposed replacement: none; the original code is deleted."))
    blocks.append(Code(r.diff, "diff", "Diff as submitted"))
    refs = [e.ref for e in events if e.payload.get("id") == r.id]
    blocks.append(source(f"modifications.json {r.id}", refs))
    return blocks


def _round_blocks(rec: RunRecords, index: int) -> list[Block]:
    events = _round_events(rec, index)
    start = events[0]
    base = str(start.payload.get("base"))
    blocks: list[Block] = [Heading(3, f"Round {index}")]
    base_run = next((e for e in events if e.kind == "suite_run" and e.payload.get("version") == base), None)
    base_text = (f"{base_run.payload.get('passed')}/{base_run.payload.get('total')} tests passed ({MEASURED})"
                 if base_run is not None and base_run.payload.get("executed") else "suite not executed")
    blocks.append(Para(f"Current implementation: {base}; {base_text}."))
    answers: list[list[str]] = []
    for e in events:
        if e.kind not in ("agent_output", "agent_unavailable") or e.payload.get("call_type") != "modify":
            continue
        who = rec.who(e.role)
        if e.kind == "agent_unavailable":
            answers.append([who, "UNAVAILABLE", str(e.payload.get("error")), e.ref])
        else:
            n = len((e.payload.get("data") or {}).get("proposals") or [])
            answers.append([who, f"{n} proposal(s)" if n else "no proposal", "-", e.ref])
    blocks.append(table(["agent", "answer", "error", "record"], answers, caption="Modification calls"))
    records = [r for r in rec.modifications if r.round_index == index]
    for r in records:
        blocks.extend(_record_blocks(rec, r, events))
    trials = [e for e in events if e.kind == "composition_trial"]
    if trials:
        blocks.append(table(["trial version", "modifications", f"test status ({MEASURED})", "record"],
                            [[str(e.payload.get("version")), ", ".join(e.payload.get("modifications") or []),
                              str(e.payload.get("test_status")), e.ref] for e in trials],
                            caption="Composition trials (passing proposals tested together)"))
    created = next((e for e in events if e.kind == "round_version_created"), None)
    completed = next((e for e in events if e.kind == "round_completed"), None)
    if created is not None:
        mods = ", ".join(created.payload.get("modifications") or [])
        blocks.append(Para(f"Resulting version: {created.payload.get('version')} = {base} with {mods}."))
        blocks.append(Code(str(created.payload.get("diff", "")), "diff",
                           f"{base} -> {created.payload.get('version')}"))
    elif completed is not None:
        blocks.append(Para(f"No modification was accepted; {base} stays the current version."))
    else:
        blocks.append(Note("This round has no completion record: the run stopped inside it.", "warning"))
    if completed is not None and completed.payload.get("unavailable"):
        blocks.append(Note("Unavailable in this round: " + ", ".join(completed.payload["unavailable"]), "warning"))
    stop = next((e for e in events if e.kind == "early_stop"), None)
    if stop is not None:
        blocks.append(Note(f"Rounds stopped: {stop.payload.get('reason')}"))
    blocks.append(source([e.ref for e in events if e.kind in ("round_started", "round_version_created",
                                                               "round_completed", "early_stop")]))
    return blocks


def section_rounds(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "5. Iterative optimization process")]
    indices = sorted({int(e.payload.get("round", 0)) for e in rec.of_kind("round_started")})
    if not indices:
        return blocks + not_reached(rec, "the optimization rounds")
    s = rec.summary
    if s is not None:
        blocks.append(Para(f"Rounds requested: {s.rounds_requested}; rounds run: {s.rounds_run}. "
                           f"Stop reason: {s.stop_reason or 'not recorded'}."))
    blocks.append(Para("Each agent was asked once per round for modifications as unified diffs against the current "
                       "version. Every applied diff was tested in the sandbox. The orchestrator accepted a change "
                       "only on test evidence and measurements; challenge verdicts are recorded but do not decide."))
    for index in indices:
        blocks.extend(_round_blocks(rec, index))
    return blocks


# ---------------------------------------------------------------- 7


def _final_lines(rec: RunRecords, modification_id: str) -> str:
    label = final_label(rec)
    meta = rec.versions.get(label) if label else None
    if meta is None:
        return "-"
    lines = [n for n, o in enumerate(meta.line_origins, start=1) if o.modification_id == modification_id]
    return format_lines(label or "", lines) if lines else f"{label}: no line left"


def section_history(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "7. Complete modification history")]
    if not rec.modifications:
        if rec.of_kind("round_started"):
            blocks.append(Para("No agent proposed a modification, so the history is empty."))
            return blocks + [source("modifications.json")]
        return blocks + not_reached(rec, "the optimization rounds, so no modification exists")
    rows = [[r.id, str(r.round_index), rec.who(r.proposed_by), r.base_version, r.candidate_version or "-",
             r.new_version or "-", one_line(r.justification, 160), r.test_status.value,
             r.decision.value.upper(), one_line(r.decision_reason, 220)] for r in rec.modifications]
    blocks.append(table(["id", "round", "agent", "original version", "candidate", "new version", "modification",
                         f"tests ({MEASURED})", "status", "reason"], rows,
                        caption="All modifications, accepted and rejected, in order"))
    accepted = [r for r in rec.modifications if r.decision is Decision.ACCEPTED]
    if accepted:
        blocks.append(Heading(3, "Exact locations of accepted changes"))
        hunk_rows = [[r.id, str(h.index + 1), h.original_location, h.new_location,
                      "yes" if h.whitespace_relaxed else "no"] for r in accepted for h in r.hunks]
        blocks.append(table(["id", "hunk", "original location", "location in candidate", "whitespace relaxed"],
                            hunk_rows))
        final = final_label(rec)
        blocks.append(table(["id", "agent", "integrated into", "location there", f"lines in final version {final}"],
                            [[r.id, r.proposed_by.value, r.new_version or "-", r.new_location or "-",
                              _final_lines(rec, r.id)] for r in accepted]))
        blocks.append(Para("Locations name the version they refer to. Lines in the final version come from the "
                           "stored per-line origins, so a line is credited to a modification only if that "
                           "modification's record inserted it."))
    rejected = [r for r in rec.modifications if r.decision is Decision.REJECTED]
    if rejected:
        blocks.append(Para("Rejected: " + listing([f"{r.id} ({r.proposed_by.value})" for r in rejected])
                           + ". Their records are kept unchanged above and in modifications.json."))
    pending = [r.id for r in rec.modifications if r.decision is Decision.PENDING]
    if pending:
        blocks.append(Note("Not decided because the run stopped first: " + ", ".join(pending), "warning"))
    blocks.append(source("modifications.json", "versions/*.json (line origins)"))
    return blocks
