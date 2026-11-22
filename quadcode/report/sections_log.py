"""Section 4: the agent communication log, built from the logged messages only."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from quadcode.report.data import AgentMessage, LoggedEvent, RunRecords, message_of
from quadcode.report.doc import Block, Heading, Note, Para, source, table
from quadcode.report.fmt import listing, not_reached, one_line
from quadcode.schemas import ROLE_VERSION_LABEL, Decision, Role, Verdict

_LOG_KINDS = ("agent_output", "agent_unavailable", "proposal_recorded", "challenge_requested", "challenge_verdict",
              "modification_decided", "clarification_needed", "early_stop", "phase_skipped")


def _review_item_text(item: dict[str, Any]) -> str:
    parts = [f"{name}: {listing(item.get(key) or [])}" for name, key in
             [("strengths", "strengths"), ("weaknesses", "weaknesses"), ("bugs", "bugs"),
              ("suggestions", "suggestions")] if item.get(key)]
    if item.get("justification"):
        parts.append(f"justification: {item['justification']}")
    return "\n".join(parts) or "(no findings)"


def _message_rows(m: AgentMessage) -> list[tuple[str, str, str]]:
    """(to/about, kind, message) rows for one agent message."""
    if not m.ok:
        return [("-", f"{m.call_type} UNAVAILABLE", m.error or "")]
    data = m.data or {}
    match m.call_type:
        case "analysis":
            return [("all", "specification", f"'{data.get('title', '')}': {data.get('summary', '')} "
                     f"(full specification in section 1)")]
        case "initial":
            label = ROLE_VERSION_LABEL[m.role]
            tests = f"; {len(data.get('tests') or [])} test cases" if m.role is Role.VALIDATOR else ""
            return [(label, "initial implementation",
                     f"{data.get('algorithm', '')}; time {data.get('time_complexity', '')}, space "
                     f"{data.get('space_complexity', '')} (claimed){tests}; full code in section 2")]
        case "review":
            return [(str(item.get("target", "?")), "review", _review_item_text(item))
                    for item in data.get("items") or []] or [("-", "review", "(no review items)")]
        case "modify":
            proposals = data.get("proposals") or []
            if not proposals:
                return [("current version", "no proposal", "proposed no modification this round")]
            return [("current version", "proposal", f"{p.get('justification', '')}\nexpected impact: "
                     f"{p.get('expected_impact', '')}" + (f"\ndisadvantages: {p['disadvantages']}"
                                                         if p.get("disadvantages") else ""))
                    for p in proposals]
    return [("-", m.call_type, one_line(str(data), 300))]


def _event_rows(rec: RunRecords, e: LoggedEvent) -> list[tuple[str, str, str, str]]:
    """(from, to/about, kind, message) for one logged event of the communication log."""
    p = e.payload
    match e.kind:
        case "agent_output" | "agent_unavailable":
            if p.get("call_type") == "challenge":
                return []  # shown through the challenge_verdict events, which also cover unavailability
            m = message_of(e)
            return [(m.who, to, kind, text) for to, kind, text in _message_rows(m)] if m else []
        case "proposal_recorded":
            where = (f"applies at {p.get('original_location')}; candidate {p.get('candidate')}"
                     if not p.get("apply_error") else f"diff did not apply: {p.get('apply_error')}")
            return [("orchestrator", str(p.get("id")), "proposal recorded",
                     f"proposal by {rec.who(e.role)} recorded as {p.get('id')} against {p.get('base')}; {where}")]
        case "challenge_requested":
            return [("orchestrator", ", ".join(p.get("challengers") or []), "challenge request",
                     f"asked to judge {p.get('id')} by {e.role.value if e.role else '?'}: {p.get('reason')}")]
        case "challenge_verdict":
            verdict = str(p.get("verdict"))
            label = "challenge verdict UNAVAILABLE" if verdict == Verdict.UNAVAILABLE.value else "challenge verdict"
            return [(rec.who(e.role), str(p.get("id")), label, f"{verdict}: {p.get('comment', '')}")]
        case "modification_decided":
            verdicts = ", ".join(f"{v['role']}:{v['verdict']}" for v in p.get("verdicts") or []) or "none"
            return [("orchestrator", str(p.get("id")), f"decision: {p.get('decision')}",
                     f"{p.get('reason')}\n(challenge verdicts: {verdicts}; decided on test evidence, not by vote)")]
        case "clarification_needed":
            return [(rec.who(e.role), "user", "clarification needed", "\n".join(p.get("questions") or []))]
        case "early_stop":
            return [("orchestrator", "all", "early stop", str(p.get("reason")))]
        case "phase_skipped":
            return [("orchestrator", "all", f"phase {e.phase.value} skipped", str(p.get("reason")))]
    return []


def _side_by_side(rec: RunRecords) -> list[Block]:
    by_target: dict[str, list[list[str]]] = defaultdict(list)
    for role in Role:
        found = rec.review(role)
        if found is None:
            continue
        review, msg = found
        for item in review.items:
            by_target[item.target].append([rec.who(role), listing(item.strengths), listing(item.weaknesses),
                                           listing(item.bugs), listing(item.suggestions),
                                           item.justification or "-", msg.logged.ref])
    unavailable = [m for m in rec.messages("review") if not m.ok and rec.review(m.role) is None]
    blocks: list[Block] = []
    for target in sorted(by_target, key=lambda t: list(ROLE_VERSION_LABEL.values()).index(t)
                         if t in ROLE_VERSION_LABEL.values() else 99):
        rows = by_target[target] + [[m.who, "UNAVAILABLE: no review", "-", "-", "-", "-", m.logged.ref]
                                    for m in unavailable]
        blocks.append(table(["reviewer", "strengths", "weaknesses", "bugs", "suggestions", "justification",
                             "record"], rows, caption=f"Reviews of {target}"))
    return blocks


def _disagreements(rec: RunRecords) -> list[Block]:
    rows: list[list[str]] = []
    for r in rec.modifications:
        given = [v for v in r.reviewers if v.verdict is not Verdict.UNAVAILABLE]
        distinct = {v.verdict for v in given}
        split = len(distinct) > 1
        overruled = [v for v in given if (v.verdict is Verdict.ACCEPT and r.decision is Decision.REJECTED)
                     or (v.verdict is Verdict.REJECT and r.decision is Decision.ACCEPTED)]
        if not split and not overruled:
            continue
        kind = "split verdicts" if split else "verdict differs from decision"
        rows.append([r.id, f"{r.proposed_by.value} (proposer)", kind,
                     "\n".join(f"{v.role.value}: {v.verdict.value}: {v.comment}" for v in r.reviewers),
                     f"{r.decision.value}: {r.decision_reason}"])
    blocks: list[Block] = [Heading(3, "Disagreements")]
    if rows:
        blocks.append(table(["modification", "proposed by", "kind", "verdicts and arguments", "outcome"], rows))
        blocks.append(Para("Disagreements are kept as they were logged. The orchestrator decided each case on "
                           "test results and measurements; verdicts never decide."))
    else:
        blocks.append(Para("No split or overruled challenge verdicts were recorded."))
    blocks.append(Note("Only challenge verdicts are checked for disagreement automatically. Disagreement inside "
                       "free-text reviews is not detected by meaning; compare the reviews side by side above."))
    return blocks


def section_communication(rec: RunRecords) -> list[Block]:
    blocks: list[Block] = [Heading(2, "4. Agent communication log")]
    events = rec.of_kind(*_LOG_KINDS)
    rows: list[list[str]] = []
    for e in events:
        for sender, to, kind, text in _event_rows(rec, e):
            rows.append([str(len(rows) + 1), e.ts, e.phase.value, sender, to, kind, text, e.ref])
    if not rows:
        return blocks + not_reached(rec, "any agent communication")
    blocks.append(Para("Every message in the order it was logged. Agents never talked to each other directly: "
                       "the orchestrator sent each one the material listed in the spec and recorded the answers. "
                       "Code is not repeated here (see sections 2, 5 and 8)."))
    blocks.append(table(["#", "time", "phase", "from", "to / about", "kind", "message", "record"], rows))
    side = _side_by_side(rec)
    if side:
        blocks.append(Heading(3, "Reviews side by side"))
        blocks.extend(side)
    blocks.extend(_disagreements(rec))
    blocks.append(source("events.jsonl", "modifications.json"))
    return blocks
