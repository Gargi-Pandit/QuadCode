"""Cross-cutting report tables: agent availability, tokens and cost, contributions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from quadcode.config import AppConfig
from quadcode.report.data import RunRecords
from quadcode.report.doc import Block, Note, Para, Table, source, table
from quadcode.report.fmt import MEASURED, one_line
from quadcode.schemas import ROLE_VERSION_LABEL, CallRecord, CallStatus, Decision, Role, Verdict

_ATTEMPT_STATUSES = {CallStatus.OK, CallStatus.PROVIDER_ERROR, CallStatus.INVALID_OUTPUT}


# ---------------------------------------------------------------- availability


def availability_blocks(rec: RunRecords) -> list[Block]:
    msgs = rec.messages()
    if not msgs:
        return [Para("No agent call finished, so there is no availability record.")]
    rows = [[m.logged.ts, m.role.value, ROLE_VERSION_LABEL[m.role], f"{m.provider}/{m.model}", m.call_type, m.tag,
             "UNAVAILABLE" if not m.ok else m.logged.payload.get("status", "done"), one_line(m.error, 160) or "-",
             m.logged.ref] for m in msgs]
    per_role: list[list[str]] = []
    for role in Role:
        mine = [m for m in msgs if m.role is role]
        bad = [m.tag for m in mine if not m.ok]
        per_role.append([role.value, rec.agent_identity(role), str(sum(1 for m in mine if m.ok)), str(len(bad)),
                         ", ".join(bad) or "-"])
    return [
        table(["time", "role", "slot", "provider/model", "call", "phase/tag", "status", "error", "record"], rows,
              caption="Every agent call, as logged"),
        table(["role", "provider/model", "answered", "unavailable", "unavailable in"], per_role,
              caption="Per agent"),
        Para("An UNAVAILABLE call means the agent gave no usable answer (API error, rate limit, timeout, invalid "
             "output or call budget). Missing feedback is never counted as approval."),
        source("events.jsonl (agent_output, agent_unavailable)"),
    ]


# ---------------------------------------------------------------- tokens and cost


@dataclass
class _Usage:
    attempts: int = 0
    ok: int = 0
    provider_errors: int = 0
    invalid: int = 0
    budget_stops: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    estimated: bool = False

    def add(self, c: CallRecord) -> None:
        if c.cache_hit:
            self.cache_hits += 1
            return
        if c.status is CallStatus.BUDGET_EXHAUSTED:
            self.budget_stops += 1
            return
        if c.status in _ATTEMPT_STATUSES:
            self.attempts += 1
        self.ok += c.status is CallStatus.OK
        self.provider_errors += c.status is CallStatus.PROVIDER_ERROR
        self.invalid += c.status is CallStatus.INVALID_OUTPUT
        self.tokens_in += c.tokens_in
        self.tokens_out += c.tokens_out
        self.latency_s += c.latency_s
        # A failed attempt logs 0 tokens flagged as estimated; it adds nothing, so it does not make the total an
        # estimate. Only counted tokens that were estimated do.
        if c.tokens_in or c.tokens_out:
            self.estimated = self.estimated or c.tokens_estimated


def _cost(usage: _Usage, provider: str, model: str, pricing: AppConfig | None) -> float | None:
    price = pricing.price_for(provider, model) if pricing is not None else None
    if price is None:
        return None
    return (usage.tokens_in * price.input_per_mtok + usage.tokens_out * price.output_per_mtok) / 1_000_000


def _row(name: str, ident: str, u: _Usage, cost: float | None, cost_text: str | None = None) -> list[str]:
    tokens_note = " (estimated)" if u.estimated else ""
    return [name, ident, str(u.attempts), str(u.ok), str(u.provider_errors), str(u.invalid), str(u.budget_stops),
            str(u.cache_hits), f"{u.tokens_in}{tokens_note}", f"{u.tokens_out}{tokens_note}", f"{u.latency_s:.3f}",
            cost_text if cost_text is not None else ("not configured" if cost is None else f"{cost:.6f}")]


def token_blocks(rec: RunRecords, pricing: AppConfig | None) -> list[Block]:
    if not rec.calls:
        return [Para("calls.jsonl holds no calls."), source("calls.jsonl")]
    groups: dict[tuple[Role, str, str], _Usage] = {}
    total = _Usage()
    for c in rec.calls:
        groups.setdefault((c.role, c.provider, c.model), _Usage()).add(c)
        total.add(c)
    rows: list[list[str]] = []
    costs: list[float | None] = []
    for (role, provider, model), usage in sorted(groups.items(), key=lambda kv: list(Role).index(kv[0][0])):
        cost = _cost(usage, provider, model, pricing)
        costs.append(cost)
        rows.append(_row(role.value, f"{provider}/{model}", usage, cost))
    known = [c for c in costs if c is not None]
    if not known:
        total_cost = "not configured"
    elif len(known) == len(costs):
        total_cost = f"{sum(known):.6f}"
    else:
        total_cost = f"{sum(known):.6f} (models without a price not included)"
    rows.append(_row("total", "-", total, None, total_cost))
    blocks: list[Block] = [
        table(["agent", "provider/model", "provider attempts", "ok", "provider errors", "invalid outputs",
               "budget stops", "cache hits", "tokens in", "tokens out", f"latency s ({MEASURED})", "cost USD"],
              rows, caption="Calls, tokens and cost from calls.jsonl"),
        Para("Provider attempts include retries, because each one spends quota. Cache hits made no provider "
             "call and their tokens are not counted. Failed attempts are recorded with 0 tokens because the "
             "provider reported no usage. Latency is the measured wall time of the attempts."),
    ]
    if total.estimated:
        blocks.append(Note("Token counts marked 'estimated' are character-based estimates, not counts reported "
                           "by a provider."))
    if known:
        blocks.append(Note("Cost is estimated from the prices in config.yaml ('pricing') at report time. "
                           "It is not a billed amount."))
    else:
        blocks.append(Para("Cost: no prices are configured for these models ('pricing' in config.yaml), so no "
                           "cost is shown."))
    blocks.append(source("calls.jsonl"))
    return blocks


# ---------------------------------------------------------------- contributions


def final_label(rec: RunRecords) -> str | None:
    ev = rec.last("final_written")
    if ev is not None:
        return str(ev.payload.get("version"))
    if rec.summary is not None and rec.summary.final_version:
        return rec.summary.final_version
    return None


def final_line_credit(rec: RunRecords) -> tuple[str | None, Counter[tuple[Role, bool]]]:
    """Lines of the stored final version per (agent, came from an accepted modification)."""
    label = final_label(rec)
    meta = rec.versions.get(label) if label else None
    credit: Counter[tuple[Role, bool]] = Counter()
    if meta is None:
        return label, credit
    for origin in meta.line_origins:
        if origin.role is not None:
            credit[(origin.role, origin.modification_id is not None)] += 1
    return label, credit


def contribution_table(rec: RunRecords) -> Table:
    label, credit = final_line_credit(rec)
    spec = rec.spec()
    rows: list[list[str]] = []
    for role in Role:
        slot = ROLE_VERSION_LABEL[role]
        initial = rec.initial(role)
        suite = rec.initial_suite(slot)
        if initial is None:
            init_text = "UNAVAILABLE" if any(not m.ok for m in rec.messages("initial") if m.role is role) else "none"
        elif suite is not None and suite.executed:
            init_text = f"{slot} ({sum(r.status.value == 'pass' for r in suite.results)}/{len(suite.results)} passed)"
        else:
            init_text = f"{slot} (not executed)"
        review = rec.review(role)
        review_text = (str(len(review[0].items)) if review is not None else
                       "UNAVAILABLE" if any(not m.ok for m in rec.messages("review") if m.role is role) else "none")
        mine = [r for r in rec.modifications if r.proposed_by is role]
        verdicts = [v for r in rec.modifications for v in r.reviewers if v.role is role]
        given = sum(1 for v in verdicts if v.verdict is not Verdict.UNAVAILABLE)
        missed = len(verdicts) - given
        lines = ("-" if label is None or label not in rec.versions else
                 f"{credit[(role, False)]} from its initial version, {credit[(role, True)]} from accepted changes")
        rows.append([
            role.value, rec.agent_identity(role),
            "yes" if spec is not None and spec[1].role is role else "-",
            init_text, review_text, str(len(mine)),
            str(sum(1 for r in mine if r.decision is Decision.ACCEPTED)),
            str(sum(1 for r in mine if r.decision is Decision.REJECTED)),
            f"{given}" + (f" (+{missed} unavailable)" if missed else ""), lines,
        ])
    caption = f"Contributions (lines counted in stored version {label})" if label else "Contributions"
    return table(["agent", "provider/model", "analysis", "initial version", "review items", "proposals",
                  "accepted", "rejected", "challenge verdicts", "lines in final version"], rows, caption=caption)
