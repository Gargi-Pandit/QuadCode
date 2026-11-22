"""One optimization round: proposals, candidates, tests, challenges, decisions, round version.

Decisions are made by the orchestrator on test evidence and measurements only. Challenge
verdicts are recorded on the modification and logged, but never decide. Every step checks
the ledger and the version store first, so a resumed round does not redo finished work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quadcode.diffs import make_diff
from quadcode.gateway import BudgetExhausted
from quadcode.linemap import ComposeConflict, ComposeResult, compose
from quadcode.modifications import build_record
from quadcode.prompts import (
    NO_REVIEW_CONTEXT,
    challenge_prompt,
    later_round_context,
    modify_prompt,
    results_table,
    round_one_context,
    truncate,
)
from quadcode.runctx import RunContext
from quadcode.runstate import RoundState
from quadcode.schemas import (
    Ablation,
    ChallengeResponse,
    CallType,
    Decision,
    ModificationRecord,
    ModifyOutput,
    Phase,
    ReviewerVerdict,
    Role,
    SuiteRun,
    TestStatus,
    Verdict,
)
from quadcode.selection import BaseComparison, compare_to_base, is_noop
from quadcode.testing import suite_status
from quadcode.versions import StoredVersion

PHASE = Phase.ROUNDS


@dataclass
class _Tested:
    record: ModificationRecord
    suite: SuiteRun
    comparison: BaseComparison
    noop: bool


def round_label(index: int) -> str:
    return f"V{index}"


def _failing_text(suite: SuiteRun, limit: int, max_tests: int) -> str:
    failing = [r for r in suite.results if r.status is not TestStatus.PASS]
    if not suite.executed:
        return f"not executed: {suite.not_executed_reason}"
    shown = "; ".join(f"{r.test_id} {r.status.value}: {truncate(r.error or '', limit)}" for r in failing[:max_tests])
    more = f"; ... {len(failing) - max_tests} more" if len(failing) > max_tests else ""
    return shown + more


class RoundRunner:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------ prompt context

    def _context_text(self, rs: RoundState, base: StoredVersion) -> str:
        ctx = self.ctx
        if rs.index == 1:
            if ctx.state.ablation is Ablation.NO_REVIEW:
                return NO_REVIEW_CONTEXT
            source = base.meta.parents[0] if base.meta.parents else base.label
            return round_one_context(ctx.state.reviews, source)
        prev = ctx.state.rounds_state[rs.index - 2]
        prev_code = ctx.store.load(prev.base).code
        rejected = [ctx.ledger.get(m) for m in prev.rejected]
        return later_round_context(prev.base, make_diff(prev_code, base.code, f"a/{prev.base}", f"b/{base.label}"),
                                   rejected, ctx.config.prompts, with_results=ctx.show_results)

    # ------------------------------------------------------------ steps

    def _collect_proposals(self, rs: RoundState, base: StoredVersion, base_suite: SuiteRun) -> None:
        ctx = self.ctx
        if rs.modify_done:
            return
        tag = f"round-{rs.index}"
        results = results_table({base.label: base_suite}, ctx.config.prompts) if ctx.show_results else None
        prompt = modify_prompt(ctx.spec, rs.index, base.label, base.code, ctx.tests, results,
                               self._context_text(rs, base), ctx.config.prompts)
        missing = [r for r in ctx.roles if r not in rs.proposals]
        result = ctx.pool.run({r: (lambda a: a.ask(CallType.MODIFY, prompt, ModifyOutput, tag)) for r in missing})
        for role, outcome in result.outcomes.items():
            if outcome.ok:
                assert outcome.data is not None
                rs.proposals = {**rs.proposals, role: outcome.data}
                rs.unavailable = [r for r in rs.unavailable if r is not role]
            elif role not in rs.unavailable:
                rs.unavailable = [*rs.unavailable, role]
        ctx.finish_pool(PHASE, result, {r: tag for r in missing})
        rs.modify_done = True
        ctx.save()

    def _record_proposals(self, rs: RoundState, base: StoredVersion) -> list[ModificationRecord]:
        ctx = self.ctx
        if not rs.planned_ids:
            next_n = int(ctx.ledger.next_id()[2:])
            planned: dict[str, str] = {}
            for role in Role:
                out = rs.proposals.get(role)
                for k in range(len(out.proposals) if out else 0):
                    planned[f"{role.value}:{k}"] = f"M-{next_n:03d}"
                    next_n += 1
            rs.planned_ids = planned
            ctx.save()
        if rs.records_done:
            return [ctx.ledger.get(mod_id) for mod_id in rs.planned_ids.values()]
        records: list[ModificationRecord] = []
        for key, mod_id in rs.planned_ids.items():
            role_name, k = key.split(":")
            role = Role(role_name)
            proposal = rs.proposals[role].proposals[int(k)]
            built = build_record(proposal, modification_id=mod_id, proposed_by=role, round_index=rs.index,
                                 base_label=base.label, base_code=base.code)
            record = ctx.ledger.add(built.record)
            if built.result is not None:
                ctx.store.save_candidate(base.label, mod_id, role, built.result,
                                         note=f"{base.label} plus {mod_id} only, for testing")
            ctx.event(PHASE, "proposal_recorded", role, round=rs.index, id=mod_id, base=base.label,
                      candidate=record.candidate_version, original_location=record.original_location,
                      new_location=record.new_location, apply_error=record.apply_error,
                      justification=record.justification, expected_impact=record.expected_impact,
                      disadvantages=record.disadvantages, diff=record.diff)
            if record.decision is Decision.REJECTED:
                ctx.event(PHASE, "modification_decided", role, round=rs.index, id=mod_id,
                          decision=record.decision.value, reason=record.decision_reason, verdicts=[])
            records.append(record)
        rs.records_done = True
        ctx.save()
        return records

    def _test_candidates(self, records: Sequence[ModificationRecord], base: StoredVersion,
                         base_suite: SuiteRun) -> list[_Tested]:
        ctx = self.ctx
        tested: list[_Tested] = []
        for rec in records:
            if rec.decision is not Decision.PENDING or rec.candidate_version is None:
                continue
            cand = ctx.store.load(rec.candidate_version)
            suite = ctx.run_tests(PHASE, cand.label, cand.code)
            rec = ctx.ledger.set_test_status(rec.id, suite_status(suite))
            comparison = compare_to_base(base_suite, suite, ctx.perf_ids, ctx.config.selection)
            noop = is_noop(base.code, cand.code, base.meta.language)
            ctx.event(PHASE, "candidate_tested", rec.proposed_by, round=rec.round_index, id=rec.id,
                      version=cand.label, test_status=rec.test_status.value,
                      failing=_failing_text(suite, ctx.config.prompts.max_error_chars, len(suite.results)) or None,
                      comparison=comparison.describe(), regression=comparison.regression, noop=noop)
            tested.append(_Tested(rec, suite, comparison, noop))
        return tested

    def _compose(self, base: StoredVersion, label: str, recs: Sequence[ModificationRecord]) -> ComposeResult:
        return compose(base.code, base.meta.line_origins, label, [(r.id, r.proposed_by, r.diff) for r in recs])

    def _choose(self, rs: RoundState, base: StoredVersion,
                eligible: Sequence[_Tested]) -> tuple[list[_Tested], dict[str, str]]:
        """Which eligible modifications can be integrated together; the others with the reason."""
        ctx = self.ctx
        label = round_label(rs.index)
        if len(eligible) <= 1:
            return list(eligible), {}
        trial_no = 0

        def trial(chosen: Sequence[_Tested]) -> tuple[bool, str]:
            nonlocal trial_no
            try:
                composed = self._compose(base, label, [t.record for t in chosen])
            except ComposeConflict as exc:
                return False, f"conflicts with {', '.join(t.record.id for t in chosen if t.record.id != exc.modification_id)}: {exc.error}"
            trial_no += 1
            trial_label = f"{label}~trial-{trial_no}"
            suite = ctx.run_tests(PHASE, trial_label, composed.text)
            ids = [t.record.id for t in chosen]
            ctx.event(PHASE, "composition_trial", round=rs.index, version=trial_label, modifications=ids,
                      test_status=suite_status(suite).value)
            if suite.all_passed:
                return True, ""
            return False, (f"combined with {', '.join(ids[:-1])} the suite fails: "
                           f"{_failing_text(suite, ctx.config.prompts.max_error_chars, ctx.config.prompts.max_failing_tests)}")

        ok, _ = trial(eligible)
        if ok:
            return list(eligible), {}
        chosen: list[_Tested] = []
        dropped: dict[str, str] = {}
        for t in eligible:
            if not chosen:
                chosen.append(t)  # passed every test on its own
                continue
            ok, reason = trial([*chosen, t])
            if ok:
                chosen.append(t)
            else:
                dropped[t.record.id] = reason
        return chosen, dropped

    def _challenge(self, rs: RoundState, item: _Tested, reason: str) -> None:
        ctx = self.ctx
        rec = ctx.ledger.get(item.record.id)
        tag = f"round-{rs.index}:{rec.id}"
        already = {v.role for v in rec.reviewers}
        roles = [r for r in ctx.roles if r is not rec.proposed_by and r not in already]
        if not roles:
            return
        if ctx.show_results:
            prompt = challenge_prompt(ctx.spec, rec, reason,
                                      results_table({rec.candidate_version or rec.id: item.suite}, ctx.config.prompts))
        else:
            prompt = challenge_prompt(ctx.spec, rec, None, None)
        ctx.event(PHASE, "challenge_requested", rec.proposed_by, round=rs.index, id=rec.id, reason=reason,
                  challengers=[r.value for r in roles])
        result = ctx.pool.run({r: (lambda a: a.ask(CallType.CHALLENGE, prompt, ChallengeResponse, tag))
                               for r in roles})
        ctx.log_pool(PHASE, result, {r: tag for r in result.outcomes})
        for role, outcome in result.outcomes.items():
            if outcome.ok:
                assert isinstance(outcome.data, ChallengeResponse)
                data = outcome.data
                comment = data.comment + (f" Evidence: {data.evidence}" if data.evidence else "")
                verdict = ReviewerVerdict(role=role, verdict=data.verdict, comment=comment)
            else:
                verdict = ReviewerVerdict(role=role, verdict=Verdict.UNAVAILABLE, comment=outcome.error or "")
            ctx.ledger.add_verdict(rec.id, verdict)
            ctx.event(PHASE, "challenge_verdict", role, round=rs.index, id=rec.id, verdict=verdict.verdict.value,
                      comment=verdict.comment)
        ctx.raise_if_budget(result)

    def _decide(self, rs: RoundState, rec_id: str, accept: bool, reason: str) -> None:
        ctx = self.ctx
        rec = ctx.ledger.get(rec_id)
        if rec.decision is Decision.PENDING:
            rec = ctx.ledger.accept(rec_id, reason) if accept else ctx.ledger.reject(rec_id, reason)
            ctx.event(PHASE, "modification_decided", rec.proposed_by, round=rs.index, id=rec_id,
                      decision=rec.decision.value, reason=rec.decision_reason,
                      verdicts=[v.model_dump(mode="json") for v in rec.reviewers])

    # ------------------------------------------------------------ the round

    def run(self, rs: RoundState) -> RoundState:
        """Run (or resume) round ``rs.index`` on ``rs.base``. Raises BudgetExhausted to abort."""
        ctx = self.ctx
        if rs.decided:
            return rs
        base = ctx.store.load(rs.base)
        ctx.event(PHASE, "round_started", round=rs.index, base=base.label)
        base_suite = ctx.run_tests(PHASE, base.label, base.code)
        self._collect_proposals(rs, base, base_suite)
        records = self._record_proposals(rs, base)
        tested = self._test_candidates(records, base, base_suite)

        pre_reject: dict[str, str] = {}
        contested: dict[str, str] = {}
        eligible: list[_Tested] = []
        cfg = ctx.config.prompts
        for t in tested:
            if t.record.test_status is not TestStatus.PASS:
                reason = (f"candidate {t.record.candidate_version} does not pass the suite: "
                          f"{_failing_text(t.suite, cfg.max_error_chars, cfg.max_failing_tests)}")
                pre_reject[t.record.id] = reason
                contested[t.record.id] = reason
            elif t.noop:
                pre_reject[t.record.id] = "no executable change: the code is AST-identical to the base " \
                                          "(formatting or comments only)"
            elif t.comparison.regression:
                reason = f"measured regression against {base.label}: {t.comparison.describe()}"
                pre_reject[t.record.id] = reason
                contested[t.record.id] = reason
            else:
                eligible.append(t)

        chosen, dropped = self._choose(rs, base, eligible)
        for rec_id, reason in dropped.items():
            pre_reject[rec_id] = reason
            contested[rec_id] = reason

        by_id = {t.record.id: t for t in tested}
        try:
            for rec_id, reason in contested.items():
                self._challenge(rs, by_id[rec_id], reason)
        except BudgetExhausted:
            ctx.save()
            raise

        label = round_label(rs.index)
        n_tests = len(ctx.tests)
        for t in chosen:
            reason = (f"candidate passes all {n_tests} tests; {t.comparison.describe()}; integrated into {label}")
            self._decide(rs, t.record.id, True, reason)
        for rec_id, reason in pre_reject.items():
            self._decide(rs, rec_id, False, reason)

        accepted = [r for r in ctx.ledger.by_round(rs.index) if r.decision is Decision.ACCEPTED]
        rs.accepted = [r.id for r in accepted]
        rs.rejected = [r.id for r in ctx.ledger.by_round(rs.index) if r.decision is Decision.REJECTED]
        if accepted:
            composed = self._compose(base, label, accepted)
            stored = ctx.store.save_round(label, base.label, composed,
                                          note=f"{base.label} with {', '.join(rs.accepted)} integrated")
            for rec in accepted:
                if ctx.ledger.get(rec.id).new_version is None:
                    ctx.ledger.place(rec.id, label, composed.location_of(rec.id, label))
            rs.round_version = stored.label
            ctx.event(PHASE, "round_version_created", round=rs.index, version=label, base=base.label,
                      modifications=rs.accepted, diff=make_diff(base.code, stored.code, f"a/{base.label}",
                                                                f"b/{label}"))
        rs.decided = True
        ctx.event(PHASE, "round_completed", round=rs.index, accepted=rs.accepted, rejected=rs.rejected,
                  unavailable=[r.value for r in rs.unavailable], version=rs.round_version)
        ctx.save()
        return rs


def stop_reason(rs: RoundState, active_agents: int = len(Role)) -> str | None:
    """Why the rounds end after ``rs``, or None if they continue."""
    if rs.accepted:
        return None
    if len(rs.unavailable) == active_agents:
        return (f"every agent was unavailable in round {rs.index}; stopped without evidence of convergence")
    text = f"round {rs.index} accepted no modification (early stop)"
    if rs.unavailable:
        text += f"; unavailable in this round: {', '.join(r.value for r in rs.unavailable)}"
    return text
