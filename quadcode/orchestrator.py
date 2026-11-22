"""The orchestrator: phases 1-8 of a run, each step logged and checkpointed.

1 analysis (one call) -> 2 blind initial solutions (one call per agent) -> 2.5 local tests
(zero tokens) -> 3 cross review (one call per agent) -> reference V0 chosen on evidence ->
4+6 optimization rounds with early stop -> 7 final benchmark -> 8 integration with
attribution comments. Missing agent output is recorded as UNAVAILABLE, never as approval.
"""

from __future__ import annotations

import json
from pathlib import Path

from quadcode.agents import AgentPool
from quadcode.attribution import annotate
from quadcode.benchmark import benchmark_versions
from quadcode.config import AppConfig
from quadcode.gateway import BudgetExhausted, LLMGateway
from quadcode.modifications import ModificationLedger
from quadcode.prompts import analysis_prompt, initial_prompt, results_table, review_prompt
from quadcode.rounds import RoundRunner, stop_reason
from quadcode.runctx import SUITE_FILE, TESTS_FILE, RunContext
from quadcode.runstate import (
    RoundState,
    RunState,
    check_resumable,
    config_fingerprint,
    load_state,
    problem_fingerprint,
    save_state,
)
from quadcode.sandbox import Sandbox
from quadcode.schemas import (
    ROLE_VERSION_LABEL,
    Ablation,
    AgentAvailability,
    AgentStatus,
    CallType,
    Decision,
    InitialSolution,
    Phase,
    ProblemSpec,
    Review,
    Role,
    RunStatus,
    RunSummary,
    SuiteRun,
    TestCase,
    TestStatus,
)
from quadcode.selection import rank_versions
from quadcode.testing import load_test_cases
from quadcode.versions import VersionStore

PROBLEM_FILE = "problem.md"
CONFIG_SNAPSHOT = "config.json"
BENCHMARK_FILE = "benchmark.json"
SUMMARY_FILE = "summary.json"
FINAL_STEM = "final_solution"
FINAL_LABEL = "final_solution"


def _load_suite_runs(path: Path) -> list[SuiteRun]:
    if not path.is_file():
        return []
    return [SuiteRun.model_validate(r) for r in json.loads(path.read_text(encoding="utf-8"))]


class Orchestrator:
    def __init__(self, config: AppConfig, gateway: LLMGateway, sandbox: Sandbox, run_dir: Path, problem: str,
                 state: RunState) -> None:
        self.config = config
        self.problem = problem
        self.run_dir = run_dir
        suite_path = run_dir / SUITE_FILE
        self.ctx = RunContext(
            config=config, run_dir=run_dir, runlog=gateway.runlog, store=VersionStore(run_dir),
            ledger=ModificationLedger(run_dir), pool=AgentPool(gateway, config.run.parallel), sandbox=sandbox,
            state=state, tests=[t for t in (load_test_cases(suite_path) if suite_path.is_file() else [])
                                if t.id not in state.disputed_tests],
            suite_runs=_load_suite_runs(run_dir / TESTS_FILE))
        self.gateway = gateway

    # ------------------------------------------------------------ construction

    @classmethod
    def create(cls, config: AppConfig, gateway: LLMGateway, sandbox: Sandbox, run_dir: Path, run_id: str,
               problem: str, rounds: int, ablation: Ablation = Ablation.NONE) -> Orchestrator:
        if (run_dir / "state.json").exists():
            raise FileExistsError(f"{run_dir} already holds a run; use --resume {run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / PROBLEM_FILE).write_text(problem, encoding="utf-8")
        (run_dir / CONFIG_SNAPSHOT).write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
        state = RunState(run_id=run_id, problem_sha256=problem_fingerprint(problem),
                         config_sha256=config_fingerprint(config), rounds=rounds, ablation=ablation)
        for role in ablation.excluded_roles:
            agent = config.agent(role)
            state.availability = [*state.availability, AgentAvailability(
                role=role, label=ROLE_VERSION_LABEL[role], provider=agent.provider, model=agent.model,
                phase="all", status=AgentStatus.EXCLUDED, error=None)]
        save_state(run_dir, state)
        return cls(config, gateway, sandbox, run_dir, problem, state)

    @classmethod
    def resume(cls, config: AppConfig, gateway: LLMGateway, sandbox: Sandbox, run_dir: Path) -> Orchestrator:
        state = load_state(run_dir)
        problem = (run_dir / PROBLEM_FILE).read_text(encoding="utf-8")
        check_resumable(state, problem, config)
        return cls(config, gateway, sandbox, run_dir, problem, state)

    @property
    def state(self) -> RunState:
        return self.ctx.state

    # ------------------------------------------------------------ helpers

    def _stop(self, status: RunStatus, reason: str, unresolved: list[str] | None = None) -> None:
        self.state.status = status
        self.state.stop_reason = reason
        if unresolved is not None:
            self.state.unresolved = unresolved
        self.ctx.save()

    def _begin(self, phase: Phase) -> None:
        self.ctx.event(phase, "phase_started")

    def _end(self, phase: Phase, step: str) -> None:
        self.state.mark_done(step)
        self.ctx.save()
        self.ctx.event(phase, "phase_completed")

    # ------------------------------------------------------------ phase 1

    def analysis(self) -> None:
        phase = Phase.ANALYSIS
        self._begin(phase)
        prompt = analysis_prompt(self.problem)
        for role in self.ctx.roles:
            agent = self.ctx.pool.agents[role]
            outcome = agent.ask(CallType.ANALYSIS, prompt, ProblemSpec, "analysis")
            self.ctx.note_outcome(phase, outcome, "analysis")
            if outcome.ok:
                assert isinstance(outcome.data, ProblemSpec)
                self.state.spec = outcome.data
                self.state.analysis_role = role
                break
        spec = self.state.spec
        if spec is None:
            self._stop(RunStatus.FAILED, "the analysis call failed for every agent slot",
                       ["no problem specification"])
        elif spec.clarification_needed:
            self.ctx.event(phase, "clarification_needed", self.state.analysis_role,
                           questions=spec.clarification_questions)
            self._stop(RunStatus.NEEDS_CLARIFICATION, "the analysis found essential information missing",
                       list(spec.clarification_questions) or ["clarification requested without questions"])
        self._end(phase, "analysis")

    # ------------------------------------------------------------ phase 2

    def initial(self) -> None:
        phase = Phase.INITIAL
        self._begin(phase)
        spec = self.ctx.spec
        missing = [r for r in self.ctx.roles if r not in self.state.initial]
        result = self.ctx.pool.run({
            r: (lambda a: a.ask(CallType.INITIAL, initial_prompt(self.problem, spec, a.role is Role.VALIDATOR),
                                InitialSolution, "initial"))
            for r in missing})
        for role, outcome in result.outcomes.items():
            if outcome.ok:
                assert isinstance(outcome.data, InitialSolution)
                self.state.initial = {**self.state.initial, role: outcome.data}
                self.ctx.store.save_initial(ROLE_VERSION_LABEL[role], outcome.data.code, outcome.data.language,
                                            note=f"blind initial implementation of the {role.value} slot "
                                                 f"({outcome.provider}/{outcome.model})")
        self.ctx.finish_pool(phase, result, {r: "initial" for r in missing})
        self._save_suite(phase)
        self._end(phase, "initial")

    def _save_suite(self, phase: Phase) -> None:
        sol = self.state.initial.get(Role.VALIDATOR)
        if sol is None or not sol.tests:
            return
        kept: list[TestCase] = []
        seen: set[str] = set()
        for t in sol.tests:
            if t.id in seen:
                self.ctx.event(phase, "test_dropped", Role.VALIDATOR, test_id=t.id, reason="duplicate test id")
                continue
            seen.add(t.id)
            kept.append(t)
        (self.run_dir / SUITE_FILE).write_text(
            json.dumps([t.model_dump(mode="json") for t in kept], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        self.ctx.tests = kept
        self.ctx.event(phase, "test_suite_saved", Role.VALIDATOR, count=len(kept), file=SUITE_FILE,
                       categories=sorted({t.category.value for t in kept}))

    # ------------------------------------------------------------ phase 2.5

    def local_tests(self) -> None:
        phase = Phase.LOCAL_TEST
        self._begin(phase)
        labels = [ROLE_VERSION_LABEL[r] for r in Role if r in self.state.initial]
        if not labels:
            self._stop(RunStatus.FAILED, "no agent produced an initial implementation",
                       ["no initial implementation"])
        elif not self.ctx.tests:
            self._stop(RunStatus.UNVERIFIED, "no test suite: the validator produced no tests, so no version "
                       "can be verified and no modification can be accepted",
                       ["no test suite; nothing was executed"])
        else:
            for label in labels:
                if label in self.state.initial_suites:
                    continue
                stored = self.ctx.store.load(label)
                suite = self.ctx.run_tests(phase, label, stored.code, language=stored.meta.language)
                self.state.initial_suites = {**self.state.initial_suites, label: suite}
                self.ctx.save()
            executed = [s for s in self.state.initial_suites.values() if s.executed]
            if len(executed) >= 2:
                suspect = [t.id for t in self.ctx.tests
                           if all(next(r.status for r in s.results if r.test_id == t.id) is not TestStatus.PASS
                                  for s in executed)]
                if suspect:
                    self._set_aside(phase, suspect)
            if executed and not self.ctx.tests:
                self._stop(RunStatus.UNVERIFIED, "every test failed on every initial implementation, so the "
                           "whole suite is disputed and nothing can be verified",
                           [f"disputed test {t}" for t in self.state.disputed_tests])
            if not executed:
                reasons = sorted({s.not_executed_reason or "" for s in self.state.initial_suites.values()})
                self._stop(RunStatus.UNVERIFIED, "no initial implementation could be executed",
                           [f"not executed: {r}" for r in reasons])
        self._end(phase, "local_tests")

    def _set_aside(self, phase: Phase, test_ids: list[str]) -> None:
        """Drop tests that every independent implementation failed from all later decisions.

        When four separately written solutions all disagree with a test, the test's expected value is the
        likelier mistake. Keeping it would reward changes that bend the code toward a wrong answer, so it is
        set aside: it no longer ranks versions, decides modifications or verifies the final version. It stays
        in test_suite.json and tests.json, and the run lists it as unresolved.
        """
        disputed = set(test_ids)
        self.state.disputed_tests = sorted(disputed)
        self.ctx.tests = [t for t in self.ctx.tests if t.id not in disputed]
        self.state.initial_suites = {
            label: suite.model_copy(update={"results": [r for r in suite.results if r.test_id not in disputed]})
            for label, suite in self.state.initial_suites.items()}
        self.ctx.event(phase, "suspect_tests", test_ids=sorted(disputed), set_aside=True,
                       note="failed on every executed initial implementation, so the expected values are probably "
                            "wrong. Set aside: not used to rank versions, accept modifications or verify the final "
                            "version; listed as unresolved.")
        self.ctx.save()

    def _disputed_notes(self) -> list[str]:
        return [f"disputed test {t}: every independent initial implementation failed it, so it was set aside "
                "(its expected value is probably wrong)" for t in self.state.disputed_tests]

    # ------------------------------------------------------------ phase 3

    def review(self) -> None:
        phase = Phase.REVIEW
        self._begin(phase)
        if self.state.ablation is Ablation.NO_REVIEW:
            self.ctx.event(phase, "phase_skipped", reason=self.state.ablation.description)
            self._end(phase, "review")
            return
        spec = self.ctx.spec
        codes: dict[str, str | None] = {
            ROLE_VERSION_LABEL[r]: (self.state.initial[r].code if r in self.state.initial else None)
            for r in self.ctx.roles}
        table = results_table(self.state.initial_suites, self.config.prompts) if self.ctx.show_results else None
        missing = [r for r in self.ctx.roles if r not in self.state.reviews]
        result = self.ctx.pool.run({
            r: (lambda a: a.ask(CallType.REVIEW, review_prompt(spec, ROLE_VERSION_LABEL[a.role], codes, table),
                                Review, "review"))
            for r in missing})
        for role, outcome in result.outcomes.items():
            if outcome.ok:
                assert isinstance(outcome.data, Review)
                self.state.reviews = {**self.state.reviews, role: outcome.data}
        self.ctx.finish_pool(phase, result, {r: "review" for r in missing})
        self._end(phase, "review")

    def select_reference(self) -> None:
        phase = Phase.ROUNDS
        suites = self.state.initial_suites
        codes = {label: self.ctx.store.load(label).code for label in suites}
        evidence = rank_versions(suites, codes, self.ctx.perf_ids, self.config.selection,
                                 purpose="V0: reference implementation for the optimization rounds")
        assert evidence.winner is not None
        self.ctx.store.save_reference(evidence.winner, note=f"copy of {evidence.winner}, chosen on test evidence")
        self.state.reference = evidence
        self.ctx.event(phase, "reference_selected", version="V0", source=evidence.winner,
                       verified=evidence.verified, evidence=evidence.model_dump(mode="json"))
        self.state.mark_done("reference")
        self.ctx.save()

    # ------------------------------------------------------------ phases 4-6

    def rounds(self) -> None:
        phase = Phase.ROUNDS
        self._begin(phase)
        runner = RoundRunner(self.ctx)
        reason = f"completed all {self.state.rounds} rounds"
        for index in range(1, self.state.rounds + 1):
            if index <= len(self.state.rounds_state):
                rs = self.state.rounds_state[index - 1]
            else:
                prev = self.state.rounds_state[-1].round_version if self.state.rounds_state else None
                rs = RoundState(index=index, base=prev or "V0")
                self.state.rounds_state = [*self.state.rounds_state, rs]
                self.ctx.save()
            runner.run(rs)
            stop = stop_reason(rs, len(self.ctx.roles))
            if stop is not None:
                reason = stop
                self.ctx.event(phase, "early_stop", round=index, reason=stop)
                break
        self.state.stop_reason = reason
        last = next((rs.round_version for rs in reversed(self.state.rounds_state) if rs.round_version), None)
        self.state.final_version = last or "V0"
        self._end(phase, "rounds")

    # ------------------------------------------------------------ phase 7

    def benchmark(self) -> None:
        phase = Phase.FINAL_BENCHMARK
        self._begin(phase)
        final = self.state.final_version or "V0"
        labels = [*self.state.initial_suites, final]
        versions = {label: self.ctx.store.load(label).code for label in labels}
        report = benchmark_versions(versions, entry_point=self.ctx.spec.entry_point,
                                    language=self.ctx.spec.language, tests=self.ctx.tests,
                                    sandbox=self.ctx.sandbox, cfg=self.config.sandbox)
        (self.run_dir / BENCHMARK_FILE).write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        rows = [{"version": s.version, "test_id": s.test_id, "status": s.status.value, "repeats": s.repeats,
                 "median_s": s.median_s, "min_s": s.min_s, "max_s": s.max_s,
                 "peak_memory_bytes": s.peak_memory_bytes} for s in report.stats]
        passed = {s.version: s.count(TestStatus.PASS) for s in report.suites}
        self.ctx.event(phase, "final_benchmark", file=BENCHMARK_FILE, backend=report.backend,
                       isolation_warning=report.isolation_warning, method=report.method, repeats=report.repeats,
                       order=report.order, passed=passed, total=len(self.ctx.tests), stats=rows)
        self._end(phase, "benchmark")

    # ------------------------------------------------------------ phase 8

    def integrate(self) -> None:
        phase = Phase.INTEGRATE
        self._begin(phase)
        final = self.ctx.store.load(self.state.final_version or "V0")
        accepted = {r.id: r.proposed_by for r in self.ctx.ledger.all()
                    if r.decision is Decision.ACCEPTED and r.new_version is not None}
        annotated = annotate(final.code, final.meta.line_origins, accepted, final.meta.language)
        path = self.run_dir / f"{FINAL_STEM}{final.meta.ext}"
        path.write_text(annotated.text, encoding="utf-8")
        suite = self.ctx.run_tests(phase, FINAL_LABEL, annotated.text, language=final.meta.language)
        failing = [f"{r.test_id} {r.status.value}: {r.error or ''}".strip() for r in suite.results
                   if r.status is not TestStatus.PASS]
        self.ctx.event(phase, "final_written", version=final.label, file=path.name,
                       annotations=[{"id": a.modification_id, "role": a.role.value, "line": a.line,
                                     "skipped_reason": a.skipped_reason} for a in annotated.annotations],
                       verified=suite.all_passed)
        if suite.all_passed:
            self.state.status = RunStatus.COMPLETED
            self.state.unresolved = self._disputed_notes()
        else:
            self.state.status = RunStatus.UNVERIFIED
            self.state.unresolved = (failing or [f"not executed: {suite.not_executed_reason}"]) + self._disputed_notes()
        self._end(phase, "integrate")

    # ------------------------------------------------------------ driver

    def run(self) -> RunSummary:
        state = self.state
        if state.status in (RunStatus.ABORTED_BUDGET, RunStatus.RUNNING):
            state.status = RunStatus.RUNNING
            self.ctx.event(Phase.ANALYSIS, "run_started", run_id=state.run_id, resumed=bool(state.done),
                           done=list(state.done), rounds=state.rounds, ablation=state.ablation.value,
                           excluded=[r.value for r in state.ablation.excluded_roles])
        steps = [("analysis", self.analysis), ("initial", self.initial), ("local_tests", self.local_tests),
                 ("review", self.review), ("reference", self.select_reference), ("rounds", self.rounds),
                 ("benchmark", self.benchmark), ("integrate", self.integrate)]
        current = "analysis"
        try:
            for name, step in steps:
                if state.status is not RunStatus.RUNNING:
                    break
                if state.is_done(name):
                    continue
                current = name
                step()
        except BudgetExhausted:
            limit = self.gateway.budget.limit
            reason = f"call budget of {limit} exhausted during step '{current}'; resume with --resume {state.run_id}"
            self.ctx.event(Phase.ANALYSIS, "run_aborted", reason=reason, step=current)
            self._stop(RunStatus.ABORTED_BUDGET, reason)
        return self.write_summary()

    def write_summary(self) -> RunSummary:
        state = self.state
        records = self.ctx.ledger.all()
        final_file = next((p.name for p in sorted(self.run_dir.glob(f"{FINAL_STEM}.*"))), None)
        summary = RunSummary(
            run_id=state.run_id, status=state.status, problem_title=state.spec.title if state.spec else None,
            rounds_requested=state.rounds, rounds_run=sum(1 for rs in state.rounds_state if rs.decided),
            stop_reason=state.stop_reason,
            reference_version="V0" if state.reference else None,
            reference_source=state.reference.winner if state.reference else None,
            final_version=state.final_version if state.is_done("integrate") else None,
            final_file=final_file if state.is_done("integrate") else None,
            final_verified=state.status is RunStatus.COMPLETED,
            accepted=[r.id for r in records if r.decision is Decision.ACCEPTED],
            rejected=[r.id for r in records if r.decision is Decision.REJECTED],
            unresolved=list(state.unresolved), availability=list(state.availability), ablation=state.ablation,
            sandbox_backend=self.ctx.sandbox.name, isolation_warning=self.ctx.sandbox.isolation_warning)
        (self.run_dir / SUMMARY_FILE).write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        self.ctx.event(Phase.INTEGRATE, "run_finished", status=summary.status.value, stop_reason=summary.stop_reason,
                       final_version=summary.final_version, final_verified=summary.final_verified)
        return summary
