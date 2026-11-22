"""Score finished runs on hidden tests.

Code is taken from a run's stored versions and final file, executed against the problem's hidden tests in
the sandbox (once per distinct code, identical inputs for every version), and turned into one
``ResultRecord`` per condition. Nothing here calls a model.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from quadcode.cache import sha256_text
from quadcode.config import AppConfig
from quadcode.evaluation.conditions import RUN_ABLATION, SINGLE_ROLE, Condition
from quadcode.evaluation.problems import BenchProblem
from quadcode.evaluation.records import (
    BenchState,
    CallCost,
    PairState,
    PairStatus,
    ProblemResults,
    ResultRecord,
)
from quadcode.orchestrator import FINAL_STEM
from quadcode.runlog import RunLog
from quadcode.runstate import load_state
from quadcode.sandbox import Sandbox
from quadcode.schemas import ROLE_VERSION_LABEL, CallRecord, CallStatus, CallType, Role, SuiteRun, TestCategory, TestStatus
from quadcode.testing import run_suite
from quadcode.versions import VersionError, VersionStore

_ATTEMPTS = {CallStatus.OK, CallStatus.PROVIDER_ERROR, CallStatus.INVALID_OUTPUT}


def call_cost(calls: list[CallRecord]) -> CallCost:
    cost = CallCost()
    for c in calls:
        if c.cache_hit:
            cost.cache_hits += 1
        elif c.status in _ATTEMPTS:
            cost.provider_attempts += 1
            cost.ok += c.status is CallStatus.OK
            cost.tokens_in += c.tokens_in
            cost.tokens_out += c.tokens_out
            if c.tokens_in or c.tokens_out:
                cost.tokens_estimated = cost.tokens_estimated or c.tokens_estimated
    return cost


def _code(run_dir: Path, label: str) -> tuple[str, str] | None:
    """(code, language) of a stored version, or None if the run never created it."""
    store = VersionStore(run_dir)
    if not store.exists(label):
        return None
    try:
        stored = store.load(label)
    except VersionError:
        return None
    return stored.code, stored.meta.language


def _final(run_dir: Path) -> tuple[str, str, str] | None:
    """(file name, code, language) of the delivered final solution."""
    files = sorted(run_dir.glob(f"{FINAL_STEM}.*"))
    if not files:
        return None
    state = load_state(run_dir)
    language = state.spec.language if state.spec else "python"
    return files[0].name, files[0].read_text(encoding="utf-8"), language


class HiddenScorer:
    def __init__(self, problem: BenchProblem, sandbox: Sandbox, config: AppConfig, previous: ProblemResults | None):
        self.problem = problem
        self.sandbox = sandbox
        self.config = config
        keep = previous is not None and previous.hidden_sha256 == problem.hidden_sha256
        self.results = ProblemResults(problem=problem.id, hidden_sha256=problem.hidden_sha256,
                                      entry_point=problem.entry_point,
                                      suites=dict(previous.suites) if keep and previous else {},
                                      records=list(previous.records) if keep and previous else [])
        self.perf_ids = [t.id for t in problem.hidden_tests if t.category is TestCategory.PERFORMANCE]

    # ------------------------------------------------------------ execution

    def suite_for(self, code: str, language: str, label: str) -> SuiteRun:
        sha = sha256_text(code)
        if sha not in self.results.suites:
            cfg = self.config.sandbox
            self.results.suites[sha] = run_suite(
                code=code, entry_point=self.problem.entry_point, language=language,
                tests=list(self.problem.hidden_tests), sandbox=self.sandbox, cfg=cfg, version=label,
                repeats=self.config.evaluation.benchmark_repeats or cfg.benchmark_repeats)
        return self.results.suites[sha]

    def _scored(self, condition: Condition, pair: PairState, version: str, code: str, language: str,
                cost: CallCost, notes: list[str]) -> ResultRecord:
        suite = self.suite_for(code, language, version)
        passed = sum(1 for r in suite.results if r.status is TestStatus.PASS)
        perf = {r.test_id: r.runtime_s for r in suite.results
                if r.test_id in self.perf_ids and r.status is TestStatus.PASS and r.runtime_s is not None}
        if not suite.executed:
            notes = [*notes, f"not executed: {suite.not_executed_reason}"]
        return ResultRecord(problem=self.problem.id, condition=condition, status="scored", run_id=pair.run_id,
                            run_status=pair.run_status, version=version, code_sha256=sha256_text(code),
                            executed=suite.executed, passed=passed, total=len(self.problem.hidden_tests),
                            perf_tests=list(self.perf_ids), perf_median_s=perf, notes=notes, cost=cost)

    def _none(self, condition: Condition, pair: PairState | None, status: str, reason: str,
              cost: CallCost | None = None) -> ResultRecord:
        return ResultRecord(problem=self.problem.id, condition=condition, status=status,  # type: ignore[arg-type]
                            reason=reason, run_id=pair.run_id if pair else None,
                            run_status=pair.run_status if pair else None,
                            total=len(self.problem.hidden_tests) if status == "no_solution" else 0,
                            cost=cost or CallCost())

    def _put(self, record: ResultRecord) -> None:
        self.results.records = [r for r in self.results.records if r.condition is not record.condition] + [record]

    # ------------------------------------------------------------ conditions

    def score(self, state: BenchState, runs_root: Path) -> ProblemResults:
        """Score every condition whose run finished; keep earlier scored records; mark the rest missing."""
        full = state.pair(self.problem.id, Condition.FULL)
        full_dir = runs_root / full.run_id
        derived_ok = full.status is PairStatus.FINISHED
        full_calls = RunLog(full_dir).read_calls() if derived_ok else []
        for cond in state.run_conditions:
            pair = state.pair(self.problem.id, cond)
            prev = self.results.record(cond)
            if prev is not None and prev.counted:
                continue
            self._put(self._score_run(cond, pair, runs_root / pair.run_id, full_dir))
        for cond, role in SINGLE_ROLE.items():
            prev = self.results.record(cond)
            if prev is not None and prev.counted:
                continue
            if not derived_ok:
                self._put(self._none(cond, full, "missing", _missing_reason(full)))
                continue
            label = ROLE_VERSION_LABEL[role]
            cost = call_cost([c for c in full_calls if c.call_type is CallType.ANALYSIS
                              or (c.call_type is CallType.INITIAL and c.role is role)])
            found = _code(full_dir, label)
            if found is None:
                why = ("the slot was excluded" if role in RUN_ABLATION[Condition.FULL].excluded_roles else
                       f"the {role.value} slot produced no initial solution ({_run_end(full_dir)})")
                self._put(self._none(cond, full, "no_solution", why, cost))
            else:
                self._put(self._scored(cond, full, label, found[0], found[1], cost, []))
        self._score_select_only(full, full_dir, full_calls, derived_ok)
        self._score_oracle(full, full_calls, derived_ok)
        order = list(Condition)
        self.results.records = sorted(self.results.records, key=lambda r: order.index(r.condition))
        return self.results

    def _score_run(self, cond: Condition, pair: PairState, run_dir: Path, full_dir: Path) -> ResultRecord:
        if pair.status is not PairStatus.FINISHED:
            return self._none(cond, pair, "missing", _missing_reason(pair))
        cost = call_cost(RunLog(run_dir).read_calls())
        notes = _entry_note(run_dir, self.problem.entry_point)
        controlled: bool | None = None
        if cond is not Condition.FULL:
            controlled = _same_initials(run_dir, full_dir)
            if controlled is False:
                notes.append("initial solutions differ from the full run's, so this ablation is not a controlled "
                             "comparison")
        found = _final(run_dir)
        if found is None:
            rec = self._none(cond, pair, "no_solution", f"no final solution: {_run_end(run_dir)}", cost)
            return rec.model_copy(update={"controlled": controlled, "notes": notes})
        name, code, language = found
        rec = self._scored(cond, pair, name, code, language, cost, notes)
        return rec.model_copy(update={"controlled": controlled})

    def _score_select_only(self, full: PairState, full_dir: Path, full_calls: list[CallRecord],
                           derived_ok: bool) -> None:
        cond = Condition.SELECT_ONLY
        prev = self.results.record(cond)
        if prev is not None and prev.counted:
            return
        if not derived_ok:
            self._put(self._none(cond, full, "missing", _missing_reason(full)))
            return
        cost = call_cost([c for c in full_calls if c.call_type in (CallType.ANALYSIS, CallType.INITIAL)])
        found = _code(full_dir, "V0")
        if found is None:
            self._put(self._none(cond, full, "no_solution", f"no V0 was selected ({_run_end(full_dir)})", cost))
            return
        state = load_state(full_dir)
        rec = self._scored(cond, full, "V0", found[0], found[1], cost, [])
        source = state.reference.winner if state.reference else None
        self._put(rec.model_copy(update={"source_version": source}))

    def _score_oracle(self, full: PairState, full_calls: list[CallRecord], derived_ok: bool) -> None:
        cond = Condition.ORACLE
        if not derived_ok:
            self._put(self._none(cond, full, "missing", _missing_reason(full)))
            return
        singles = [self.results.record(c) for c in SINGLE_ROLE]
        cost = call_cost([c for c in full_calls if c.call_type in (CallType.ANALYSIS, CallType.INITIAL)])
        scored = [r for r in singles if r is not None and r.status == "scored"]
        if not scored:
            self._put(self._none(cond, full, "no_solution", "no initial solution exists", cost))
            return
        best = max(scored, key=lambda r: r.passed)  # max keeps the first in slot order on ties
        self._put(best.model_copy(update={
            "condition": cond, "source_version": best.version, "cost": cost,
            "notes": [*best.notes, f"picked {best.version} by hidden tests ({best.passed}/{best.total})"]}))


def _missing_reason(pair: PairState) -> str:
    match pair.status:
        case PairStatus.PENDING:
            return "not run"
        case PairStatus.ABORTED:
            return "run stopped by the call budget before it finished (resume to complete it)"
        case PairStatus.ERROR:
            return f"run could not be started: {pair.error}"
    return "run finished"


def _run_end(run_dir: Path) -> str:
    try:
        state = load_state(run_dir)
    except Exception:  # noqa: BLE001 - a damaged run is reported, not raised
        return "run state unreadable"
    return f"run ended {state.status.value}: {state.stop_reason or 'no reason recorded'}"


def _entry_note(run_dir: Path, entry_point: str) -> list[str]:
    try:
        state = load_state(run_dir)
    except Exception:  # noqa: BLE001
        return []
    if state.spec is not None and state.spec.entry_point != entry_point:
        return [f"the run's analysis named entry point '{state.spec.entry_point}', the hidden tests call "
                f"'{entry_point}'"]
    return []


def _same_initials(run_dir: Path, full_dir: Path) -> bool:
    """True if every initial version present in both runs has identical code (the ablation reused them)."""
    same = True
    for role in Role:
        label = ROLE_VERSION_LABEL[role]
        a, b = _code(run_dir, label), _code(full_dir, label)
        if a is None:
            continue
        if b is None or sha256_text(a[0]) != sha256_text(b[0]):
            same = False
    return same


def records_by_condition(results: Mapping[str, ProblemResults]) -> dict[Condition, dict[str, ResultRecord]]:
    out: dict[Condition, dict[str, ResultRecord]] = {}
    for pid, res in results.items():
        for rec in res.records:
            out.setdefault(rec.condition, {})[pid] = rec
    return out
