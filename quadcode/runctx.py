"""Services shared by the orchestrator and the optimization rounds of one run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quadcode.agents import AgentPool, PoolResult
from quadcode.config import AppConfig
from quadcode.gateway import BudgetExhausted, CallOutcome
from quadcode.modifications import ModificationLedger
from quadcode.runlog import RunLog
from quadcode.runstate import RunState, save_state
from quadcode.sandbox import Sandbox
from quadcode.schemas import (
    ROLE_VERSION_LABEL,
    AgentAvailability,
    AgentStatus,
    Phase,
    ProblemSpec,
    Role,
    SuiteRun,
    TestCase,
    TestCategory,
    TestStatus,
)
from quadcode.testing import run_suite, write_test_results
from quadcode.versions import VersionStore

TESTS_FILE = "tests.json"
SUITE_FILE = "test_suite.json"


@dataclass
class RunContext:
    config: AppConfig
    run_dir: Path
    runlog: RunLog
    store: VersionStore
    ledger: ModificationLedger
    pool: AgentPool
    sandbox: Sandbox
    state: RunState
    tests: list[TestCase] = field(default_factory=list)
    suite_runs: list[SuiteRun] = field(default_factory=list)

    @property
    def spec(self) -> ProblemSpec:
        if self.state.spec is None:
            raise RuntimeError("the problem specification is not available yet")
        return self.state.spec

    @property
    def perf_ids(self) -> list[str]:
        return [t.id for t in self.tests if t.category is TestCategory.PERFORMANCE]

    def save(self) -> None:
        save_state(self.run_dir, self.state)

    def event(self, phase: Phase, kind: str, role: Role | None = None, **payload: Any) -> None:
        self.runlog.event(phase, kind, role, payload)

    # ------------------------------------------------------------ agents

    def note_outcome(self, phase: Phase, outcome: CallOutcome[Any], tag: str) -> None:
        """Log an agent's message (or its unavailability) and record its availability."""
        self.state.availability = [*self.state.availability, AgentAvailability(
            role=outcome.role, label=ROLE_VERSION_LABEL[outcome.role], provider=outcome.provider,
            model=outcome.model, phase=tag, status=outcome.status, error=outcome.error)]
        common = {"call_type": outcome.call_type.value, "tag": tag, "provider": outcome.provider,
                  "model": outcome.model}
        if outcome.ok:
            assert outcome.data is not None
            self.event(phase, "agent_output", outcome.role, **common, status=outcome.status.value,
                       data=outcome.data.model_dump(mode="json"))
        else:
            self.event(phase, "agent_unavailable", outcome.role, **common, status=AgentStatus.UNAVAILABLE.value,
                       error=outcome.error)

    def log_pool(self, phase: Phase, result: PoolResult, tags: dict[Role, str]) -> None:
        for role, outcome in result.outcomes.items():
            self.note_outcome(phase, outcome, tags[role])

    def raise_if_budget(self, result: PoolResult) -> None:
        """Save what finished, then abort if any call of the pool hit the call budget."""
        if result.budget_exhausted:
            self.save()
            raise BudgetExhausted("call budget exhausted")

    def finish_pool(self, phase: Phase, result: PoolResult, tags: dict[Role, str]) -> None:
        """Log every outcome; raise BudgetExhausted after the finished outcomes are logged and saved."""
        self.log_pool(phase, result, tags)
        self.raise_if_budget(result)

    # ------------------------------------------------------------ sandbox

    def run_tests(self, phase: Phase, label: str, code: str, language: str | None = None,
                  repeats: int | None = None) -> SuiteRun:
        """Run the suite on ``code`` in the sandbox, append it to tests.json and log the result."""
        suite = run_suite(code=code, entry_point=self.spec.entry_point, language=language or self.spec.language,
                          tests=self.tests, sandbox=self.sandbox, cfg=self.config.sandbox, version=label,
                          repeats=repeats)
        self.suite_runs.append(suite)
        write_test_results(self.run_dir / TESTS_FILE, self.suite_runs)
        failing = {r.test_id: r.status.value for r in suite.results if r.status is not TestStatus.PASS}
        self.event(phase, "suite_run", version=label, executed=suite.executed, backend=suite.backend,
                   passed=suite.count(TestStatus.PASS), total=len(suite.results), failing=failing,
                   not_executed_reason=suite.not_executed_reason, isolation_warning=suite.isolation_warning)
        return suite
