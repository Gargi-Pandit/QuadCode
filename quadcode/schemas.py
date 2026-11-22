"""Pydantic data contracts shared by every QuadCode component.

Two kinds of models live here:
- LLM output contracts (what an agent must return as JSON), e.g. ``InitialSolution``.
- Log records written by the orchestrator, e.g. ``CallRecord`` and ``RunEvent``.
Records are frozen: updates create new objects via ``model_copy(update=...)``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> str:
    """ISO-8601 UTC timestamp with millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Role(StrEnum):
    GENERATOR = "generator"
    REVIEWER = "reviewer"
    OPTIMIZER = "optimizer"
    VALIDATOR = "validator"


# Immutable initial-version labels for the four agent slots.
ROLE_VERSION_LABEL: dict[Role, str] = {
    Role.GENERATOR: "G1",
    Role.REVIEWER: "C1",
    Role.OPTIMIZER: "P1",
    Role.VALIDATOR: "D1",
}


class CallType(StrEnum):
    ANALYSIS = "analysis"
    INITIAL = "initial"
    REVIEW = "review"
    MODIFY = "modify"
    CHALLENGE = "challenge"


class AgentStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    CACHED = "cached"
    UNAVAILABLE = "unavailable"


class CallStatus(StrEnum):
    OK = "ok"
    PROVIDER_ERROR = "provider_error"
    INVALID_OUTPUT = "invalid_output"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Phase(StrEnum):
    ANALYSIS = "1-analysis"
    INITIAL = "2-initial"
    LOCAL_TEST = "2.5-local-test"
    REVIEW = "3-review"
    ROUNDS = "4-6-rounds"
    FINAL_BENCHMARK = "7-final-benchmark"
    INTEGRATE = "8-integrate"


class Verdict(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_CHANGES = "needs_changes"
    UNAVAILABLE = "unavailable"


class Decision(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class TestCategory(StrEnum):
    __test__ = False

    NORMAL = "normal"
    BOUNDARY = "boundary"
    ADVERSARIAL = "adversarial"
    PERFORMANCE = "performance"


class TestStatus(StrEnum):
    __test__ = False

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_RUN = "not_run"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _LLMOutput(BaseModel):
    """LLM output contracts: frozen, but tolerant of extra keys models like to add."""

    model_config = ConfigDict(frozen=True, extra="ignore")


# ---------------------------------------------------------------- LLM outputs


class ProblemSpec(_LLMOutput):
    title: str
    language: str
    summary: str
    entry_point: str = Field(description="Name of the function the solution must define.")
    signature: str
    inputs: str
    outputs: str
    functional_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    performance_constraints: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    clarification_needed: bool = False
    clarification_questions: list[str] = Field(default_factory=list)


class TestCase(_LLMOutput):
    __test__ = False  # not a pytest class

    id: str
    category: TestCategory
    description: str = ""
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    generator: str | None = Field(
        default=None,
        description="Optional Python expression evaluated inside the sandbox that yields "
        "(args, kwargs); used for large performance inputs.",
    )
    expected: Any = None
    expect_exception: str | None = None
    compare: Literal["exact", "unordered", "float"] = "exact"
    hidden: bool = False


class InitialSolution(_LLMOutput):
    language: str
    code: str
    algorithm: str
    data_structures: list[str] = Field(default_factory=list)
    time_complexity: str
    space_complexity: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    tests: list[TestCase] = Field(default_factory=list)


class ReviewItem(_LLMOutput):
    target: str = Field(description="Version label being reviewed, e.g. G1.")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    bugs: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    justification: str = ""


class Review(_LLMOutput):
    items: list[ReviewItem]


class ModificationProposal(_LLMOutput):
    diff: str = Field(description="Unified diff against the current version.")
    justification: str
    expected_impact: str
    disadvantages: str = ""


class ModifyOutput(_LLMOutput):
    proposals: list[ModificationProposal] = Field(default_factory=list)


class ChallengeResponse(_LLMOutput):
    verdict: Verdict
    comment: str
    evidence: str = ""


# ---------------------------------------------------------------- records


class ReviewerVerdict(_Frozen):
    role: Role
    verdict: Verdict
    comment: str = ""


class ModificationRecord(_Frozen):
    id: str = Field(pattern=r"^M-\d{3,}$")
    proposed_by: Role
    round_index: int = Field(ge=1)
    base_version: str
    new_version: str | None = None
    diff: str
    original_code: str
    replacement_code: str
    original_location: str
    new_location: str | None = None
    justification: str
    expected_impact: str
    disadvantages: str = ""
    reviewers: list[ReviewerVerdict] = Field(default_factory=list)
    test_status: TestStatus = TestStatus.NOT_RUN
    decision: Decision = Decision.PENDING
    decision_reason: str = ""
    created_at: str = Field(default_factory=utc_now)
    decided_at: str | None = None


class TestResult(_Frozen):
    __test__ = False

    test_id: str
    version: str
    status: TestStatus
    actual: Any = None
    error: str | None = None
    runtime_s: float | None = Field(default=None, description="Median of runtimes_s.")
    runtimes_s: list[float] = Field(default_factory=list, description="Every timed run, warmup excluded.")
    peak_memory_bytes: int | None = Field(default=None, description="Python heap peak (tracemalloc), one run.")
    measured: bool = Field(description="True only if the value came from real execution.")
    truncated: bool = Field(default=False, description="'actual' was shortened for storage.")
    non_json: bool = Field(default=False, description="The return value had non-JSON parts, stored as repr.")


SandboxBackendKind = Literal["docker", "subprocess", "none"]


class SuiteRun(_Frozen):
    """One execution of a test suite against one code version inside the sandbox."""

    version: str
    language: str
    entry_point: str
    backend: SandboxBackendKind
    executed: bool
    not_executed_reason: str | None = None
    isolation_warning: str | None = None
    sandbox_env: dict[str, Any] = Field(
        default_factory=dict, description="Interpreter and platform as reported from inside the sandbox."
    )
    method: str = ""
    repeats: int = Field(ge=1)
    warmup: int = Field(ge=0)
    seed: int = 0
    results: list[TestResult] = Field(default_factory=list)
    wall_time_s: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    output_truncated: bool = False
    stderr_tail: str = ""
    max_rss_kb: int | None = Field(default=None, description="Process peak RSS (getrusage) inside the sandbox.")
    started_at: str = Field(default_factory=utc_now)

    def count(self, status: TestStatus) -> int:
        return sum(1 for r in self.results if r.status == status)

    @property
    def all_passed(self) -> bool:
        return self.executed and bool(self.results) and all(r.status == TestStatus.PASS for r in self.results)


class BenchmarkStats(_Frozen):
    """Summary statistics of repeated timed runs of one test against one version."""

    version: str
    test_id: str
    status: TestStatus
    repeats: int = Field(ge=0)
    runtimes_s: list[float]
    median_s: float
    min_s: float
    max_s: float
    mean_s: float
    stdev_s: float | None = None
    peak_memory_bytes: int | None = None
    measured: bool = True


class BenchmarkReport(_Frozen):
    backend: SandboxBackendKind
    isolation_warning: str | None = None
    method: str
    repeats: int = Field(ge=1)
    warmup: int = Field(ge=0)
    seed: int
    order: list[str] = Field(description="Version labels in the order they were executed.")
    suites: list[SuiteRun]
    stats: list[BenchmarkStats]
    started_at: str = Field(default_factory=utc_now)


class CallRecord(_Frozen):
    ts: str = Field(default_factory=utc_now)
    role: Role
    provider: str
    model: str
    call_type: CallType
    tag: str = ""
    attempt: int = Field(ge=1)
    prompt_hash: str
    cache_key: str
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    tokens_estimated: bool
    latency_s: float = Field(ge=0)
    cache_hit: bool
    status: CallStatus
    error: str | None = None


class RunEvent(_Frozen):
    ts: str = Field(default_factory=utc_now)
    phase: Phase
    kind: str
    role: Role | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
