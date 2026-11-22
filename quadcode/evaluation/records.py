"""What the harness writes: its checkpoint (bench.json) and one scored record per problem and condition."""

from __future__ import annotations

import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quadcode.evaluation.conditions import Condition
from quadcode.schemas import RunStatus, SuiteRun, utc_now

BENCH_FILE = "bench.json"
BENCH_EVENTS = "events.jsonl"
HIDDEN_DIR = "hidden"
RUNS_DIR = "runs"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PairStatus(StrEnum):
    PENDING = "pending"     # not started
    FINISHED = "finished"   # the run ended (completed, unverified, failed or needs clarification)
    ABORTED = "aborted"     # stopped by the call budget; --resume continues it
    ERROR = "error"         # the harness could not start or resume the run; --resume tries again


class PairState(_Model):
    problem: str
    condition: Condition
    run_id: str
    status: PairStatus = PairStatus.PENDING
    run_status: RunStatus | None = None
    error: str | None = None


class BenchStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED_BUDGET = "aborted_budget"


class BenchState(_Model):
    bench_id: str
    mock: bool
    rounds: int
    problems: list[str]
    problem_sha256: dict[str, str]
    hidden_sha256: dict[str, str]
    hidden_tests: dict[str, int] = Field(default_factory=dict, description="Number of hidden tests per problem.")
    run_conditions: list[Condition]
    config_sha256: str
    bootstrap_samples: int = Field(ge=100)
    confidence: float = Field(gt=0, lt=1)
    seed: int = 0
    pairs: dict[str, PairState] = Field(default_factory=dict)
    status: BenchStatus = BenchStatus.RUNNING
    stop_reason: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @staticmethod
    def key(problem: str, condition: Condition) -> str:
        return f"{problem}--{condition.value}"

    def pair(self, problem: str, condition: Condition) -> PairState:
        return self.pairs[self.key(problem, condition)]


class CallCost(_Model):
    """Provider attempts and tokens from calls.jsonl. Cache hits cost nothing and are counted apart."""

    provider_attempts: int = 0
    ok: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_estimated: bool = False


ResultStatus = Literal["scored", "no_solution", "missing"]


class ResultRecord(_Model):
    """The hidden-test outcome of one condition on one problem. ``missing`` records are never scored."""

    problem: str
    condition: Condition
    status: ResultStatus
    reason: str = ""
    run_id: str | None = None
    run_status: RunStatus | None = None
    version: str | None = Field(default=None, description="Stored version or file whose code was scored.")
    source_version: str | None = Field(default=None, description="For the oracle: the initial version it picked.")
    code_sha256: str | None = None
    executed: bool = False
    passed: int = 0
    total: int = 0
    perf_tests: list[str] = Field(default_factory=list)
    perf_median_s: dict[str, float] = Field(default_factory=dict, description="Passing performance tests only.")
    controlled: bool | None = Field(default=None, description="Ablations: initial solutions identical to the "
                                    "full run's. None when not applicable.")
    notes: list[str] = Field(default_factory=list)
    cost: CallCost = Field(default_factory=CallCost)

    @property
    def counted(self) -> bool:
        return self.status != "missing"

    @property
    def pass_rate(self) -> float | None:
        if not self.counted:
            return None
        return self.passed / self.total if self.total else 0.0

    @property
    def solved(self) -> bool | None:
        return (self.total > 0 and self.passed == self.total) if self.counted else None

    @property
    def perf_total_s(self) -> float | None:
        """Sum of median runtimes over the hidden performance tests, only if every one of them passed."""
        if not self.perf_tests or set(self.perf_median_s) != set(self.perf_tests):
            return None
        return sum(self.perf_median_s[t] for t in self.perf_tests)


class ProblemResults(_Model):
    """hidden/<problem>.json: every hidden-suite run (by code sha256) and the records built from them."""

    problem: str
    hidden_sha256: str
    entry_point: str
    suites: dict[str, SuiteRun] = Field(default_factory=dict)
    records: list[ResultRecord] = Field(default_factory=list)

    def record(self, condition: Condition) -> ResultRecord | None:
        return next((r for r in self.records if r.condition is condition), None)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".tmp-{path.stem}-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_bench(bench_dir: Path, state: BenchState) -> None:
    state.updated_at = utc_now()
    write_atomic(bench_dir / BENCH_FILE, state.model_dump_json(indent=2) + "\n")


def load_bench(bench_dir: Path) -> BenchState:
    path = bench_dir / BENCH_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {BENCH_FILE} in {bench_dir}")
    return BenchState.model_validate_json(path.read_text(encoding="utf-8"))


def hidden_path(bench_dir: Path, problem: str) -> Path:
    return bench_dir / HIDDEN_DIR / f"{problem}.json"


def load_problem_results(bench_dir: Path, problem: str) -> ProblemResults | None:
    path = hidden_path(bench_dir, problem)
    if not path.is_file():
        return None
    return ProblemResults.model_validate_json(path.read_text(encoding="utf-8"))
