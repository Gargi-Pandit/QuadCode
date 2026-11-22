"""Checkpoint of a run (``state.json``), written atomically after every step.

The state holds every valid agent output, so ``--resume`` never repeats a completed call,
even with the cache switched off. Calls that failed have no stored output and are tried
again on resume while their phase is still open; a closed phase is never reopened.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from quadcode.cache import sha256_text
from quadcode.config import AppConfig, ProviderConfig
from quadcode.schemas import (
    Ablation,
    AgentAvailability,
    InitialSolution,
    ModifyOutput,
    ProblemSpec,
    Review,
    Role,
    RunStatus,
    SelectionEvidence,
    SuiteRun,
    utc_now,
)

STATE_FILE = "state.json"


class ResumeError(RuntimeError):
    """The run cannot be resumed with the given problem or configuration."""


class _State(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RoundState(_State):
    index: int = Field(ge=1)
    base: str
    proposals: dict[Role, ModifyOutput] = Field(default_factory=dict)
    unavailable: list[Role] = Field(default_factory=list)
    modify_done: bool = False
    planned_ids: dict[str, str] = Field(default_factory=dict, description="'<role>:<n>' -> modification id")
    records_done: bool = False
    decided: bool = False
    accepted: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    round_version: str | None = None


class RunState(_State):
    run_id: str
    problem_sha256: str
    config_sha256: str
    rounds: int = Field(ge=1)
    ablation: Ablation = Ablation.NONE
    status: RunStatus = RunStatus.RUNNING
    done: list[str] = Field(default_factory=list, description="Completed steps, in order.")
    spec: ProblemSpec | None = None
    analysis_role: Role | None = None
    initial: dict[Role, InitialSolution] = Field(default_factory=dict)
    reviews: dict[Role, Review] = Field(default_factory=dict)
    initial_suites: dict[str, SuiteRun] = Field(default_factory=dict)
    disputed_tests: list[str] = Field(default_factory=list, description="Tests every executed initial "
                                      "implementation failed. Set aside: not used to rank, accept or verify.")
    reference: SelectionEvidence | None = None
    rounds_state: list[RoundState] = Field(default_factory=list)
    availability: list[AgentAvailability] = Field(default_factory=list)
    final_version: str | None = None
    stop_reason: str = ""
    unresolved: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @property
    def roles(self) -> list[Role]:
        """The agent slots taking part in this run, in slot order."""
        return [r for r in Role if r not in self.ablation.excluded_roles]

    def is_done(self, step: str) -> bool:
        return step in self.done

    def mark_done(self, step: str) -> None:
        if step not in self.done:
            self.done = [*self.done, step]


def config_fingerprint(config: AppConfig) -> str:
    """Hash of everything that changes what agents are asked or how code is judged.

    The ``run`` block (rounds, budget, cache, parallelism, paths), the sandbox backend choice and
    ``pricing`` (used only for report cost estimates) and ``evaluation`` (used only by the benchmark harness)
    are excluded, so a run can be resumed with a
    different call budget, with the cache switched off, on another available backend or after
    prices were added.
    """
    data = config.model_dump(mode="json", exclude={"run": True, "sandbox": {"backend"}, "pricing": True,
                                                    "evaluation": True})
    # Settings added after runs already existed are left out while they hold their default, so
    # those runs stay resumable.
    for agent in data["agents"].values():
        if not agent.get("request_options"):
            agent.pop("request_options", None)
        if not agent.get("inline_system"):
            agent.pop("inline_system", None)
        if not agent.get("max_tokens"):
            agent.pop("max_tokens", None)
        if not agent.get("fallbacks"):
            agent.pop("fallbacks", None)
    for prov in data["providers"].values():
        if prov.get("max_retry_after_s") == ProviderConfig.model_fields["max_retry_after_s"].default:
            prov.pop("max_retry_after_s", None)
    return sha256_text(json.dumps(data, sort_keys=True))


def problem_fingerprint(problem: str) -> str:
    return sha256_text(problem)


def save_state(run_dir: Path, state: RunState) -> Path:
    path = run_dir / STATE_FILE
    state.updated_at = utc_now()
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=".tmp-state-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(state.model_dump_json(indent=2))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_state(run_dir: Path) -> RunState:
    path = run_dir / STATE_FILE
    if not path.is_file():
        raise ResumeError(f"no {STATE_FILE} in {run_dir}; nothing to resume")
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))


def check_resumable(state: RunState, problem: str, config: AppConfig) -> None:
    if state.problem_sha256 != problem_fingerprint(problem):
        raise ResumeError("the problem text differs from the one this run started with")
    if state.config_sha256 != config_fingerprint(config):
        raise ResumeError("config.yaml (providers, agents, token caps, sandbox, selection or prompts) differs "
                          "from the configuration this run started with")
