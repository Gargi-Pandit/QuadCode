from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from quadcode.agents import AgentPool
from quadcode.config import AppConfig
from quadcode.gateway import BudgetExhausted, CallOutcome
from quadcode.runstate import (
    ResumeError,
    RoundState,
    RunState,
    check_resumable,
    config_fingerprint,
    load_state,
    problem_fingerprint,
    save_state,
)
from quadcode.schemas import AgentStatus, CallType, ModifyOutput, Role

from .conftest import GatewayFactory


def test_state_round_trip_and_done_steps(tmp_path: Path, app_config: AppConfig) -> None:
    state = RunState(run_id="r", problem_sha256=problem_fingerprint("p"), config_sha256=config_fingerprint(app_config),
                     rounds=2)
    state.mark_done("analysis")
    state.mark_done("analysis")
    state.rounds_state = [RoundState(index=1, base="V0", proposals={Role.GENERATOR: ModifyOutput()})]
    save_state(tmp_path, state)
    loaded = load_state(tmp_path)
    assert loaded.done == ["analysis"] and loaded.is_done("analysis") and not loaded.is_done("initial")
    assert loaded.rounds_state[0].proposals[Role.GENERATOR].proposals == []
    check_resumable(loaded, "p", app_config)
    with pytest.raises(ResumeError):
        check_resumable(loaded, "q", app_config)
    assert not list(tmp_path.glob(".tmp-*"))


def test_load_missing_state(tmp_path: Path) -> None:
    with pytest.raises(ResumeError, match="nothing to resume"):
        load_state(tmp_path)


def test_fingerprint_ignores_run_block_and_backend(app_config: AppConfig) -> None:
    data = app_config.model_dump(mode="json")
    data["run"]["rounds"] = 5
    data["sandbox"]["backend"] = "subprocess"
    same = AppConfig.model_validate(data)
    assert config_fingerprint(same) == config_fingerprint(app_config)
    data["selection"] = {"runtime_tolerance": 0.5}
    assert config_fingerprint(AppConfig.model_validate(data)) != config_fingerprint(app_config)


@pytest.mark.parametrize("parallel", [True, False])
def test_pool_returns_role_order_and_runs_in_parallel(make_gateway: GatewayFactory, parallel: bool) -> None:
    pool = AgentPool(make_gateway(), parallel=parallel)
    threads: set[str] = set()

    def task(agent: Any) -> CallOutcome[Any]:
        threads.add(threading.current_thread().name)
        return agent.ask(CallType.MODIFY, "prompt", ModifyOutput, "round-2")

    result = pool.run({r: task for r in reversed(list(Role))})
    assert list(result.outcomes) == list(Role)
    assert all(o.status is AgentStatus.DONE for o in result.outcomes.values())
    assert not result.budget_exhausted
    assert all(t.startswith("agent") for t in threads) == parallel


@pytest.mark.parametrize("parallel", [True, False])
def test_pool_reports_budget_without_losing_finished_calls(make_gateway: GatewayFactory, parallel: bool) -> None:
    pool = AgentPool(make_gateway(budget=2), parallel=parallel)
    result = pool.run({r: (lambda a: a.ask(CallType.MODIFY, "p", ModifyOutput, "round-2")) for r in Role})
    assert result.budget_exhausted
    assert len(result.outcomes) == 2 and all(o.ok for o in result.outcomes.values())


def test_agent_knows_its_slot(make_gateway: GatewayFactory) -> None:
    pool = AgentPool(make_gateway(), parallel=False)
    agent = pool.agents[Role.OPTIMIZER]
    assert agent.provider == "mock" and agent.model == "mock-optimizer-v1"
    with pytest.raises(BudgetExhausted):
        AgentPool(make_gateway(budget=0, run_name="b"), parallel=False).agents[Role.GENERATOR].ask(
            CallType.MODIFY, "p", ModifyOutput, "round-2")
