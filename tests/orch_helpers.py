"""Build an orchestrator on the mock provider, the subprocess sandbox and a fake clock."""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quadcode.cache import DiskCache
from quadcode.config import AppConfig, Secrets, parse_config
from quadcode.gateway import CallBudget, LLMGateway
from quadcode.orchestrator import Orchestrator
from quadcode.providers.base import LLMClient
from quadcode.providers.mock import FailureSpec, MockProvider, MockScenario
from quadcode.providers.mock_scenarios import get_scenario
from quadcode.ratelimit import build_limiters
from quadcode.runlog import RunLog
from quadcode.sandbox import SubprocessSandbox
from quadcode.schemas import CallType, Role, RunEvent

from .conftest import BASE_CONFIG, FakeClock

# Timing noise must never decide anything in tests: a huge runtime tolerance makes runtime a
# tie, so selection falls through to measured memory and code size, which are deterministic.
TEST_CONFIG: dict[str, Any] = {
    **copy.deepcopy(BASE_CONFIG),
    "sandbox": {"backend": "subprocess", "test_timeout_s": 1.0, "wall_timeout_s": 30.0, "test_repeats": 2,
                "benchmark_repeats": 2, "warmup": 1},
    "selection": {"runtime_tolerance": 1000.0},
}


def make_config(**overrides: Any) -> AppConfig:
    data = copy.deepcopy(TEST_CONFIG)
    for key, value in overrides.items():
        data[key] = {**data.get(key, {}), **value} if isinstance(value, dict) else value
    return parse_config(data)



def scenario_with(responses: dict[tuple[Role, CallType, str], str] | None = None,
                  failures: dict[tuple[Role, CallType, str], FailureSpec] | None = None,
                  drop: list[tuple[Role, CallType, str]] | None = None) -> MockScenario:
    """The two_sum scenario with some responses or failures replaced."""
    base = get_scenario("two_sum")
    resp = {**base.responses, **(responses or {})}
    fail = {**base.failures, **(failures or {})}
    for key in drop or []:
        resp.pop(key, None)
        fail.pop(key, None)
    return MockScenario(name="two_sum-variant", responses=resp, failures=fail)


@dataclass
class Built:
    orch: Orchestrator
    gateway: LLMGateway
    run_dir: Path
    clock: FakeClock

    def provider(self, role: Role) -> MockProvider:
        client = self.gateway.clients[role][0]
        assert isinstance(client, MockProvider)
        return client


def build(root: Path, *, scenario: MockScenario | None = None, config: AppConfig | None = None,
          budget: int | None = None, cache_enabled: bool = False, run_name: str = "run", rounds: int = 3,
          resume: bool = False, problem: str = "Two sum problem statement.") -> Built:
    cfg = config or make_config()
    scen = scenario or get_scenario("two_sum")
    clock = FakeClock()
    clients: dict[Role, LLMClient] = {r: MockProvider("mock", r, scen) for r in Role}
    run_dir = root / run_name
    gateway = LLMGateway(
        config=cfg, clients=clients,
        limiters=build_limiters({n: (p.rpm, p.tpm) for n, p in cfg.providers.items()}, clock=clock,
                                sleep=clock.sleep),
        cache=DiskCache(root / "cache", enabled=cache_enabled), runlog=RunLog(run_dir), budget=CallBudget(budget),
        secrets=Secrets({}), sleep=clock.sleep, clock=clock, rng=random.Random(0))
    sandbox = SubprocessSandbox(cfg.sandbox)
    if resume:
        orch = Orchestrator.resume(cfg, gateway, sandbox, run_dir)
    else:
        orch = Orchestrator.create(cfg, gateway, sandbox, run_dir, run_name, problem, rounds)
    return Built(orch, gateway, run_dir, clock)


def events(run_dir: Path, kind: str | None = None) -> list[RunEvent]:
    evs = RunLog(run_dir).read_events()
    return [e for e in evs if kind is None or e.kind == kind]


def calls(run_dir: Path, pred: Callable[[dict[str, Any]], bool] = lambda c: True) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines() if line.strip()]
    return [r for r in rows if pred(r)]
