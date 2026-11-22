"""Shared fixtures. All tests are offline and use the mock provider."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from quadcode.cache import DiskCache
from quadcode.config import AppConfig, Secrets, parse_config
from quadcode.gateway import CallBudget, LLMGateway
from quadcode.providers.base import LLMClient
from quadcode.providers.mock import MockProvider, MockScenario
from quadcode.providers.mock_scenarios import get_scenario
from quadcode.ratelimit import build_limiters
from quadcode.runlog import RunLog
from quadcode.schemas import Role

BASE_CONFIG: dict[str, Any] = {
    "providers": {
        "mock": {
            "kind": "mock",
            "scenario": "two_sum",
            "rpm": 600,
            "tpm": 1_000_000,
            "timeout_s": 5,
            "max_retries": 3,
            "backoff_base_s": 0.5,
            "backoff_cap_s": 4,
        }
    },
    "agents": {r.value: {"provider": "mock", "model": f"mock-{r.value}-v1"} for r in Role},
    "max_tokens": {"analysis": 100, "initial": 300, "review": 200, "modify": 200, "challenge": 80},
    "run": {"rounds": 3},
}


class FakeClock:
    """Monotonic fake time; ``sleep`` advances it and records the requested delays."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def config_dict() -> dict[str, Any]:
    return copy.deepcopy(BASE_CONFIG)


@pytest.fixture
def app_config(config_dict: dict[str, Any]) -> AppConfig:
    return parse_config(config_dict)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


GatewayFactory = Callable[..., LLMGateway]


@pytest.fixture
def make_gateway(tmp_path: Path, app_config: AppConfig, clock: FakeClock) -> GatewayFactory:
    def factory(
        scenario: MockScenario | None = None,
        budget: int | None = None,
        secrets: Secrets | None = None,
        config: AppConfig | None = None,
        cache_enabled: bool = True,
        run_name: str = "run",
    ) -> LLMGateway:
        cfg = config or app_config
        scen = scenario or get_scenario("two_sum")
        clients: dict[Role, LLMClient] = {r: MockProvider("mock", r, scen) for r in Role}
        limits = {n: (p.rpm, p.tpm) for n, p in cfg.providers.items()}
        return LLMGateway(
            config=cfg,
            clients=clients,
            limiters=build_limiters(limits, clock=clock, sleep=clock.sleep),
            cache=DiskCache(tmp_path / "cache", enabled=cache_enabled),
            runlog=RunLog(tmp_path / run_name),
            budget=CallBudget(budget),
            secrets=secrets or Secrets({}),
            sleep=clock.sleep,
            clock=clock,
            rng=random.Random(0),
        )

    return factory
