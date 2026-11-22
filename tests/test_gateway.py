from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from quadcode.config import AppConfig, Secrets
from quadcode.gateway import BudgetExhausted, LLMGateway
from quadcode.providers.base import LLMClient, LLMRequest, LLMResponse, PermanentError
from quadcode.providers.mock import MockProvider, MockScenario
from quadcode.schemas import (
    AgentStatus,
    CallStatus,
    CallType,
    InitialSolution,
    ModifyOutput,
    Review,
    Role,
)
from tests.conftest import FakeClock, GatewayFactory


def _initial(gw: LLMGateway, role: Role, prompt: str = "solve two sum"):  # type: ignore[no-untyped-def]
    return gw.call(role, CallType.INITIAL, "system", prompt, InitialSolution)


def test_success_logs_provider_and_model(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    out = _initial(gw, Role.REVIEWER)
    assert out.status is AgentStatus.DONE and out.ok
    assert out.data is not None and "def two_sum" in out.data.code
    assert (out.provider, out.model) == ("mock", "mock-reviewer-v1")
    [rec] = gw.runlog.read_calls()
    assert rec.status is CallStatus.OK and not rec.cache_hit
    assert (rec.role, rec.provider, rec.model) == (Role.REVIEWER, "mock", "mock-reviewer-v1")
    assert rec.tokens_estimated is True and rec.tokens_out > 0


def test_fenced_response_is_parsed(make_gateway: GatewayFactory) -> None:
    out = _initial(make_gateway(), Role.VALIDATOR)
    assert out.data is not None and len(out.data.tests) == 9


def test_cache_hit_is_logged_and_free(make_gateway: GatewayFactory) -> None:
    gw = make_gateway(budget=1)
    first = _initial(gw, Role.GENERATOR)
    second = _initial(gw, Role.GENERATOR)
    assert first.status is AgentStatus.DONE and second.status is AgentStatus.CACHED
    assert second.data == first.data
    recs = gw.runlog.read_calls()
    assert [r.cache_hit for r in recs] == [False, True]
    assert gw.budget.used == 1
    client = gw.clients[Role.GENERATOR][0]
    assert isinstance(client, MockProvider) and client.calls_received == 1


def test_resume_uses_cache_across_gateways(make_gateway: GatewayFactory) -> None:
    _initial(make_gateway(run_name="a"), Role.OPTIMIZER)
    gw2 = make_gateway(run_name="b")
    assert _initial(gw2, Role.OPTIMIZER).status is AgentStatus.CACHED
    client = gw2.clients[Role.OPTIMIZER][0]
    assert isinstance(client, MockProvider) and client.calls_received == 0


def test_cache_disabled(make_gateway: GatewayFactory) -> None:
    gw = make_gateway(cache_enabled=False)
    _initial(gw, Role.GENERATOR)
    assert _initial(gw, Role.GENERATOR).status is AgentStatus.DONE


def test_budget_exhaustion_raises_and_logs(make_gateway: GatewayFactory) -> None:
    gw = make_gateway(budget=1)
    _initial(gw, Role.GENERATOR)
    with pytest.raises(BudgetExhausted):
        _initial(gw, Role.REVIEWER)
    last = gw.runlog.read_calls()[-1]
    assert last.status is CallStatus.BUDGET_EXHAUSTED and last.role is Role.REVIEWER
    # cached results remain available after the budget is spent
    assert _initial(gw, Role.GENERATOR).status is AgentStatus.CACHED


def test_transient_rate_limit_retried(make_gateway: GatewayFactory, clock: FakeClock) -> None:
    gw = make_gateway()
    out = gw.call(Role.VALIDATOR, CallType.MODIFY, "s", "p", ModifyOutput, tag="round-1")
    assert out.status is AgentStatus.DONE and out.data is not None
    recs = gw.runlog.read_calls()
    assert [(r.attempt, r.status) for r in recs] == [(1, CallStatus.PROVIDER_ERROR), (2, CallStatus.OK)]
    assert "RateLimitError" in (recs[0].error or "")
    assert len(clock.sleeps) == 1 and 0 <= clock.sleeps[0] <= 0.5


def test_always_failing_agent_is_unavailable(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    out = gw.call(Role.VALIDATOR, CallType.REVIEW, "s", "p", Review)
    assert out.status is AgentStatus.UNAVAILABLE
    assert out.data is None and not out.ok
    assert out.error is not None and "4 attempt(s)" in out.error
    recs = gw.runlog.read_calls()
    assert len(recs) == 4 and all(r.status is CallStatus.PROVIDER_ERROR for r in recs)
    # failure is not cached: a later call contacts the provider again
    gw.call(Role.VALIDATOR, CallType.REVIEW, "s", "p", Review)
    client = gw.clients[Role.VALIDATOR][0]
    assert isinstance(client, MockProvider) and client.calls_received == 8


def test_invalid_output_is_unavailable_and_not_cached(make_gateway: GatewayFactory) -> None:
    scen = MockScenario(name="bad", responses={(Role.GENERATOR, CallType.INITIAL, "*"): "I refuse to answer in JSON."})
    gw = make_gateway(scenario=scen)
    out = _initial(gw, Role.GENERATOR)
    assert out.status is AgentStatus.UNAVAILABLE and out.raw_text == "I refuse to answer in JSON."
    records = gw.runlog.read_calls()
    assert [(r.status, r.repair) for r in records] == [(CallStatus.INVALID_OUTPUT, 0), (CallStatus.INVALID_OUTPUT, 1)]
    assert _initial(gw, Role.GENERATOR).status is AgentStatus.UNAVAILABLE  # retried, not served from cache


def test_schema_mismatch_is_invalid_output(make_gateway: GatewayFactory) -> None:
    scen = MockScenario(name="bad", responses={(Role.GENERATOR, CallType.INITIAL, "*"): '{"code": "x"}'})
    out = _initial(make_gateway(scenario=scen), Role.GENERATOR)
    assert out.status is AgentStatus.UNAVAILABLE and "InitialSolution" in (out.error or "")


class _LeakyClient(LLMClient):
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise PermanentError("401 invalid key sk-live-123456")


def test_secrets_redacted_from_logs_and_outcomes(make_gateway: GatewayFactory, tmp_path: Path) -> None:
    gw = make_gateway(secrets=Secrets({"MY_KEY": "sk-live-123456"}))
    gw.clients[Role.GENERATOR] = [_LeakyClient("mock")]
    out = _initial(gw, Role.GENERATOR)
    assert out.status is AgentStatus.UNAVAILABLE
    assert "sk-live-123456" not in (out.error or "")
    assert len(gw.runlog.read_calls()) == 1  # permanent errors are not retried
    for f in tmp_path.rglob("*.jsonl"):
        assert "sk-live-123456" not in f.read_text()


def test_requires_distinct_client_instances(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    shared = gw.clients[Role.GENERATOR][0]
    with pytest.raises(ValueError, match="own client"):
        LLMGateway(gw.config, {r: shared for r in Role}, gw.limiters, gw.cache, gw.runlog, gw.budget, gw.secrets)


def test_parallel_calls_all_logged(make_gateway: GatewayFactory) -> None:
    gw = make_gateway()
    with ThreadPoolExecutor(max_workers=4) as pool:
        outs = list(pool.map(lambda r: _initial(gw, r), list(Role)))
    assert all(o.status is AgentStatus.DONE for o in outs)
    assert {r.role for r in gw.runlog.read_calls()} == set(Role)


def test_from_config_builds_four_independent_clients(app_config: AppConfig, tmp_path: Path) -> None:
    gw = LLMGateway.from_config(app_config, Secrets({}), tmp_path / "run", cache_root=tmp_path / "c")
    assert len({id(c) for c in gw.clients.values()}) == 4
    out = _initial(gw, Role.GENERATOR)
    assert out.status is AgentStatus.DONE
    line = json.loads((tmp_path / "run" / "calls.jsonl").read_text().splitlines()[0])
    assert line["model"] == "mock-generator-v1" and line["provider"] == "mock"
