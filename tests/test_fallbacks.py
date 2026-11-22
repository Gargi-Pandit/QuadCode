"""Fallback models per agent slot: config rules, the gateway's order of tries, and how reports name them."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from quadcode.cache import DiskCache
from quadcode.cli_solve import mock_config
from quadcode.config import AppConfig, ConfigError, Secrets, missing_keys, parse_config
from quadcode.gateway import BudgetExhausted, CallBudget, LLMGateway
from quadcode.providers.base import LLMClient, LLMRequest, LLMResponse, PermanentError, TransientError
from quadcode.providers.mock import MockProvider
from quadcode.providers.mock_scenarios import get_scenario
from quadcode.ratelimit import build_limiters
from quadcode.report.data import LoggedEvent, RunRecords
from quadcode.runlog import RunLog
from quadcode.runstate import config_fingerprint
from quadcode.schemas import AgentStatus, CallStatus, CallType, Phase, Role, RunEvent

from .conftest import BASE_CONFIG, FakeClock

OK = '{"ok": true}'


class Reply(BaseModel):
    ok: bool


class Scripted(LLMClient):
    """Answers with the given steps in order (the last one repeats); an exception step is raised."""

    def __init__(self, name: str, *steps: str | Exception) -> None:
        super().__init__(name)
        self.steps = list(steps)
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return LLMResponse(text=step, tokens_in=10, tokens_out=5, tokens_estimated=False, model=request.model,
                           finish_reason="stop")


def _data(**generator: Any) -> dict[str, Any]:
    data: dict[str, Any] = copy.deepcopy(BASE_CONFIG)
    data["providers"]["backup"] = {**data["providers"]["mock"], "max_retries": 0}
    data["agents"]["generator"]["fallbacks"] = [{"provider": "backup", "model": "backup-model-a"},
                                                {"provider": "backup", "model": "backup-model-b"}]
    data["agents"]["generator"].update(generator)
    return data


def _gateway(tmp_path: Path, chain: list[LLMClient], budget: int | None = None,
             config: AppConfig | None = None) -> tuple[LLMGateway, FakeClock]:
    cfg = config or parse_config(_data())
    clock = FakeClock()
    clients: dict[Role, list[LLMClient]] = {
        r: [MockProvider("mock", r, get_scenario("two_sum"))] for r in Role if r is not Role.GENERATOR}
    clients[Role.GENERATOR] = chain
    gw = LLMGateway(config=cfg, clients=clients,
                    limiters=build_limiters({n: (p.rpm, p.tpm) for n, p in cfg.providers.items()}, clock=clock,
                                            sleep=clock.sleep),
                    cache=DiskCache(tmp_path / "cache"), runlog=RunLog(tmp_path / "run"), budget=CallBudget(budget),
                    secrets=Secrets({}), sleep=clock.sleep, clock=clock, rng=random.Random(0))
    return gw, clock


def _ask(gw: LLMGateway, prompt: str = "p1", choice: int | None = None):  # type: ignore[no-untyped-def]
    return gw.call(Role.GENERATOR, CallType.REVIEW, "system", prompt, Reply, tag="t", choice=choice)


def _trail(gw: LLMGateway) -> list[tuple[str, CallStatus, int, bool]]:
    return [(r.model, r.status, r.choice, r.cache_hit) for r in gw.runlog.read_calls()]


# ------------------------------------------------------------------ gateway


def test_provider_failure_falls_back_and_stays_on_the_fallback(tmp_path: Path) -> None:
    first, fb_a, fb_b = Scripted("mock", PermanentError("HTTP 503: overloaded")), Scripted("backup", OK), \
        Scripted("backup", OK)
    gw, _ = _gateway(tmp_path, [first, fb_a, fb_b])
    out = _ask(gw)
    assert out.ok and out.status is AgentStatus.DONE
    assert (out.provider, out.model) == ("backup", "backup-model-a")
    assert _trail(gw) == [("mock-generator-v1", CallStatus.PROVIDER_ERROR, 0, False),
                          ("backup-model-a", CallStatus.OK, 1, False)]
    assert gw.active_choice(Role.GENERATOR) == 1
    assert _ask(gw, "p2").model == "backup-model-a"
    assert (first.calls, fb_a.calls, fb_b.calls) == (1, 2, 0)  # the failed provider is not asked again


def test_invalid_answer_falls_back_for_that_call_only(tmp_path: Path) -> None:
    first, fb_a = Scripted("mock", "no json here", "still no json", OK), Scripted("backup", OK)
    gw, _ = _gateway(tmp_path, [first, fb_a, Scripted("backup", OK)])
    assert _ask(gw).model == "backup-model-a"
    assert [s for _, s, _, _ in _trail(gw)] == [CallStatus.INVALID_OUTPUT, CallStatus.INVALID_OUTPUT, CallStatus.OK]
    assert gw.active_choice(Role.GENERATOR) == 0
    assert _ask(gw, "p2").model == "mock-generator-v1"
    assert (first.calls, fb_a.calls) == (3, 1)


class Recording(Scripted):
    def __init__(self, name: str, *steps: str | Exception) -> None:
        super().__init__(name, *steps)
        self.prompts: list[str] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.prompts.append(request.prompt)
        return super().complete(request)


def test_invalid_answer_is_repaired_by_the_same_model(tmp_path: Path) -> None:
    first = Recording("mock", '{"ok": null}', OK)
    fb_a = Scripted("backup", OK)
    gw, _ = _gateway(tmp_path, [first, fb_a, Scripted("backup", OK)])
    out = _ask(gw)
    assert out.ok and out.model == "mock-generator-v1" and fb_a.calls == 0
    records = gw.runlog.read_calls()
    assert [(r.status, r.repair) for r in records] == [(CallStatus.INVALID_OUTPUT, 0), (CallStatus.OK, 1)]
    original, fix = first.prompts
    assert original.startswith("p1\n\nOUTPUT LIMIT: your whole reply") and "200 tokens" in original
    assert fix.startswith("p1\n") and "could not be used" in fix and fix.count("OUTPUT LIMIT") == 1
    assert "not 10**9" in fix  # the JSON rules models most often break are restated
    assert "Input should be a valid boolean" in fix  # the validation error is passed on
    assert '{"ok": null}' in fix  # and so is the rejected answer


def test_cut_off_answer_is_asked_to_be_shorter(tmp_path: Path) -> None:
    class CutOff(Recording):
        def complete(self, request: LLMRequest) -> LLMResponse:
            resp = super().complete(request)
            if len(self.prompts) == 1:
                return LLMResponse(text='{"ok": tr', tokens_in=10, tokens_out=5, tokens_estimated=False,
                                   model=request.model, finish_reason="length")
            return resp

    first = CutOff("mock", OK)
    gw, _ = _gateway(tmp_path, [first, Scripted("backup", OK), Scripted("backup", OK)])
    assert _ask(gw).ok
    fix = first.prompts[1]
    assert "cut off at the output limit" in fix and "Your previous answer was" not in fix


def test_repairs_can_be_switched_off(tmp_path: Path) -> None:
    data = _data()
    data["run"]["repair_attempts"] = 0
    first = Scripted("mock", '{"ok": null}', OK)
    gw, _ = _gateway(tmp_path, [first, Scripted("backup", OK), Scripted("backup", OK)], config=parse_config(data))
    assert _ask(gw).model == "backup-model-a" and first.calls == 1


def test_a_repair_counts_against_the_budget(tmp_path: Path) -> None:
    first = Scripted("mock", '{"ok": null}', OK)
    gw, _ = _gateway(tmp_path, [first, Scripted("backup", OK), Scripted("backup", OK)], budget=1)
    with pytest.raises(BudgetExhausted):
        _ask(gw)
    assert first.calls == 1


def test_every_model_failing_is_unavailable_and_names_each_failure(tmp_path: Path) -> None:
    chain = [Scripted("mock", PermanentError("first down")), Scripted("backup", PermanentError("a down")),
             Scripted("backup", PermanentError("b down"))]
    gw, _ = _gateway(tmp_path, list(chain))
    out = _ask(gw)
    assert out.status is AgentStatus.UNAVAILABLE and out.data is None
    assert out.error is not None and out.error.startswith("all 3 configured models failed")
    for label, msg in [("mock/mock-generator-v1", "first down"), ("backup/backup-model-a", "a down"),
                       ("backup/backup-model-b", "b down")]:
        assert f"[{label}]" in out.error and msg in out.error
    assert (out.provider, out.model) == ("backup", "backup-model-b")
    assert gw.active_choice(Role.GENERATOR) == 2  # the last model stays in use
    assert not _ask(gw, "p2").ok
    assert [c.calls for c in chain] == [1, 1, 2]


def test_transient_errors_are_retried_before_falling_back(tmp_path: Path) -> None:
    first = Scripted("mock", *[TransientError("HTTP 503")] * 4)
    gw, clock = _gateway(tmp_path, [first, Scripted("backup", OK), Scripted("backup", OK)])
    assert _ask(gw).model == "backup-model-a"
    assert first.calls == 4 and len(clock.sleeps) == 3  # max_retries=3 on the first choice's provider


def test_cached_answer_of_a_fallback_is_reused_without_asking_the_first_choice(tmp_path: Path) -> None:
    gw, _ = _gateway(tmp_path, [Scripted("mock", PermanentError("down")), Scripted("backup", OK),
                                Scripted("backup", OK)])
    assert _ask(gw).model == "backup-model-a"
    fresh = Scripted("mock", OK)
    gw2, _ = _gateway(tmp_path, [fresh, Scripted("backup", OK), Scripted("backup", OK)])
    out = _ask(gw2)
    assert out.status is AgentStatus.CACHED and out.model == "backup-model-a"
    assert fresh.calls == 0
    assert _trail(gw2)[-1] == ("backup-model-a", CallStatus.OK, 1, True)


def test_pinned_choice_never_falls_back(tmp_path: Path) -> None:
    fb_a, fb_b = Scripted("backup", OK), Scripted("backup", OK)
    gw, _ = _gateway(tmp_path, [Scripted("mock", PermanentError("down")), fb_a, fb_b])
    assert _ask(gw, choice=0).status is AgentStatus.UNAVAILABLE
    assert fb_a.calls == 0 and gw.active_choice(Role.GENERATOR) == 0
    assert _ask(gw, "p2", choice=2).model == "backup-model-b"
    with pytest.raises(ValueError, match="no model #3"):
        _ask(gw, choice=3)


def test_call_budget_is_not_bypassed_by_fallbacks(tmp_path: Path) -> None:
    fb_a = Scripted("backup", OK)
    gw, _ = _gateway(tmp_path, [Scripted("mock", PermanentError("down")), fb_a, Scripted("backup", OK)], budget=1)
    with pytest.raises(BudgetExhausted):
        _ask(gw)
    assert fb_a.calls == 0
    assert _trail(gw)[-1][1] is CallStatus.BUDGET_EXHAUSTED


def test_each_model_needs_its_own_client(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lists 3 model"):
        _gateway(tmp_path, [Scripted("mock", OK)])
    shared = Scripted("backup", OK)
    with pytest.raises(ValueError, match="own client"):
        _gateway(tmp_path, [Scripted("mock", OK), shared, shared])


# ------------------------------------------------------------------ config


def test_fallbacks_are_validated_like_first_choices() -> None:
    data = _data()
    data["agents"]["generator"]["fallbacks"][0]["provider"] = "nope"
    with pytest.raises(ConfigError, match="'generator' fallback 1 references unknown provider 'nope'"):
        parse_config(data)
    data = _data()
    data["agents"]["generator"]["fallbacks"][1]["model"] = "backup-model-a"
    with pytest.raises(ConfigError, match="lists backup/backup-model-a twice"):
        parse_config(data)
    data = _data()
    data["agents"]["generator"]["fallbacks"][0]["request_options"] = {"x": 1}
    with pytest.raises(ConfigError, match="fallback 1: request_options and inline_system are not used by mock"):
        parse_config(data)


def test_a_fallback_may_not_be_another_slots_model() -> None:
    data = _data()
    data["agents"]["reviewer"]["fallbacks"] = [{"provider": "backup", "model": "vendor/mock-generator-v1:free"}]
    with pytest.raises(ConfigError, match="'generator' and 'reviewer' both list model 'mock-generator-v1'"):
        parse_config(data)
    data["allow_duplicate_models"] = True
    assert parse_config(data).agent(Role.REVIEWER).fallbacks[0].model == "vendor/mock-generator-v1:free"


def test_the_same_model_on_another_host_may_back_up_its_own_slot() -> None:
    data = _data()
    data["agents"]["generator"]["fallbacks"] = [{"provider": "backup", "model": "vendor/mock-generator-v1:free"}]
    assert len(parse_config(data).agent(Role.GENERATOR).choices()) == 2


def test_caps_come_from_the_model_in_use() -> None:
    data = _data()
    data["agents"]["generator"]["fallbacks"][0]["max_tokens"] = {"review": 999}
    cfg = parse_config(data)
    fallback = cfg.agent(Role.GENERATOR).fallbacks[0]
    assert cfg.cap_for(CallType.REVIEW, Role.GENERATOR, fallback) == 999
    assert cfg.cap_for(CallType.REVIEW, Role.GENERATOR) == 200


def test_missing_keys_include_fallback_providers() -> None:
    data = _data()
    data["providers"]["real"] = {"kind": "openai_compat", "base_url": "https://x.test/v1", "api_key_env": "QC_FB",
                                 "rpm": 10, "tpm": 1000}
    data["agents"]["generator"]["fallbacks"] = [{"provider": "real", "model": "real-model"}]
    cfg = parse_config(data)
    assert missing_keys(cfg, Secrets({})) == [("real", "QC_FB")]
    assert missing_keys(cfg, Secrets({}), [Role.REVIEWER]) == []


def test_fingerprint_ignores_an_empty_fallback_list_only() -> None:
    plain = copy.deepcopy(BASE_CONFIG)
    empty = copy.deepcopy(BASE_CONFIG)
    empty["agents"]["generator"]["fallbacks"] = []
    assert config_fingerprint(parse_config(plain)) == config_fingerprint(parse_config(empty))
    assert config_fingerprint(parse_config(_data())) != config_fingerprint(parse_config(plain))


def test_describe_agents_and_mock_mode() -> None:
    cfg = parse_config(_data())
    rows = {r["role"]: r for r in cfg.describe_agents()}
    assert rows["generator"]["fallbacks"] == "backup/backup-model-a, backup/backup-model-b"
    assert rows["reviewer"]["fallbacks"] == "-"
    data = _data()
    data["providers"]["real"] = {"kind": "openai_compat", "base_url": "https://x.test/v1", "api_key_env": "QC_FB",
                                 "rpm": 10, "tpm": 1000}
    data["agents"]["reviewer"]["provider"] = "real"
    assert all(not a.fallbacks for a in mock_config(parse_config(data), None).agents.values())


# ------------------------------------------------------------------ report


def _message(line: int, role: Role, provider: str, model: str, ok: bool) -> LoggedEvent:
    payload: dict[str, Any] = {"call_type": "initial", "tag": "initial", "provider": provider, "model": model}
    if not ok:
        payload["error"] = "down"
    return LoggedEvent(line, RunEvent(phase=Phase.INITIAL, kind="agent_output" if ok else "agent_unavailable",
                                      role=role, payload=payload))


def test_report_names_the_model_that_answered(tmp_path: Path) -> None:
    rec = RunRecords(run_dir=tmp_path, run_id="r", config=parse_config(_data()), events=[
        _message(1, Role.GENERATOR, "mock", "mock-generator-v1", False),
        _message(2, Role.GENERATOR, "backup", "backup-model-a", True),
        _message(3, Role.REVIEWER, "mock", "mock-reviewer-v1", True),
    ])
    assert rec.agent_identity(Role.GENERATOR) == "backup/backup-model-a (fallback; first choice mock/mock-generator-v1)"
    assert rec.agent_identity(Role.REVIEWER) == "mock/mock-reviewer-v1"
    assert rec.agent_identity(Role.OPTIMIZER) == "mock/mock-optimizer-v1"  # no answer yet: the configured model


def test_a_failed_attempt_does_not_make_provider_token_counts_estimates() -> None:
    from quadcode.report.tables import _Usage
    from quadcode.schemas import CallRecord

    def rec(status: CallStatus, tokens: int, estimated: bool) -> CallRecord:
        return CallRecord(role=Role.REVIEWER, provider="p", model="m", call_type=CallType.REVIEW, attempt=1,
                          prompt_hash="h", cache_key="k", tokens_in=tokens, tokens_out=tokens,
                          tokens_estimated=estimated, latency_s=0.1, cache_hit=False, status=status)

    usage = _Usage()
    usage.add(rec(CallStatus.OK, 100, False))
    usage.add(rec(CallStatus.PROVIDER_ERROR, 0, True))
    assert usage.estimated is False and usage.tokens_in == 100
    usage.add(rec(CallStatus.OK, 50, True))
    assert usage.estimated is True
