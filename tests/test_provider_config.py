"""Config additions for real providers: request options, inline system, optional caps, key checks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from quadcode.config import ConfigError, Secrets, load_config, missing_keys, parse_config
from quadcode.jsonutil import JSONExtractionError, parse_json_object
from quadcode.runstate import config_fingerprint
from quadcode.schemas import CallType, Role, TestCase

from .conftest import BASE_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[1]


def _real(**agent: Any) -> dict[str, Any]:
    data = copy.deepcopy(BASE_CONFIG)
    data["providers"]["compat"] = {"kind": "openai_compat", "base_url": "https://c.test/v1",
                                   "api_key_env": "C_KEY", "rpm": 10, "tpm": 1000}
    data["providers"]["g"] = {"kind": "gemini", "api_key_env": "G_KEY", "rpm": 10, "tpm": 1000}
    data["agents"]["optimizer"] = {"provider": "compat", "model": "o", **agent}
    return data


def test_request_options_and_inline_system_accepted() -> None:
    cfg = parse_config(_real(request_options={"reasoning_effort": "low"}, inline_system=True))
    agent = cfg.agent(Role.OPTIMIZER)
    assert agent.request_options == {"reasoning_effort": "low"} and agent.inline_system


@pytest.mark.parametrize("key", ["model", "messages", "max_tokens", "temperature", "stream"])
def test_reserved_openai_options_rejected(key: str) -> None:
    with pytest.raises(ConfigError, match="request_options may not set"):
        parse_config(_real(request_options={key: 1}))


def test_reserved_gemini_options_rejected() -> None:
    data = _real()
    data["agents"]["generator"] = {"provider": "g", "model": "gm", "request_options": {"maxOutputTokens": 5}}
    with pytest.raises(ConfigError, match="maxOutputTokens"):
        parse_config(data)
    data["agents"]["generator"]["request_options"] = {"responseMimeType": "application/json"}
    parse_config(data)


def test_mock_agents_reject_real_only_settings() -> None:
    data = copy.deepcopy(BASE_CONFIG)
    data["agents"]["reviewer"]["inline_system"] = True
    with pytest.raises(ConfigError, match="not used by mock providers"):
        parse_config(data)


def test_smoke_cap_is_optional_with_default() -> None:
    cfg = parse_config(copy.deepcopy(BASE_CONFIG))
    assert CallType.SMOKE not in cfg.max_tokens and cfg.cap_for(CallType.SMOKE) == 512
    data = copy.deepcopy(BASE_CONFIG)
    data["max_tokens"]["smoke"] = 64
    assert parse_config(data).cap_for(CallType.SMOKE) == 64
    assert parse_config(data).cap_for(CallType.INITIAL) == 300


def test_max_retry_after_must_be_positive() -> None:
    data = _real()
    data["providers"]["compat"]["max_retry_after_s"] = 0
    with pytest.raises(ConfigError):
        parse_config(data)


def test_missing_keys_by_role() -> None:
    data = _real()
    data["agents"]["generator"] = {"provider": "g", "model": "gm"}
    cfg = parse_config(data)
    assert missing_keys(cfg, Secrets({})) == [("g", "G_KEY"), ("compat", "C_KEY")]
    assert missing_keys(cfg, Secrets({"C_KEY": "x"})) == [("g", "G_KEY")]
    assert missing_keys(cfg, Secrets({}), [Role.OPTIMIZER]) == [("compat", "C_KEY")]
    assert missing_keys(cfg, Secrets({}), [Role.REVIEWER]) == []  # mock needs no key


def test_fingerprint_ignores_new_settings_at_default_but_not_when_set() -> None:
    base = parse_config(_real())
    explicit_defaults = _real(request_options={}, inline_system=False)
    explicit_defaults["providers"]["compat"]["max_retry_after_s"] = 90
    assert config_fingerprint(parse_config(explicit_defaults)) == config_fingerprint(base)
    assert config_fingerprint(parse_config(_real(inline_system=True))) != config_fingerprint(base)
    assert config_fingerprint(parse_config(_real(request_options={"top_p": 0.9}))) != config_fingerprint(base)
    changed = _real()
    changed["providers"]["compat"]["max_retry_after_s"] = 30
    assert config_fingerprint(parse_config(changed)) != config_fingerprint(base)


def test_repo_config_uses_free_real_providers() -> None:
    cfg = load_config(REPO_ROOT / "config.yaml")
    choices = {r: cfg.agent(r).choices() for r in Role}
    used = [c for cs in choices.values() for c in cs]
    # all four slots and every fallback run on Groq's free tier
    assert {cfg.providers[c.provider].api_key_env for c in used} == {"GROQ_API_KEY"}
    assert all(len(cs) >= 2 for cs in choices.values())  # every slot has a fallback
    # one provider block per Groq model, so slots sharing a model share its per-model limits
    block_of = {c.model: c.provider for c in used}
    assert all(block_of[c.model] == c.provider for c in used)
    assert len(set(block_of.values())) == len(block_of)
    # Groq has three chat models for four slots: the one deliberate duplicate is explicit
    assert cfg.allow_duplicate_models
    assert len({cfg.agent(r).model for r in Role}) == 3
    # Qwen may not ask for more than its 1,000 output tokens per minute
    assert all(cfg.cap_for(t, Role.REVIEWER) <= 1000 for t in CallType)
    assert cfg.run.state_output_limit


def test_think_trace_is_ignored_when_parsing() -> None:
    text = '<think>maybe {"ok": false} is right?</think>\n{"ok": true}'
    assert parse_json_object(text) == {"ok": True}


def test_agent_cap_overrides_global_cap() -> None:
    cfg = parse_config(_real(max_tokens={"initial": 900}))
    assert cfg.cap_for(CallType.INITIAL, Role.OPTIMIZER) == 900
    assert cfg.cap_for(CallType.INITIAL, Role.REVIEWER) == 300  # global cap for the others
    assert cfg.cap_for(CallType.REVIEW, Role.OPTIMIZER) == 200  # override only for the listed call type
    assert cfg.cap_for(CallType.INITIAL) == 300
    with pytest.raises(ConfigError, match="max_tokens must be positive"):
        parse_config(_real(max_tokens={"initial": 0}))
    assert config_fingerprint(parse_config(_real(max_tokens={}))) == config_fingerprint(parse_config(_real()))
    assert config_fingerprint(cfg) != config_fingerprint(parse_config(_real()))


def test_malformed_answer_reports_its_json_error_not_an_inner_object() -> None:
    # the whole answer is broken (trailing comma); "{}" inside its code string must not be taken as the answer
    text = '{"language": "python", "code": "seen = {}\\nreturn seen", "tests": [1, 2,],}'
    with pytest.raises(JSONExtractionError, match=r"not valid JSON: .* at line 1 column \d+, at <<HERE>> in: "):
        parse_json_object(text)


def test_python_arithmetic_in_json_is_located() -> None:
    with pytest.raises(JSONExtractionError) as exc:
        parse_json_object('{"args": [[10**9, 5], 10]}')
    assert "[[10<<HERE>>**9" in str(exc.value)


def test_prose_brace_before_the_answer_is_skipped() -> None:
    assert parse_json_object('Plan: {draft} then {"ok": true}') == {"ok": True}


def test_null_args_mean_no_arguments() -> None:
    case = TestCase.model_validate({"id": "t1", "category": "performance", "args": None, "kwargs": None,
                                    "generator": "(([1] * 10, 2), {})", "expected": []})
    assert case.args == [] and case.kwargs == {}
