"""Real provider clients against fake HTTP transports: request shape, key placement, parsing."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from quadcode.cache import DiskCache
from quadcode.config import ProviderConfig, Secrets, parse_config
from quadcode.gateway import CallBudget, LLMGateway
from quadcode.providers import MissingAPIKeyError, build_clients
from quadcode.providers.anthropic import AnthropicClient
from quadcode.providers.base import LLMClient, LLMRequest, PermanentError
from quadcode.providers.gemini import GeminiClient
from quadcode.providers.openai_compat import OpenAICompatClient
from quadcode.ratelimit import build_limiters
from quadcode.runlog import RunLog
from quadcode.schemas import AgentStatus, CallStatus, CallType, ProblemSpec, Role

from .conftest import BASE_CONFIG, FakeClock

KEY = "sk-test-SECRET-1234567890"
Handler = Callable[[httpx.Request], httpx.Response]


class Recorder:
    """A MockTransport handler that records requests and replays canned responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def _pcfg(kind: str, **extra: Any) -> ProviderConfig:
    base: dict[str, Any] = {"kind": kind, "api_key_env": "K", "rpm": 60, "tpm": 100000, "timeout_s": 5}
    if kind == "openai_compat":
        base["base_url"] = "https://compat.test/v1"
    return ProviderConfig(**{**base, **extra})


def _req(**kw: Any) -> LLMRequest:
    base: dict[str, Any] = {"system": "SYS", "prompt": "USER", "model": "m-1", "max_tokens": 321,
                            "temperature": 0.3, "timeout_s": 5, "call_type": CallType.REVIEW, "tag": "round-1"}
    return LLMRequest(**{**base, **kw})


def _assert_key_only_in_headers(rec: Recorder, header: str, value: str) -> None:
    for request in rec.requests:
        assert KEY not in str(request.url)
        assert KEY not in request.content.decode()
        assert request.headers[header] == value


# ------------------------------------------------------------------ openai_compat

def _chat_ok(text: str = '{"x": 1}', usage: dict[str, Any] | None = None, finish: str = "stop") -> httpx.Response:
    body: dict[str, Any] = {"model": "m-1-served", "choices": [{"message": {"role": "assistant", "content": text},
                                                                "finish_reason": finish}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def test_openai_compat_request_and_response() -> None:
    rec = Recorder(_chat_ok(usage={"prompt_tokens": 11, "completion_tokens": 7}))
    client = OpenAICompatClient("groq", _pcfg("openai_compat"), KEY, httpx.MockTransport(rec))
    resp = client.complete(_req(options={"reasoning_effort": "low"}))
    request = rec.requests[0]
    assert request.method == "POST" and str(request.url) == "https://compat.test/v1/chat/completions"
    _assert_key_only_in_headers(rec, "authorization", f"Bearer {KEY}")
    assert rec.body == {"reasoning_effort": "low", "model": "m-1", "max_tokens": 321, "temperature": 0.3,
                        "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]}
    assert "round-1" not in request.content.decode()  # the orchestrator tag is never sent
    assert (resp.text, resp.tokens_in, resp.tokens_out, resp.tokens_estimated) == ('{"x": 1}', 11, 7, False)
    assert resp.finish_reason == "stop" and not resp.truncated and resp.model == "m-1-served"


def test_openai_compat_inline_system() -> None:
    rec = Recorder(_chat_ok(usage={"prompt_tokens": 1, "completion_tokens": 1}))
    OpenAICompatClient("p", _pcfg("openai_compat"), KEY, httpx.MockTransport(rec)).complete(_req(inline_system=True))
    assert rec.body["messages"] == [{"role": "user", "content": "SYS\n\nUSER"}]


def test_openai_compat_missing_usage_is_estimated_and_length_flagged() -> None:
    rec = Recorder(_chat_ok(text="{" * 40, finish="length"))
    resp = OpenAICompatClient("p", _pcfg("openai_compat"), KEY, httpx.MockTransport(rec)).complete(_req())
    assert resp.tokens_estimated and resp.tokens_out == 10 and resp.truncated


def test_openai_compat_content_parts() -> None:
    rec = Recorder(httpx.Response(200, json={"choices": [{"message": {"content": [
        {"type": "text", "text": '{"a":'}, {"type": "text", "text": " 1}"}]}, "finish_reason": "stop"}]}))
    resp = OpenAICompatClient("p", _pcfg("openai_compat"), KEY, httpx.MockTransport(rec)).complete(_req())
    assert resp.text == '{"a": 1}'


@pytest.mark.parametrize("body", [
    {"choices": []},
    {"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]},
    {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
])
def test_openai_compat_empty_content_is_permanent(body: dict[str, Any]) -> None:
    rec = Recorder(httpx.Response(200, json=body))
    with pytest.raises(PermanentError):
        OpenAICompatClient("p", _pcfg("openai_compat"), KEY, httpx.MockTransport(rec)).complete(_req())


def test_openai_compat_list_models() -> None:
    rec = Recorder(httpx.Response(200, json={"data": [{"id": "b"}, {"id": "a"}, {"id": "a"}, {"x": 1}]}))
    client = OpenAICompatClient("p", _pcfg("openai_compat"), KEY, httpx.MockTransport(rec))
    assert client.list_models() == ["a", "b"]
    assert rec.requests[0].method == "GET" and str(rec.requests[0].url) == "https://compat.test/v1/models"
    _assert_key_only_in_headers(rec, "authorization", f"Bearer {KEY}")


# ------------------------------------------------------------------ gemini

def _gemini_ok(parts: list[dict[str, Any]], finish: str = "STOP",
               usage: dict[str, Any] | None = None) -> httpx.Response:
    body: dict[str, Any] = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}],
                            "modelVersion": "gemma-x-served"}
    if usage is not None:
        body["usageMetadata"] = usage
    return httpx.Response(200, json=body)


def test_gemini_request_and_response() -> None:
    rec = Recorder(_gemini_ok([{"text": "thinking...", "thought": True}, {"text": '{"y": 2}'}],
                              usage={"promptTokenCount": 20, "candidatesTokenCount": 5, "thoughtsTokenCount": 3}))
    client = GeminiClient("gemini", _pcfg("gemini"), KEY, httpx.MockTransport(rec))
    resp = client.complete(_req(model="gemma-4-31b-it", options={"responseMimeType": "application/json"}))
    request = rec.requests[0]
    assert str(request.url) == ("https://generativelanguage.googleapis.com/v1beta/models/"
                                "gemma-4-31b-it:generateContent")
    _assert_key_only_in_headers(rec, "x-goog-api-key", KEY)
    assert rec.body == {
        "contents": [{"role": "user", "parts": [{"text": "USER"}]}],
        "systemInstruction": {"parts": [{"text": "SYS"}]},
        "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 321, "temperature": 0.3},
    }
    assert resp.text == '{"y": 2}'  # thought parts are not the answer
    assert (resp.tokens_in, resp.tokens_out, resp.tokens_estimated) == (20, 8, False)
    assert resp.model == "gemma-x-served"


def test_gemini_inline_system_and_custom_base_url() -> None:
    rec = Recorder(_gemini_ok([{"text": "{}"}], usage={"promptTokenCount": 1, "candidatesTokenCount": 1}))
    cfg = _pcfg("gemini", base_url="https://proxy.test/v1beta/")
    GeminiClient("g", cfg, KEY, httpx.MockTransport(rec)).complete(_req(model="models/g-1", inline_system=True))
    assert str(rec.requests[0].url) == "https://proxy.test/v1beta/models/g-1:generateContent"
    assert "systemInstruction" not in rec.body
    assert rec.body["contents"][0]["parts"][0]["text"] == "SYS\n\nUSER"


def test_gemini_max_tokens_finish_is_truncated() -> None:
    rec = Recorder(_gemini_ok([{"text": '{"a": '}], finish="MAX_TOKENS"))
    resp = GeminiClient("g", _pcfg("gemini"), KEY, httpx.MockTransport(rec)).complete(_req())
    assert resp.truncated and resp.tokens_estimated


def test_gemini_blocked_prompt_is_permanent() -> None:
    rec = Recorder(httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(PermanentError, match="blockReason=SAFETY"):
        GeminiClient("g", _pcfg("gemini"), KEY, httpx.MockTransport(rec)).complete(_req())


def test_gemini_safety_stop_without_text_is_permanent() -> None:
    rec = Recorder(httpx.Response(200, json={"candidates": [{"finishReason": "SAFETY"}]}))
    with pytest.raises(PermanentError, match="finishReason=SAFETY"):
        GeminiClient("g", _pcfg("gemini"), KEY, httpx.MockTransport(rec)).complete(_req())


def test_gemini_list_models_pages_and_filters() -> None:
    rec = Recorder(
        httpx.Response(200, json={"models": [
            {"name": "models/gemma-4-31b-it", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-1", "supportedGenerationMethods": ["embedContent"]}],
            "nextPageToken": "p2"}),
        httpx.Response(200, json={"models": [{"name": "models/gemini-flash",
                                              "supportedGenerationMethods": ["generateContent", "countTokens"]}]}),
    )
    client = GeminiClient("g", _pcfg("gemini"), KEY, httpx.MockTransport(rec))
    assert client.list_models() == ["gemini-flash", "gemma-4-31b-it"]
    assert rec.requests[1].url.params["pageToken"] == "p2"
    _assert_key_only_in_headers(rec, "x-goog-api-key", KEY)


# ------------------------------------------------------------------ anthropic

def test_anthropic_request_and_response() -> None:
    rec = Recorder(httpx.Response(200, json={
        "model": "a-1", "stop_reason": "end_turn", "usage": {"input_tokens": 9, "output_tokens": 4},
        "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": '{"z": 3}'}]}))
    client = AnthropicClient("anthropic", _pcfg("anthropic"), KEY, httpx.MockTransport(rec))
    resp = client.complete(_req(options={"top_k": 5}))
    assert str(rec.requests[0].url) == "https://api.anthropic.com/v1/messages"
    _assert_key_only_in_headers(rec, "x-api-key", KEY)
    assert rec.requests[0].headers["anthropic-version"] == "2023-06-01"
    assert rec.body == {"top_k": 5, "model": "m-1", "max_tokens": 321, "temperature": 0.3, "system": "SYS",
                        "messages": [{"role": "user", "content": "USER"}]}
    assert (resp.text, resp.tokens_in, resp.tokens_out, resp.tokens_estimated) == ('{"z": 3}', 9, 4, False)
    assert resp.finish_reason == "end_turn"


def test_anthropic_max_tokens_and_empty() -> None:
    rec = Recorder(httpx.Response(200, json={"stop_reason": "max_tokens", "content": [{"type": "text", "text": "{"}]}))
    resp = AnthropicClient("a", _pcfg("anthropic"), KEY, httpx.MockTransport(rec)).complete(_req())
    assert resp.truncated
    rec = Recorder(httpx.Response(200, json={"stop_reason": "refusal", "content": []}))
    with pytest.raises(PermanentError, match="stop_reason=refusal"):
        AnthropicClient("a", _pcfg("anthropic"), KEY, httpx.MockTransport(rec)).complete(_req())


def test_anthropic_list_models() -> None:
    rec = Recorder(httpx.Response(200, json={"data": [{"id": "a-2"}, {"id": "a-1"}]}))
    assert AnthropicClient("a", _pcfg("anthropic"), KEY, httpx.MockTransport(rec)).list_models() == ["a-1", "a-2"]


# ------------------------------------------------------------------ registry

def _real_config(**agent_extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(json.dumps(BASE_CONFIG))
    data["providers"] = {
        "compat": {"kind": "openai_compat", "base_url": "https://compat.test/v1", "api_key_env": "COMPAT_KEY",
                   "rpm": 600, "tpm": 1_000_000, "timeout_s": 5, "max_retries": 2, "backoff_base_s": 0.5,
                   "backoff_cap_s": 4},
        "g": {"kind": "gemini", "api_key_env": "G_KEY", "rpm": 600, "tpm": 1_000_000},
    }
    data["agents"] = {r.value: {"provider": "compat", "model": f"model-{r.value}", **agent_extra} for r in Role}
    data["agents"]["generator"] = {"provider": "g", "model": "gemma"}
    return data


def test_build_clients_gives_each_slot_its_own_client() -> None:
    cfg = parse_config(_real_config())
    clients = build_clients(cfg, Secrets({"COMPAT_KEY": KEY, "G_KEY": "gk"}))
    assert all(len(chain) == 1 for chain in clients.values())
    assert len({id(c) for chain in clients.values() for c in chain}) == 4
    assert isinstance(clients[Role.GENERATOR][0], GeminiClient)
    assert all(isinstance(clients[r][0], OpenAICompatClient) for r in (Role.REVIEWER, Role.OPTIMIZER, Role.VALIDATOR))
    for chain in clients.values():
        for c in chain:
            c.close()


def test_build_clients_without_key_raises() -> None:
    with pytest.raises(MissingAPIKeyError, match="G_KEY"):
        build_clients(parse_config(_real_config()), Secrets({"COMPAT_KEY": KEY}))


# ------------------------------------------------------------------ gateway over HTTP

ANALYSIS_JSON = json.dumps({"title": "Two sum", "summary": "find two indices", "language": "python",
                            "entry_point": "two_sum", "signature": "def two_sum(nums, target)",
                            "inputs": "nums, target", "outputs": "pair"})


def _gateway(tmp_path: Path, handler: Handler, clock: FakeClock, **agent_extra: Any) -> LLMGateway:
    cfg = parse_config(_real_config(**agent_extra))
    transport = httpx.MockTransport(handler)
    clients: dict[Role, LLMClient] = {
        r: OpenAICompatClient(cfg.agent(r).provider, cfg.provider_for(r), KEY, transport)
        for r in Role if r is not Role.GENERATOR
    }
    clients[Role.GENERATOR] = GeminiClient("g", cfg.provider_for(Role.GENERATOR), "gk", transport)
    return LLMGateway(config=cfg, clients=clients,
                      limiters=build_limiters({n: (p.rpm, p.tpm) for n, p in cfg.providers.items()},
                                              clock=clock, sleep=clock.sleep),
                      cache=DiskCache(tmp_path / "cache"), runlog=RunLog(tmp_path / "run"), budget=CallBudget(None),
                      secrets=Secrets({"COMPAT_KEY": KEY, "G_KEY": "gk"}), sleep=clock.sleep, clock=clock,
                      rng=random.Random(0))


def test_gateway_retries_429_then_logs_provider_usage(tmp_path: Path, clock: FakeClock) -> None:
    rec = Recorder(httpx.Response(429, json={"error": {"message": "slow"}}, headers={"Retry-After": "2"}),
                   _chat_ok(ANALYSIS_JSON, usage={"prompt_tokens": 50, "completion_tokens": 30}))
    gw = _gateway(tmp_path, rec, clock)
    out = gw.call(Role.REVIEWER, CallType.ANALYSIS, "sys", "analyse", ProblemSpec, tag="analysis")
    assert out.status is AgentStatus.DONE and out.data is not None and out.data.entry_point == "two_sum"
    assert 2.0 in clock.sleeps  # Retry-After is honored as a lower bound
    first, second = gw.runlog.read_calls()
    assert first.status is CallStatus.PROVIDER_ERROR and "RateLimitError: HTTP 429" in (first.error or "")
    assert second.status is CallStatus.OK and (second.tokens_in, second.tokens_out) == (50, 30)
    assert second.tokens_estimated is False and second.finish_reason == "stop"
    assert (second.provider, second.model) == ("compat", "model-reviewer")


def test_gateway_marks_daily_quota_unavailable_without_retry(tmp_path: Path, clock: FakeClock) -> None:
    rec = Recorder(httpx.Response(429, json={"error": {"message": "free-models-per-day exceeded"}}))
    gw = _gateway(tmp_path, rec, clock)
    out = gw.call(Role.OPTIMIZER, CallType.ANALYSIS, "sys", "analyse", ProblemSpec)
    assert out.status is AgentStatus.UNAVAILABLE and "daily quota" in (out.error or "")
    assert len(rec.requests) == 1 and clock.sleeps == []


def test_gateway_reports_truncated_output(tmp_path: Path, clock: FakeClock) -> None:
    rec = Recorder(_chat_ok('{"summary": "cut', usage={"prompt_tokens": 5, "completion_tokens": 200},
                            finish="length"))
    gw = _gateway(tmp_path, rec, clock)
    out = gw.call(Role.VALIDATOR, CallType.ANALYSIS, "sys", "analyse", ProblemSpec)
    assert out.status is AgentStatus.UNAVAILABLE
    record, repair = gw.runlog.read_calls()  # the original answer and one request to fix it
    assert record.status is CallStatus.INVALID_OUTPUT and record.finish_reason == "length"
    assert "output hit max_tokens=100" in (record.error or "")
    assert repair.repair == 1 and "cut off at the output limit" in str(rec.body["messages"])


def test_gateway_redacts_key_echoed_in_error(tmp_path: Path, clock: FakeClock) -> None:
    rec = Recorder(httpx.Response(401, json={"error": {"message": f"invalid key {KEY}"}}))
    gw = _gateway(tmp_path, rec, clock)
    out = gw.call(Role.REVIEWER, CallType.ANALYSIS, "sys", "analyse", ProblemSpec)
    assert out.status is AgentStatus.UNAVAILABLE and KEY not in (out.error or "")
    log = (tmp_path / "run" / "calls.jsonl").read_text()
    assert KEY not in log and "<redacted:COMPAT_KEY>" in log


def test_request_options_reach_body_and_cache_key(tmp_path: Path, clock: FakeClock) -> None:
    ok = _chat_ok(ANALYSIS_JSON, usage={"prompt_tokens": 1, "completion_tokens": 1})
    rec = Recorder(ok)
    gw = _gateway(tmp_path, rec, clock, request_options={"reasoning_effort": "low"})
    gw.call(Role.OPTIMIZER, CallType.ANALYSIS, "sys", "analyse", ProblemSpec)
    assert rec.body["reasoning_effort"] == "low"
    plain = _gateway(tmp_path / "other", Recorder(ok), clock)
    plain.call(Role.OPTIMIZER, CallType.ANALYSIS, "sys", "analyse", ProblemSpec)
    [with_opts] = gw.runlog.read_calls()
    [without] = plain.runlog.read_calls()
    assert with_opts.cache_key != without.cache_key


def test_reasoning_that_fills_the_cap_is_explained() -> None:
    rec = Recorder(httpx.Response(200, json={"choices": [{"message": {"content": None, "reasoning": "..."},
                                                          "finish_reason": "length"}]}))
    with pytest.raises(PermanentError, match="hit max_tokens before any answer text"):
        OpenAICompatClient("p", _pcfg("openai_compat"), KEY, httpx.MockTransport(rec)).complete(_req())


def test_agent_cap_reaches_request_and_cache_key(tmp_path: Path, clock: FakeClock) -> None:
    ok = _chat_ok(ANALYSIS_JSON, usage={"prompt_tokens": 1, "completion_tokens": 1})
    rec = Recorder(ok)
    gw = _gateway(tmp_path, rec, clock, max_tokens={"analysis": 777})
    gw.call(Role.VALIDATOR, CallType.ANALYSIS, "sys", "analyse", ProblemSpec)
    assert rec.body["max_tokens"] == 777
    plain = _gateway(tmp_path / "other", Recorder(ok), clock)
    plain.call(Role.VALIDATOR, CallType.ANALYSIS, "sys", "analyse", ProblemSpec)
    assert gw.runlog.read_calls()[0].cache_key != plain.runlog.read_calls()[0].cache_key


def test_rejected_output_excerpt_is_logged_and_redacted(tmp_path: Path, clock: FakeClock) -> None:
    text = f"leaked {KEY} " + "prose " * 200 + "tail " * 100  # the key sits in the kept head
    rec = Recorder(_chat_ok(text, usage={"prompt_tokens": 5, "completion_tokens": 400}))
    gw = _gateway(tmp_path, rec, clock)
    out = gw.call(Role.REVIEWER, CallType.ANALYSIS, "sys", "analyse", ProblemSpec)
    assert out.status is AgentStatus.UNAVAILABLE
    record, _ = gw.runlog.read_calls()
    assert KEY not in json.dumps(rec.body)  # the repair request quotes the answer with the key redacted
    excerpt = record.output_excerpt or ""
    assert excerpt.startswith("leaked <redacted:COMPAT_KEY> prose") and excerpt.rstrip().endswith("tail")
    assert "characters omitted" in excerpt and len(excerpt) < 800
    assert KEY not in (tmp_path / "run" / "calls.jsonl").read_text()


def test_short_rejected_output_is_kept_whole(tmp_path: Path, clock: FakeClock) -> None:
    rec = Recorder(_chat_ok("not json at all", usage={"prompt_tokens": 5, "completion_tokens": 4}))
    gw = _gateway(tmp_path, rec, clock)
    gw.call(Role.REVIEWER, CallType.ANALYSIS, "sys", "analyse", ProblemSpec)
    assert gw.runlog.read_calls()[0].output_excerpt == "not json at all"
