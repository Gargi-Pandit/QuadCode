"""Error mapping and request plumbing shared by the real providers (offline, httpx.MockTransport)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest

from quadcode.providers.base import (
    PermanentError,
    ProviderTimeout,
    RateLimitError,
    TransientError,
)
from quadcode.providers.http import classify_error, make_http_client, request_json, retry_after_seconds

from .conftest import NetworkBlocked

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return make_http_client("https://api.test/v1", {"Authorization": "Bearer k"}, 5.0, httpx.MockTransport(handler))


def _call(handler: Handler, max_retry_after_s: float = 90.0) -> dict[str, Any]:
    return request_json(_client(handler), "POST", "chat/completions", max_retry_after_s, body={"a": 1})


def _respond(status: int, body: Any = None, headers: dict[str, str] | None = None) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        content = body if isinstance(body, str) else json.dumps(body if body is not None else {})
        return httpx.Response(status, content=content.encode(), headers=headers)

    return handler


def test_success_returns_json_and_sends_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    assert _call(handler) == {"ok": True}
    assert str(seen[0].url) == "https://api.test/v1/chat/completions"
    assert json.loads(seen[0].content) == {"a": 1}


def test_429_with_retry_after_seconds() -> None:
    with pytest.raises(RateLimitError) as exc:
        _call(_respond(429, {"error": {"message": "slow down"}}, {"Retry-After": "7"}))
    assert exc.value.retry_after == 7.0
    assert "HTTP 429: slow down" in str(exc.value)


def test_429_with_retry_after_ms() -> None:
    with pytest.raises(RateLimitError) as exc:
        _call(_respond(429, {}, {"retry-after-ms": "1500"}))
    assert exc.value.retry_after == 1.5


def test_429_with_http_date() -> None:
    when = datetime.now(UTC) + timedelta(seconds=30)
    with pytest.raises(RateLimitError) as exc:
        _call(_respond(429, {}, {"Retry-After": format_datetime(when, usegmt=True)}))
    assert exc.value.retry_after is not None and 25 <= exc.value.retry_after <= 31


def test_429_with_google_retry_delay_in_body() -> None:
    body = {"error": {"code": 429, "message": "Resource exhausted",
                      "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "23s"}]}}
    with pytest.raises(RateLimitError) as exc:
        _call(_respond(429, body))
    assert exc.value.retry_after == 23.0


def test_429_without_hint_has_no_retry_after() -> None:
    with pytest.raises(RateLimitError) as exc:
        _call(_respond(429, "busy"))
    assert exc.value.retry_after is None


@pytest.mark.parametrize("body", [
    {"error": {"code": 429, "message": "Quota exceeded for metric: generate_content_free_tier_requests",
               "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}},
    {"error": {"code": 429, "message": "Rate limit exceeded: free-models-per-day. Add 10 credits"}},
    {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
                          "Limit 200000, Used 199000"}},
])
def test_daily_quota_is_not_retried(body: dict[str, Any]) -> None:
    with pytest.raises(PermanentError, match="daily quota exhausted"):
        _call(_respond(429, body, {"Retry-After": "5"}))


def test_per_minute_quota_is_retried() -> None:
    body = {"error": {"code": 429, "message": "Quota exceeded",
                      "details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}]}}
    with pytest.raises(RateLimitError):
        _call(_respond(429, body))


def test_retry_after_longer_than_cap_is_not_retried() -> None:
    with pytest.raises(PermanentError, match="wait 600s, more than max_retry_after_s=90"):
        _call(_respond(429, {}, {"Retry-After": "600"}), max_retry_after_s=90)


@pytest.mark.parametrize("status", [408, 425, 500, 502, 503, 529])
def test_transient_status(status: int) -> None:
    with pytest.raises(TransientError, match=f"HTTP {status}"):
        _call(_respond(status, {"error": {"message": "upstream"}}))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_status(status: int) -> None:
    with pytest.raises(PermanentError, match=f"HTTP {status}"):
        _call(_respond(status, {"error": {"message": "nope"}}))


def test_openrouter_upstream_detail_is_kept() -> None:
    body = {"error": {"message": "Provider returned error", "metadata": {"raw": "model overloaded"}}}
    with pytest.raises(TransientError, match=r"Provider returned error \(model overloaded\)"):
        _call(_respond(503, body))


def test_error_inside_200_body_is_mapped_by_code() -> None:
    with pytest.raises(RateLimitError):
        _call(_respond(200, {"error": {"code": 429, "message": "rate limited upstream"}}))
    with pytest.raises(TransientError, match="HTTP 502"):
        _call(_respond(200, {"error": {"message": "something broke"}}))


def test_non_json_success_body_is_transient() -> None:
    with pytest.raises(TransientError, match="not a JSON object"):
        _call(_respond(200, "<html>gateway</html>"))


def test_long_error_body_is_truncated() -> None:
    with pytest.raises(PermanentError) as exc:
        _call(_respond(400, "x" * 5000))
    assert len(str(exc.value)) < 400


def test_timeout_maps_to_provider_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ProviderTimeout):
        _call(handler)


def test_connection_error_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(TransientError, match="network error: ConnectError"):
        _call(handler)


def test_classify_error_direct() -> None:
    assert isinstance(classify_error(429, "m", "", 3.0, 90), RateLimitError)
    assert isinstance(classify_error(429, "daily limit", "", None, 90), PermanentError)
    assert isinstance(classify_error(418, "teapot", "", None, 90), PermanentError)


def test_retry_after_header_wins_over_body() -> None:
    body = {"error": {"details": [{"retryDelay": "40s"}]}}
    assert retry_after_seconds(httpx.Headers({"retry-after": "3"}), body) == 3.0
    assert retry_after_seconds(httpx.Headers({}), body) == 40.0
    assert retry_after_seconds(httpx.Headers({"retry-after": "soon"}), {}) is None


def test_real_transport_is_blocked_in_tests() -> None:
    client = make_http_client("https://api.test/v1", {}, 1.0)
    with pytest.raises(NetworkBlocked):
        client.get("models")
