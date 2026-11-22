"""Shared HTTP plumbing for real providers: one request helper and one error mapping.

Every provider error ends up as a ``ProviderError`` subclass, so the gateway's retry and
UNAVAILABLE handling is the same for all of them:

- 429 -> ``RateLimitError`` with the server's requested wait, unless the quota is a daily
  one or the requested wait is longer than ``max_retry_after_s``; both become
  ``PermanentError`` so no quota or time is spent on retries that cannot succeed.
- 408, 425 and 5xx, network failures -> ``TransientError``; timeouts -> ``ProviderTimeout``.
- Other 4xx -> ``PermanentError``.

API keys travel in request headers only, never in URLs, so they cannot appear in the
exception text httpx builds from a URL. Error text is still redacted by the gateway.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from quadcode.providers.base import (
    PermanentError,
    ProviderError,
    ProviderTimeout,
    RateLimitError,
    TransientError,
)

SNIPPET_CHARS = 300
TRANSIENT_STATUS = frozenset({408, 425})
# Quota identifiers and messages that mean "come back tomorrow", e.g. Gemini's
# "GenerateRequestsPerDayPerProjectPerModel-FreeTier", OpenRouter's "free-models-per-day",
# Groq's "tokens per day (TPD)".
DAILY_QUOTA_RE = re.compile(r"per[ _-]?day|\bdaily\b|\b[RT]PD\b", re.IGNORECASE)
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)s\s*$")


def make_http_client(
    base_url: str,
    headers: dict[str, str],
    timeout_s: float,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """One ``httpx.Client`` per agent slot. ``timeout_s`` bounds connect, read, write and pool waits."""
    return httpx.Client(
        base_url=base_url.rstrip("/") + "/",
        headers=headers,
        timeout=httpx.Timeout(timeout_s),
        transport=transport,
        follow_redirects=False,
    )


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _error_message(payload: Any, body: str) -> str:
    """The provider's own error message when it has one, else the raw body."""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg:
                meta = err.get("metadata")
                raw = meta.get("raw") if isinstance(meta, dict) else None
                return f"{msg} ({raw})" if isinstance(raw, str) and raw else msg
        elif isinstance(err, str) and err:
            return err
        detail = payload.get("detail") or payload.get("message")
        if isinstance(detail, str) and detail:
            return detail
    return body


def _retry_after_header(headers: httpx.Headers) -> float | None:
    ms = headers.get("retry-after-ms")
    if ms is not None:
        try:
            return max(0.0, float(ms) / 1000.0)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _retry_after_body(payload: Any) -> float | None:
    """Google APIs put the wait in ``error.details[].retryDelay`` (e.g. "23s")."""
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return None
    details = payload["error"].get("details")
    if not isinstance(details, list):
        return None
    for item in details:
        if isinstance(item, dict) and isinstance(item.get("retryDelay"), str):
            match = _DURATION_RE.match(item["retryDelay"])
            if match:
                return float(match.group(1))
    return None


def retry_after_seconds(headers: httpx.Headers, payload: Any) -> float | None:
    header = _retry_after_header(headers)
    return header if header is not None else _retry_after_body(payload)


def classify_error(
    status: int,
    message: str,
    raw_body: str,
    retry_after: float | None,
    max_retry_after_s: float,
) -> ProviderError:
    """Map an HTTP failure to the error class that decides whether it is retried."""
    text = f"HTTP {status}: {snippet(message)}"
    if status == 429:
        if DAILY_QUOTA_RE.search(raw_body) or DAILY_QUOTA_RE.search(message):
            return PermanentError(f"daily quota exhausted, not retried: {text}")
        if retry_after is not None and retry_after > max_retry_after_s:
            return PermanentError(f"rate limited and the server asks to wait {retry_after:.0f}s, more than "
                                  f"max_retry_after_s={max_retry_after_s:g}; not retried: {text}")
        return RateLimitError(text, retry_after=retry_after)
    if status in TRANSIENT_STATUS or 500 <= status <= 599:
        return TransientError(text)
    return PermanentError(text)


def _parse_json(body: str) -> Any:
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    max_retry_after_s: float,
    body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one request and return the decoded JSON object, or raise a ``ProviderError``."""
    try:
        resp = client.request(method, url, json=body, params=params)
    except httpx.TimeoutException as exc:
        raise ProviderTimeout(f"no answer within the configured timeout ({type(exc).__name__})") from exc
    except httpx.HTTPError as exc:
        raise TransientError(f"network error: {type(exc).__name__}: {snippet(str(exc))}") from exc

    text = resp.text
    payload = _parse_json(text)
    if resp.status_code >= 400:
        raise classify_error(resp.status_code, _error_message(payload, text), text,
                             retry_after_seconds(resp.headers, payload), max_retry_after_s)
    if not isinstance(payload, dict):
        raise TransientError(f"HTTP {resp.status_code} with a body that is not a JSON object: {snippet(text)}")
    err = payload.get("error")
    if err:
        # Some gateways answer 200 and put the upstream failure in the body.
        code = err.get("code") if isinstance(err, dict) else None
        status = code if isinstance(code, int) and 400 <= code <= 599 else 502
        raise classify_error(status, _error_message(payload, text), text,
                             retry_after_seconds(resp.headers, payload), max_retry_after_s)
    return payload


def token_count(value: Any) -> int | None:
    """A provider-reported token count, or None if missing or malformed."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
