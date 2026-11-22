"""Deterministic mock provider driven by scripted scenarios.

Token counts are estimates (len/4) and flagged ``tokens_estimated=True`` so they are
never presented as real provider usage.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

from quadcode.providers.base import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    PermanentError,
    ProviderTimeout,
    RateLimitError,
    TransientError,
    estimate_tokens,
)
from quadcode.schemas import CallType, Role

FailureKind = Literal["rate_limit", "transient", "timeout", "permanent"]
WILDCARD = "*"


@dataclass(frozen=True)
class FailureSpec:
    kind: FailureKind
    times: int | None = None  # None = fail on every attempt


@dataclass(frozen=True)
class MockScenario:
    name: str
    # (role, call_type, tag) -> raw response text; tag WILDCARD matches any tag.
    responses: dict[tuple[Role, CallType, str], str] = field(default_factory=dict)
    failures: dict[tuple[Role, CallType, str], FailureSpec] = field(default_factory=dict)


def _lookup[V](table: dict[tuple[Role, CallType, str], V], role: Role, call_type: CallType, tag: str) -> V | None:
    exact = table.get((role, call_type, tag))
    if exact is not None:
        return exact
    return table.get((role, call_type, WILDCARD))


class MockProvider(LLMClient):
    """Fake client for one agent slot. Constructed per role, like a real client."""

    def __init__(self, provider_name: str, role: Role, scenario: MockScenario) -> None:
        super().__init__(provider_name)
        self.role = role
        self.scenario = scenario
        self._lock = threading.Lock()
        self._failures_raised: dict[tuple[CallType, str], int] = {}
        self.calls_received = 0

    def _maybe_fail(self, request: LLMRequest) -> None:
        spec = _lookup(self.scenario.failures, self.role, request.call_type, request.tag)
        if spec is None:
            return
        key = (request.call_type, request.tag)
        with self._lock:
            count = self._failures_raised.get(key, 0)
            if spec.times is not None and count >= spec.times:
                return
            self._failures_raised[key] = count + 1
        where = f"{self.role.value}/{request.call_type.value}/{request.tag or '-'}"
        if spec.kind == "rate_limit":
            raise RateLimitError(f"mock 429 for {where}", retry_after=None)
        if spec.kind == "transient":
            raise TransientError(f"mock 503 for {where}")
        if spec.kind == "timeout":
            raise ProviderTimeout(f"mock timeout after {request.timeout_s}s for {where}")
        raise PermanentError(f"mock 400 for {where}")

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            self.calls_received += 1
        self._maybe_fail(request)
        text = _lookup(self.scenario.responses, self.role, request.call_type, request.tag)
        if text is None:
            raise PermanentError(
                f"mock scenario '{self.scenario.name}' has no response for "
                f"{self.role.value}/{request.call_type.value}/{request.tag or '-'}"
            )
        return LLMResponse(
            text=text,
            tokens_in=estimate_tokens(request.system + request.prompt),
            tokens_out=estimate_tokens(text),
            tokens_estimated=True,
            model=request.model,
        )
