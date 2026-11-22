"""Provider-agnostic LLM client interface and error hierarchy."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from quadcode.schemas import CallType


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4) used for rate-limit budgeting and mock usage."""
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class LLMRequest:
    system: str
    prompt: str
    model: str
    max_tokens: int
    temperature: float
    timeout_s: float
    call_type: CallType
    tag: str = ""  # orchestrator context such as "round-1"; not sent to real APIs


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    tokens_estimated: bool  # True when counts are approximations, not provider-reported
    model: str


class ProviderError(Exception):
    """Base class for provider failures."""


class RateLimitError(ProviderError):
    """HTTP 429 or equivalent quota signal."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(ProviderError):
    """5xx or network failure; worth retrying."""


class ProviderTimeout(ProviderError):
    """The provider did not answer within ``timeout_s``."""


class PermanentError(ProviderError):
    """Auth failure, bad request or other non-retryable error."""


class LLMClient(ABC):
    """One independently configured client per agent slot."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Perform one completion. Must raise a ``ProviderError`` subclass on failure
        and enforce ``request.timeout_s`` (raising ``ProviderTimeout``)."""
