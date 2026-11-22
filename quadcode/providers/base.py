"""Provider-agnostic LLM client interface and error hierarchy."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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
    options: Mapping[str, Any] = field(default_factory=dict)  # provider-specific body fields from config
    inline_system: bool = False  # send the system text inside the user message (models without system role)

    def messages(self) -> tuple[str | None, str]:
        """(system text or None, user text), honoring ``inline_system``."""
        if self.inline_system:
            return None, f"{self.system}\n\n{self.prompt}" if self.system else self.prompt
        return (self.system or None), self.prompt


# Finish reasons meaning "the output was cut off at max_tokens", across providers.
TRUNCATED_FINISH_REASONS = frozenset({"length", "MAX_TOKENS", "max_tokens"})


def empty_output_message(field_name: str, finish_reason: str | None) -> str:
    """Error text for a response without answer text; says so plainly when the token cap was the cause."""
    base = f"response has no text content ({field_name}={finish_reason})"
    if finish_reason in TRUNCATED_FINISH_REASONS:
        return (f"{base}: the output hit max_tokens before any answer text, so reasoning likely used the whole "
                "budget; lower the model's reasoning effort or raise max_tokens")
    return base


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    tokens_estimated: bool  # True when counts are approximations, not provider-reported
    model: str
    finish_reason: str | None = None  # as reported by the provider; None when it gives none

    @property
    def truncated(self) -> bool:
        return self.finish_reason in TRUNCATED_FINISH_REASONS


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

    def list_models(self) -> list[str]:
        """Model ids the provider offers. Costs no completion call."""
        raise PermanentError(f"provider '{self.provider_name}' cannot list its models")

    def close(self) -> None:
        """Release network resources. Safe to call more than once."""
