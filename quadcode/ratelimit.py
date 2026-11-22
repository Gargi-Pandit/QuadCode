"""Per-provider rate limiting (requests/min + tokens/min) and retry with backoff.

Clock, sleep and RNG are injectable so tests run instantly and deterministically.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from quadcode.providers.base import PermanentError, ProviderError, RateLimitError

T = TypeVar("T")
Clock = Callable[[], float]
Sleep = Callable[[float], None]


class TokenBucket:
    """Classic token bucket. Not thread-safe on its own; ``ProviderLimiter`` locks it.

    The balance may go negative via ``debit`` when actual usage exceeds an estimate;
    later requests then wait until the debt is refilled.
    """

    def __init__(self, capacity: float, refill_per_s: float, clock: Clock) -> None:
        if capacity <= 0 or refill_per_s <= 0:
            raise ValueError("capacity and refill rate must be positive")
        self.capacity = float(capacity)
        self.refill_per_s = float(refill_per_s)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_s)
        self._last = now

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens

    def clamp(self, amount: float) -> float:
        """A request larger than capacity can never fit; treat it as a full bucket."""
        return min(float(amount), self.capacity)

    def wait_time(self, amount: float) -> float:
        """Seconds until ``amount`` tokens are available (0 if available now)."""
        self._refill()
        need = self.clamp(amount) - self._tokens
        return 0.0 if need <= 0 else need / self.refill_per_s

    def take(self, amount: float) -> None:
        self._refill()
        self._tokens -= self.clamp(amount)

    def debit(self, amount: float) -> None:
        """Adjust balance by actual-minus-estimate (may be negative = refund)."""
        self._refill()
        self._tokens = min(self.capacity, self._tokens - amount)


class ProviderLimiter:
    """Requests-per-minute and tokens-per-minute limits for one provider block."""

    def __init__(self, rpm: int, tpm: int, clock: Clock = time.monotonic, sleep: Sleep = time.sleep) -> None:
        self._lock = threading.Lock()
        self._sleep = sleep
        self.requests = TokenBucket(rpm, rpm / 60.0, clock)
        self.tokens = TokenBucket(tpm, tpm / 60.0, clock)

    def acquire(self, est_tokens: int) -> float:
        """Block until one request and ``est_tokens`` tokens fit. Returns total seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                wait = max(self.requests.wait_time(1), self.tokens.wait_time(est_tokens))
                if wait <= 0:
                    self.requests.take(1)
                    self.tokens.take(est_tokens)
                    return waited
            self._sleep(wait)
            waited += wait

    def settle(self, est_tokens: int, actual_tokens: int) -> None:
        """Correct the token bucket once the real usage is known."""
        with self._lock:
            self.tokens.debit(actual_tokens - self.tokens.clamp(est_tokens))


def build_limiters(
    limits: Mapping[str, tuple[int, int]],
    clock: Clock = time.monotonic,
    sleep: Sleep = time.sleep,
) -> dict[str, ProviderLimiter]:
    """One limiter per provider name from ``{name: (rpm, tpm)}``."""
    return {name: ProviderLimiter(rpm, tpm, clock, sleep) for name, (rpm, tpm) in limits.items()}


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int
    base_s: float
    cap_s: float


def is_retryable(err: ProviderError) -> bool:
    return not isinstance(err, PermanentError)


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random, retry_after: float | None = None) -> float:
    """Full-jitter exponential backoff; a server Retry-After is a lower bound."""
    ceiling = min(policy.cap_s, policy.base_s * (2**attempt))
    delay = rng.uniform(0.0, ceiling)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return delay


def retry_with_backoff(
    fn: Callable[[int], T],
    policy: RetryPolicy,
    sleep: Sleep = time.sleep,
    rng: random.Random | None = None,
    on_retry: Callable[[int, ProviderError, float], None] | None = None,
) -> T:
    """Call ``fn(attempt)`` (attempt starts at 1) until it succeeds or retries are exhausted.

    Only ``ProviderError`` subclasses are retried, and never ``PermanentError``.
    The last error is re-raised.
    """
    rng = rng or random.Random()
    attempt = 1
    while True:
        try:
            return fn(attempt)
        except ProviderError as err:
            if not is_retryable(err) or attempt > policy.max_retries:
                raise
            retry_after = err.retry_after if isinstance(err, RateLimitError) else None
            delay = backoff_delay(attempt - 1, policy, rng, retry_after)
            if on_retry is not None:
                on_retry(attempt, err, delay)
            sleep(delay)
            attempt += 1
