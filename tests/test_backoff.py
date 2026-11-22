from __future__ import annotations

import random

import pytest

from quadcode.providers.base import (
    PermanentError,
    ProviderError,
    ProviderTimeout,
    RateLimitError,
    TransientError,
)
from quadcode.ratelimit import RetryPolicy, backoff_delay, retry_with_backoff

POLICY = RetryPolicy(max_retries=3, base_s=1.0, cap_s=5.0)


def test_delay_bounds_grow_and_cap() -> None:
    rng = random.Random(42)
    for attempt, ceiling in [(0, 1.0), (1, 2.0), (2, 4.0), (3, 5.0), (10, 5.0)]:
        for _ in range(200):
            d = backoff_delay(attempt, POLICY, rng)
            assert 0.0 <= d <= ceiling


def test_retry_after_is_lower_bound() -> None:
    assert backoff_delay(0, POLICY, random.Random(1), retry_after=7.5) >= 7.5


@pytest.mark.parametrize("err", [RateLimitError("429"), TransientError("503"), ProviderTimeout("t")])
def test_retries_then_succeeds(err: ProviderError) -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def fn(attempt: int) -> str:
        calls.append(attempt)
        if attempt < 3:
            raise err
        return "ok"

    assert retry_with_backoff(fn, POLICY, sleep=sleeps.append, rng=random.Random(0)) == "ok"
    assert calls == [1, 2, 3]
    assert len(sleeps) == 2


def test_permanent_not_retried() -> None:
    calls: list[int] = []

    def fn(attempt: int) -> str:
        calls.append(attempt)
        raise PermanentError("401")

    with pytest.raises(PermanentError):
        retry_with_backoff(fn, POLICY, sleep=lambda s: None)
    assert calls == [1]


def test_gives_up_after_max_retries() -> None:
    calls: list[int] = []
    retries: list[tuple[int, float]] = []

    def fn(attempt: int) -> str:
        calls.append(attempt)
        raise RateLimitError("429", retry_after=2.0)

    with pytest.raises(RateLimitError):
        retry_with_backoff(fn, POLICY, sleep=lambda s: None, rng=random.Random(0),
                           on_retry=lambda a, e, d: retries.append((a, d)))
    assert calls == [1, 2, 3, 4]
    assert [a for a, _ in retries] == [1, 2, 3]
    assert all(d >= 2.0 for _, d in retries)


def test_non_provider_errors_propagate_immediately() -> None:
    def fn(attempt: int) -> str:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        retry_with_backoff(fn, POLICY, sleep=lambda s: None)
