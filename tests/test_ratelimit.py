from __future__ import annotations

import threading

import pytest

from quadcode.ratelimit import ProviderLimiter, TokenBucket
from tests.conftest import FakeClock


def test_bucket_starts_full_and_refills(clock: FakeClock) -> None:
    b = TokenBucket(capacity=10, refill_per_s=2, clock=clock)
    assert b.wait_time(10) == 0
    b.take(10)
    assert b.wait_time(4) == pytest.approx(2.0)
    clock.now += 1.0
    assert b.tokens == pytest.approx(2.0)
    clock.now += 100
    assert b.tokens == pytest.approx(10.0)  # capped at capacity


def test_bucket_clamps_oversized_requests(clock: FakeClock) -> None:
    b = TokenBucket(capacity=5, refill_per_s=1, clock=clock)
    assert b.wait_time(50) == 0
    b.take(50)
    assert b.tokens == pytest.approx(0.0)


def test_bucket_rejects_bad_params(clock: FakeClock) -> None:
    with pytest.raises(ValueError):
        TokenBucket(0, 1, clock)


def test_rpm_limit_blocks(clock: FakeClock) -> None:
    lim = ProviderLimiter(rpm=2, tpm=1_000_000, clock=clock, sleep=clock.sleep)
    assert lim.acquire(10) == 0
    assert lim.acquire(10) == 0
    waited = lim.acquire(10)
    assert waited == pytest.approx(30.0)  # 1 request at 2/min = 30 s
    assert clock.sleeps == [pytest.approx(30.0)]


def test_tpm_limit_blocks(clock: FakeClock) -> None:
    lim = ProviderLimiter(rpm=1000, tpm=600, clock=clock, sleep=clock.sleep)
    lim.acquire(600)
    waited = lim.acquire(300)
    assert waited == pytest.approx(30.0)  # 300 tokens at 10 tokens/s


def test_settle_charges_underestimates(clock: FakeClock) -> None:
    lim = ProviderLimiter(rpm=1000, tpm=600, clock=clock, sleep=clock.sleep)
    lim.acquire(100)
    lim.settle(100, 400)  # actual usage was 300 more than estimated
    assert lim.tokens.tokens == pytest.approx(200.0)
    lim.settle(100, 0)  # refund
    assert lim.tokens.tokens == pytest.approx(300.0)


def test_concurrent_acquire_respects_capacity() -> None:
    clock = FakeClock()
    lock = threading.Lock()

    def sleep(s: float) -> None:
        with lock:
            clock.sleep(s)

    lim = ProviderLimiter(rpm=5, tpm=10**9, clock=clock, sleep=sleep)
    start = clock.now
    threads = [threading.Thread(target=lim.acquire, args=(1,)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 5 immediately, 5 more need refill at 5/min = 12 s each -> at least 60 s of fake time.
    assert clock.now - start >= 60.0 - 1e-6
