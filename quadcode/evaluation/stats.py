"""Summary statistics over problems: percentile bootstrap intervals and paired win/tie/loss counts.

Pure Python and seeded, so the same records always give the same numbers.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """Mean of ``n`` values with a percentile-bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    n: int


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]) of already sorted values."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_mean(values: Sequence[float], samples: int, confidence: float, seed: str) -> Interval | None:
    """Mean with a percentile bootstrap CI (resampling problems with replacement). None for no values."""
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    if n == 1 or len(set(values)) == 1:
        return Interval(mean, mean, mean, n)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(samples))
    alpha = (1 - confidence) / 2
    return Interval(mean, _percentile(means, alpha), _percentile(means, 1 - alpha), n)


def bootstrap_geomean(ratios: Sequence[float], samples: int, confidence: float, seed: str) -> Interval | None:
    """Geometric mean of positive ratios, with the bootstrap CI computed on their logarithms."""
    if any(r <= 0 for r in ratios):
        raise ValueError("ratios must be positive")
    logs = bootstrap_mean([math.log(r) for r in ratios], samples, confidence, seed)
    if logs is None:
        return None
    return Interval(math.exp(logs.mean), math.exp(logs.low), math.exp(logs.high), logs.n)


@dataclass(frozen=True)
class WinTieLoss:
    """Per-problem comparison of A against B on problems where both have a value. Win = A higher."""

    wins: tuple[str, ...]
    ties: tuple[str, ...]
    losses: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.wins) + len(self.ties) + len(self.losses)


def paired(a: Mapping[str, float], b: Mapping[str, float]) -> WinTieLoss:
    common = sorted(set(a) & set(b))
    return WinTieLoss(wins=tuple(p for p in common if a[p] > b[p]),
                      ties=tuple(p for p in common if a[p] == b[p]),
                      losses=tuple(p for p in common if a[p] < b[p]))
