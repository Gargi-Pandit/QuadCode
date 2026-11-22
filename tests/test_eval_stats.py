"""Bootstrap intervals and paired comparisons on hand-computed data."""

from __future__ import annotations

import math

import pytest

from quadcode.evaluation.stats import _percentile, bootstrap_geomean, bootstrap_mean, paired


def test_percentile_interpolates() -> None:
    assert _percentile([0.0, 10.0], 0.5) == 5.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.25) == 2.0
    assert _percentile([7.0], 0.9) == 7.0


def test_bootstrap_empty_single_and_constant() -> None:
    assert bootstrap_mean([], 1000, 0.95, "s") is None
    one = bootstrap_mean([0.4], 1000, 0.95, "s")
    assert one is not None and (one.mean, one.low, one.high, one.n) == (0.4, 0.4, 0.4, 1)
    same = bootstrap_mean([1.0, 1.0, 1.0], 1000, 0.95, "s")
    assert same is not None and same.low == same.high == 1.0


def test_bootstrap_mean_and_bounds() -> None:
    values = [0.0, 0.5, 1.0, 1.0, 0.25, 0.75]
    iv = bootstrap_mean(values, 5000, 0.95, "x")
    assert iv is not None
    assert iv.mean == pytest.approx(sum(values) / 6)
    assert 0.0 <= iv.low < iv.mean < iv.high <= 1.0
    # A wider confidence level never gives a narrower interval from the same resamples.
    wide = bootstrap_mean(values, 5000, 0.99, "x")
    assert wide is not None and wide.low <= iv.low and wide.high >= iv.high


def test_bootstrap_is_deterministic_per_seed() -> None:
    values = [0.1, 0.9, 0.4, 0.7]
    assert bootstrap_mean(values, 2000, 0.95, "a") == bootstrap_mean(values, 2000, 0.95, "a")
    assert bootstrap_mean(values, 2000, 0.95, "a") != bootstrap_mean(values, 2000, 0.95, "b")


def test_bootstrap_two_values_hits_extremes() -> None:
    # With values 0 and 1, resample means are k/2; the 2.5% and 97.5% percentiles are 0 and 1.
    iv = bootstrap_mean([0.0, 1.0], 4000, 0.95, "z")
    assert iv is not None and iv.low == 0.0 and iv.high == 1.0


def test_geometric_mean_of_ratios() -> None:
    iv = bootstrap_geomean([2.0, 0.5], 2000, 0.95, "g")
    assert iv is not None and iv.mean == pytest.approx(1.0)
    assert iv.low == pytest.approx(0.5) and iv.high == pytest.approx(2.0)
    assert bootstrap_geomean([], 10, 0.95, "g") is None
    single = bootstrap_geomean([4.0], 10, 0.95, "g")
    assert single is not None and single.mean == pytest.approx(4.0)
    with pytest.raises(ValueError):
        bootstrap_geomean([0.0], 10, 0.95, "g")
    assert math.isfinite(iv.mean)


def test_paired_win_tie_loss_only_on_common_problems() -> None:
    a = {"p1": 1.0, "p2": 0.5, "p3": 0.2, "p4": 1.0}
    b = {"p1": 0.5, "p2": 0.5, "p3": 0.9, "p5": 0.0}
    wtl = paired(a, b)
    assert wtl.wins == ("p1",) and wtl.ties == ("p2",) and wtl.losses == ("p3",) and wtl.n == 3
    assert paired({}, b).n == 0
