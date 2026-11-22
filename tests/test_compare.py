from __future__ import annotations

from typing import Any

import pytest

from quadcode.compare import (
    Tolerance,
    compare_values,
    digest_matches,
    exception_matches,
    strict_equal,
    unordered_equal,
)
from quadcode.sandbox_runner import normalize, value_digest


@pytest.mark.parametrize(
    ("a", "b", "equal"),
    [
        ([0, 1], [0, 1], True),
        ([0, 1], [1, 0], False),
        (True, 1, False),
        (1, True, False),
        (False, False, True),
        (1, 1.0, True),
        (None, None, True),
        (None, [], False),
        ("1", 1, False),
        ({"a": [1, 2]}, {"a": [1, 2]}, True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ([[1], [2]], [[1], [2, 3]], False),
    ],
)
def test_strict_equal(a: Any, b: Any, equal: bool) -> None:
    assert strict_equal(a, b) is equal


def test_unordered_is_multiset_at_top_level() -> None:
    assert unordered_equal([3, 1, 2], [1, 2, 3])
    assert unordered_equal([[1, 2], [0]], [[0], [1, 2]])
    assert not unordered_equal([1, 1, 2], [1, 2, 2])
    assert not unordered_equal([[2, 1]], [[1, 2]])  # nested order still matters
    assert unordered_equal(5, 5)


def test_float_mode_uses_tolerance() -> None:
    tol = Tolerance(rel=1e-6, abs=1e-9)
    assert compare_values([0.1 + 0.2], [0.3], "float", tol)
    assert not compare_values([0.1 + 0.2], [0.3], "exact", tol)
    assert not compare_values([1.0], [1.1], "float", tol)
    assert not compare_values([True], [1.0], "float", tol)


def test_expected_is_normalized_before_comparing() -> None:
    assert compare_values([0, 1], (0, 1), "exact", Tolerance())
    assert compare_values([1, 2], {2, 1}, "exact", Tolerance())


def test_digest_matches_normalized_expected() -> None:
    actual, _ = normalize(tuple(range(50)))
    assert digest_matches(value_digest(actual), list(range(50)))
    assert not digest_matches(value_digest(actual), list(range(51)))


@pytest.mark.parametrize(
    ("expected", "type_name", "mro", "match"),
    [
        ("TypeError", "TypeError", ["TypeError", "Exception", "BaseException"], True),
        ("builtins.TypeError", "TypeError", ["TypeError"], True),
        ("LookupError", "KeyError", ["KeyError", "LookupError", "Exception"], True),
        ("ValueError", "TypeError", ["TypeError", "Exception"], False),
        ("", "TypeError", ["TypeError"], False),
    ],
)
def test_exception_matches(expected: str, type_name: str, mro: list[str], match: bool) -> None:
    assert exception_matches(expected, type_name, mro) is match
