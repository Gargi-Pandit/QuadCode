"""Host-side comparison of sandbox outputs with expected values.

Expected values never enter the sandbox; only normalized actual values (or their
digest, for very large outputs) come back and are compared here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from quadcode.sandbox_runner import canonical_json, normalize, value_digest

CompareMode = Literal["exact", "unordered", "float"]


@dataclass(frozen=True)
class Tolerance:
    rel: float = 1e-9
    abs: float = 1e-9


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def strict_equal(a: Any, b: Any, tol: Tolerance | None = None) -> bool:
    """Structural equality that keeps bool distinct from int (``True != 1`` here).

    Numbers compare numerically (``1 == 1.0``); with ``tol`` they compare with
    ``math.isclose``.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        if tol is not None:
            return math.isclose(a, b, rel_tol=tol.rel, abs_tol=tol.abs)
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(strict_equal(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k], tol) for k in a)
    if type(a) is not type(b):
        return False
    return bool(a == b)


def unordered_equal(a: Any, b: Any) -> bool:
    """Top-level lists compared as multisets; anything else falls back to exact."""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(strict_equal(x, y) for x, y in zip(sorted(a, key=canonical_json), sorted(b, key=canonical_json),
                                                      strict=True))
    return strict_equal(a, b)


def compare_values(actual: Any, expected: Any, mode: CompareMode, tol: Tolerance) -> bool:
    """Compare a normalized actual value with an expected value (normalized here)."""
    exp, _ = normalize(expected)
    if mode == "exact":
        return strict_equal(actual, exp)
    if mode == "unordered":
        return unordered_equal(actual, exp)
    return strict_equal(actual, exp, tol)


def digest_matches(actual_digest: str, expected: Any) -> bool:
    """Exact comparison when only the digest of a large actual value came back."""
    exp, _ = normalize(expected)
    return actual_digest == value_digest(exp)


def exception_matches(expected_name: str, type_name: str, mro: Sequence[str]) -> bool:
    """True if the raised type, or any of its base classes, has the expected name.

    ``expected_name`` may be qualified (``builtins.ValueError``); only the last part counts.
    """
    wanted = expected_name.rsplit(".", 1)[-1].strip()
    return bool(wanted) and (wanted == type_name or wanted in mro)
