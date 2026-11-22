"""Catch test cases whose values are not the JSON values the function signature asks for.

Models sometimes write an argument as Python source inside a string, e.g. ``"['a', 'b']"`` for a
``list[str]`` parameter, or ``"2"`` for an ``int``. Such a suite fails on every implementation and is
then set aside as disputed, so the run loses its only evidence. The check is deliberately narrow: it
only flags a JSON string where the annotation names a non-string type, and it skips tests that expect
an exception (passing a wrong type on purpose is a valid adversarial test) and generated inputs.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from quadcode.schemas import InitialSolution, TestCase

_ARRAY = {"list", "tuple", "set", "frozenset", "sequence", "iterable", "collection", "deque", "mutablesequence"}
_OBJECT = {"dict", "mapping", "mutablemapping", "defaultdict", "ordereddict", "counter"}
_NUMBER = {"int", "float", "complex"}
MAX_REPORTED = 3


@dataclass(frozen=True)
class Signature:
    params: tuple[tuple[str, str, str | None], ...]  # (name, annotation text, kind)
    returns: tuple[str, str | None] | None  # (annotation text, kind)


def _kind(annotation: ast.expr | None) -> tuple[str, str | None] | None:
    if annotation is None:
        return None
    text = ast.unparse(annotation)
    if "|" in text or "Optional" in text or "Union" in text:
        return text, None  # several types allowed: never flag
    base = text.split("[", 1)[0].strip().split(".")[-1].lower()
    if base in _ARRAY:
        return text, "array"
    if base in _OBJECT:
        return text, "object"
    if base in _NUMBER:
        return text, "number"
    if base == "bool":
        return text, "boolean"
    if base == "str":
        return text, "string"
    return text, None


def parse_signature(signature: str) -> Signature | None:
    """Parameter and return kinds of a ``def f(a: int, ...) -> T`` signature; None if it cannot be parsed."""
    src = signature.strip().rstrip(":").strip()
    if src.startswith("async "):
        src = src[len("async "):]
    if not src.startswith("def "):
        src = "def " + src
    try:
        tree = ast.parse(src + ":\n    pass\n")
    except SyntaxError:
        return None
    fn = tree.body[0] if tree.body else None
    if not isinstance(fn, ast.FunctionDef):
        return None
    params: list[tuple[str, str, str | None]] = []
    for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
        found = _kind(a.annotation)
        params.append((a.arg, found[0] if found else "", found[1] if found else None))
    return Signature(tuple(params), _kind(fn.returns))


def _brief(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 60 else text[:57] + "..."


def _wrong(value: Any, kind: str | None) -> bool:
    return kind is not None and kind != "string" and isinstance(value, str)


def check_tests(tests: Sequence[TestCase], signature: str) -> str | None:
    """A description of the first few mistyped values, or None if every checked value fits the signature."""
    sig = parse_signature(signature)
    if sig is None:
        return None
    by_name = {name: (text, kind) for name, text, kind in sig.params}
    problems: list[str] = []
    for t in tests:
        if t.expect_exception or t.generator:
            continue
        for i, value in enumerate(t.args):
            if i < len(sig.params):
                name, text, kind = sig.params[i]
                if _wrong(value, kind):
                    problems.append(f"test {t.id}: argument '{name}' must be {text} but is the string {_brief(value)}")
        for name, value in t.kwargs.items():
            text, kind = by_name.get(name, ("", None))
            if _wrong(value, kind):
                problems.append(f"test {t.id}: keyword '{name}' must be {text} but is the string {_brief(value)}")
        if sig.returns is not None and _wrong(t.expected, sig.returns[1]):
            problems.append(f"test {t.id}: expected must be {sig.returns[0]} but is the string {_brief(t.expected)}")
    if not problems:
        return None
    more = f" (and {len(problems) - MAX_REPORTED} more)" if len(problems) > MAX_REPORTED else ""
    return ("test values do not match the signature: " + "; ".join(problems[:MAX_REPORTED]) + more
            + ". Write every argument and expected value as a JSON value of its type (a list is a JSON array "
            "such as [\"a\", \"b\"], a number is 2), never as Python source text inside a string.")


def suite_check(signature: str) -> Callable[[InitialSolution], str | None]:
    """Answer check for the validator's initial call: its tests must fit the specification's signature."""
    def check(solution: InitialSolution) -> str | None:
        return check_tests(solution.tests, signature)
    return check
