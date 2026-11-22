"""Pure helpers of the in-sandbox runner (no solution code runs in the host)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from quadcode import sandbox_runner
from quadcode.sandbox_runner import canonical_json, normalize, value_digest


class Opaque:
    def __repr__(self) -> str:
        return "Opaque()"


class BadRepr:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


def test_plain_json_passes_through() -> None:
    value = {"a": [1, 2.5, "x", None, True]}
    assert normalize(value) == (value, False)


def test_tuples_and_sets_become_lists() -> None:
    assert normalize((1, (2, 3))) == ([1, [2, 3]], False)
    assert normalize({3, 1, 2}) == ([1, 2, 3], False)
    assert normalize(frozenset()) == ([], False)


def test_unorderable_set_is_sorted_deterministically() -> None:
    first, _ = normalize({1, "a", (2,)})
    second, _ = normalize({(2,), "a", 1})
    assert first == second


def test_dict_keys_follow_json_rules() -> None:
    assert normalize({1: "a", None: "b", 2.5: "c"}) == ({"1": "a", "null": "b", "2.5": "c"}, False)
    assert normalize({True: 0}) == ({"true": 0}, False)
    out, non_json = normalize({(1, 2): 0})
    assert out == {"(1, 2)": 0} and non_json


def test_non_json_values_are_flagged() -> None:
    assert normalize(Opaque()) == ("Opaque()", True)
    assert normalize(float("nan")) == ("nan", True)
    assert normalize([float("inf")]) == (["inf"], True)
    out, flag = normalize(BadRepr())
    assert out == "<unrepresentable BadRepr>" and flag


def test_deep_nesting_is_cut_off() -> None:
    value: list[object] = []
    cur = value
    for _ in range(100):
        nxt: list[object] = []
        cur.append(nxt)
        cur = nxt
    _, flag = normalize(value)
    assert flag


def test_canonical_json_and_digest_are_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert value_digest({"b": 1, "a": 2}) == value_digest({"a": 2, "b": 1})
    assert value_digest([1, 2]) != value_digest([2, 1])


def test_runner_is_stdlib_only() -> None:
    """The runner is copied into the sandbox image, where only the stdlib exists."""
    tree = ast.parse(Path(sandbox_runner.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported - set(sys.stdlib_module_names)
