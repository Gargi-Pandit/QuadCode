"""Scripted mock scenarios for benchmark problems, built from a compact description.

Like ``two_sum`` they ignore the prompt text. Each one is shaped so that the harness has something to
measure on hidden tests:

- ``merge_intervals``: G1 misses touching intervals ([1, 4] + [4, 5]); C1 also forgets ``max`` for
  contained intervals; P1 is cubic; D1 assumes sorted input. V0 = G1. Round 1: the generator fixes the
  touching case (accepted); the optimizer proposes a comment-only change, which still fails the touching
  test on V0, is challenged (split verdicts) and rejected. Round 2 is empty: early stop. The final version passes every hidden test, no single agent does.
- ``lis_length``: G1 is a correct O(n^2) DP that times out on the performance test; C1 is the
  O(n log n) patience method; P1 uses ``bisect_right`` (non-strict); D1 counts adjacent increases.
  V0 = C1. Round 1: the optimizer proposes switching C1 to ``bisect_right``, which fails the
  all-equal test, is challenged and rejected; nothing is accepted: early stop. The final version is C1,
  so the full pipeline ties the reviewer's single-agent baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from quadcode.diffs import make_diff
from quadcode.providers.mock import WILDCARD, FailureSpec, MockScenario
from quadcode.schemas import CallType, Role

G, R, O, V = Role.GENERATOR, Role.REVIEWER, Role.OPTIMIZER, Role.VALIDATOR  # noqa: E741


def _js(obj: Any) -> str:
    return json.dumps(obj, indent=2)


@dataclass(frozen=True)
class Initial:
    code: str
    algorithm: str
    time: str
    space: str
    pros: tuple[str, ...] = ()
    cons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Proposal:
    new_code: str
    justification: str
    impact: str
    disadvantages: str = ""


@dataclass
class ScenarioSpec:
    name: str
    analysis: dict[str, Any]
    initials: dict[Role, Initial]
    tests: list[dict[str, Any]]
    reviews: dict[Role, list[dict[str, Any]]]
    base_code: str  # the code V0 is expected to copy; round-1 diffs are written against it
    round_one: dict[Role, list[Proposal]] = field(default_factory=dict)
    challenges: dict[tuple[str, Role], dict[str, str]] = field(default_factory=dict)


def build_scenario(spec: ScenarioSpec) -> MockScenario:
    responses: dict[tuple[Role, CallType, str], str] = {}
    for role in Role:
        responses[(role, CallType.ANALYSIS, WILDCARD)] = _js(spec.analysis)
        ini = spec.initials[role]
        out: dict[str, Any] = {
            "language": "python", "code": ini.code, "algorithm": ini.algorithm, "data_structures": [],
            "time_complexity": ini.time, "space_complexity": ini.space, "pros": list(ini.pros),
            "cons": list(ini.cons), "assumptions": []}
        if role is V:
            out["tests"] = spec.tests
        responses[(role, CallType.INITIAL, WILDCARD)] = _js(out)
        responses[(role, CallType.REVIEW, WILDCARD)] = _js({"items": spec.reviews.get(role, [])})
        responses[(role, CallType.MODIFY, "round-1")] = _js({"proposals": [
            {"diff": make_diff(spec.base_code, p.new_code, "a/solution.py", "b/solution.py"),
             "justification": p.justification, "expected_impact": p.impact, "disadvantages": p.disadvantages}
            for p in spec.round_one.get(role, [])]})
        responses[(role, CallType.MODIFY, WILDCARD)] = _js({"proposals": []})
    for (mod_id, role), answer in spec.challenges.items():
        responses[(role, CallType.CHALLENGE, f"round-1:{mod_id}")] = _js(answer)
    return MockScenario(name=spec.name, responses=responses, failures={})


def _analysis(title: str, entry: str, signature: str, summary: str) -> dict[str, Any]:
    return {"title": title, "language": "python", "summary": summary, "entry_point": entry,
            "signature": signature, "inputs": "see signature", "outputs": "see signature",
            "functional_requirements": [summary], "constraints": [], "performance_constraints": [],
            "edge_cases": ["empty input"], "dependencies": [], "assumptions": [],
            "clarification_needed": False, "clarification_questions": []}


# ---------------------------------------------------------------- merge_intervals

MERGE_G1 = '''def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Sort by start, then sweep and extend the last merged interval."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
'''

MERGE_G1_FIXED = MERGE_G1.replace("if merged and start < merged[-1][1]:", "if merged and start <= merged[-1][1]:")

MERGE_C1 = '''def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for start, end in sorted(intervals):
        if out and start < out[-1][1]:
            out[-1][1] = end
        else:
            out.append([start, end])
    return out
'''

MERGE_P1 = '''def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    items = [list(x) for x in intervals]
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a[0] <= b[1] and b[0] <= a[1]:
                    items[i] = [min(a[0], b[0]), b[1]]
                    del items[j]
                    changed = True
                    break
            if changed:
                break
    return sorted(items)
'''

MERGE_D1 = '''def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for start, end in intervals:
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
'''

MERGE_TESTS: list[dict[str, Any]] = [
    {"id": "T01", "category": "normal", "description": "classic", "args": [[[1, 3], [2, 6], [8, 10]]],
     "expected": [[1, 6], [8, 10]]},
    {"id": "T02", "category": "boundary", "description": "touching", "args": [[[1, 4], [4, 5]]], "expected": [[1, 5]]},
    {"id": "T03", "category": "normal", "description": "unsorted", "args": [[[8, 10], [1, 3], [2, 6]]],
     "expected": [[1, 6], [8, 10]]},
    {"id": "T04", "category": "boundary", "description": "empty", "args": [[]], "expected": []},
    {"id": "T05", "category": "boundary", "description": "contained", "args": [[[1, 10], [2, 3]]], "expected": [[1, 10]]},
    {"id": "T06", "category": "performance", "description": "2e4 shuffled overlapping",
     "generator": "([[[i, i + 2] for i in rng.sample(range(20000), 20000)]], {})", "expected": [[0, 20001]]},
]


def build_merge_intervals() -> MockScenario:
    return build_scenario(ScenarioSpec(
        name="merge_intervals",
        analysis=_analysis("Merge intervals", "merge_intervals",
                           "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]",
                           "Merge overlapping or touching closed intervals, sorted by start."),
        initials={
            G: Initial(MERGE_G1, "Sort and sweep", "O(n log n)", "O(n)", ("Simple",), ("Strict overlap test",)),
            R: Initial(MERGE_C1, "Sort and sweep", "O(n log n)", "O(n)", ("Short",)),
            O: Initial(MERGE_P1, "Repeated pairwise merging", "O(n^3)", "O(n)", ("No sorting assumptions",),
                       ("Very slow",)),
            V: Initial(MERGE_D1, "Single sweep", "O(n)", "O(n)", ("Linear",), ("Assumes sorted input",)),
        },
        tests=MERGE_TESTS,
        reviews={
            G: [{"target": "D1", "bugs": ["Does not sort the input"]}],
            R: [{"target": "G1", "bugs": ["Touching intervals [1,4] and [4,5] are not merged"]}],
            O: [{"target": "C1", "bugs": ["Overwrites the end instead of taking the max"]}],
            V: [{"target": "P1", "weaknesses": ["Cubic time"]}],
        },
        base_code=MERGE_G1,
        round_one={
            G: [Proposal(MERGE_G1_FIXED, "Touching intervals must merge: use <= in the overlap test.",
                         "Fixes T02; no complexity change.")],
            O: [Proposal(MERGE_G1.replace("    merged: list[list[int]] = []\n",
                                          "    merged: list[list[int]] = []  # result, sorted by start\n"),
                         "Document the invariant.", "Readability only.")],
        },
        challenges={
            ("M-002", G): {"verdict": "reject", "comment": "A comment does not fix the touching case.",
                           "evidence": "T02 still fails"},
            ("M-002", R): {"verdict": "accept", "comment": "Harmless documentation.", "evidence": ""},
            ("M-002", V): {"verdict": "reject", "comment": "No behaviour change; the suite still fails.",
                           "evidence": "T02"},
        },
    ))


# ---------------------------------------------------------------- lis_length

LIS_G1 = '''def lis_length(nums: list[int]) -> int:
    best = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    return max(best, default=0)
'''

LIS_C1 = '''from bisect import bisect_left


def lis_length(nums: list[int]) -> int:
    """Patience method: tails[k] is the smallest tail of an increasing run of length k + 1."""
    tails: list[int] = []
    for x in nums:
        k = bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
        else:
            tails[k] = x
    return len(tails)
'''

LIS_P1 = LIS_C1.replace("bisect_left", "bisect_right")

LIS_D1 = '''def lis_length(nums: list[int]) -> int:
    if not nums:
        return 0
    run = best = 1
    for a, b in zip(nums, nums[1:]):
        run = run + 1 if b > a else 1
        best = max(best, run)
    return best
'''

LIS_TESTS: list[dict[str, Any]] = [
    {"id": "T01", "category": "normal", "description": "classic", "args": [[10, 9, 2, 5, 3, 7, 101, 18]], "expected": 4},
    {"id": "T02", "category": "boundary", "description": "all equal", "args": [[7, 7, 7]], "expected": 1},
    {"id": "T03", "category": "boundary", "description": "empty", "args": [[]], "expected": 0},
    {"id": "T04", "category": "normal", "description": "decreasing", "args": [[3, 2, 1]], "expected": 1},
    {"id": "T05", "category": "performance", "description": "2e4 increasing",
     "generator": "([list(range(20000))], {})", "expected": 20000},
]


def build_lis_length() -> MockScenario:
    return build_scenario(ScenarioSpec(
        name="lis_length",
        analysis=_analysis("Longest increasing subsequence", "lis_length", "def lis_length(nums: list[int]) -> int",
                           "Length of the longest strictly increasing subsequence."),
        initials={
            G: Initial(LIS_G1, "Quadratic DP", "O(n^2)", "O(n)", ("Classic",), ("Quadratic",)),
            R: Initial(LIS_C1, "Patience sorting with bisect_left", "O(n log n)", "O(n)", ("Fast",)),
            O: Initial(LIS_P1, "Patience sorting", "O(n log n)", "O(n)", ("Fast",)),
            V: Initial(LIS_D1, "Longest increasing run", "O(n)", "O(1)", ("Linear",)),
        },
        tests=LIS_TESTS,
        reviews={
            G: [{"target": "D1", "bugs": ["Counts contiguous runs, not subsequences"]}],
            R: [{"target": "P1", "bugs": ["bisect_right allows equal values"]}],
            O: [{"target": "G1", "weaknesses": ["O(n^2) for 1e5 elements"]}],
            V: [{"target": "C1", "strengths": ["Correct and O(n log n)"]}],
        },
        base_code=LIS_C1,
        round_one={O: [Proposal(LIS_P1, "bisect_right avoids replacing equal tails.", "Fewer writes.")]},
        challenges={
            ("M-001", G): {"verdict": "reject", "comment": "Equal values would extend the run.",
                           "evidence": "[7, 7, 7] gives 3"},
            ("M-001", R): {"verdict": "reject", "comment": "Strictness is required.", "evidence": "T02"},
            ("M-001", V): {"verdict": "needs_changes", "comment": "Keep bisect_left.", "evidence": "T02 fails"},
        },
    ))


BENCH_SCENARIOS: dict[str, MockScenario] = {
    "merge_intervals": build_merge_intervals(),
    "lis_length": build_lis_length(),
}

__all__ = ["BENCH_SCENARIOS", "FailureSpec", "ScenarioSpec", "build_scenario"]
