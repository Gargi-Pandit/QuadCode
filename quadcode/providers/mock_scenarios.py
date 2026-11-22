"""Scripted mock scenarios.

The ``two_sum`` scenario is designed to exercise every orchestrator path:
- P1 (optimizer slot) is deliberately buggy: it returns indices into the *sorted* array.
- C1 (reviewer slot) is a correct but O(n^2) brute force that times out on the
  performance test, so selection must rely on measured evidence.
- Phase 3 reviews disagree: the reviewer calls C1 acceptable, the optimizer calls it
  unacceptable.
- The validator is UNAVAILABLE for the Phase 3 review (always rate-limited).
- Round 1: the reviewer proposes a validation change (passes tests), the optimizer
  proposes a memory "optimization" that breaks duplicate inputs (fails tests, gets
  challenged, must be rejected). The validator hits one transient 429 first.
- Round 2+: every agent proposes nothing, which triggers early stop.
- Some responses are wrapped in prose and markdown fences to exercise JSON extraction.
"""

from __future__ import annotations

import json
from typing import Any

from quadcode.providers.mock import WILDCARD, FailureSpec, MockScenario
from quadcode.schemas import CallType, Role

G = Role.GENERATOR
R = Role.REVIEWER
O = Role.OPTIMIZER  # noqa: E741
V = Role.VALIDATOR


def _js(obj: dict[str, Any]) -> str:
    return json.dumps(obj, indent=2)


def _fenced(obj: dict[str, Any], prose: str) -> str:
    return f"{prose}\n\n```json\n{_js(obj)}\n```\n\nLet me know if anything is unclear."


TWO_SUM_PROBLEM = (
    "Write a Python function two_sum(nums: list[int], target: int) -> list[int] that returns "
    "the indices [i, j] with i < j such that nums[i] + nums[j] == target, using the first such "
    "pair found when scanning j from left to right. Return [] if no pair exists. "
    "Inputs may contain up to 10^5 integers, including negatives and duplicates."
)

_ANALYSIS: dict[str, Any] = {
    "title": "Two Sum (indices)",
    "language": "python",
    "summary": "Find two distinct indices whose values sum to target.",
    "entry_point": "two_sum",
    "signature": "def two_sum(nums: list[int], target: int) -> list[int]",
    "inputs": "nums: list of integers (len 0..1e5), target: integer",
    "outputs": "[i, j] with i < j, or [] if no pair exists",
    "functional_requirements": [
        "Return indices, not values",
        "i < j and i != j",
        "Return [] when no pair exists",
    ],
    "constraints": ["len(nums) <= 100000", "values fit in Python int"],
    "performance_constraints": ["Should handle 1e5 elements well under one second"],
    "edge_cases": ["empty list", "single element", "duplicates", "negative values", "no solution"],
    "dependencies": [],
    "assumptions": ["If several pairs exist, the pair with the smallest j is returned"],
    "clarification_needed": False,
    "clarification_questions": [],
}

G1_CODE = '''def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices [i, j] (i < j) with nums[i] + nums[j] == target, or [] if none."""
    seen: dict[int, int] = {}
    for i, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], i]
        seen[value] = i
    return []
'''

C1_CODE = '''def two_sum(nums: list[int], target: int) -> list[int]:
    """Brute force: check every pair, ordered by j then i."""
    n = len(nums)
    for j in range(1, n):
        for i in range(j):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
'''

P1_CODE = '''def two_sum(nums: list[int], target: int) -> list[int]:
    """Sort, then move two pointers inward."""
    order = sorted(nums)
    lo, hi = 0, len(order) - 1
    while lo < hi:
        total = order[lo] + order[hi]
        if total == target:
            return [lo, hi]
        if total < target:
            lo += 1
        else:
            hi -= 1
    return []
'''

D1_CODE = '''def two_sum(nums: list[int], target: int) -> list[int]:
    """Single pass with a value -> first index map."""
    first_index: dict[int, int] = {}
    for j, value in enumerate(nums):
        i = first_index.get(target - value)
        if i is not None:
            return [i, j]
        if value not in first_index:
            first_index[value] = j
    return []
'''

_TESTS: list[dict[str, Any]] = [
    {"id": "T01", "category": "normal", "description": "basic", "args": [[2, 7, 11, 15], 9], "expected": [0, 1]},
    {"id": "T02", "category": "normal", "description": "unsorted input", "args": [[3, 2, 4], 6], "expected": [1, 2]},
    {"id": "T03", "category": "boundary", "description": "duplicates", "args": [[3, 3], 6], "expected": [0, 1]},
    {"id": "T04", "category": "boundary", "description": "empty list", "args": [[], 1], "expected": []},
    {"id": "T05", "category": "boundary", "description": "single element", "args": [[5], 10], "expected": []},
    {"id": "T06", "category": "boundary", "description": "negatives and zero target", "args": [[-3, 4, 3, 90], 0], "expected": [0, 2]},
    {"id": "T07", "category": "adversarial", "description": "no solution", "args": [[1, 2, 3], 100], "expected": []},
    {"id": "T08", "category": "adversarial", "description": "None input", "args": [None, 1], "expect_exception": "TypeError"},
    {
        "id": "T09",
        "category": "performance",
        "description": "20000 ascending ints, pair at the end",
        "generator": "([list(range(20000)), 39997], {})",
        "expected": [19998, 19999],
    },
]


def _initial(code: str, algorithm: str, ds: list[str], time_c: str, space_c: str,
             pros: list[str], cons: list[str], assumptions: list[str],
             tests: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "language": "python",
        "code": code,
        "algorithm": algorithm,
        "data_structures": ds,
        "time_complexity": time_c,
        "space_complexity": space_c,
        "pros": pros,
        "cons": cons,
        "assumptions": assumptions,
    }
    if tests is not None:
        out["tests"] = tests
    return out


def _review(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"items": items}


M001_DIFF = '''--- a/solution.py
+++ b/solution.py
@@ -1,5 +1,7 @@
 def two_sum(nums: list[int], target: int) -> list[int]:
     """Return indices [i, j] (i < j) with nums[i] + nums[j] == target, or [] if none."""
+    if not isinstance(nums, list):
+        raise TypeError("nums must be a list of integers")
     seen: dict[int, int] = {}
     for i, value in enumerate(nums):
         complement = target - value
'''

M002_DIFF = '''--- a/solution.py
+++ b/solution.py
@@ -6,4 +6,5 @@
         if complement in seen:
             return [seen[complement], i]
-        seen[value] = i
+        if complement != value:
+            seen[value] = i
     return []
'''


def build_two_sum() -> MockScenario:
    responses: dict[tuple[Role, CallType, str], str] = {}
    for role in Role:
        responses[(role, CallType.ANALYSIS, WILDCARD)] = _js(_ANALYSIS)

    responses[(G, CallType.INITIAL, WILDCARD)] = _js(_initial(
        G1_CODE, "Single pass hash map of value -> index", ["dict"], "O(n)", "O(n)",
        ["Linear time", "Simple"], ["Extra O(n) memory"], ["Inputs are lists of ints"]))
    responses[(R, CallType.INITIAL, WILDCARD)] = _js(_initial(
        C1_CODE, "Brute force over all pairs", ["none"], "O(n^2)", "O(1)",
        ["Obviously correct", "No extra memory"], ["Quadratic time"], ["Small inputs"]))
    responses[(O, CallType.INITIAL, WILDCARD)] = _fenced(_initial(
        P1_CODE, "Sort then two pointers", ["sorted list"], "O(n log n)", "O(n)",
        ["No hash map needed"], ["Sorting cost"], ["Indices may refer to sorted order"]),
        prose="Here is my solution:")
    responses[(V, CallType.INITIAL, WILDCARD)] = _fenced(_initial(
        D1_CODE, "Single pass map storing first index per value", ["dict"], "O(n)", "O(n)",
        ["Linear time", "Keeps earliest index for duplicates"], ["Extra memory"], [],
        tests=_TESTS), prose="Solution and adversarial test suite follow.")

    responses[(G, CallType.REVIEW, WILDCARD)] = _js(_review([
        {"target": "C1", "weaknesses": ["O(n^2)"], "suggestions": ["Use a hash map"]},
        {"target": "P1", "bugs": ["Returns indices into the sorted copy, not the input"],
         "suggestions": ["Sort (value, index) pairs"]},
        {"target": "D1", "strengths": ["Correct on duplicates"]},
    ]))
    responses[(R, CallType.REVIEW, WILDCARD)] = _js(_review([
        {"target": "G1", "strengths": ["Clear"], "suggestions": ["Validate input type explicitly"]},
        {"target": "C1", "strengths": ["Simplest to verify"],
         "justification": "Quadratic time is acceptable for typical interview-sized inputs."},
        {"target": "P1", "bugs": ["Index mismatch after sorting"]},
    ]))
    responses[(O, CallType.REVIEW, WILDCARD)] = _js(_review([
        {"target": "C1", "weaknesses": ["O(n^2) violates the 1e5 performance constraint"],
         "justification": "Disagree with the reviewer: 1e5 elements means ~5e9 pair checks."},
        {"target": "G1", "suggestions": ["Avoid storing values that are their own complement"]},
    ]))

    responses[(G, CallType.MODIFY, "round-1")] = _js({"proposals": []})
    responses[(R, CallType.MODIFY, "round-1")] = _js({"proposals": [{
        "diff": M001_DIFF,
        "justification": "Fail fast with a clear message on non-list input.",
        "expected_impact": "Clearer error; no complexity change.",
        "disadvantages": "Rejects tuples that previously worked.",
    }]})
    responses[(O, CallType.MODIFY, "round-1")] = _js({"proposals": [{
        "diff": M002_DIFF,
        "justification": "Values equal to their own complement never need to be stored.",
        "expected_impact": "Slightly fewer dict insertions.",
        "disadvantages": "None expected.",
    }]})
    responses[(V, CallType.MODIFY, "round-1")] = _js({"proposals": []})
    for role in Role:
        responses[(role, CallType.MODIFY, WILDCARD)] = _js({"proposals": []})

    responses[(G, CallType.CHALLENGE, WILDCARD)] = _js({
        "verdict": "accept", "comment": "The memory saving seems reasonable.", "evidence": ""})
    responses[(R, CallType.CHALLENGE, WILDCARD)] = _js({
        "verdict": "reject", "comment": "Breaks [3, 3] with target 6: the first 3 is never stored.",
        "evidence": "Failing test T03"})
    responses[(V, CallType.CHALLENGE, WILDCARD)] = _js({
        "verdict": "reject", "comment": "Regression on duplicate values.", "evidence": "T03 fails"})
    responses[(O, CallType.CHALLENGE, WILDCARD)] = _js({
        "verdict": "accept", "comment": "Input validation is cheap and clearer.", "evidence": ""})

    failures: dict[tuple[Role, CallType, str], FailureSpec] = {
        (V, CallType.REVIEW, WILDCARD): FailureSpec("rate_limit", times=None),
        (V, CallType.MODIFY, "round-1"): FailureSpec("rate_limit", times=1),
    }
    return MockScenario(name="two_sum", responses=responses, failures=failures)


SCENARIOS: dict[str, MockScenario] = {"two_sum": build_two_sum()}
PROBLEMS: dict[str, str] = {"two_sum": TWO_SUM_PROBLEM}


def get_scenario(name: str) -> MockScenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown mock scenario '{name}'; available: {sorted(SCENARIOS)}") from None
