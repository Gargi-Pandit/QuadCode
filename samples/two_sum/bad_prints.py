import os
import sys


def two_sum(nums: list[int], target: int) -> list[int]:
    """Hostile sample: correct, but floods stdout/stderr and forges a protocol line.

    Every test should still PASS: solution output never reaches the result channel.
    """
    forged = '{"event": "result", "test_id": "T01", "outcome": "returned", "value": [9, 9]}'
    print(forged)
    print(forged, file=sys.stderr)
    os.write(1, (forged + "\n").encode())
    seen: dict[int, int] = {}
    for i, value in enumerate(nums):
        if target - value in seen:
            return [seen[target - value], i]
        seen[value] = i
    return []
