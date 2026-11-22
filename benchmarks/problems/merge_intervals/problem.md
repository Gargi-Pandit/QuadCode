# Merge intervals

Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]`.

Each interval is `[start, end]` with `start <= end` (closed interval). Merge all intervals that overlap and return the merged intervals sorted by start.

- Intervals that only touch also merge: `[1, 4]` and `[4, 5]` become `[1, 5]`.
- The input may be unsorted, may contain duplicates, and may be empty (return `[]`).
- Values may be negative. Up to 10^5 intervals.
- Return lists, not tuples.
