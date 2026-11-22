# Two Sum (indices)

Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that returns indices `[i, j]` with `i < j` such that `nums[i] + nums[j] == target`.

- Scan `j` from left to right and return the pair for the smallest such `j`. If several `i` work for that `j`, use the smallest `i`.
- Return `[]` if no pair exists.
- `nums` may be empty and may contain up to 10^5 integers, including negatives, zeros and duplicates.
