def two_sum(nums: list[int], target: int) -> list[int]:
    """Brute force: check every pair, ordered by j then i."""
    n = len(nums)
    for j in range(1, n):
        for i in range(j):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
