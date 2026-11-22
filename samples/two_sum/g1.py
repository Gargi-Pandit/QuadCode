def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices [i, j] (i < j) with nums[i] + nums[j] == target, or [] if none."""
    seen: dict[int, int] = {}
    for i, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], i]
        seen[value] = i
    return []
