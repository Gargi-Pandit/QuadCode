def first_pair(nums: list[int], target: int) -> list[int]:
    """Pair with the smallest j."""
    if len(nums) < 2:
        return []
    seen: dict[int, int] = {}
    for j, value in enumerate(nums):
        if target - value in seen:
            return [seen[target - value], j]
        seen.setdefault(value, j)
    return []


def last_pair(nums: list[int], target: int) -> list[int]:
    """Pair with the largest i."""
    if len(nums) < 2:
        return []
    seen: dict[int, int] = {}
    for i in range(len(nums) - 1, -1, -1):
        if target - nums[i] in seen:
            return [i, seen[target - nums[i]]]
        seen.setdefault(nums[i], i)
    return []
