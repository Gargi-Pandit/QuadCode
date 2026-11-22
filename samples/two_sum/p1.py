def two_sum(nums: list[int], target: int) -> list[int]:
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
