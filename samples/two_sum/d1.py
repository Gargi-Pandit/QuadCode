def two_sum(nums: list[int], target: int) -> list[int]:
    """Single pass with a value -> first index map."""
    first_index: dict[int, int] = {}
    for j, value in enumerate(nums):
        i = first_index.get(target - value)
        if i is not None:
            return [i, j]
        if value not in first_index:
            first_index[value] = j
    return []
