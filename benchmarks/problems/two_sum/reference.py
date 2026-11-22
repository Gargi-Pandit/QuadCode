def two_sum(nums, target):
    first = {}
    for j, value in enumerate(nums):
        need = target - value
        if need in first:
            return [first[need], j]
        if value not in first:
            first[value] = j
    return []
