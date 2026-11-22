def two_sum(nums: list[int], target: int) -> list[int]:
    """Hostile sample: allocates until the sandbox memory cap stops it. Expect ERROR."""
    hoard = []
    while True:
        hoard.append(bytearray(50_000_000))
