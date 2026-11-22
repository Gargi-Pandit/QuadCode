import os


def two_sum(nums: list[int], target: int) -> list[int]:
    """Hostile sample: kills the interpreter. Expect ERROR on T01 and NOT_RUN for the rest."""
    os._exit(3)
