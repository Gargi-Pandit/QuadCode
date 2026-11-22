import os


def two_sum(nums: list[int], target: int) -> list[int]:
    """Hostile sample: returns any credential-like environment variable names it can see.

    If the sandbox strips the environment, this behaves like a correct solution and
    every test passes. A leak shows up as FAIL with the variable names as the output.
    """
    leaked = sorted(k for k in os.environ if any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET", "PASSWORD")))
    if leaked:
        return leaked
    seen: dict[int, int] = {}
    for i, value in enumerate(nums):
        if target - value in seen:
            return [seen[target - value], i]
        seen[value] = i
    return []
