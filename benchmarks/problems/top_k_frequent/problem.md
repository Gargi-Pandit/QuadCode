# Top k frequent words

Write a Python function `top_k_frequent(words: list[str], k: int) -> list[str]` that returns the `k` most frequent distinct words.

- Order: higher frequency first; words with equal frequency in ascending Python string order (so `"A" < "B" < "a"`).
- If `k` is larger than the number of distinct words, return all of them in that order. `k` is never negative; `k == 0` returns `[]`.
- Up to 2 * 10^5 words.
