# Sample diffs

Unified diffs for `quadcode apply-diff`. All but the last two were written against `../g1.py`.
`m001_validation.diff` and `m002_skip_self.diff` are the same text the mock scenario's
reviewer and optimizer propose in round 1.

| diff | base | expected result |
|---|---|---|
| `m001_validation.diff` | `g1.py` | applies (offset +0); inserts 2 lines at `G1:after L2`; all 9 tests PASS |
| `m002_skip_self.diff` | `g1.py` | applies (offset +0); replaces `G1:L8`; T03 FAIL (`[3, 3]` returns `[]`), the rest PASS |
| `wrong_header.diff` | `g1.py` | header says line 12, the context is at line 4: applies with offset -8; all 9 PASS |
| `stale_context.diff` | `g1.py` | does not apply: written against code that names the dict `lookup`; exit code 1 |
| `two_hunks.diff` | `g1.py` | applies, 2 hunks; the second lands at `patched:L10` because the first added 2 lines; all 9 PASS |
| `ambiguous.diff` | `pair_helpers.py` | does not apply: no line numbers and the context matches at lines 3 and 15; exit code 1 |
| `ambiguous_with_line.diff` | `pair_helpers.py` | same change, header names line 15: applies to `last_pair` only |

Examples:

```bash
quadcode apply-diff samples/two_sum/g1.py samples/two_sum/diffs/m001_validation.diff --tests samples/two_sum/tests.json --entry two_sum
```

```bash
quadcode apply-diff samples/two_sum/g1.py samples/two_sum/diffs/m001_validation.diff --run-dir runs/demo --proposed-by reviewer --tests samples/two_sum/tests.json --entry two_sum
```

```bash
quadcode versions runs/demo --blame G1+M-001
```
