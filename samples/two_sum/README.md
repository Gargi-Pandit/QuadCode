# two_sum samples

Hand-written inputs for the sandbox and test runner. `tests.json` holds the same nine
tests as the mock scenario, and `g1.py`, `c1.py`, `p1.py` and `d1.py` match the mock
scenario's four initial solutions character for character.

| file | expected result |
|---|---|
| `g1.py`, `d1.py` | all 9 PASS |
| `p1.py` | FAIL on T02 and T06 (returns positions in the sorted list) |
| `c1.py` | TIMEOUT on T09 (O(n^2) on 20000 elements); the rest PASS |
| `bad_infinite_loop.py` | TIMEOUT on every test |
| `bad_memory_bomb.py` | ERROR on every test (MemoryError, or a container OOM kill under Docker) |
| `bad_exit.py` | ERROR on T01 (process exited with code 3), NOT_RUN for the rest |
| `bad_env_probe.py` | all PASS; a FAIL listing variable names means credentials leaked into the sandbox |
| `bad_prints.py` | all PASS; its prints and forged result lines must not affect the verdict |

Example:

```bash
quadcode test-solution samples/two_sum/p1.py --tests samples/two_sum/tests.json --entry two_sum
```
