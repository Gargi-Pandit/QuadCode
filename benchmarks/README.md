# Benchmark problems

Used by `quadcode bench`. Each directory under `problems/` is one problem:

| file | who sees it | purpose |
|---|---|---|
| `problem.md` | the agents (it is the only thing a run is given) | statement; names the function and every tie rule |
| `hidden_tests.json` | nobody during a run | scored only after a run has finished; every test has `"hidden": true` |
| `reference.py` | nobody during a run | hand-written solution; `tests/test_bench_problems.py` checks that it passes every hidden test, so the hand-written expected values are cross-checked |
| `meta.yaml` | the harness | `entry_point`, `language`, optional `mock_scenario` |

Problems with a `mock_scenario` (two_sum, merge_intervals, lis_length) can run with `--mock`; the
scripted answers ignore the statement.

To add a problem: create the four files, keep inputs small enough that a correct solution runs in well
under a second, write expected values by hand, and run `pytest tests/test_bench_problems.py`.
