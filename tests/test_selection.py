from __future__ import annotations

from quadcode.config import SelectionConfig
from quadcode.schemas import SuiteRun, TestResult, TestStatus
from quadcode.selection import code_size, compare_to_base, is_noop, rank_versions

CFG = SelectionConfig(runtime_tolerance=0.25, memory_tolerance=0.10, memory_floor_bytes=100)
PERF = ["T9"]


def _suite(version: str, statuses: dict[str, TestStatus], runtime: float | None = None,
           memory: int | None = None, executed: bool = True, runs: list[float] | None = None) -> SuiteRun:
    results = []
    for tid, st in statuses.items():
        perf = tid in PERF and st is TestStatus.PASS
        results.append(TestResult(test_id=tid, version=version, status=st, measured=executed,
                                  runtime_s=runtime if perf else None,
                                  runtimes_s=(runs or [runtime]) if perf and runtime is not None else [],
                                  peak_memory_bytes=memory if perf else None))
    return SuiteRun(version=version, language="python", entry_point="f", backend="subprocess",
                    executed=executed, repeats=1, warmup=0, results=results)


P = TestStatus.PASS
F = TestStatus.FAIL
TO = TestStatus.TIMEOUT
CODE = {"G1": "a\nb\n", "C1": "a\nb\n", "P1": "a\nb\n", "D1": "a\nb\n"}


def test_tests_passed_decides_first() -> None:
    suites = {
        "G1": _suite("G1", {"T1": P, "T9": P}, runtime=1.0, memory=1000),
        "C1": _suite("C1", {"T1": P, "T9": TO}),
        "P1": _suite("P1", {"T1": F, "T9": P}, runtime=0.001, memory=10),
    }
    ev = rank_versions(suites, CODE, PERF, CFG, "V0")
    assert ev.winner == "G1"
    assert ev.verified
    assert ev.steps[0].criterion == "tests_passed"
    assert ev.steps[0].dropped == ["C1", "P1"]
    assert len(ev.steps) == 1
    assert ev.not_measured == ["security", "readability", "maintainability"]


def test_runtime_outside_tolerance_decides() -> None:
    suites = {
        "G1": _suite("G1", {"T1": P, "T9": P}, runtime=2.0, memory=1000),
        "D1": _suite("D1", {"T1": P, "T9": P}, runtime=1.0, memory=1000),
    }
    ev = rank_versions(suites, CODE, PERF, CFG, "V0")
    assert ev.winner == "D1"
    assert ev.steps[1].criterion == "runtime" and ev.steps[1].dropped == ["G1"]


def test_overlapping_runtime_ranges_are_noise() -> None:
    # Median 2x slower, but the slower version's fastest run beats the faster one's slowest run.
    suites = {
        "G1": _suite("G1", {"T9": P}, runtime=2.0, memory=1000, runs=[1.0, 2.0, 2.6]),
        "D1": _suite("D1", {"T9": P}, runtime=1.0, memory=1000, runs=[0.9, 1.0, 1.1]),
    }
    ev = rank_versions(suites, CODE, PERF, CFG, "V0")
    assert ev.steps[1].kept == ["G1", "D1"]
    assert "noise" in ev.steps[1].note
    base = suites["D1"]
    assert not compare_to_base(base, suites["G1"], PERF, CFG).runtime_regression


def test_runtime_within_tolerance_is_a_tie_then_memory() -> None:
    suites = {
        "G1": _suite("G1", {"T9": P}, runtime=1.2, memory=1000),
        "D1": _suite("D1", {"T9": P}, runtime=1.0, memory=5000),
    }
    ev = rank_versions(suites, CODE, PERF, CFG, "V0")
    assert ev.steps[1].kept == ["G1", "D1"]
    assert ev.steps[2].criterion == "memory"
    assert ev.winner == "G1"


def test_memory_floor_and_code_size_then_slot_order() -> None:
    suites = {
        "G1": _suite("G1", {"T9": P}, runtime=1.0, memory=1000),
        "D1": _suite("D1", {"T9": P}, runtime=1.0, memory=1050),
    }
    ev = rank_versions(suites, {"G1": "a\nb\nc\n", "D1": "a\n\n# c\nb\n"}, PERF, CFG, "V0")
    assert [s.criterion for s in ev.steps] == ["tests_passed", "runtime", "memory", "code_size"]
    assert ev.winner == "D1"

    ev = rank_versions(suites, CODE, PERF, CFG, "V0")
    assert ev.steps[-1].criterion == "slot_order"
    assert not ev.steps[-1].measured
    assert ev.winner == "G1"


def test_no_perf_measurement_is_stated_not_invented() -> None:
    suites = {"G1": _suite("G1", {"T1": P}), "D1": _suite("D1", {"T1": P})}
    ev = rank_versions(suites, CODE, PERF, CFG, "V0")
    runtime = ev.steps[1]
    assert runtime.criterion == "runtime" and not runtime.measured and runtime.values == {}


def test_nothing_passes_is_unverified() -> None:
    suites = {"G1": _suite("G1", {"T1": F}), "C1": _suite("C1", {"T1": P}, executed=False)}
    ev = rank_versions(suites, CODE, PERF, CFG, "V0")
    assert not ev.verified
    assert ev.steps[0].values == {"G1": 0, "C1": 0}


def test_empty() -> None:
    ev = rank_versions({}, {}, PERF, CFG, "V0")
    assert ev.winner is None and not ev.verified


def test_compare_to_base() -> None:
    base = _suite("V0", {"T9": P}, runtime=1.0, memory=1000)
    assert not compare_to_base(base, _suite("c", {"T9": P}, runtime=1.2, memory=1050), PERF, CFG).regression
    slow = compare_to_base(base, _suite("c", {"T9": P}, runtime=1.5, memory=1000), PERF, CFG)
    assert slow.runtime_regression and slow.regression
    assert "regression" in slow.describe()
    fat = compare_to_base(base, _suite("c", {"T9": P}, runtime=1.0, memory=2000), PERF, CFG)
    assert fat.memory_regression
    fast = compare_to_base(base, _suite("c", {"T9": P}, runtime=0.5, memory=1000), PERF, CFG)
    assert fast.runtime_improvement and not fast.regression
    none = compare_to_base(base, _suite("c", {"T9": F}), PERF, CFG)
    assert not none.regression and none.describe() == "no performance measurement to compare"


def test_is_noop() -> None:
    a = "def f(x):\n    return x + 1\n"
    assert is_noop(a, "def f(x):\n    # add one\n    return (x + 1)\n", "python")
    assert not is_noop(a, "def f(x):\n    return x + 2\n", "python")
    assert not is_noop(a, "def f(x:\n", "python")
    assert is_noop("int f() { return 1; }", "int f()  {\n return 1; }", "c")


def test_code_size() -> None:
    assert code_size("a\n\n  # c\nb\n") == 2
