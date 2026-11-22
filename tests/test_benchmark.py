from __future__ import annotations

import pytest

from quadcode.benchmark import benchmark_versions, summarize
from quadcode.config import SandboxConfig
from quadcode.providers import mock_scenarios
from quadcode.sandbox import SubprocessSandbox
from quadcode.schemas import TestCase, TestCategory, TestStatus


def test_summarize_statistics() -> None:
    s = summarize("G1", "T1", TestStatus.PASS, [0.3, 0.1, 0.2], 1024)
    assert s is not None
    assert (s.median_s, s.min_s, s.max_s) == (0.2, 0.1, 0.3)
    assert s.mean_s == pytest.approx(0.2)
    assert s.stdev_s == pytest.approx(0.1)
    assert s.repeats == 3 and s.peak_memory_bytes == 1024 and s.measured


def test_summarize_single_run_and_none() -> None:
    one = summarize("G1", "T1", TestStatus.PASS, [0.5], None)
    assert one is not None and one.stdev_s is None
    assert summarize("G1", "T1", TestStatus.TIMEOUT, [], None) is None


def test_benchmark_identical_inputs_across_versions(two_sum_tests: list[TestCase],
                                                    subprocess_sandbox: SubprocessSandbox,
                                                    sandbox_cfg: SandboxConfig) -> None:
    versions = {"G1": mock_scenarios.G1_CODE, "C1": mock_scenarios.C1_CODE, "D1": mock_scenarios.D1_CODE}
    report = benchmark_versions(versions, entry_point="two_sum", language="python", tests=two_sum_tests,
                                sandbox=subprocess_sandbox, cfg=sandbox_cfg,
                                categories={TestCategory.PERFORMANCE})
    assert report.order == ["G1", "C1", "D1"]
    assert report.repeats == sandbox_cfg.benchmark_repeats
    assert report.backend == "subprocess" and report.isolation_warning
    assert "identical payload" in report.method and "median" in report.method
    assert all([r.test_id for r in s.results] == ["T09"] for s in report.suites)

    by_version = {s.version: s for s in report.stats}
    assert set(by_version) == {"G1", "D1"}  # C1 timed out, so it has no runtime statistics
    for label in ("G1", "D1"):
        s = by_version[label]
        assert s.status == TestStatus.PASS
        assert s.repeats == sandbox_cfg.benchmark_repeats == len(s.runtimes_s)
        assert s.min_s <= s.median_s <= s.max_s
    c1 = next(s for s in report.suites if s.version == "C1")
    assert c1.results[0].status == TestStatus.TIMEOUT


def test_benchmark_requires_selected_tests(two_sum_tests: list[TestCase], subprocess_sandbox: SubprocessSandbox,
                                           sandbox_cfg: SandboxConfig) -> None:
    only_normal = [t for t in two_sum_tests if t.category == TestCategory.NORMAL]
    with pytest.raises(ValueError, match="no tests selected"):
        benchmark_versions({"G1": mock_scenarios.G1_CODE}, entry_point="two_sum", language="python",
                           tests=only_normal, sandbox=subprocess_sandbox, cfg=sandbox_cfg,
                           categories={TestCategory.PERFORMANCE})
