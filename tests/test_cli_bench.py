"""``quadcode bench`` and ``quadcode bench-report`` on mock agents."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from quadcode.cli import main

from .orch_helpers import TEST_CONFIG


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    data = copy.deepcopy(TEST_CONFIG)
    data["run"] = {"rounds": 1, "cache_dir": str(tmp_path / "cache"), "runs_dir": str(tmp_path / "runs")}
    data["sandbox"] = {**data["sandbox"], "test_repeats": 1, "benchmark_repeats": 1}
    data["evaluation"] = {"bootstrap_samples": 500}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _cli(cfg: Path, *args: str) -> list[str]:
    return ["--config", str(cfg), "--env-file", str(cfg.parent / "none.env"), *args]


def test_bench_mock_then_report(cfg_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_cli(cfg_path, "bench", "--mock", "--problems", "two_sum", "--conditions", "full",
                     "--bench-id", "cb1"))
    out = capsys.readouterr().out
    assert code == 0, out
    assert "bench cb1: COMPLETED" in out and "mock mode" in out
    assert "single-validator" in out and "oracle" in out and "no-review" not in out
    bench_dir = cfg_path.parent / "runs" / "cb1"
    for name in ("bench.json", "results.csv", "summary.csv", "comparisons.csv", "summary.md",
                 "hidden/two_sum.json", "runs/two_sum--full/report.md"):
        assert (bench_dir / name).is_file(), name
    before = (bench_dir / "summary.md").read_bytes()
    assert main(_cli(cfg_path, "bench-report", "cb1")) == 0
    assert (bench_dir / "summary.md").read_bytes() == before
    assert main(_cli(cfg_path, "bench-report", str(bench_dir))) == 0


def test_bench_budget_abort_exit_code_and_resume(cfg_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ("bench", "--mock", "--problems", "two_sum", "--conditions", "full")
    assert main(_cli(cfg_path, *args, "--bench-id", "cb2", "--budget-calls", "3")) == 4
    out = capsys.readouterr().out
    assert "ABORTED_BUDGET" in out and "provider attempts used: 3" in out
    assert main(_cli(cfg_path, "bench", "--resume", "cb2")) == 2  # started with --mock
    assert "started with --mock" in capsys.readouterr().err
    assert main(_cli(cfg_path, "bench", "--mock", "--resume", "cb2")) == 0
    assert "bench cb2: COMPLETED" in capsys.readouterr().out


def test_bench_usage_errors(cfg_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli(cfg_path, "bench", "--mock", "--conditions", "no-review")) == 2
    assert "must include full" in capsys.readouterr().err
    assert main(_cli(cfg_path, "bench", "--mock", "--problems", "edit_distance")) == 2
    assert "no mock scenario" in capsys.readouterr().err
    assert main(_cli(cfg_path, "bench", "--mock", "--problems", "nope")) == 2
    assert "unknown problems" in capsys.readouterr().err
    assert main(_cli(cfg_path, "bench", "--mock", "--rounds", "9")) == 2
    assert main(_cli(cfg_path, "bench-report", "missing-bench")) == 2


def test_real_bench_needs_keys(cfg_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = yaml.safe_load(cfg_path.read_text())
    data["providers"]["real"] = {"kind": "openai_compat", "base_url": "https://example.invalid/v1",
                                 "api_key_env": "QC_TEST_MISSING_KEY", "rpm": 10, "tpm": 1000}
    data["agents"]["generator"] = {"provider": "real", "model": "m"}
    cfg_path.write_text(yaml.safe_dump(data))
    assert main(_cli(cfg_path, "bench", "--problems", "two_sum", "--conditions", "full", "--bench-id", "k")) == 2
    assert "QC_TEST_MISSING_KEY" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "k").exists()
