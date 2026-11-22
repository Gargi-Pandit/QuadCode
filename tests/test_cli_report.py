"""``quadcode report`` and the report written at the end of ``quadcode solve``."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from quadcode.cli import main

from .conftest import SAMPLES
from .orch_helpers import TEST_CONFIG

PROBLEM = str(SAMPLES / "problem.md")


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    data = copy.deepcopy(TEST_CONFIG)
    data["run"] = {"rounds": 3, "cache_dir": str(tmp_path / "cache"), "runs_dir": str(tmp_path / "runs")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _cli(cfg: Path, command: str, *args: str) -> list[str]:
    return ["--config", str(cfg), "--env-file", str(cfg.parent / "none.env"), command, *args]


def _solve(cfg: Path, *args: str) -> list[str]:
    return _cli(cfg, "solve", PROBLEM, "--mock", "--no-cache", *args)


def test_solve_writes_the_report_and_report_rebuilds_it(cfg_path: Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_solve(cfg_path, "--run-id", "r1")) == 0
    out = capsys.readouterr().out
    run_dir = cfg_path.parent / "runs" / "r1"
    assert f"report: {run_dir / 'report.md'}" in out and f"report: {run_dir / 'report.html'}" in out
    md = (run_dir / "report.md").read_text()
    html = (run_dir / "report.html").read_text()
    assert "| status | COMPLETED |" in md and html.startswith("<!doctype html>")

    assert main(_cli(cfg_path, "report", "r1")) == 0
    assert (run_dir / "report.md").read_text() == md and (run_dir / "report.html").read_text() == html
    assert main(_cli(cfg_path, "report", str(run_dir), "--format", "md")) == 0
    out = capsys.readouterr().out
    assert "report.md" in out.splitlines()[-1] and "report.html" not in out.splitlines()[-1]


def test_aborted_solve_still_writes_a_report(cfg_path: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_solve(cfg_path, "--run-id", "b", "--budget-calls", "3")) == 4
    md = (cfg_path.parent / "runs" / "b" / "report.md").read_text()
    assert "| status | ABORTED_BUDGET |" in md and "quadcode solve --resume b" in md


def test_report_of_unknown_run(cfg_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli(cfg_path, "report", "nope")) == 2
    assert "no run directory" in capsys.readouterr().err


def test_report_works_without_a_valid_config(cfg_path: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_solve(cfg_path, "--run-id", "r2", "--rounds", "1")) == 0
    run_dir = cfg_path.parent / "runs" / "r2"
    bad = cfg_path.parent / "missing.yaml"
    assert main(["--config", str(bad), "report", str(run_dir)]) == 0
    captured = capsys.readouterr()
    assert "cost will show as not configured" in captured.err
    assert "not configured" in (run_dir / "report.md").read_text()
