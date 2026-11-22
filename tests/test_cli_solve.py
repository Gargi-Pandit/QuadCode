from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest
import yaml

from quadcode.cli import main
from quadcode.cli_solve import mock_config
from quadcode.config import parse_config
from quadcode.schemas import Role

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


def _cli(cfg: Path, *args: str) -> list[str]:
    return ["--config", str(cfg), "--env-file", str(cfg.parent / "none.env"), "solve", *args]


def test_solve_mock_end_to_end(cfg_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli(cfg_path, PROBLEM, "--mock", "--no-cache", "--run-id", "r1")) == 0
    out = capsys.readouterr().out
    assert "run r1: COMPLETED" in out
    assert "V0 = copy of G1 (selected on evidence)" in out
    assert "code_size: kept G1; dropped D1" in out
    assert re.search(r"validator +D1 +mock/mock-validator-v1 +review +unavailable +RateLimitError", out)
    assert "generator:accept, reviewer:reject, validator:reject" in out
    assert "stop: round 2 accepted no modification (early stop)" in out
    assert "final: V1 ->" in out and "(passes every test)" in out
    assert "scripted by scenario two_sum" in out
    run_dir = cfg_path.parent / "runs" / "r1"
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "completed"
    assert not (cfg_path.parent / "cache").exists()


def test_budget_abort_and_resume(cfg_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli(cfg_path, PROBLEM, "--mock", "--no-cache", "--run-id", "r2", "--budget-calls", "6")) == 4
    out = capsys.readouterr().out
    assert "ABORTED_BUDGET" in out and "call budget of 6 exhausted" in out
    assert main(_cli(cfg_path, "--mock", "--no-cache", "--resume", "r2")) == 0
    out = capsys.readouterr().out
    assert "resuming r2 (completed steps: analysis, initial, local_tests)" in out and "COMPLETED" in out
    assert main(_cli(cfg_path, PROBLEM, "--mock", "--run-id", "r2")) == 2
    assert "already holds a run" in capsys.readouterr().err


def test_usage_errors(cfg_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli(cfg_path, "--mock")) == 2
    assert "problem file is required" in capsys.readouterr().err
    assert main(_cli(cfg_path, str(tmp_path / "missing.md"), "--mock")) == 2
    assert main(_cli(cfg_path, "--mock", "--resume", "nope")) == 2
    assert main(_cli(cfg_path, PROBLEM, "--mock", "--rounds", "9")) == 2
    other = tmp_path / "other.md"
    other.write_text("different")
    assert main(_cli(cfg_path, PROBLEM, "--mock", "--no-cache", "--run-id", "r3", "--budget-calls", "1")) == 4
    capsys.readouterr()
    assert main(_cli(cfg_path, str(other), "--mock", "--resume", "r3")) == 2
    assert "differs" in capsys.readouterr().err


def test_real_providers_are_not_wired(cfg_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = yaml.safe_load(cfg_path.read_text())
    data["providers"]["remote"] = {"kind": "openai_compat", "base_url": "https://example.invalid/v1",
                                   "api_key_env": "X_KEY", "rpm": 1, "tpm": 1000}
    data["agents"]["generator"] = {"provider": "remote", "model": "some-model"}
    cfg_path.write_text(yaml.safe_dump(data))
    assert main(_cli(cfg_path, PROBLEM, "--no-cache", "--run-id", "r4")) == 2
    assert "run with --mock" in capsys.readouterr().err
    assert main(_cli(cfg_path, PROBLEM, "--mock", "--no-cache", "--run-id", "r5", "--rounds", "1")) == 0


def test_mock_config_replaces_real_slots() -> None:
    data = copy.deepcopy(TEST_CONFIG)
    data["providers"]["remote"] = {"kind": "gemini", "api_key_env": "K", "rpm": 1, "tpm": 1}
    data["agents"]["reviewer"] = {"provider": "remote", "model": "m"}
    cfg = mock_config(parse_config(data), None)
    assert set(cfg.providers) == {"mock"} and cfg.providers["mock"].scenario == "two_sum"
    assert {cfg.agent(r).model for r in Role} == {f"mock-{r.value}-v1" for r in Role}
    all_mock = parse_config(copy.deepcopy(TEST_CONFIG))
    assert mock_config(all_mock, None) == all_mock
    assert mock_config(all_mock, "other").providers["mock"].scenario == "other"
