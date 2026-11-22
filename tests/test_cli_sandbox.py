from __future__ import annotations

import json
from pathlib import Path

import pytest

from quadcode.cli import main

from .conftest import REPO_ROOT, SAMPLES

CONFIG = str(REPO_ROOT / "config.yaml")
TESTS = str(SAMPLES / "tests.json")


def _cli(*args: str) -> list[str]:
    return ["--config", CONFIG, *args]


def test_sandbox_info_subprocess(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli("sandbox-info", "--backend", "subprocess")) == 0
    out = capsys.readouterr().out
    assert "backend: subprocess" in out
    assert "WARNING: Weaker isolation" in out


def test_test_solution_passing(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_cli("test-solution", str(SAMPLES / "g1.py"), "--tests", TESTS, "--entry", "two_sum",
                     "--backend", "subprocess", "--repeats", "1"))
    out = capsys.readouterr().out
    assert code == 0
    assert "== g1" in out and "pass 9" in out
    assert "method:" in out


def test_test_solution_failing_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_file = tmp_path / "out" / "p1.json"
    code = main(_cli("test-solution", str(SAMPLES / "p1.py"), "--tests", TESTS, "--entry", "two_sum",
                     "--backend", "subprocess", "--repeats", "1", "--json", str(out_file)))
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL" in out and "expected [1,2], got [0,2]" in out
    data = json.loads(out_file.read_text(encoding="utf-8"))
    statuses = {r["test_id"]: r["status"] for r in data[0]["results"]}
    assert statuses["T02"] == "fail" and statuses["T01"] == "pass"


def test_test_solution_benchmark_table(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_cli("test-solution", str(SAMPLES / "g1.py"), str(SAMPLES / "d1.py"), "--tests", TESTS,
                     "--entry", "two_sum", "--backend", "subprocess", "--benchmark", "--repeats", "2"))
    out = capsys.readouterr().out
    assert code == 0
    assert "benchmark (measured):" in out
    assert "median_ms" in out and "min_ms" in out and "max_ms" in out


def test_test_solution_timeout_flag(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_cli("test-solution", str(SAMPLES / "c1.py"), "--tests", TESTS, "--entry", "two_sum",
                     "--backend", "subprocess", "--repeats", "1", "--test-timeout", "0.5"))
    out = capsys.readouterr().out
    assert code == 1
    assert "TIMEOUT" in out and "0.5s limit" in out


def test_test_solution_bad_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli("test-solution", str(tmp_path / "missing.py"), "--tests", TESTS, "--entry", "f",
                     "--backend", "subprocess")) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("[{\"id\": 1}]")
    assert main(_cli("test-solution", str(SAMPLES / "g1.py"), "--tests", str(bad), "--entry", "two_sum",
                     "--backend", "subprocess")) == 2
    assert "cannot read tests" in capsys.readouterr().err
