from __future__ import annotations

import json
from pathlib import Path

import pytest

from quadcode.cli import main

from .conftest import REPO_ROOT, SAMPLES

CONFIG = str(REPO_ROOT / "config.yaml")
TESTS = str(SAMPLES / "tests.json")
G1 = str(SAMPLES / "g1.py")
DIFFS = SAMPLES / "diffs"


def _cli(*args: str) -> list[str]:
    return ["--config", CONFIG, *args]


def _test_args() -> list[str]:
    return ["--tests", TESTS, "--entry", "two_sum", "--backend", "subprocess", "--repeats", "1"]


def test_apply_diff_shows_locations_and_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli("apply-diff", G1, str(DIFFS / "two_hunks.diff"))) == 0
    out = capsys.readouterr().out
    assert "diff applied to g1 (2 hunk(s))" in out
    assert "g1:after L2" in out and "patched:L3-L4" in out
    assert "g1:L8" in out and "patched:L10" in out
    assert "seen.setdefault(value, i)" in out


def test_apply_diff_with_tests_pass(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli("apply-diff", G1, str(DIFFS / "wrong_header.diff"), *_test_args())) == 0
    out = capsys.readouterr().out
    assert "-8" in out and "pass 9" in out


def test_apply_diff_with_tests_fail(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli("apply-diff", G1, str(DIFFS / "m002_skip_self.diff"), *_test_args())) == 1
    out = capsys.readouterr().out
    assert "T03" in out and "FAIL" in out and "fail 1" in out


@pytest.mark.parametrize("diff, base, message", [
    ("stale_context.diff", G1, "context does not match"),
    ("ambiguous.diff", str(DIFFS / "pair_helpers.py"), "ambiguous"),
])
def test_apply_diff_failures(diff: str, base: str, message: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli("apply-diff", base, str(DIFFS / diff))) == 1
    assert message in capsys.readouterr().out


def test_apply_diff_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli("apply-diff", G1, str(DIFFS / "m001_validation.diff"), "--tests", TESTS)) == 2
    assert main(_cli("apply-diff", G1, str(DIFFS / "m001_validation.diff"), "--run-dir", "x")) == 2
    assert main(_cli("apply-diff", "missing.py", str(DIFFS / "m001_validation.diff"))) == 2
    err = capsys.readouterr().err
    assert "--entry is required" in err and "--proposed-by is required" in err and "cannot read input" in err


def test_apply_diff_store_and_versions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "demo"
    out_json = tmp_path / "m001.json"
    assert main(_cli("apply-diff", G1, str(DIFFS / "m001_validation.diff"), "--run-dir", str(run),
                     "--proposed-by", "reviewer", "--json", str(out_json), *_test_args())) == 0
    out = capsys.readouterr().out
    assert "M-001 by reviewer: stored candidate G1+M-001" in out
    assert "M-001: test status pass, decision pending" in out
    assert json.loads(out_json.read_text(encoding="utf-8"))["candidate_version"] == "G1+M-001"

    # A second, unappliable diff against the same run: base re-save is a no-op, the record is kept.
    assert main(_cli("apply-diff", G1, str(DIFFS / "stale_context.diff"), "--run-dir", str(run),
                     "--proposed-by", "validator")) == 1
    assert "M-002 recorded as rejected" in capsys.readouterr().out

    assert main(_cli("versions", str(run))) == 0
    out = capsys.readouterr().out
    assert "G1+M-001" in out and "candidate" in out and "initial" in out
    assert "M-001" in out and "M-002" in out and "diff did not apply" in out
    assert "2 modification record(s), 1 pending" in out

    assert main(_cli("versions", str(run), "--blame", "G1+M-001")) == 0
    out = capsys.readouterr().out
    assert "lineage: G1 -> G1+M-001" in out
    lines = out.splitlines()
    assert any(ln.startswith("3 ") and "M-001 reviewer" in ln for ln in lines)
    assert any(ln.startswith("5 ") and "G1:L3 generator" in ln for ln in lines)


def test_apply_diff_store_needs_initial_label(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_cli("apply-diff", str(DIFFS / "pair_helpers.py"), str(DIFFS / "ambiguous_with_line.diff"),
                     "--run-dir", str(tmp_path / "r"), "--proposed-by", "generator"))
    assert code == 2
    assert "must be G1, C1, P1 or D1" in capsys.readouterr().err


def test_versions_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli("versions", str(tmp_path))) == 2
    assert "no versions directory" in capsys.readouterr().err
    (tmp_path / "versions").mkdir()
    assert main(_cli("versions", str(tmp_path), "--blame", "V9")) == 2
    assert "version store" in capsys.readouterr().err
