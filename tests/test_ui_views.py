"""Dashboard view models of the two_sum mock run, checked against the run's own files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from quadcode.report.data import RunRecords, load_run
from quadcode.schemas import TestStatus
from quadcode.ui import views

from .ui_helpers import digest, jsonl, mock_run


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return mock_run(tmp_path_factory.mktemp("views"))


@pytest.fixture(scope="module")
def rec(run_dir: Path) -> RunRecords:
    return load_run(run_dir)


def test_overview_matches_summary_json(run_dir: Path, rec: RunRecords) -> None:
    summary = json.loads((run_dir / "summary.json").read_text())
    ov = views.overview(rec)
    assert ov.status == summary["status"] == "completed"
    assert ov.final_version == summary["final_version"] and ov.final_verified is True
    assert ov.rounds == f"{summary['rounds_run']} of {summary['rounds_requested']}"
    assert ov.accepted == summary["accepted"] == ["M-001"] and ov.rejected == summary["rejected"]
    assert ov.reference == "V0 = copy of G1" and ov.mock


def test_slots_name_the_logged_models(rec: RunRecords) -> None:
    rows = {r["role"]: r for r in views.slot_rows(rec)}
    assert rows["validator"]["provider/model"] == "mock/mock-validator-v1"
    assert rows["validator"]["unavailable"] == 1  # the scripted Phase 3 outage
    assert [r["slot"] for r in views.slot_rows(rec)] == ["G1", "C1", "P1", "D1"]


def test_selection_rows_come_from_the_logged_evidence(run_dir: Path, rec: RunRecords) -> None:
    rows = views.selection_rows(rec)
    ev = next((n, e) for n, e in enumerate(jsonl(run_dir / "events.jsonl"), 1) if e["kind"] == "reference_selected")
    steps = ev[1]["payload"]["evidence"]["steps"]
    assert [r["criterion"] for r in rows] == [s["criterion"] for s in steps]
    assert all(r["record"] == f"events:{ev[0]}" for r in rows)
    assert rows[0]["basis"] == "measured" and "P1" in rows[0]["dropped"]


def test_timeline_cites_real_lines_and_filters(run_dir: Path, rec: RunRecords) -> None:
    lines = jsonl(run_dir / "events.jsonl")
    rows = views.timeline_rows(rec)
    assert len(rows) == len(lines)
    for r in rows:
        n = int(r["record"].split(":")[1])
        assert lines[n - 1]["kind"] == r["event"] and lines[n - 1]["ts"] == r["time"]
    only = views.timeline_rows(rec, roles={"validator"})
    assert only and all(r["from"].startswith("validator") for r in only)
    assert any("UNAVAILABLE" in r["summary"] for r in only)
    decided = views.timeline_rows(rec, kinds={"modification_decided"})
    assert [r["summary"].split(":")[0] for r in decided] == ["M-001 accepted", "M-002 rejected"]
    assert views.event_payload(rec, 1) == lines[0] and views.event_payload(rec, 10**6) is None
    phases, senders, kinds = views.timeline_filters(rec)
    assert phases[0] == "1-analysis" and "orchestrator" in senders and "early_stop" in kinds


def test_call_rows_equal_calls_jsonl(run_dir: Path, rec: RunRecords) -> None:
    raw = jsonl(run_dir / "calls.jsonl")
    rows = views.call_rows(rec)
    assert len(rows) == len(raw)
    assert sum(r["tokens in"] for r in rows) == sum(c["tokens_in"] for c in raw)
    assert sum(r["tokens out"] for r in rows) == sum(c["tokens_out"] for c in raw)
    assert all(r["tokens estimated"] for r in rows)  # mock tokens are estimates and say so
    assert any(r["status"] == "provider_error" for r in rows)  # the scripted 429s


def test_modifications_keep_rejected_records(rec: RunRecords) -> None:
    rows = {r["id"]: r for r in views.modification_rows(rec)}
    assert rows["M-001"]["decision"] == "accepted" and rows["M-001"]["integrated into"] == "V1"
    assert rows["M-002"]["decision"] == "rejected" and rows["M-002"]["tests (measured)"] == "fail"
    assert "optimizer" in rows["M-002"]["verdicts"] or "generator" in rows["M-002"]["verdicts"]


def test_version_diff_and_provenance_follow_the_records(rec: RunRecords) -> None:
    m1 = rec.record("M-001")
    assert m1 is not None
    diff = views.version_diff(rec, "V0", "V1")
    added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    assert added == m1.replacement_code.splitlines()
    assert views.version_diff(rec, "V1", "V1") == ""
    prov = views.provenance_rows(rec, "V1")
    assert len(prov) == rec.versions["V1"].line_count
    credited = [r for r in prov if r["modification"] == "M-001"]
    assert [r["code"] for r in credited] == m1.replacement_code.splitlines()
    assert all(r["agent"] == "reviewer" for r in credited)
    assert all(r["modification"] == "-" for r in views.provenance_rows(rec, "G1"))
    labels = [r["version"] for r in views.version_rows(rec)]
    assert labels[:5] == ["G1", "C1", "P1", "D1", "V0"] and "V0+M-002" in labels


def test_benchmark_rows_equal_benchmark_json(run_dir: Path, rec: RunRecords) -> None:
    stats = json.loads((run_dir / "benchmark.json").read_text())["stats"]
    rows = views.benchmark_rows(rec)
    measured = [s for s in stats if s["measured"]]
    assert len(rows) == len(measured)
    for r, s in zip(rows, measured, strict=True):
        assert (r["version"], r["test"]) == (s["version"], s["test_id"])
        assert r["median s (measured)"] == s["median_s"] and r["min s (measured)"] == s["min_s"]
        assert r["max s (measured)"] == s["max_s"]
    test = views.benchmark_tests(rec)[0]
    assert test == "T09"  # the performance test is offered first
    chart = views.runtime_chart_rows(rec, test)
    passing = {s["version"] for s in measured if s["test_id"] == test and s["status"] == TestStatus.PASS.value}
    assert {r["version"] for r in chart} == passing and "C1" not in passing  # C1 times out on T09


def test_labels_separate_measured_from_theoretical(rec: RunRecords) -> None:
    theo = views.theoretical_rows(rec)
    assert [r["version"] for r in theo] == ["G1", "C1", "P1", "D1"]
    assert all(any("theoretical" in k for k in r) for r in theo)
    assert all(any("measured" in k for k in r) for r in views.benchmark_rows(rec))
    suites = {r["version"]: r for r in views.initial_suite_rows(rec)}
    assert suites["P1"]["passed (measured)"] < suites["P1"]["total"]  # P1 is the scripted buggy one


def test_unfinished_run_views_do_not_invent_numbers(tmp_path: Path, run_dir: Path) -> None:
    partial = tmp_path / "partial"
    shutil.copytree(run_dir, partial)
    for name in ("summary.json", "benchmark.json", "final_solution.py"):
        (partial / name).unlink()
    rec = load_run(partial)
    ov = views.overview(rec)
    assert ov.status == "unfinished" and ov.final_version is None and ov.final_verified is None
    assert views.benchmark_rows(rec) == [] and views.runtime_chart_rows(rec, "T09") == []


def test_building_views_writes_nothing(run_dir: Path) -> None:
    before = digest(run_dir)
    rec = load_run(run_dir)
    for fn in (views.overview, views.slot_rows, views.selection_rows, views.timeline_rows, views.call_rows,
               views.modification_rows, views.version_rows, views.initial_suite_rows, views.benchmark_rows,
               views.theoretical_rows):
        fn(rec)
    assert digest(run_dir) == before
