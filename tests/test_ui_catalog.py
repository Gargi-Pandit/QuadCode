"""The dashboard's run catalog: what each directory under runs/ is, and in which order it is listed."""

from __future__ import annotations

import json
from pathlib import Path

from quadcode.ui.catalog import bench_runs, classify, config_is_mock, scan


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def _config(kind: str) -> dict[str, object]:
    return {"providers": {"p": {"kind": kind}},
            "agents": {r: {"provider": "p", "model": "m"} for r in ("generator", "reviewer", "optimizer", "validator")}}


def _solve(root: Path, name: str, status: str | None, finished: str, kind: str = "mock") -> Path:
    d = root / name
    _write(d / "events.jsonl", json.dumps({"ts": finished, "kind": "run_started"}) + "\n")
    _write(d / "config.json", _config(kind))
    _write(d / "problem.md", "# Two sum\n\nFind two indices.")
    if status is not None:
        _write(d / "summary.json", {"status": status, "problem_title": f"title {name}", "finished_at": finished})
    return d


def test_scan_classifies_every_kind_and_skips_the_rest(tmp_path: Path) -> None:
    _solve(tmp_path, "done", "completed", "2026-11-22T10:00:00.000+00:00")
    _solve(tmp_path, "real", "unverified", "2026-11-22T11:00:00.000+00:00", kind="openai_compat")
    _solve(tmp_path, "partial", None, "2026-11-22T12:00:00.000+00:00")
    _write(tmp_path / "bench1" / "bench.json", {"bench_id": "bench1", "mock": True, "problems": ["a", "b"],
                                                "run_conditions": ["full"], "status": "completed",
                                                "updated_at": "2026-11-22T09:00:00.000+00:00"})
    (tmp_path / "diffs" / "versions").mkdir(parents=True)
    _write(tmp_path / "smoke-1" / "calls.jsonl", "")
    (tmp_path / "empty").mkdir()
    _solve(tmp_path, ".ui-logs", "completed", "2026-11-22T13:00:00.000+00:00")

    entries = {e.id: e for e in scan(tmp_path)}
    assert set(entries) == {"done", "real", "partial", "bench1", "diffs", "smoke-1"}
    assert (entries["done"].kind, entries["done"].status, entries["done"].mock) == ("solve", "completed", True)
    assert entries["done"].title == "title done"
    assert entries["real"].mock is False
    assert (entries["partial"].status, entries["partial"].title) == ("unfinished", "Two sum")
    assert entries["partial"].updated == "2026-11-22T12:00:00.000+00:00"  # from its last event
    assert (entries["bench1"].kind, entries["bench1"].mock) == ("bench", True)
    assert "2 problem(s): a, b" in (entries["bench1"].title or "")
    assert (entries["diffs"].kind, entries["diffs"].status) == ("other", "versions only")
    assert (entries["smoke-1"].kind, entries["smoke-1"].status) == ("other", "calls only")
    ordered = [e.id for e in scan(tmp_path) if e.kind != "other"]
    assert ordered == ["partial", "real", "done", "bench1"]


def test_damaged_records_are_listed_with_the_error(tmp_path: Path) -> None:
    d = _solve(tmp_path, "broken", None, "2026-11-22T10:00:00.000+00:00")
    _write(d / "summary.json", "{not json")
    with (d / "events.jsonl").open("a") as fh:
        fh.write('{"ts": "2026-11-22T')  # a run killed mid-write
    _write(tmp_path / "badbench" / "bench.json", "[")
    entries = {e.id: e for e in scan(tmp_path)}
    assert entries["broken"].error is not None and entries["broken"].status == "unfinished"
    assert entries["badbench"].status == "unreadable" and entries["badbench"].error


def test_bench_sub_runs(tmp_path: Path) -> None:
    bench = tmp_path / "b"
    _write(bench / "bench.json", {"bench_id": "b", "mock": True, "problems": [], "run_conditions": []})
    _solve(bench / "runs", "two_sum--full", "completed", "2026-11-22T10:00:00.000+00:00")
    _solve(bench / "runs", "lis--full", None, "2026-11-22T10:00:00.000+00:00")
    (bench / "runs" / "not-a-run").mkdir()
    subs = bench_runs(bench)
    assert [e.id for e in subs] == ["lis--full", "two_sum--full"]
    assert all(e.bench == "b" for e in subs)
    assert classify(bench) is not None and classify(bench).kind == "bench"  # type: ignore[union-attr]


def test_config_is_mock_needs_every_slot_on_mock() -> None:
    cfg = _config("mock")
    assert config_is_mock(cfg)
    cfg["providers"] = {"p": {"kind": "mock"}, "q": {"kind": "openai_compat"}}
    cfg["agents"]["validator"]["provider"] = "q"  # type: ignore[index]
    assert not config_is_mock(cfg)
    assert not config_is_mock({})
