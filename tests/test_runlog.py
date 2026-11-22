from __future__ import annotations

import threading
from pathlib import Path

from quadcode.runlog import RunLog, read_jsonl
from quadcode.schemas import Phase, Role


def test_events_roundtrip(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "r1")
    log.event(Phase.INITIAL, "agent_status", Role.GENERATOR, {"status": "running"})
    log.event(Phase.INITIAL, "agent_status", Role.GENERATOR, {"status": "done"})
    events = log.read_events()
    assert [e.payload["status"] for e in events] == ["running", "done"]
    assert events[0].role is Role.GENERATOR


def test_concurrent_appends_produce_whole_lines(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "r2")

    def worker(n: int) -> None:
        for i in range(50):
            log.events.append({"worker": n, "i": i, "pad": "x" * 500})

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = read_jsonl(tmp_path / "r2" / "events.jsonl")
    assert len(rows) == 400


def test_truncated_trailing_line_is_skipped(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": ')
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []
