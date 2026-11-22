"""Thread-safe JSONL run log: calls.jsonl and events.jsonl under runs/<run_id>/."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from quadcode.schemas import CallRecord, Phase, Role, RunEvent

CALLS_FILE = "calls.jsonl"
EVENTS_FILE = "events.jsonl"


class JsonlWriter:
    """Append-only JSONL file. Each line is flushed immediately so live readers see it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: BaseModel | dict[str, Any]) -> None:
        data = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        line = json.dumps(data, ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all complete lines; a truncated trailing line (crash mid-write) is skipped."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


class RunLog:
    """The run directory's call and event logs."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.calls = JsonlWriter(run_dir / CALLS_FILE)
        self.events = JsonlWriter(run_dir / EVENTS_FILE)

    def log_call(self, record: CallRecord) -> None:
        self.calls.append(record)

    def event(
        self,
        phase: Phase,
        kind: str,
        role: Role | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        ev = RunEvent(phase=phase, kind=kind, role=role, payload=payload or {})
        self.events.append(ev)
        return ev

    def read_calls(self) -> list[CallRecord]:
        return [CallRecord.model_validate(r) for r in read_jsonl(self.run_dir / CALLS_FILE)]

    def read_events(self) -> list[RunEvent]:
        return [RunEvent.model_validate(r) for r in read_jsonl(self.run_dir / EVENTS_FILE)]
