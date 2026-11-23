"""Find the solve runs and benches under a runs directory, reading only small status files.

Nothing is written. A directory whose files cannot be read is still listed, with the error, so a
damaged run is visible instead of silently missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from quadcode.evaluation.records import BENCH_FILE, RUNS_DIR
from quadcode.orchestrator import CONFIG_SNAPSHOT, PROBLEM_FILE, SUMMARY_FILE
from quadcode.runlog import CALLS_FILE, EVENTS_FILE
from quadcode.schemas import Role
from quadcode.versions import VERSIONS_DIR

EntryKind = Literal["solve", "bench", "other"]
UNFINISHED = "unfinished"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    path: Path
    kind: EntryKind
    status: str
    title: str | None
    mock: bool | None
    updated: str  # ISO-8601 UTC, from the run's own records, else the directory's modification time
    error: str | None = None
    bench: str | None = None  # the bench a sub-run belongs to

    @property
    def label(self) -> str:
        mode = "" if self.mock is None else (" [mock]" if self.mock else " [real]")
        return f"{self.id} ({self.kind}, {self.status}){mode}"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_ts(text: str) -> datetime:
    try:
        ts = datetime.fromisoformat(text)
    except ValueError:
        return _EPOCH
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="milliseconds")


def _last_event_ts(run_dir: Path) -> str | None:
    path = run_dir / EVENTS_FILE
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            ts = json.loads(line).get("ts")
        except json.JSONDecodeError:
            continue  # a run killed mid-write can leave a partial last line
        if isinstance(ts, str):
            return ts
    return None


def config_is_mock(config: dict[str, Any]) -> bool:
    """True if every agent slot in a config snapshot points at a mock provider."""
    providers = config.get("providers", {})
    agents = config.get("agents", {})
    kinds = [providers.get(agents.get(r.value, {}).get("provider"), {}).get("kind") for r in Role]
    return bool(kinds) and all(k == "mock" for k in kinds)


def _problem_title(run_dir: Path) -> str | None:
    path = run_dir / PROBLEM_FILE
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return line.strip().lstrip("#").strip()[:80] or None
    return None


def is_solve_run(path: Path) -> bool:
    return any((path / name).is_file() for name in (SUMMARY_FILE, EVENTS_FILE))


def solve_entry(run_dir: Path, bench: str | None = None) -> CatalogEntry:
    status, title, mock, updated, error = UNFINISHED, None, None, None, None
    try:
        if (run_dir / SUMMARY_FILE).is_file():
            summary = _json(run_dir / SUMMARY_FILE)
            status = str(summary.get("status", UNFINISHED))
            title = summary.get("problem_title")
            updated = summary.get("finished_at")
        if (run_dir / CONFIG_SNAPSHOT).is_file():
            mock = config_is_mock(_json(run_dir / CONFIG_SNAPSHOT))
        title = title or _problem_title(run_dir)
        updated = updated or _last_event_ts(run_dir)
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return CatalogEntry(id=run_dir.name, path=run_dir, kind="solve", status=status, title=title, mock=mock,
                        updated=updated or _mtime(run_dir), error=error, bench=bench)


def bench_entry(bench_dir: Path) -> CatalogEntry:
    try:
        state = _json(bench_dir / BENCH_FILE)
        problems = state.get("problems", [])
        conditions = state.get("run_conditions", [])
        title = f"{len(problems)} problem(s): {', '.join(problems)}; runs: {', '.join(conditions)}"
        return CatalogEntry(id=bench_dir.name, path=bench_dir, kind="bench", status=str(state.get("status", "?")),
                            title=title, mock=bool(state.get("mock")),
                            updated=str(state.get("updated_at") or _mtime(bench_dir)))
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        return CatalogEntry(id=bench_dir.name, path=bench_dir, kind="bench", status="unreadable", title=None,
                            mock=None, updated=_mtime(bench_dir), error=f"{type(exc).__name__}: {exc}")


def classify(path: Path) -> CatalogEntry | None:
    """The catalog entry of one directory, or None if it holds nothing QuadCode wrote."""
    if (path / BENCH_FILE).is_file():
        return bench_entry(path)
    if is_solve_run(path):
        return solve_entry(path)
    if (path / VERSIONS_DIR).is_dir() or (path / "modifications.json").is_file():
        return CatalogEntry(id=path.name, path=path, kind="other", status="versions only", title=None, mock=None,
                            updated=_mtime(path))
    if (path / CALLS_FILE).is_file():  # e.g. `quadcode smoke`: provider calls but no pipeline run
        return CatalogEntry(id=path.name, path=path, kind="other", status="calls only", title=None, mock=None,
                            updated=_mtime(path))
    return None


def _newest_first(entries: list[CatalogEntry]) -> list[CatalogEntry]:
    return sorted(entries, key=lambda e: (_parse_ts(e.updated), e.id), reverse=True)


def scan(runs_dir: Path) -> list[CatalogEntry]:
    """Every run, bench and versions-only directory directly under ``runs_dir``, newest first.
    Hidden directories (such as the launcher's ``.ui-logs``) are skipped."""
    if not runs_dir.is_dir():
        return []
    found = [classify(p) for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return _newest_first([e for e in found if e is not None])


def bench_runs(bench_dir: Path) -> list[CatalogEntry]:
    """The pipeline runs a bench started (``<bench>/runs/<problem>--<condition>``), in name order."""
    root = bench_dir / RUNS_DIR
    if not root.is_dir():
        return []
    return [solve_entry(p, bench=bench_dir.name) for p in sorted(root.iterdir()) if p.is_dir() and is_solve_run(p)]
