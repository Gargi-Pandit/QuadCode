"""Read everything a report needs from a run directory, without changing it.

The report is built only from these records. ``state.json`` is not read: it is a checkpoint,
and every fact it holds that a report needs is also in ``events.jsonl``. Missing files are
allowed (a run that stopped early), and problems reading a file are collected in
``load_errors`` instead of aborting, so a damaged run still gets a report that says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from quadcode.config import AppConfig
from quadcode.modifications import MODIFICATIONS_FILE
from quadcode.orchestrator import BENCHMARK_FILE, CONFIG_SNAPSHOT, FINAL_STEM, PROBLEM_FILE, SUMMARY_FILE
from quadcode.runctx import SUITE_FILE, TESTS_FILE
from quadcode.runlog import CALLS_FILE, EVENTS_FILE
from quadcode.schemas import (
    Ablation,
    BenchmarkReport,
    CallRecord,
    InitialSolution,
    ModificationRecord,
    Phase,
    ProblemSpec,
    Review,
    Role,
    RunEvent,
    RunSummary,
    SelectionEvidence,
    SuiteRun,
    TestCase,
    VersionMeta,
)
from quadcode.versions import VERSIONS_DIR, VersionError, VersionStore, sort_key

CONFIG_FILE = CONFIG_SNAPSHOT

_SUITE_RUNS = TypeAdapter(list[SuiteRun])
_TEST_CASES = TypeAdapter(list[TestCase])
_RECORDS = TypeAdapter(list[ModificationRecord])


@dataclass(frozen=True)
class LoggedEvent:
    """One line of events.jsonl and its 1-based line number, for citations."""

    line: int
    event: RunEvent

    @property
    def kind(self) -> str:
        return self.event.kind

    @property
    def role(self) -> Role | None:
        return self.event.role

    @property
    def phase(self) -> Phase:
        return self.event.phase

    @property
    def payload(self) -> dict[str, Any]:
        return self.event.payload

    @property
    def ts(self) -> str:
        return self.event.ts

    @property
    def ref(self) -> str:
        return f"events:{self.line}"


@dataclass(frozen=True)
class AgentMessage:
    """An agent's logged output (or its unavailability) for one call."""

    logged: LoggedEvent
    role: Role
    call_type: str
    tag: str
    provider: str
    model: str
    ok: bool
    data: dict[str, Any] | None
    error: str | None

    @property
    def who(self) -> str:
        return f"{self.role.value} ({self.provider}/{self.model})"


def message_of(e: LoggedEvent) -> AgentMessage | None:
    """The agent message carried by an agent_output / agent_unavailable event."""
    if e.kind not in ("agent_output", "agent_unavailable") or e.role is None:
        return None
    p = e.payload
    ok = e.kind == "agent_output"
    return AgentMessage(e, e.role, str(p.get("call_type", "")), str(p.get("tag", "")), str(p.get("provider", "?")),
                        str(p.get("model", "?")), ok, p.get("data") if ok else None,
                        None if ok else str(p.get("error") or ""))


@dataclass
class RunRecords:
    run_dir: Path
    run_id: str
    problem: str | None = None
    config: AppConfig | None = None
    events: list[LoggedEvent] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    modifications: list[ModificationRecord] = field(default_factory=list)
    versions: dict[str, VersionMeta] = field(default_factory=dict)
    codes: dict[str, str] = field(default_factory=dict)
    suite: list[TestCase] | None = None
    suite_runs: list[SuiteRun] = field(default_factory=list)
    benchmark: BenchmarkReport | None = None
    summary: RunSummary | None = None
    final_file: str | None = None
    final_code: str | None = None
    load_errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ events

    def of_kind(self, *kinds: str) -> list[LoggedEvent]:
        return [e for e in self.events if e.kind in kinds]

    def last(self, kind: str) -> LoggedEvent | None:
        found = self.of_kind(kind)
        return found[-1] if found else None

    def phase_started(self, phase: Phase) -> bool:
        return any(e.kind == "phase_started" and e.phase is phase for e in self.events)

    def phase_completed(self, phase: Phase) -> bool:
        return any(e.kind == "phase_completed" and e.phase is phase for e in self.events)

    def messages(self, call_type: str | None = None, tag: str | None = None) -> list[AgentMessage]:
        out: list[AgentMessage] = []
        for e in self.of_kind("agent_output", "agent_unavailable"):
            m = message_of(e)
            if m is None or (call_type is not None and m.call_type != call_type) or (tag is not None and m.tag != tag):
                continue
            out.append(m)
        return out

    def latest_output(self, call_type: str, role: Role, tag: str | None = None) -> AgentMessage | None:
        """The last successful output of ``role`` for this call type (a resumed run may retry)."""
        found = [m for m in self.messages(call_type, tag) if m.role is role and m.ok]
        return found[-1] if found else None

    # ------------------------------------------------------------ typed views

    def spec(self) -> tuple[ProblemSpec, AgentMessage] | None:
        msgs = [m for m in self.messages("analysis") if m.ok and m.data is not None]
        if not msgs:
            return None
        return ProblemSpec.model_validate(msgs[-1].data), msgs[-1]

    def initial(self, role: Role) -> tuple[InitialSolution, AgentMessage] | None:
        msg = self.latest_output("initial", role)
        if msg is None or msg.data is None:
            return None
        return InitialSolution.model_validate(msg.data), msg

    def review(self, role: Role) -> tuple[Review, AgentMessage] | None:
        msg = self.latest_output("review", role)
        if msg is None or msg.data is None:
            return None
        return Review.model_validate(msg.data), msg

    def reference(self) -> tuple[SelectionEvidence, LoggedEvent] | None:
        ev = self.last("reference_selected")
        if ev is None:
            return None
        return SelectionEvidence.model_validate(ev.payload["evidence"]), ev

    def agent_identity(self, role: Role) -> str:
        """``provider/model`` behind a slot: the models that answered it, as logged.

        When a fallback answered, the slot's first choice from the config snapshot is named too, so a
        reader never takes the fallback's work for the first choice's.
        """
        configured = None
        if self.config is not None:
            agent = self.config.agent(role)
            configured = f"{agent.provider}/{agent.model}"
        answered = list(dict.fromkeys(f"{m.provider}/{m.model}" for m in self.messages() if m.role is role and m.ok))
        if not answered:
            if configured is not None:
                return configured
            tried = [f"{m.provider}/{m.model}" for m in self.messages() if m.role is role]
            return tried[-1] if tried else "not recorded"
        if configured is None or answered == [configured]:
            return ", ".join(answered)
        return f"{', '.join(answered)} (fallback; first choice {configured})"

    def who(self, role: Role | None) -> str:
        if role is None:
            return "orchestrator"
        return f"{role.value} ({self.agent_identity(role)})"

    def initial_suite(self, label: str) -> SuiteRun | None:
        """The Phase 2.5 suite run of an initial version (its first run under that label)."""
        return next((s for s in self.suite_runs if s.version == label), None)

    def benchmark_suite(self, label: str) -> SuiteRun | None:
        if self.benchmark is None:
            return None
        return next((s for s in self.benchmark.suites if s.version == label), None)

    def ablation(self) -> Ablation:
        """The ablation this run was started with (summary.json, else the first run_started event)."""
        if self.summary is not None:
            return self.summary.ablation
        started = next(iter(self.of_kind("run_started")), None)
        if started is None:
            return Ablation.NONE
        return Ablation(started.payload.get("ablation", Ablation.NONE.value))

    def excluded(self, role: Role) -> bool:
        return role in self.ablation().excluded_roles

    def is_mock(self) -> bool:
        if self.config is None:
            return False
        return all(self.config.provider_for(r).kind == "mock" for r in Role)

    def record(self, modification_id: str) -> ModificationRecord | None:
        return next((r for r in self.modifications if r.id == modification_id), None)


# ---------------------------------------------------------------- loading


def _read_json(path: Path, errors: list[str]) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: unreadable ({exc})")
        return None


def _read_jsonl(path: Path, errors: list[str]) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        return []
    rows: list[tuple[int, dict[str, Any]]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append((n, json.loads(line)))
        except json.JSONDecodeError:
            errors.append(f"{path.name}:{n}: incomplete line skipped")
    return rows


def _validated(model: Any, raw: Any, name: str, errors: list[str]) -> Any:
    if raw is None:
        return None
    try:
        if isinstance(model, TypeAdapter):
            return model.validate_python(raw)
        return model.model_validate(raw)
    except ValidationError as exc:
        errors.append(f"{name}: does not match its schema ({exc.error_count()} errors)")
        return None


def _load_versions(run_dir: Path, records: RunRecords) -> None:
    if not (run_dir / VERSIONS_DIR).is_dir():
        return
    store = VersionStore(run_dir)
    labels = sorted((p.stem for p in (run_dir / VERSIONS_DIR).glob("*.json")), key=_safe_sort_key)
    for label in labels:
        try:
            stored = store.load(label)
        except (VersionError, ValidationError, OSError) as exc:
            records.load_errors.append(f"versions/{label}: {exc}")
            continue
        records.versions[label] = stored.meta
        records.codes[label] = stored.code


def _safe_sort_key(label: str) -> tuple[int, int, int]:
    try:
        return sort_key(label)
    except VersionError:
        return (10**9, 0, 0)


def load_run(run_dir: Path) -> RunRecords:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run directory {run_dir}")
    errors: list[str] = []
    rec = RunRecords(run_dir=run_dir, run_id=run_dir.name, load_errors=errors)

    problem = run_dir / PROBLEM_FILE
    rec.problem = problem.read_text(encoding="utf-8") if problem.is_file() else None
    rec.config = _validated(AppConfig, _read_json(run_dir / CONFIG_FILE, errors), CONFIG_FILE, errors)

    for n, raw in _read_jsonl(run_dir / EVENTS_FILE, errors):
        ev = _validated(RunEvent, raw, f"{EVENTS_FILE}:{n}", errors)
        if ev is not None:
            rec.events.append(LoggedEvent(n, ev))
    for n, raw in _read_jsonl(run_dir / CALLS_FILE, errors):
        call = _validated(CallRecord, raw, f"{CALLS_FILE}:{n}", errors)
        if call is not None:
            rec.calls.append(call)

    mods = _validated(_RECORDS, _read_json(run_dir / MODIFICATIONS_FILE, errors), MODIFICATIONS_FILE, errors)
    rec.modifications = sorted(mods or [], key=lambda r: int(r.id[2:]))
    rec.suite = _validated(_TEST_CASES, _read_json(run_dir / SUITE_FILE, errors), SUITE_FILE, errors)
    rec.suite_runs = _validated(_SUITE_RUNS, _read_json(run_dir / TESTS_FILE, errors), TESTS_FILE, errors) or []
    rec.benchmark = _validated(BenchmarkReport, _read_json(run_dir / BENCHMARK_FILE, errors), BENCHMARK_FILE,
                               errors)
    rec.summary = _validated(RunSummary, _read_json(run_dir / SUMMARY_FILE, errors), SUMMARY_FILE, errors)
    if rec.summary is not None:
        rec.run_id = rec.summary.run_id

    _load_versions(run_dir, rec)

    finals = sorted(run_dir.glob(f"{FINAL_STEM}.*"))
    if finals:
        rec.final_file = finals[0].name
        rec.final_code = finals[0].read_text(encoding="utf-8")
    return rec
