"""``quadcode solve``: run the full pipeline on a problem, or resume a run."""

from __future__ import annotations

import argparse
import re
import secrets as pysecrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quadcode.cli_report import print_report_paths
from quadcode.config import AppConfig, ConfigError, load_config, load_secrets, parse_config
from quadcode.gateway import LLMGateway
from quadcode.modifications import ModificationLedger
from quadcode.orchestrator import Orchestrator
from quadcode.providers import UnsupportedProviderError
from quadcode.report import write_report
from quadcode.runstate import ResumeError, load_state
from quadcode.sandbox import SandboxUnavailable, select_sandbox
from quadcode.schemas import Role, RunStatus, RunSummary
from quadcode.tables import format_table

EXIT_CODES = {RunStatus.COMPLETED: 0, RunStatus.UNVERIFIED: 1, RunStatus.FAILED: 2,
              RunStatus.NEEDS_CLARIFICATION: 3, RunStatus.ABORTED_BUDGET: 4, RunStatus.RUNNING: 2}
DEFAULT_SCENARIO = "two_sum"
_MOCK_PROVIDER = {"kind": "mock", "rpm": 6000, "tpm": 10_000_000, "timeout_s": 5, "max_retries": 3,
                  "backoff_base_s": 0.01, "backoff_cap_s": 0.05}


def mock_config(config: AppConfig, scenario: str | None) -> AppConfig:
    """Point every agent slot at a mock provider. Real provider blocks are not used at all."""
    data = config.model_dump(mode="json")
    all_mock = all(config.provider_for(r).kind == "mock" for r in Role)
    if all_mock:
        if scenario:
            for name in {config.agent(r).provider for r in Role}:
                data["providers"][name]["scenario"] = scenario
        return parse_config(data)
    data["providers"] = {"mock": {**_MOCK_PROVIDER, "scenario": scenario or DEFAULT_SCENARIO}}
    data["agents"] = {r.value: {"provider": "mock", "model": f"mock-{r.value}-v1",
                                "temperature": config.agent(r).temperature} for r in Role}
    data["allow_duplicate_models"] = False
    return parse_config(data)


def _with_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    data = config.model_dump(mode="json")
    if args.no_cache:
        data["run"]["cache_enabled"] = False
    if args.backend:
        data["sandbox"]["backend"] = args.backend
    return parse_config(data)


def _new_run_id(problem_path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", problem_path.stem.lower()).strip("-")[:30] or "problem"
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{slug}-{pysecrets.token_hex(2)}"


def _short(text: str | None, limit: int = 60) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_summary(summary: RunSummary, run_dir: Path) -> None:
    print(f"\nrun {summary.run_id}: {summary.status.value.upper()}")
    if summary.isolation_warning:
        print(f"WARNING: {summary.isolation_warning}")
    rows = [[a.role.value, a.label, f"{a.provider}/{a.model}", a.phase, a.status.value, _short(a.error, 50)]
            for a in summary.availability]
    if rows:
        print("\nagent calls (from the run log):")
        print(format_table(["role", "slot", "provider/model", "phase", "status", "error"], rows))
    state = load_state(run_dir)
    if state.reference is not None:
        print(f"\nV0 = copy of {state.reference.winner} (selected on evidence):")
        for step in state.reference.steps:
            values = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in step.values.items())
            print(f"  {step.criterion}: kept {', '.join(step.kept)}"
                  + (f"; dropped {', '.join(step.dropped)}" if step.dropped else "")
                  + (f" [{values}]" if values else "") + ("" if step.measured else " (not measured)"))
        print(f"  not measured: {', '.join(state.reference.not_measured)}")
    records = ModificationLedger(run_dir).all()
    if records:
        print("\nmodifications:")
        print(format_table(
            ["id", "by", "base", "tests", "verdicts", "decision", "reason"],
            [[r.id, r.proposed_by.value, r.base_version, r.test_status.value,
              ", ".join(f"{v.role.value}:{v.verdict.value}" for v in r.reviewers) or "-",
              r.decision.value, _short(r.decision_reason, 70)] for r in records]))
    print(f"\nrounds run: {summary.rounds_run} of {summary.rounds_requested}; stop: {summary.stop_reason}")
    if summary.final_version:
        verdict = "passes every test" if summary.final_verified else "NOT verified"
        print(f"final: {summary.final_version} -> {run_dir / (summary.final_file or '')} ({verdict})")
    for issue in summary.unresolved:
        print(f"unresolved: {issue}")
    print(f"\nrun directory: {run_dir}")


def _cmd_solve(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        if args.mock:
            config = mock_config(config, args.scenario)
        config = _with_overrides(config, args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    rounds = args.rounds if args.rounds is not None else config.run.rounds
    if not 1 <= rounds <= 5:
        print("--rounds must be between 1 and 5", file=sys.stderr)
        return 2
    runs_dir = Path(args.runs_dir or config.run.runs_dir)
    if args.resume:
        run_id = args.resume
        run_dir = runs_dir / run_id
        if not run_dir.is_dir():
            print(f"no run directory {run_dir}", file=sys.stderr)
            return 2
    else:
        if not args.problem:
            print("a problem file is required unless --resume is given", file=sys.stderr)
            return 2
        problem_path = Path(args.problem)
        if not problem_path.is_file():
            print(f"no such problem file: {problem_path}", file=sys.stderr)
            return 2
        run_id = args.run_id or _new_run_id(problem_path)
        run_dir = runs_dir / run_id

    try:
        sandbox, reason = select_sandbox(config.sandbox)
        secrets = load_secrets(config, env_file=args.env_file)
        gateway = LLMGateway.from_config(config, secrets, run_dir, budget_calls=args.budget_calls)
    except SandboxUnavailable as exc:
        print(f"sandbox unavailable: {exc}", file=sys.stderr)
        return 2
    except UnsupportedProviderError as exc:
        print(f"{exc}. Real providers are not wired yet; run with --mock.", file=sys.stderr)
        return 2

    print(f"sandbox: {sandbox.name} ({reason})")
    if args.mock:
        scen = {config.provider_for(r).scenario for r in Role}
        print(f"mock mode: agent responses are scripted by scenario {', '.join(sorted(s or '' for s in scen))} "
              "and do not depend on the problem text")
    try:
        if args.resume:
            if args.problem:
                given = Path(args.problem).read_text(encoding="utf-8")
                if given != (run_dir / "problem.md").read_text(encoding="utf-8"):
                    print("the problem file differs from the one this run started with", file=sys.stderr)
                    return 2
            orch = Orchestrator.resume(config, gateway, sandbox, run_dir)
            print(f"resuming {run_id} (completed steps: {', '.join(orch.state.done) or 'none'})")
        else:
            problem = Path(args.problem).read_text(encoding="utf-8")
            orch = Orchestrator.create(config, gateway, sandbox, run_dir, run_id, problem, rounds)
            print(f"run {run_id}")
    except (ResumeError, FileExistsError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 2
    summary = orch.run()
    print_summary(summary, run_dir)
    try:
        print_report_paths(write_report(run_dir, pricing=config))
    except Exception as exc:  # the run's records are complete; a report failure must not change the outcome
        print(f"report could not be written: {type(exc).__name__}: {exc}; "
              f"retry with: quadcode report {summary.run_id}", file=sys.stderr)
    return EXIT_CODES[summary.status]


def add_solve_command(sub: Any) -> None:
    p = sub.add_parser("solve", help="run the four-agent pipeline on a problem statement")
    p.add_argument("problem", nargs="?", help="problem statement file (text or markdown)")
    p.add_argument("--mock", action="store_true", help="use scripted mock agents (no API calls)")
    p.add_argument("--scenario", default=None, help=f"mock scenario (default: from config, else {DEFAULT_SCENARIO})")
    p.add_argument("--rounds", type=int, default=None, help="optimization rounds, 1-5 (default: config run.rounds)")
    p.add_argument("--budget-calls", type=int, default=None,
                   help="stop gracefully after this many provider attempts (cache hits are free)")
    p.add_argument("--resume", metavar="RUN_ID", default=None, help="continue an interrupted run")
    p.add_argument("--run-id", default=None, help="name of the new run directory")
    p.add_argument("--runs-dir", default=None, help="parent directory of runs (default: config run.runs_dir)")
    p.add_argument("--backend", choices=["auto", "docker", "subprocess"], help="override sandbox.backend")
    p.add_argument("--no-cache", action="store_true", help="do not read or write the response cache")
    p.set_defaults(func=_cmd_solve)
