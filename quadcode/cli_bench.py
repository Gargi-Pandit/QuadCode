"""``quadcode bench`` (run and score the benchmark) and ``quadcode bench-report`` (rebuild its outputs)."""

from __future__ import annotations

import argparse
import secrets as pysecrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quadcode.cli_solve import mock_config, report_missing_keys
from quadcode.config import AppConfig, ConfigError, load_config, load_secrets, parse_config
from quadcode.evaluation.conditions import Condition, run_conditions
from quadcode.evaluation.harness import BenchError, BenchHarness, GatewayFactory, check_resumable, new_state
from quadcode.evaluation.outputs import write_outputs
from quadcode.evaluation.problems import ProblemError, load_problems
from quadcode.evaluation.records import BenchStatus, load_bench, save_bench
from quadcode.evaluation.summary import summarize
from quadcode.gateway import CallBudget, LLMGateway
from quadcode.providers import MissingAPIKeyError, UnsupportedProviderError
from quadcode.sandbox import SandboxUnavailable, select_sandbox
from quadcode.tables import format_table

EXIT = {BenchStatus.COMPLETED: 0, BenchStatus.ABORTED_BUDGET: 4, BenchStatus.RUNNING: 2}


def _split(text: str | None) -> list[str] | None:
    return [x.strip() for x in text.split(",") if x.strip()] if text else None


def _overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    data = config.model_dump(mode="json")
    if args.no_cache:
        data["run"]["cache_enabled"] = False
    if args.backend:
        data["sandbox"]["backend"] = args.backend
    return parse_config(data)


def print_summary(bench_dir: Path) -> None:
    state = load_bench(bench_dir)
    s = summarize(bench_dir, state)
    rows = []
    for c in s.conditions:
        x = s.summaries[c]
        rate = "-" if x.pass_rate is None else f"{x.pass_rate.mean:.1%} [{x.pass_rate.low:.1%}, {x.pass_rate.high:.1%}]"
        solved = "-" if x.solved is None else f"{x.solved.mean:.1%}"
        w = s.comparisons.get(c)
        wtl = "-" if w is None else f"{len(w.wins)}/{len(w.ties)}/{len(w.losses)}"
        rows.append([c.value, rate, solved, str(x.counted), str(x.missing), wtl])
    print(f"\nbench {state.bench_id}: {state.status.value.upper()} - {state.stop_reason}")
    if state.mock:
        print("mock mode: scripted agents; hidden tests and timings are real sandbox executions")
    print(format_table(["condition", "hidden pass rate [CI]", "solved", "problems", "missing",
                        "full W/T/L vs"], rows))


def _cmd_bench(args: argparse.Namespace) -> int:
    try:
        base = _overrides(load_config(args.config), args)
        if args.mock:
            base = mock_config(base, None)  # each problem gets its own scenario in the harness
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    secrets = load_secrets(base, env_file=args.env_file)
    if not args.mock and not report_missing_keys(base, secrets):
        return 2
    ev = base.evaluation
    runs_dir = Path(args.runs_dir or base.run.runs_dir)
    try:
        if args.resume:
            bench_dir = runs_dir / args.resume
            state = load_bench(bench_dir)
            if args.mock != state.mock:
                print(f"this bench was started {'with' if state.mock else 'without'} --mock", file=sys.stderr)
                return 2
            problems = load_problems(Path(args.problems_dir or ev.problems_dir), names=state.problems,
                                     mock=state.mock)
            check_resumable(state, problems, base)
        else:
            rounds = args.rounds if args.rounds is not None else base.run.rounds
            if not 1 <= rounds <= 5:
                print("--rounds must be between 1 and 5", file=sys.stderr)
                return 2
            conds = run_conditions(_split(args.conditions) or ev.conditions)
            if Condition.FULL not in conds:
                print("--conditions must include full (the baselines come from it)", file=sys.stderr)
                return 2
            problems = load_problems(Path(args.problems_dir or ev.problems_dir), names=_split(args.problems),
                                     limit=args.limit if args.limit is not None else
                                     (None if args.problems else ev.limit), mock=args.mock)
            bench_id = args.bench_id or f"bench-{datetime.now(UTC):%Y%m%d-%H%M%S}-{pysecrets.token_hex(2)}"
            bench_dir = runs_dir / bench_id
            if bench_dir.exists():
                print(f"{bench_dir} already exists; use --resume {bench_id}", file=sys.stderr)
                return 2
            bench_dir.mkdir(parents=True)
            state = new_state(bench_id, problems, conds, rounds, args.mock, base)
            save_bench(bench_dir, state)
    except (ProblemError, BenchError, FileNotFoundError, ValueError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 2

    try:
        sandbox, reason = select_sandbox(base.sandbox)
    except SandboxUnavailable as exc:
        print(f"sandbox unavailable: {exc}", file=sys.stderr)
        return 2
    budget = CallBudget(args.budget_calls)

    def make_gateway(cfg: AppConfig, run_dir: Path, shared: CallBudget) -> LLMGateway:
        return LLMGateway.from_config(cfg, secrets, run_dir, budget=shared)

    factory: GatewayFactory = make_gateway
    print(f"bench {state.bench_id} in {bench_dir}")
    print(f"sandbox: {sandbox.name} ({reason})")
    print(f"problems: {', '.join(state.problems)}; runs per problem: "
          f"{', '.join(c.value for c in state.run_conditions)}; budget: "
          f"{args.budget_calls if args.budget_calls is not None else 'unlimited'} provider attempts")
    harness = BenchHarness(base_config=base, bench_dir=bench_dir, state=state, problems=problems, sandbox=sandbox,
                           make_gateway=factory, budget=budget)
    try:
        harness.run()
    except (UnsupportedProviderError, MissingAPIKeyError) as exc:
        print(f"cannot build provider clients: {exc}", file=sys.stderr)
        return 2
    outputs = write_outputs(bench_dir)
    print_summary(bench_dir)
    print(f"\nprovider attempts used: {budget.used}")
    for path in outputs.paths():
        print(f"wrote {path}")
    return EXIT[harness.state.status]


def _cmd_bench_report(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    if runs_dir is None:
        try:
            runs_dir = Path(load_config(args.config).run.runs_dir)
        except ConfigError:
            runs_dir = Path("runs")
    bench_dir = Path(args.bench) if Path(args.bench).is_dir() else runs_dir / args.bench
    try:
        outputs = write_outputs(bench_dir)
    except FileNotFoundError as exc:
        print(f"cannot read bench: {exc}", file=sys.stderr)
        return 2
    print_summary(bench_dir)
    for path in outputs.paths():
        print(f"wrote {path}")
    return 0


def add_bench_commands(sub: Any) -> None:
    p = sub.add_parser("bench", help="run the benchmark problems under each condition and score on hidden tests")
    p.add_argument("--mock", action="store_true", help="scripted mock agents (only problems with a mock scenario)")
    p.add_argument("--problems", default=None, help="comma-separated problem ids (default: the first --limit)")
    p.add_argument("--limit", type=int, default=None, help="number of problems (default: config evaluation.limit)")
    p.add_argument("--conditions", default=None,
                   help="comma-separated pipeline runs: full,no-review,no-test-feedback,no-optimizer "
                        "(default: config evaluation.conditions; baselines always come from full)")
    p.add_argument("--rounds", type=int, default=None, help="optimization rounds per run, 1-5")
    p.add_argument("--budget-calls", type=int, default=None,
                   help="provider attempts for this whole invocation; stops gracefully and still writes results")
    p.add_argument("--bench-id", default=None, help="name of the new bench directory")
    p.add_argument("--resume", metavar="BENCH_ID", default=None, help="continue a stopped bench")
    p.add_argument("--problems-dir", default=None, help="benchmark directory (default: config evaluation.problems_dir)")
    p.add_argument("--runs-dir", default=None, help="parent directory (default: config run.runs_dir)")
    p.add_argument("--backend", choices=["auto", "docker", "subprocess"], help="override sandbox.backend")
    p.add_argument("--no-cache", action="store_true",
                   help="do not use the response cache (ablations then repeat the analysis and initial calls)")
    p.set_defaults(func=_cmd_bench)
    r = sub.add_parser("bench-report", help="rebuild a bench's CSV, Markdown and charts from its records")
    r.add_argument("bench", help="bench id (under runs_dir) or path to its directory")
    r.add_argument("--runs-dir", default=None)
    r.set_defaults(func=_cmd_bench_report)
