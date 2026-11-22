"""CLI commands that execute code in the sandbox: ``sandbox-info`` and ``test-solution``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from quadcode.benchmark import benchmark_versions
from quadcode.config import ConfigError, SandboxConfig, load_config
from quadcode.sandbox import SandboxUnavailable, select_sandbox
from quadcode.schemas import SuiteRun, TestCase, TestStatus
from quadcode.tables import format_table
from quadcode.testing import load_test_cases, run_suite


def _sandbox_config(args: argparse.Namespace) -> SandboxConfig:
    cfg = load_config(args.config).sandbox
    updates: dict[str, Any] = {}
    if getattr(args, "backend", None):
        updates["backend"] = args.backend
    if getattr(args, "test_timeout", None):
        updates["test_timeout_s"] = args.test_timeout
    return SandboxConfig.model_validate({**cfg.model_dump(), **updates})


def _cmd_sandbox_info(args: argparse.Namespace) -> int:
    try:
        cfg = _sandbox_config(args)
        sandbox, reason = select_sandbox(cfg)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except SandboxUnavailable as exc:
        print(f"sandbox unavailable: {exc}", file=sys.stderr)
        return 2
    print(f"backend: {sandbox.name}")
    print(f"reason: {reason}")
    if sandbox.isolation_warning:
        print(f"WARNING: {sandbox.isolation_warning}")
    print(f"limits: memory={cfg.memory_mb}MB cpus={cfg.cpus} pids={cfg.pids_limit} "
          f"per-call={cfg.test_timeout_s}s wall={cfg.wall_timeout_s}s")
    print(f"measurement: warmup={cfg.warmup} test_repeats={cfg.test_repeats} "
          f"benchmark_repeats={cfg.benchmark_repeats} seed={cfg.seed}")
    return 0


def _fmt_ms(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds * 1000:.4f}"


def _fmt_kib(n: int | None) -> str:
    return "-" if n is None else f"{n / 1024:.1f}"


def _short(text: str | None, limit: int = 70) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_suite(suite: SuiteRun, source: str, tests: list[TestCase]) -> None:
    category = {t.id: t.category.value for t in tests}
    env = suite.sandbox_env
    where = f"python {env.get('python', '?')}" if suite.executed else "NOT EXECUTED"
    print(f"== {suite.version} ({source})  backend={suite.backend}  {where}")
    if suite.isolation_warning:
        print(f"WARNING: {suite.isolation_warning}")
    if not suite.executed:
        print(f"reason: {suite.not_executed_reason}")
    rows = [[r.test_id, category.get(r.test_id, "?"), r.status.value.upper(), _fmt_ms(r.runtime_s),
             _fmt_kib(r.peak_memory_bytes), _short(r.error)] for r in suite.results]
    print(format_table(["test", "category", "status", "median_ms", "peak_KiB", "note"], rows))
    counts = " | ".join(f"{s.value} {suite.count(s)}" for s in TestStatus)
    wall = "-" if suite.wall_time_s is None else f"{suite.wall_time_s:.2f}s"
    print(f"{counts} | sandbox wall {wall} | runtimes and memory are measured; '-' means not measured"
          if suite.executed else f"{counts} | nothing was executed, so nothing was measured")
    print()


def _cmd_test_solution(args: argparse.Namespace) -> int:
    try:
        cfg = _sandbox_config(args)
        tests = load_test_cases(args.tests)
        sandbox, reason = select_sandbox(cfg)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, ValidationError) as exc:
        print(f"cannot read tests: {exc}", file=sys.stderr)
        return 2
    except SandboxUnavailable as exc:
        print(f"sandbox unavailable: {exc}", file=sys.stderr)
        return 2
    sources: dict[str, Path] = {}
    for f in args.files:
        path = Path(f)
        if not path.is_file():
            print(f"no such file: {path}", file=sys.stderr)
            return 2
        sources[path.stem] = path
    codes = {label: p.read_text(encoding="utf-8") for label, p in sources.items()}
    print(f"sandbox: {sandbox.name} ({reason})\n")

    if args.benchmark:
        report = benchmark_versions(codes, entry_point=args.entry, language=args.language, tests=tests,
                                    sandbox=sandbox, cfg=cfg, repeats=args.repeats)
        suites = report.suites
        for suite in suites:
            print_suite(suite, str(sources[suite.version]), tests)
        rows = [[s.test_id, s.version, s.status.value.upper(), str(s.repeats), _fmt_ms(s.median_s),
                 _fmt_ms(s.min_s), _fmt_ms(s.max_s), _fmt_kib(s.peak_memory_bytes)]
                for s in sorted(report.stats, key=lambda s: (s.test_id, s.version))]
        print("benchmark (measured):")
        print(format_table(["test", "version", "status", "runs", "median_ms", "min_ms", "max_ms", "peak_KiB"], rows))
        print(f"\nmethod: {report.method}")
        payload: Any = report.model_dump(mode="json")
    else:
        suites = []
        for label, code in codes.items():
            suite = run_suite(code=code, entry_point=args.entry, language=args.language, tests=tests,
                              sandbox=sandbox, cfg=cfg, version=label, repeats=args.repeats)
            suites.append(suite)
            print_suite(suite, str(sources[label]), tests)
        if suites and suites[0].method:
            print(f"method: {suites[0].method}")
        payload = [s.model_dump(mode="json") for s in suites]

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    return 0 if all(s.all_passed for s in suites) else 1


def add_sandbox_commands(sub: Any) -> None:
    """Register the sandbox commands on the main parser's subparsers object."""
    backend_help = "override sandbox.backend from config.yaml"
    info = sub.add_parser("sandbox-info", help="show which sandbox backend would run generated code, and why")
    info.add_argument("--backend", choices=["auto", "docker", "subprocess"], help=backend_help)
    info.set_defaults(func=_cmd_sandbox_info)

    ts = sub.add_parser("test-solution", help="run a JSON test suite against solution files in the sandbox")
    ts.add_argument("files", nargs="+", help="solution source files; each file's stem is its version label")
    ts.add_argument("--tests", required=True, help="JSON file with a list of test cases")
    ts.add_argument("--entry", required=True, help="name of the function to call")
    ts.add_argument("--language", default="python")
    ts.add_argument("--repeats", type=int, default=None,
                    help="timed runs per test (default: test_repeats, or benchmark_repeats with --benchmark)")
    ts.add_argument("--test-timeout", type=float, default=None, help="per-call limit in seconds")
    ts.add_argument("--benchmark", action="store_true", help="also print median/min/max per test and version")
    ts.add_argument("--backend", choices=["auto", "docker", "subprocess"], help=backend_help)
    ts.add_argument("--json", default=None, help="write the full results as JSON to this path")
    ts.set_defaults(func=_cmd_test_solution)
