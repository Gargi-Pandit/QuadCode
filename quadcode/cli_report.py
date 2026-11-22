"""``quadcode report``: rebuild a run's report from its logs, with no API calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from quadcode.config import AppConfig, ConfigError, load_config
from quadcode.report import ReportPaths, write_report


def resolve_run_dir(run: str, runs_dir: str | None, config: AppConfig | None) -> Path:
    """A run is named by its directory or by its id under the runs directory."""
    direct = Path(run)
    if direct.is_dir() and any((direct / name).is_file() for name in ("events.jsonl", "problem.md", "state.json")):
        return direct
    base = Path(runs_dir or (config.run.runs_dir if config else "runs"))
    return base / run


def print_report_paths(paths: ReportPaths) -> None:
    for path in (paths.markdown, paths.html):
        if path is not None:
            print(f"report: {path}")


def _cmd_report(args: argparse.Namespace) -> int:
    config: AppConfig | None
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        # Prices are the only thing taken from config.yaml; the report still works without them.
        print(f"note: {exc}; cost will show as not configured", file=sys.stderr)
        config = None
    run_dir = resolve_run_dir(args.run, args.runs_dir, config)
    if not run_dir.is_dir():
        print(f"no run directory {run_dir}", file=sys.stderr)
        return 2
    try:
        paths = write_report(run_dir, pricing=config, fmt=args.format)
    except OSError as exc:
        print(f"cannot write the report: {exc}", file=sys.stderr)
        return 2
    print_report_paths(paths)
    return 0


def add_report_command(sub: Any) -> None:
    p = sub.add_parser("report", help="rebuild report.md and report.html of a run from its logs (no API calls)")
    p.add_argument("run", help="run id (under the runs directory) or path to a run directory")
    p.add_argument("--runs-dir", default=None, help="parent directory of runs (default: config run.runs_dir)")
    p.add_argument("--format", choices=["md", "html", "both"], default="both")
    p.set_defaults(func=_cmd_report)
