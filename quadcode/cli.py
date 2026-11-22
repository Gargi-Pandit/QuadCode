"""QuadCode command line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from quadcode.cli_sandbox import add_sandbox_commands
from quadcode.cli_versions import add_version_commands
from quadcode.config import ConfigError, load_config, load_secrets
from quadcode.tables import format_table


def _cmd_check_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    secrets = load_secrets(config, env_file=args.env_file)
    rows = config.describe_agents()
    headers = ["role", "label", "provider", "kind", "model", "rpm", "tpm"]
    print(format_table(headers, [[str(r[h]) for h in headers] for r in rows]))
    print()
    for name, prov in config.providers.items():
        if prov.api_key_env:
            state = "set" if secrets.has(prov.api_key_env) else "MISSING"
            print(f"provider {name}: key {prov.api_key_env} {state}")
    print(f"rounds={config.run.rounds} budget_calls={config.run.budget_calls} "
          f"cache={'on' if config.run.cache_enabled else 'off'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quadcode", description="Multi-agent code generation and optimization")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--env-file", default=".env", help="path to .env with API keys")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-config", help="validate config.yaml and show the agent slots")
    check.set_defaults(func=_cmd_check_config)
    add_sandbox_commands(sub)
    add_version_commands(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
