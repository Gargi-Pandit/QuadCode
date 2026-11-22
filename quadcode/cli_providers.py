"""``quadcode models`` and ``quadcode smoke``: check real providers with as few requests as possible.

``models`` lists a provider's catalog (no completion call). ``smoke`` sends ONE tiny JSON call to
each model of the selected agent slots (first choice and every fallback) through the normal gateway,
with retries and the cache off, so it makes at most one request per model. Its calls are logged to
``runs/smoke-<ts>/calls.jsonl`` like any other call.
"""

from __future__ import annotations

import argparse
import secrets as pysecrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from quadcode.cache import DiskCache
from quadcode.cli_solve import report_missing_keys
from quadcode.config import AppConfig, ConfigError, ModelChoice, load_config, load_secrets, parse_config
from quadcode.gateway import BudgetExhausted, CallBudget, LLMGateway
from quadcode.providers import MissingAPIKeyError, build_client
from quadcode.providers.base import LLMClient, LLMRequest, LLMResponse, PermanentError, ProviderError
from quadcode.ratelimit import build_limiters
from quadcode.runlog import RunLog
from quadcode.schemas import ROLE_VERSION_LABEL, AgentStatus, CallRecord, CallType, Role
from quadcode.tables import format_table

SMOKE_SYSTEM = "You are a connectivity check for an automated system. Answer with JSON only."
SMOKE_PROMPT = 'Reply with exactly this JSON object and nothing else: {"ok": true}'


class SmokeReply(BaseModel):
    ok: bool


class _NotSelected(LLMClient):
    """Stands in for slots that are not part of this smoke check; it is never called."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise PermanentError("this agent slot is not part of the smoke check")


def _parse_roles(text: str | None) -> list[Role]:
    if not text:
        return list(Role)
    roles: list[Role] = []
    for part in text.split(","):
        name = part.strip().lower()
        try:
            role = Role(name)
        except ValueError:
            raise ValueError(f"unknown role '{name}' (choose from {', '.join(r.value for r in Role)})") from None
        if role not in roles:
            roles.append(role)
    return roles


def smoke_config(config: AppConfig, roles: list[Role]) -> AppConfig:
    """Retries off for the providers under test and the cache off, so each model costs one request."""
    data = config.model_dump(mode="json")
    for name in {c.provider for r in roles for c in config.agent(r).choices()}:
        data["providers"][name]["max_retries"] = 0
    data["run"]["cache_enabled"] = False
    return parse_config(data)


def _smoke_run_dir(config: AppConfig, runs_dir: str | None) -> Path:
    base = Path(runs_dir or config.run.runs_dir)
    return base / f"smoke-{datetime.now(UTC):%Y%m%d-%H%M%S}-{pysecrets.token_hex(2)}"


def _position(index: int) -> str:
    return "first choice" if index == 0 else f"fallback {index}"


def _last_record(records: list[CallRecord], role: Role, index: int) -> CallRecord | None:
    mine = [r for r in records if r.role is role and r.choice == index]
    return mine[-1] if mine else None


def _cmd_smoke(args: argparse.Namespace) -> int:
    try:
        roles = _parse_roles(args.roles)
        config = load_config(args.config)
    except (ValueError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    targets: list[tuple[Role, int, ModelChoice]] = []
    for role in roles:
        for index, choice in enumerate(config.agent(role).choices()):
            if config.providers[choice.provider].kind == "mock":
                where = role.value + ("" if index == 0 else f" {_position(index)}")
                print(f"skipped {where}: provider '{choice.provider}' is a mock; nothing to check")
            else:
                targets.append((role, index, choice))
    if not targets:
        print("no agent slot uses a real provider; nothing to check", file=sys.stderr)
        return 2
    roles = list(dict.fromkeys(role for role, _, _ in targets))
    config = smoke_config(config, roles)
    secrets = load_secrets(config, env_file=args.env_file)
    if not report_missing_keys(config, secrets, roles):
        return 2

    run_dir = _smoke_run_dir(config, args.runs_dir)
    clients: dict[Role, list[LLMClient]] = {}
    try:
        for role in Role:
            choices = config.agent(role).choices()
            if role in roles:
                clients[role] = [build_client(c.provider, config.providers[c.provider], role, secrets)
                                 for c in choices]
            else:
                clients[role] = [_NotSelected("-") for _ in choices]
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    gateway = LLMGateway(
        config=config,
        clients=clients,
        limiters=build_limiters({n: (p.rpm, p.tpm) for n, p in config.providers.items()}),
        cache=DiskCache(Path(config.run.cache_dir), enabled=False),
        runlog=RunLog(run_dir),
        budget=CallBudget(len(targets)),
        secrets=secrets,
    )
    print(f"smoke check: one request per model ({len(targets)} in total), no retries, no cache")
    statuses: dict[tuple[Role, int], str] = {}
    try:
        for role, index, choice in targets:
            print(f"  {role.value} ({_position(index)}): {choice.label} ...", flush=True)
            try:
                outcome = gateway.call(role, CallType.SMOKE, SMOKE_SYSTEM, SMOKE_PROMPT, SmokeReply, tag="smoke",
                                       choice=index)
            except BudgetExhausted:
                statuses[(role, index)] = "not sent (budget)"
                continue
            if outcome.status is AgentStatus.DONE and outcome.data is not None and outcome.data.ok:
                statuses[(role, index)] = "ok"
            elif outcome.ok:
                statuses[(role, index)] = "unexpected reply"
            else:
                statuses[(role, index)] = "UNAVAILABLE"
    finally:
        gateway.close()

    records = gateway.runlog.read_calls()
    rows: list[list[str]] = []
    for role, index, choice in targets:
        rec = _last_record(records, role, index)
        status = statuses.get((role, index), "no call")
        if rec is None:
            rows.append([role.value, ROLE_VERSION_LABEL[role], _position(index), choice.label, status,
                         "-", "-", "-", "-", ""])
            continue
        tokens = f"{rec.tokens_in}/{rec.tokens_out}"
        source = "estimated" if rec.tokens_estimated else "provider"
        rows.append([role.value, ROLE_VERSION_LABEL[role], _position(index), f"{rec.provider}/{rec.model}", status,
                     f"{rec.latency_s:.2f}", tokens, source, rec.finish_reason or "-", rec.error or ""])
    print()
    print(format_table(["role", "slot", "model", "provider/model", "status", "latency_s", "tokens in/out",
                        "token counts", "finish", "error"], rows))
    print(f"\ncall log: {gateway.runlog.calls.path}")
    return 0 if all(statuses.get((r, i)) == "ok" for r, i, _ in targets) else 1


def _cmd_models(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    cfg = config.providers.get(args.provider)
    if cfg is None:
        print(f"no provider block '{args.provider}' in config (have: {', '.join(sorted(config.providers))})",
              file=sys.stderr)
        return 2
    secrets = load_secrets(config, env_file=args.env_file)
    try:
        client = build_client(args.provider, cfg, Role.GENERATOR, secrets)
    except MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        models = client.list_models()
    except ProviderError as exc:
        print(f"error: {type(exc).__name__}: {secrets.redact(str(exc))}", file=sys.stderr)
        return 1
    finally:
        client.close()
    needle = (args.contains or "").lower()
    shown = [m for m in models if needle in m.lower()]
    for model in shown:
        print(model)
    print(f"{len(shown)} of {len(models)} models listed by provider '{args.provider}'", file=sys.stderr)
    return 0


def add_provider_commands(sub: Any) -> None:
    """Register ``models`` and ``smoke`` on the main parser's subparsers object."""
    models = sub.add_parser("models", help="list the model ids a configured provider offers (no completion call)")
    models.add_argument("provider", help="provider block name from config.yaml")
    models.add_argument("--contains", default=None, help="only show ids containing this text (case-insensitive)")
    models.set_defaults(func=_cmd_models)

    smoke = sub.add_parser("smoke", help="send one tiny call to each configured model (fallbacks included) to "
                           "check keys, models and limits")
    smoke.add_argument("--roles", default=None, help="comma-separated roles to check (default: all four)")
    smoke.add_argument("--runs-dir", default=None, help="parent directory of the smoke log (default: run.runs_dir)")
    smoke.set_defaults(func=_cmd_smoke)
