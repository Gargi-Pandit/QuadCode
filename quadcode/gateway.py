"""LLMGateway: the single path every agent call goes through.

Order per call: cache -> budget -> rate limiter -> provider (with retry/backoff)
-> JSON extraction + schema validation -> cache store. Every provider attempt and
every cache hit is written to calls.jsonl. Failures come back as UNAVAILABLE
outcomes; they are never turned into empty "approval" data.

A slot may list fallback models in config.yaml. When its current model gives no usable
answer, the same call goes to the next model in the list; every attempt is logged with
the provider and model that made it. A provider failure (API error, rate limit, timeout)
moves the slot to its next model for the rest of the run, so a provider that is down
costs one failed call rather than one per phase; an invalid answer moves it only for
that call. Before that, an answer that fails validation is sent back to the same model with
the validation error (``run.repair_attempts`` times), since a single malformed field should not
cost the slot its model.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from quadcode.cache import CacheEntry, DiskCache, cache_key, sha256_text
from quadcode.config import AppConfig, ModelChoice, Secrets
from quadcode.jsonutil import JSONExtractionError, parse_model
from quadcode.prompts import output_limit_note, repair_prompt
from quadcode.providers import build_clients
from quadcode.providers.base import LLMClient, LLMRequest, LLMResponse, ProviderError, estimate_tokens
from quadcode.ratelimit import Clock, ProviderLimiter, RetryPolicy, Sleep, build_limiters, retry_with_backoff
from quadcode.runlog import RunLog
from quadcode.schemas import AgentStatus, CallRecord, CallStatus, CallType, Role

T = TypeVar("T", bound=BaseModel)


EXCERPT_HEAD = 400
EXCERPT_TAIL = 300


def output_excerpt(text: str) -> str:
    """Start and end of a rejected output, enough to see why it failed without storing all of it."""
    if len(text) <= EXCERPT_HEAD + EXCERPT_TAIL:
        return text
    omitted = len(text) - EXCERPT_HEAD - EXCERPT_TAIL
    return f"{text[:EXCERPT_HEAD]}\n[... {omitted} characters omitted ...]\n{text[-EXCERPT_TAIL:]}"


class BudgetExhausted(RuntimeError):
    """The --budget-calls limit was reached; the orchestrator should stop and report."""


class CallBudget:
    """Counts provider attempts (cache hits are free). ``limit=None`` means unlimited."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def try_consume(self) -> bool:
        with self._lock:
            if self.limit is not None and self.used >= self.limit:
                return False
            self.used += 1
            return True


@dataclass(frozen=True)
class CallOutcome(Generic[T]):
    role: Role
    call_type: CallType
    provider: str
    model: str
    status: AgentStatus  # DONE, CACHED or UNAVAILABLE
    data: T | None
    raw_text: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.status in (AgentStatus.DONE, AgentStatus.CACHED) and self.data is not None


@dataclass(frozen=True)
class _Prepared:
    """Everything one call needs for one of the slot's models."""

    role: Role
    index: int
    choice: ModelChoice
    call_type: CallType
    tag: str
    max_tokens: int
    key: str
    base: dict[str, object]
    prompt: str  # as sent, with the output limit note when enabled
    repair: int = 0


class LLMGateway:
    def __init__(
        self,
        config: AppConfig,
        clients: Mapping[Role, LLMClient | Sequence[LLMClient]],
        limiters: Mapping[str, ProviderLimiter],
        cache: DiskCache,
        runlog: RunLog,
        budget: CallBudget,
        secrets: Secrets,
        sleep: Sleep = time.sleep,
        clock: Clock = time.perf_counter,
        rng: random.Random | None = None,
    ) -> None:
        missing = [r.value for r in Role if r not in clients]
        if missing:
            raise ValueError(f"missing clients for roles: {missing}")
        chains: dict[Role, list[LLMClient]] = {}
        for role in Role:
            given = clients[role]
            chain = list(given) if isinstance(given, (list, tuple)) else [given]
            expected = len(config.agent(role).choices())
            if len(chain) != expected:
                raise ValueError(f"agent '{role.value}' lists {expected} model(s) but got {len(chain)} client(s)")
            chains[role] = chain
        ids = {id(c) for chain in chains.values() for c in chain}
        if len(ids) != sum(len(chain) for chain in chains.values()):
            raise ValueError("each agent slot needs its own client instance")
        self.config = config
        self.clients = chains
        self._active: dict[Role, int] = dict.fromkeys(Role, 0)
        self._active_lock = threading.Lock()
        self.limiters = dict(limiters)
        self.cache = cache
        self.runlog = runlog
        self.budget = budget
        self.secrets = secrets
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        secrets: Secrets,
        run_dir: Path,
        budget_calls: int | None = None,
        cache_root: Path | None = None,
    ) -> LLMGateway:
        limits = {name: (p.rpm, p.tpm) for name, p in config.providers.items()}
        return cls(
            config=config,
            clients=build_clients(config, secrets),
            limiters=build_limiters(limits),
            cache=DiskCache(cache_root or Path(config.run.cache_dir), enabled=config.run.cache_enabled),
            runlog=RunLog(run_dir),
            budget=CallBudget(budget_calls if budget_calls is not None else config.run.budget_calls),
            secrets=secrets,
        )

    def close(self) -> None:
        """Close every client's network resources."""
        for chain in self.clients.values():
            for client in chain:
                client.close()

    def active_choice(self, role: Role) -> int:
        """Index of the model the slot currently starts with (0 = its first choice)."""
        with self._active_lock:
            return self._active[role]

    def _move_past(self, role: Role, index: int) -> None:
        with self._active_lock:
            self._active[role] = max(self._active[role], index + 1)

    def _record(self, **fields: object) -> None:
        err = fields.get("error")
        if isinstance(err, str):
            fields["error"] = self.secrets.redact(err)
        self.runlog.log_call(CallRecord.model_validate(fields))

    def call(
        self,
        role: Role,
        call_type: CallType,
        system: str,
        prompt: str,
        schema: type[T],
        tag: str = "",
        choice: int | None = None,
    ) -> CallOutcome[T]:
        """Ask ``role`` and return its first usable answer, trying the slot's fallbacks in order.

        ``choice`` pins the call to one of the slot's models (no fallback), e.g. for smoke checks.
        """
        choices = self.config.agent(role).choices()
        if choice is not None:
            if not 0 <= choice < len(choices):
                raise ValueError(f"agent '{role.value}' has no model #{choice}")
            prep = self._prepare(role, choice, call_type, system, prompt, tag)
            hit = self._from_cache(prep, schema)
            return hit if hit is not None else self._send_and_repair(prep, system, prompt, schema)[0]

        order = [self._prepare(role, i, call_type, system, prompt, tag)
                 for i in range(self.active_choice(role), len(choices))]
        # An answer this exact call already got from any of the remaining models is reused first.
        for prep in order:
            hit = self._from_cache(prep, schema)
            if hit is not None:
                return hit
        failures: list[CallOutcome[T]] = []
        for prep in order:
            result, provider_failed = self._send_and_repair(prep, system, prompt, schema)
            if result.ok:
                return result
            failures.append(result)
            if provider_failed and prep.index + 1 < len(choices):
                self._move_past(role, prep.index)  # the last model stays in use, so every call is still tried
        if len(failures) == 1:
            return failures[0]
        last = failures[-1]
        combined = "; ".join(f"[{o.provider}/{o.model}] {o.error}" for o in failures)
        return CallOutcome(role, call_type, last.provider, last.model, AgentStatus.UNAVAILABLE, None,
                           last.raw_text, f"all {len(failures)} configured models failed: {combined}")

    def _send_and_repair(self, prep: _Prepared, system: str, prompt: str,
                         schema: type[T]) -> tuple[CallOutcome[T], bool]:
        """Send to one model; if its answer fails validation, ask it to fix the answer before giving up on it."""
        result, provider_failed = self._send(prep, system, schema)
        for n in range(1, self.config.run.repair_attempts + 1):
            if result.ok or provider_failed or result.error is None:
                break
            previous = self.secrets.redact(result.raw_text) if result.raw_text else None
            fix = repair_prompt(prompt, previous, result.error, result.error.startswith("output hit max_tokens"))
            rprep = self._prepare(prep.role, prep.index, prep.call_type, system, fix, prep.tag, repair=n)
            hit = self._from_cache(rprep, schema)
            if hit is not None:
                return hit, False
            result, provider_failed = self._send(rprep, system, schema)
        return result, provider_failed

    def _prepare(self, role: Role, index: int, call_type: CallType, system: str, prompt: str,
                 tag: str, repair: int = 0) -> _Prepared:
        choice = self.config.agent(role).choices()[index]
        max_tokens = self.config.cap_for(call_type, role, choice)
        params: dict[str, object] = {"max_tokens": max_tokens, "temperature": choice.temperature, "tag": tag}
        # Only non-default settings join the key, so keys made before these settings existed stay valid.
        if choice.request_options:
            params["request_options"] = choice.request_options
        if choice.inline_system:
            params["inline_system"] = True
        if self.config.run.state_output_limit:
            prompt = prompt + output_limit_note(max_tokens)
        key = cache_key(choice.provider, choice.model, system, prompt, params)
        base: dict[str, object] = {
            "role": role,
            "provider": choice.provider,
            "model": choice.model,
            "call_type": call_type,
            "tag": tag,
            "prompt_hash": sha256_text(system + "\n" + prompt),
            "cache_key": key,
        }
        if index:
            base["choice"] = index
        if repair:
            base["repair"] = repair
        return _Prepared(role, index, choice, call_type, tag, max_tokens, key, base, prompt, repair)

    def _outcome(self, prep: _Prepared, status: AgentStatus, data: T | None, raw: str | None,
                 error: str | None) -> CallOutcome[T]:
        return CallOutcome(prep.role, prep.call_type, prep.choice.provider, prep.choice.model, status, data, raw,
                           self.secrets.redact(error) if error else None)

    def _from_cache(self, prep: _Prepared, schema: type[T]) -> CallOutcome[T] | None:
        cached = self.cache.get(prep.key)
        if cached is None:
            return None
        try:
            data = parse_model(cached.text, schema)
        except (JSONExtractionError, ValidationError):
            return None
        self._record(**prep.base, attempt=1, tokens_in=cached.tokens_in, tokens_out=cached.tokens_out,
                     tokens_estimated=cached.tokens_estimated, latency_s=0.0, cache_hit=True, status=CallStatus.OK)
        return self._outcome(prep, AgentStatus.CACHED, data, cached.text, None)

    def _send(self, prep: _Prepared, system: str, schema: type[T]) -> tuple[CallOutcome[T], bool]:
        """One model's attempt (with that provider's retries). Returns the outcome and whether the provider failed."""
        choice = prep.choice
        pcfg = self.config.providers[choice.provider]
        client = self.clients[prep.role][prep.index]
        limiter = self.limiters[choice.provider]
        base = prep.base
        est_total = estimate_tokens(system + prep.prompt) + prep.max_tokens
        request = LLMRequest(system=system, prompt=prep.prompt, model=choice.model, max_tokens=prep.max_tokens,
                             temperature=choice.temperature, timeout_s=pcfg.timeout_s,
                             call_type=prep.call_type, tag=prep.tag, options=dict(choice.request_options),
                             inline_system=choice.inline_system)
        last_attempt = 0

        def attempt_once(attempt: int) -> tuple[LLMResponse, float]:
            nonlocal last_attempt
            last_attempt = attempt
            if not self.budget.try_consume():
                self._record(**base, attempt=attempt, tokens_in=0, tokens_out=0, tokens_estimated=True,
                             latency_s=0.0, cache_hit=False, status=CallStatus.BUDGET_EXHAUSTED,
                             error=f"call budget of {self.budget.limit} exhausted")
                raise BudgetExhausted(f"call budget of {self.budget.limit} exhausted")
            limiter.acquire(est_total)
            start = self._clock()
            try:
                resp = client.complete(request)
            except ProviderError as err:
                self._record(**base, attempt=attempt, tokens_in=0, tokens_out=0, tokens_estimated=True,
                             latency_s=max(0.0, self._clock() - start), cache_hit=False,
                             status=CallStatus.PROVIDER_ERROR, error=f"{type(err).__name__}: {err}")
                raise
            latency = max(0.0, self._clock() - start)
            limiter.settle(est_total, resp.tokens_in + resp.tokens_out)
            return resp, latency

        policy = RetryPolicy(pcfg.max_retries, pcfg.backoff_base_s, pcfg.backoff_cap_s)
        try:
            resp, latency = retry_with_backoff(attempt_once, policy, sleep=self._sleep, rng=self._rng)
        except ProviderError as err:
            return self._outcome(prep, AgentStatus.UNAVAILABLE, None, None,
                                 f"{type(err).__name__} after {last_attempt} attempt(s): {err}"), True

        usage = {"tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out,
                 "tokens_estimated": resp.tokens_estimated, "latency_s": latency, "cache_hit": False,
                 "finish_reason": resp.finish_reason}
        try:
            data = parse_model(resp.text, schema)
        except (JSONExtractionError, ValidationError) as err:
            msg = f"invalid output for {schema.__name__}: {str(err)[:300]}"
            if resp.truncated:
                msg = (f"output hit max_tokens={prep.max_tokens} (finish_reason={resp.finish_reason}) and was cut "
                       f"off; {msg}")
            self._record(**base, attempt=last_attempt, **usage, status=CallStatus.INVALID_OUTPUT, error=msg,
                         output_excerpt=self.secrets.redact(output_excerpt(resp.text)))
            return self._outcome(prep, AgentStatus.UNAVAILABLE, None, resp.text, msg), False

        self._record(**base, attempt=last_attempt, **usage, status=CallStatus.OK)
        self.cache.put(prep.key, CacheEntry(text=resp.text, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                                            tokens_estimated=resp.tokens_estimated, model=resp.model))
        return self._outcome(prep, AgentStatus.DONE, data, resp.text, None), False
