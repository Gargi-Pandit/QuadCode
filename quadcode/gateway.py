"""LLMGateway: the single path every agent call goes through.

Order per call: cache -> budget -> rate limiter -> provider (with retry/backoff)
-> JSON extraction + schema validation -> cache store. Every provider attempt and
every cache hit is written to calls.jsonl. Failures come back as UNAVAILABLE
outcomes; they are never turned into empty "approval" data.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from quadcode.cache import CacheEntry, DiskCache, cache_key, sha256_text
from quadcode.config import AppConfig, Secrets
from quadcode.jsonutil import JSONExtractionError, parse_model
from quadcode.providers import build_clients
from quadcode.providers.base import LLMClient, LLMRequest, LLMResponse, ProviderError, estimate_tokens
from quadcode.ratelimit import Clock, ProviderLimiter, RetryPolicy, Sleep, build_limiters, retry_with_backoff
from quadcode.runlog import RunLog
from quadcode.schemas import AgentStatus, CallRecord, CallStatus, CallType, Role

T = TypeVar("T", bound=BaseModel)


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


class LLMGateway:
    def __init__(
        self,
        config: AppConfig,
        clients: Mapping[Role, LLMClient],
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
        ids = {id(c) for c in clients.values()}
        if len(ids) != len(clients):
            raise ValueError("each agent slot needs its own client instance")
        self.config = config
        self.clients = dict(clients)
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
    ) -> CallOutcome[T]:
        agent = self.config.agent(role)
        pcfg = self.config.provider_for(role)
        provider = agent.provider
        max_tokens = self.config.max_tokens[call_type]
        params = {"max_tokens": max_tokens, "temperature": agent.temperature, "tag": tag}
        key = cache_key(provider, agent.model, system, prompt, params)
        base = {
            "role": role,
            "provider": provider,
            "model": agent.model,
            "call_type": call_type,
            "tag": tag,
            "prompt_hash": sha256_text(system + "\n" + prompt),
            "cache_key": key,
        }

        def outcome(status: AgentStatus, data: T | None, raw: str | None, error: str | None) -> CallOutcome[T]:
            return CallOutcome(role, call_type, provider, agent.model, status, data, raw,
                               self.secrets.redact(error) if error else None)

        cached = self.cache.get(key)
        if cached is not None:
            try:
                data = parse_model(cached.text, schema)
            except (JSONExtractionError, ValidationError):
                data = None
            if data is not None:
                self._record(**base, attempt=1, tokens_in=cached.tokens_in, tokens_out=cached.tokens_out,
                             tokens_estimated=cached.tokens_estimated, latency_s=0.0, cache_hit=True,
                             status=CallStatus.OK)
                return outcome(AgentStatus.CACHED, data, cached.text, None)

        client = self.clients[role]
        limiter = self.limiters[provider]
        est_in = estimate_tokens(system + prompt)
        est_total = est_in + max_tokens
        request = LLMRequest(system=system, prompt=prompt, model=agent.model, max_tokens=max_tokens,
                             temperature=agent.temperature, timeout_s=pcfg.timeout_s,
                             call_type=call_type, tag=tag)
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
            return outcome(AgentStatus.UNAVAILABLE, None, None,
                           f"{type(err).__name__} after {last_attempt} attempt(s): {err}")

        usage = {"tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out,
                 "tokens_estimated": resp.tokens_estimated, "latency_s": latency, "cache_hit": False}
        try:
            data = parse_model(resp.text, schema)
        except (JSONExtractionError, ValidationError) as err:
            msg = f"invalid output for {schema.__name__}: {str(err)[:300]}"
            self._record(**base, attempt=last_attempt, **usage, status=CallStatus.INVALID_OUTPUT, error=msg)
            return outcome(AgentStatus.UNAVAILABLE, None, resp.text, msg)

        self._record(**base, attempt=last_attempt, **usage, status=CallStatus.OK)
        self.cache.put(key, CacheEntry(text=resp.text, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out,
                                       tokens_estimated=resp.tokens_estimated, model=resp.model))
        return outcome(AgentStatus.DONE, data, resp.text, None)
