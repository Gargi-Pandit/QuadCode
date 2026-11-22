"""Load and validate ``config.yaml`` and the API keys referenced by it.

``config.yaml`` is the only source of provider and model names. API keys are read
from the process environment and ``.env`` into a ``Secrets`` object that lives in
the orchestrator process only; it never serializes or prints key values.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from quadcode.schemas import ROLE_VERSION_LABEL, CallType, Role

ProviderKind = Literal["mock", "gemini", "openai_compat", "anthropic"]


class ConfigError(ValueError):
    """Raised for any invalid or unreadable configuration."""


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderConfig(_Strict):
    kind: ProviderKind
    base_url: str | None = None
    api_key_env: str | None = None
    scenario: str | None = Field(default=None, description="Mock scenario name (mock kind only).")
    rpm: int = Field(gt=0)
    tpm: int = Field(gt=0)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=4, ge=0)
    backoff_base_s: float = Field(default=1.0, ge=0)
    backoff_cap_s: float = Field(default=30.0, ge=0)
    max_retry_after_s: float = Field(default=90.0, gt=0, description="A 429 asking to wait longer than this "
                                     "is not retried; the agent is marked unavailable for that call.")

    @model_validator(mode="after")
    def _check_kind_fields(self) -> ProviderConfig:
        if self.kind == "mock":
            if not self.scenario:
                raise ValueError("mock providers require 'scenario'")
        else:
            if not self.api_key_env:
                raise ValueError(f"{self.kind} providers require 'api_key_env'")
            if self.kind == "openai_compat" and not self.base_url:
                raise ValueError("openai_compat providers require 'base_url'")
        return self


# Request fields the client sets itself; request_options may not override them.
RESERVED_OPTIONS: dict[str, frozenset[str]] = {
    "openai_compat": frozenset({"model", "messages", "max_tokens", "max_completion_tokens", "temperature",
                                "stream", "n"}),
    "anthropic": frozenset({"model", "messages", "max_tokens", "temperature", "system", "stream"}),
    "gemini": frozenset({"maxOutputTokens", "temperature", "candidateCount"}),
}


class ModelChoice(_Strict):
    """One provider/model and the request settings used with it."""

    provider: str
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    request_options: dict[str, Any] = Field(
        default_factory=dict, description="Extra provider request fields, e.g. reasoning_effort. Merged into the "
        "request body (openai_compat, anthropic) or generationConfig (gemini).")
    inline_system: bool = Field(default=False, description="Send the system text inside the user message, for "
                                "models that reject a system role.")
    max_tokens: dict[CallType, int] = Field(
        default_factory=dict, description="Per-call-type output caps for this model only; they override the "
        "global max_tokens (e.g. a validator that writes a whole test suite needs more room).")

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


class AgentConfig(ModelChoice):
    """An agent slot: its first-choice model plus ordered fallbacks tried when that model fails."""

    fallbacks: list[ModelChoice] = Field(
        default_factory=list, description="Tried in order when the model before them returns no usable answer. "
        "A provider failure moves the slot to its next model for the rest of the run.")

    def choices(self) -> list[ModelChoice]:
        """The slot's models in the order they are tried; index 0 is the first choice."""
        return [self, *self.fallbacks]


def model_identity(model: str) -> str:
    """A model id without host prefix or variant suffix, e.g. 'google/gemma-4-31b-it:free' -> 'gemma-4-31b-it'."""
    return model.rsplit("/", 1)[-1].split(":", 1)[0].strip().lower()


class RunConfig(_Strict):
    rounds: int = Field(default=3, ge=1, le=5)
    state_output_limit: bool = Field(default=True, description="Tell the model each call's max_tokens and ask "
                                     "for a reply (reasoning included) that fits, so it shortens its answer "
                                     "instead of being cut off.")
    repair_attempts: int = Field(default=1, ge=0, le=3, description="When an answer is not valid JSON for its "
                                 "schema, ask the same model this many more times, telling it what was wrong, "
                                 "before moving on to a fallback model.")
    parallel: bool = True
    budget_calls: int | None = Field(default=None, ge=1)
    cache_enabled: bool = True
    cache_dir: str = ".quadcode_cache"
    runs_dir: str = "runs"


SandboxBackendName = Literal["auto", "docker", "subprocess"]


class SandboxConfig(_Strict):
    """Limits and measurement settings for executing untrusted generated code."""

    backend: SandboxBackendName = "auto"
    docker_image: str = Field(default="python:3.12-slim", min_length=1)
    memory_mb: int = Field(default=512, ge=32)
    cpus: float = Field(default=1.0, gt=0)
    pids_limit: int = Field(default=64, ge=8)
    test_timeout_s: float = Field(default=5.0, gt=0, description="Limit for a single call of the entry point.")
    wall_timeout_s: float = Field(default=120.0, gt=0, description="Limit for the whole sandbox process.")
    test_repeats: int = Field(default=3, ge=1, description="Timed runs per test in suite runs.")
    benchmark_repeats: int = Field(default=7, ge=1, description="Timed runs per test in benchmarks.")
    warmup: int = Field(default=1, ge=0)
    max_output_bytes: int = Field(default=8_000_000, ge=10_000)
    max_actual_chars: int = Field(default=2_000, ge=50, description="Stored 'actual' values are truncated past this.")
    float_rel_tol: float = Field(default=1e-9, ge=0)
    float_abs_tol: float = Field(default=1e-9, ge=0)
    seed: int = 0


class SelectionConfig(_Strict):
    """Tolerances for evidence-based comparisons. Differences inside a tolerance count as ties."""

    runtime_tolerance: float = Field(default=0.25, ge=0, description="Relative runtime difference treated "
                                     "as noise.")
    memory_tolerance: float = Field(default=0.10, ge=0, description="Relative peak-memory difference treated "
                                    "as noise.")
    memory_floor_bytes: int = Field(default=4096, ge=0, description="Absolute peak-memory difference always "
                                    "treated as noise.")


class PromptConfig(_Strict):
    """Size limits for what is sent to agents."""

    max_error_chars: int = Field(default=200, ge=20, description="Failing-test output is truncated to this.")
    max_failing_tests: int = Field(default=5, ge=1, description="Failing tests listed per version.")


class ModelPrice(_Strict):
    """Price of one provider/model in USD per million tokens, used only for report cost estimates."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


# Call types whose max_tokens entry may be omitted, with the cap used then.
OPTIONAL_CAPS: dict[CallType, int] = {CallType.SMOKE: 512}


class AppConfig(_Strict):
    providers: dict[str, ProviderConfig]
    agents: dict[Role, AgentConfig]
    max_tokens: dict[CallType, int]
    run: RunConfig = RunConfig()
    sandbox: SandboxConfig = SandboxConfig()
    selection: SelectionConfig = SelectionConfig()
    prompts: PromptConfig = PromptConfig()
    allow_duplicate_models: bool = False
    pricing: dict[str, ModelPrice] = Field(
        default_factory=dict, description="'<provider>/<model>' -> price. Reports show cost as not configured "
        "for models without an entry.")

    @model_validator(mode="after")
    def _check_consistency(self) -> AppConfig:
        missing_roles = [r.value for r in Role if r not in self.agents]
        if missing_roles:
            raise ValueError(f"missing agent slots: {missing_roles}")
        missing_caps = [c.value for c in CallType if c not in self.max_tokens and c not in OPTIONAL_CAPS]
        if missing_caps:
            raise ValueError(f"missing max_tokens for call types: {missing_caps}")
        bad_caps = [c.value for c, v in self.max_tokens.items() if v <= 0]
        if bad_caps:
            raise ValueError(f"max_tokens must be positive for: {bad_caps}")
        for role, agent in self.agents.items():
            seen_in_slot: set[tuple[str, str]] = set()
            for index, choice in enumerate(agent.choices()):
                self._check_choice(role, index, choice)
                if (choice.provider, choice.model) in seen_in_slot:
                    raise ValueError(f"agent '{role.value}' lists {choice.label} twice")
                seen_in_slot.add((choice.provider, choice.model))
        bad_keys = [k for k in self.pricing if "/" not in k.strip("/")]
        if bad_keys:
            raise ValueError(f"pricing keys must look like '<provider>/<model>': {bad_keys}")
        if not self.allow_duplicate_models:
            owner: dict[str, tuple[Role, ModelChoice]] = {}
            for role, agent in self.agents.items():
                for choice in agent.choices():
                    ident = model_identity(choice.model)
                    other = owner.get(ident)
                    if other is not None and other[0] is not role:
                        raise ValueError(
                            f"agents '{other[0].value}' and '{role.value}' both list model '{ident}' "
                            f"({other[1].label} and {choice.label}), so two slots could be the same model; "
                            "set allow_duplicate_models: true to permit this explicitly"
                        )
                    owner.setdefault(ident, (role, choice))
        return self

    def _check_choice(self, role: Role, index: int, choice: ModelChoice) -> None:
        where = f"agent '{role.value}'" + (f" fallback {index}" if index else "")
        if choice.provider not in self.providers:
            raise ValueError(f"{where} references unknown provider '{choice.provider}'")
        kind = self.providers[choice.provider].kind
        if kind == "mock" and (choice.request_options or choice.inline_system):
            raise ValueError(f"{where}: request_options and inline_system are not used by mock providers")
        bad_caps = [c.value for c, v in choice.max_tokens.items() if v <= 0]
        if bad_caps:
            raise ValueError(f"{where}: max_tokens must be positive for: {bad_caps}")
        reserved = sorted(RESERVED_OPTIONS.get(kind, frozenset()) & set(choice.request_options))
        if reserved:
            raise ValueError(f"{where}: request_options may not set {reserved}; they come from the model, "
                             "temperature and max_tokens settings")

    def agent(self, role: Role) -> AgentConfig:
        return self.agents[role]

    def provider_for(self, role: Role) -> ProviderConfig:
        return self.providers[self.agents[role].provider]

    def cap_for(self, call_type: CallType, role: Role | None = None, choice: ModelChoice | None = None) -> int:
        """Output token cap: the model's own override, else the global cap, else the optional default.

        ``choice`` is one of a slot's models; with only ``role`` given, the slot's first choice is used.
        """
        source = choice if choice is not None else (self.agents[role] if role is not None else None)
        if source is not None and call_type in source.max_tokens:
            return source.max_tokens[call_type]
        return self.max_tokens.get(call_type, OPTIONAL_CAPS.get(call_type, 0))

    def price_for(self, provider: str, model: str) -> ModelPrice | None:
        return self.pricing.get(f"{provider}/{model}")

    def describe_agents(self) -> list[dict[str, Any]]:
        """Role -> actual provider/model rows, for display in CLI, reports and UI."""
        rows: list[dict[str, Any]] = []
        for role in Role:
            agent = self.agents[role]
            prov = self.providers[agent.provider]
            rows.append(
                {
                    "role": role.value,
                    "label": ROLE_VERSION_LABEL[role],
                    "provider": agent.provider,
                    "kind": prov.kind,
                    "model": agent.model,
                    "fallbacks": ", ".join(c.label for c in agent.fallbacks) or "-",
                    "rpm": prov.rpm,
                    "tpm": prov.tpm,
                }
            )
        return rows


def parse_config(data: Mapping[str, Any]) -> AppConfig:
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {p}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config {p} must be a mapping at top level")
    return parse_config(raw)


class Secrets:
    """API keys held in orchestrator memory. Never printed, serialized or logged."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values: dict[str, str] = {k: v for k, v in values.items() if v}

    def get(self, env_name: str | None) -> str | None:
        if env_name is None:
            return None
        return self._values.get(env_name)

    def has(self, env_name: str) -> bool:
        return env_name in self._values

    def redact(self, text: str) -> str:
        """Replace any key value occurring in ``text`` with a marker."""
        for name, value in self._values.items():
            if value and value in text:
                text = text.replace(value, f"<redacted:{name}>")
        return text

    def __repr__(self) -> str:
        return f"Secrets(names={sorted(self._values)})"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("Secrets cannot be pickled")


def load_secrets(
    config: AppConfig,
    env_file: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> Secrets:
    """Collect only the keys named by ``api_key_env`` in the config.

    Values from the real environment take precedence over ``.env``. ``os.environ``
    is never modified, so keys cannot leak into child processes through it.
    """
    env = dict(os.environ if environ is None else environ)
    file_values: dict[str, str | None] = {}
    if env_file is not None and Path(env_file).is_file():
        file_values = dotenv_values(env_file)
    wanted = {p.api_key_env for p in config.providers.values() if p.api_key_env}
    found: dict[str, str] = {}
    for name in wanted:
        value = env.get(name) or file_values.get(name)
        if value:
            found[name] = value
    return Secrets(found)


def missing_keys(config: AppConfig, secrets: Secrets, roles: Iterable[Role] | None = None) -> list[tuple[str, str]]:
    """(provider, env var) for every provider used by ``roles`` (fallbacks included) whose API key is not set."""
    selected = list(Role) if roles is None else list(roles)
    out: list[tuple[str, str]] = []
    for name in dict.fromkeys(c.provider for r in selected for c in config.agent(r).choices()):
        env = config.providers[name].api_key_env
        if env and not secrets.has(env):
            out.append((name, env))
    return out
