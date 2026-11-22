"""Provider client factory. Each agent slot gets its own client instance."""

from __future__ import annotations

from collections.abc import Callable

from quadcode.config import AppConfig, ProviderConfig, Secrets
from quadcode.providers.anthropic import AnthropicClient
from quadcode.providers.base import LLMClient
from quadcode.providers.gemini import GeminiClient
from quadcode.providers.mock import MockProvider
from quadcode.providers.mock_scenarios import get_scenario
from quadcode.providers.openai_compat import OpenAICompatClient
from quadcode.schemas import Role

ClientFactory = Callable[[str, ProviderConfig, Role, Secrets], LLMClient]


class UnsupportedProviderError(ValueError):
    """No client implementation is registered for this provider kind."""


class MissingAPIKeyError(ValueError):
    """The provider's API key is not set in the environment or .env."""


def _build_mock(name: str, cfg: ProviderConfig, role: Role, secrets: Secrets) -> LLMClient:
    assert cfg.scenario is not None  # enforced by ProviderConfig validation
    return MockProvider(name, role, get_scenario(cfg.scenario))


def _api_key(name: str, cfg: ProviderConfig, secrets: Secrets) -> str:
    key = secrets.get(cfg.api_key_env)
    if not key:
        raise MissingAPIKeyError(f"provider '{name}' needs {cfg.api_key_env}, which is not set in the "
                                 "environment or .env")
    return key


def _build_openai_compat(name: str, cfg: ProviderConfig, role: Role, secrets: Secrets) -> LLMClient:
    return OpenAICompatClient(name, cfg, _api_key(name, cfg, secrets))


def _build_gemini(name: str, cfg: ProviderConfig, role: Role, secrets: Secrets) -> LLMClient:
    return GeminiClient(name, cfg, _api_key(name, cfg, secrets))


def _build_anthropic(name: str, cfg: ProviderConfig, role: Role, secrets: Secrets) -> LLMClient:
    return AnthropicClient(name, cfg, _api_key(name, cfg, secrets))


REGISTRY: dict[str, ClientFactory] = {
    "mock": _build_mock,
    "openai_compat": _build_openai_compat,
    "gemini": _build_gemini,
    "anthropic": _build_anthropic,
}


def build_client(name: str, cfg: ProviderConfig, role: Role, secrets: Secrets) -> LLMClient:
    factory = REGISTRY.get(cfg.kind)
    if factory is None:
        raise UnsupportedProviderError(
            f"no client registered for provider kind '{cfg.kind}' (registered: {sorted(REGISTRY)})"
        )
    return factory(name, cfg, role, secrets)


def build_clients(config: AppConfig, secrets: Secrets) -> dict[Role, list[LLMClient]]:
    """One independent client per model in each slot's list (also when slots share a provider block).

    The lists follow ``AgentConfig.choices()``: the first-choice model, then the fallbacks in order.
    """
    return {
        role: [build_client(c.provider, config.providers[c.provider], role, secrets)
               for c in config.agent(role).choices()]
        for role in Role
    }
