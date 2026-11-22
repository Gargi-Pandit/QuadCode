"""Provider client factory. Each agent slot gets its own client instance."""

from __future__ import annotations

from collections.abc import Callable

from quadcode.config import AppConfig, ProviderConfig, Secrets
from quadcode.providers.base import LLMClient
from quadcode.providers.mock import MockProvider
from quadcode.providers.mock_scenarios import get_scenario
from quadcode.schemas import Role

ClientFactory = Callable[[str, ProviderConfig, Role, Secrets], LLMClient]


class UnsupportedProviderError(ValueError):
    """No client implementation is registered for this provider kind."""


def _build_mock(name: str, cfg: ProviderConfig, role: Role, secrets: Secrets) -> LLMClient:
    assert cfg.scenario is not None  # enforced by ProviderConfig validation
    return MockProvider(name, role, get_scenario(cfg.scenario))


REGISTRY: dict[str, ClientFactory] = {"mock": _build_mock}


def build_client(name: str, cfg: ProviderConfig, role: Role, secrets: Secrets) -> LLMClient:
    factory = REGISTRY.get(cfg.kind)
    if factory is None:
        raise UnsupportedProviderError(
            f"no client registered for provider kind '{cfg.kind}' (registered: {sorted(REGISTRY)})"
        )
    return factory(name, cfg, role, secrets)


def build_clients(config: AppConfig, secrets: Secrets) -> dict[Role, LLMClient]:
    """Construct four independent clients, one per agent slot."""
    return {
        role: build_client(config.agents[role].provider, config.provider_for(role), role, secrets)
        for role in Role
    }
