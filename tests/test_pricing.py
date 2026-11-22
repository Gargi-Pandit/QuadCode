"""Optional per-model prices: validation, lookup, and no effect on resumability."""

from __future__ import annotations

import pytest

from quadcode.config import ConfigError
from quadcode.runstate import config_fingerprint

from .orch_helpers import make_config


def test_prices_are_optional_and_looked_up_by_provider_and_model() -> None:
    assert make_config().pricing == {}
    cfg = make_config(pricing={"mock/mock-generator-v1": {"input_per_mtok": 0.5, "output_per_mtok": 1.5}})
    price = cfg.price_for("mock", "mock-generator-v1")
    assert price is not None and price.input_per_mtok == 0.5 and price.output_per_mtok == 1.5
    assert cfg.price_for("mock", "mock-reviewer-v1") is None


@pytest.mark.parametrize("pricing", [
    {"no-slash": {"input_per_mtok": 1, "output_per_mtok": 1}},
    {"mock/m": {"input_per_mtok": -1, "output_per_mtok": 1}},
    {"mock/m": {"input_per_mtok": 1}},
])
def test_invalid_prices_are_config_errors(pricing: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        make_config(pricing=pricing)


def test_prices_do_not_change_the_resume_fingerprint() -> None:
    priced = make_config(pricing={"mock/mock-generator-v1": {"input_per_mtok": 3, "output_per_mtok": 9}})
    assert config_fingerprint(priced) == config_fingerprint(make_config())
