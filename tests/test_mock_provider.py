from __future__ import annotations

import pytest

from quadcode.jsonutil import parse_model
from quadcode.providers.base import LLMRequest, PermanentError, ProviderTimeout, RateLimitError
from quadcode.providers.mock import WILDCARD, FailureSpec, MockProvider, MockScenario
from quadcode.providers.mock_scenarios import (
    G1_CODE,
    M001_DIFF,
    M002_DIFF,
    get_scenario,
)
from quadcode.schemas import (
    ROLE_VERSION_LABEL,
    CallType,
    ChallengeResponse,
    InitialSolution,
    ModifyOutput,
    ProblemSpec,
    Review,
    Role,
    Verdict,
)


def _req(call_type: CallType, tag: str = "") -> LLMRequest:
    return LLMRequest(system="sys", prompt="prompt", model="m", max_tokens=10, temperature=0.0,
                      timeout_s=1.0, call_type=call_type, tag=tag)


def test_deterministic() -> None:
    scen = get_scenario("two_sum")
    a = MockProvider("mock", Role.GENERATOR, scen).complete(_req(CallType.INITIAL))
    b = MockProvider("mock", Role.GENERATOR, scen).complete(_req(CallType.INITIAL))
    assert a == b
    assert a.tokens_estimated is True
    assert a.tokens_in == 3  # len("sysprompt") / 4 rounded up


def test_unknown_scenario() -> None:
    with pytest.raises(KeyError, match="unknown mock scenario"):
        get_scenario("nope")


def test_all_scripted_outputs_parse() -> None:
    scen = get_scenario("two_sum")
    for role in Role:
        prov = MockProvider("mock", role, scen)
        spec = parse_model(prov.complete(_req(CallType.ANALYSIS)).text, ProblemSpec)
        assert spec.entry_point == "two_sum"
        sol = parse_model(prov.complete(_req(CallType.INITIAL)).text, InitialSolution)
        assert "def two_sum" in sol.code
        assert parse_model(prov.complete(_req(CallType.MODIFY, "round-2")).text, ModifyOutput).proposals == []
        parse_model(prov.complete(_req(CallType.CHALLENGE, "M-002")).text, ChallengeResponse)
        if role is not Role.VALIDATOR:
            review = parse_model(prov.complete(_req(CallType.REVIEW)).text, Review)
            assert review.items
            assert all(i.target in ROLE_VERSION_LABEL.values() for i in review.items)


def test_scenario_contains_required_paths() -> None:
    scen = get_scenario("two_sum")
    validator_init = parse_model(
        MockProvider("mock", Role.VALIDATOR, scen).complete(_req(CallType.INITIAL)).text, InitialSolution)
    categories = {t.category.value for t in validator_init.tests}
    assert categories == {"normal", "boundary", "adversarial", "performance"}
    # Disagreement on the reviewer's and optimizer's challenge verdicts for the same proposal.
    verdicts = {
        r: parse_model(MockProvider("mock", r, scen).complete(_req(CallType.CHALLENGE, "M-002")).text,
                       ChallengeResponse).verdict
        for r in (Role.GENERATOR, Role.REVIEWER)
    }
    assert verdicts[Role.GENERATOR] is Verdict.ACCEPT and verdicts[Role.REVIEWER] is Verdict.REJECT
    # Round-1 proposals carry diffs against G1.
    assert G1_CODE.splitlines()[0] in M001_DIFF
    assert "seen[value] = i" in M002_DIFF


def test_injected_transient_failure_then_success() -> None:
    prov = MockProvider("mock", Role.VALIDATOR, get_scenario("two_sum"))
    with pytest.raises(RateLimitError):
        prov.complete(_req(CallType.MODIFY, "round-1"))
    out = parse_model(prov.complete(_req(CallType.MODIFY, "round-1")).text, ModifyOutput)
    assert out.proposals == []


def test_injected_permanent_unavailability() -> None:
    prov = MockProvider("mock", Role.VALIDATOR, get_scenario("two_sum"))
    for _ in range(5):
        with pytest.raises(RateLimitError):
            prov.complete(_req(CallType.REVIEW))


def test_missing_response_is_permanent_error() -> None:
    prov = MockProvider("mock", Role.GENERATOR, MockScenario(name="empty"))
    with pytest.raises(PermanentError, match="no response"):
        prov.complete(_req(CallType.INITIAL))


def test_timeout_injection_and_exact_tag_precedence() -> None:
    scen = MockScenario(
        name="t",
        responses={(Role.GENERATOR, CallType.MODIFY, WILDCARD): "any",
                   (Role.GENERATOR, CallType.MODIFY, "round-1"): "exact"},
        failures={(Role.GENERATOR, CallType.MODIFY, "round-2"): FailureSpec("timeout", times=1)},
    )
    prov = MockProvider("mock", Role.GENERATOR, scen)
    assert prov.complete(_req(CallType.MODIFY, "round-1")).text == "exact"
    assert prov.complete(_req(CallType.MODIFY, "round-3")).text == "any"
    with pytest.raises(ProviderTimeout):
        prov.complete(_req(CallType.MODIFY, "round-2"))
    assert prov.complete(_req(CallType.MODIFY, "round-2")).text == "any"


def test_unregistered_provider_kind_is_reported() -> None:
    from quadcode.config import ProviderConfig, Secrets
    from quadcode.providers import UnsupportedProviderError, build_client

    cfg = ProviderConfig(kind="gemini", api_key_env="GEMINI_API_KEY", rpm=1, tpm=1)
    with pytest.raises(UnsupportedProviderError, match="gemini"):
        build_client("gemini", cfg, Role.GENERATOR, Secrets({}))
