"""Test values must be JSON values of the signature's types; mistyped suites are repaired, never cached."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quadcode.cache import CacheEntry
from quadcode.config import AppConfig
from quadcode.gateway import LLMGateway
from quadcode.providers.base import LLMClient, LLMRequest, LLMResponse
from quadcode.providers.mock import WILDCARD
from quadcode.schemas import AgentStatus, CallType, InitialSolution, Role, RunStatus, TestCase
from quadcode.testargs import check_tests, parse_signature, suite_check

from .conftest import GatewayFactory
from .orch_helpers import build, scenario_with

SIG = "def top_k_frequent(words: list[str], k: int) -> list[str]"


def _t(tid: str = "T01", **kw: Any) -> TestCase:
    return TestCase(id=tid, category="normal", **kw)


def test_parse_signature_kinds() -> None:
    sig = parse_signature(SIG)
    assert sig is not None
    assert [(n, k) for n, _, k in sig.params] == [("words", "array"), ("k", "number")]
    assert sig.returns == ("list[str]", "array")
    other = parse_signature("merge(a: dict[str, int], *, flag: bool, name: str, x: Optional[int], y: int | None, z)")
    assert other is not None
    assert [(n, k) for n, _, k in other.params] == [("a", "object"), ("flag", "boolean"), ("name", "string"),
                                                    ("x", None), ("y", None), ("z", None)]
    assert parse_signature("not a signature (") is None


def test_stringified_values_are_flagged() -> None:
    msg = check_tests([_t(args=["['apple','banana']", "2"], expected=["apple"])], SIG)
    assert msg is not None
    assert "argument 'words' must be list[str]" in msg and "argument 'k' must be int" in msg
    assert "JSON array" in msg
    assert check_tests([_t(kwargs={"k": "3"}, args=[["a"]], expected=["a"])], SIG) is not None
    assert "expected must be list[str]" in (check_tests([_t(args=[["a"], 1], expected="['a']")], SIG) or "")


def test_valid_and_exempt_tests_pass() -> None:
    good = _t(args=[["apple", "banana"], 2], expected=["apple"])
    wrong_on_purpose = _t("T02", args=["not a list", 1], expect_exception="TypeError")
    generated = _t("T03", generator="([['a'] * 10, 1], {})", expected=["a"])
    assert check_tests([good, wrong_on_purpose, generated], SIG) is None
    assert check_tests([_t(args=["III"], expected=3)], "def roman_to_int(s: str) -> int") is None
    assert check_tests([_t(args=["x"])], "f(a: Optional[list[int]])") is None
    assert check_tests([_t(args=["x"])], "this is not python") is None


def test_report_is_capped() -> None:
    tests = [_t(f"T{i:02d}", args=["[1]", "1"], expected="[1]") for i in range(1, 6)]
    msg = check_tests(tests, SIG)
    assert msg is not None and "(and 12 more)" in msg


# ---------------------------------------------------------------- gateway


BAD = {"language": "python", "code": "def top_k_frequent(words, k):\n    return []\n", "algorithm": "a",
       "time_complexity": "O(n)", "space_complexity": "O(n)",
       "tests": [{"id": "T01", "category": "normal", "args": ["['a', 'b']", "2"], "expected": ["a"]}]}
GOOD = {**BAD, "tests": [{"id": "T01", "category": "normal", "args": [["a", "b"], 2], "expected": ["a"]}]}


class _Fixes(LLMClient):
    """Answers with mistyped tests, and with correct ones once told what was wrong."""

    def __init__(self) -> None:
        super().__init__("mock")
        self.prompts: list[str] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.prompts.append(request.prompt)
        fixed = "do not match the signature" in request.prompt
        return LLMResponse(text=json.dumps(GOOD if fixed else BAD), tokens_in=1, tokens_out=1, tokens_estimated=False,
                           model=request.model)


def _with_client(gw: LLMGateway, client: LLMClient) -> LLMGateway:
    gw.clients = {**gw.clients, Role.VALIDATOR: [client]}
    return gw


def _calls(gw: LLMGateway) -> list[dict[str, Any]]:
    return [c.model_dump(mode="json") for c in gw.runlog.read_calls()]


def test_mistyped_answer_is_repaired_and_only_the_fix_is_cached(make_gateway: GatewayFactory) -> None:
    client = _Fixes()
    gw = _with_client(make_gateway(), client)
    out = gw.call(Role.VALIDATOR, CallType.INITIAL, "sys", "write tests", InitialSolution, tag="initial",
                  check=suite_check(SIG))
    assert out.ok and out.data is not None and out.data.tests[0].args == [["a", "b"], 2]
    calls = _calls(gw)
    assert [(c["status"], c.get("repair")) for c in calls] == [("invalid_output", 0), ("ok", 1)]
    assert "argument 'words' must be list[str]" in calls[0]["error"]
    assert "argument 'words' must be list[str]" in client.prompts[1]
    again = gw.call(Role.VALIDATOR, CallType.INITIAL, "sys", "write tests", InitialSolution, tag="initial",
                    check=suite_check(SIG))
    assert again.status is AgentStatus.CACHED and again.data == out.data  # the repaired answer, from the cache


def test_cached_answer_failing_the_check_is_not_reused(make_gateway: GatewayFactory, app_config: AppConfig,
                                                       tmp_path: Path) -> None:
    client = _Fixes()
    gw = _with_client(make_gateway(), client)
    prep = gw._prepare(Role.VALIDATOR, 0, CallType.INITIAL, "sys", "write tests", "initial")
    gw.cache.put(prep.key, CacheEntry(text=json.dumps(BAD), tokens_in=1, tokens_out=1, tokens_estimated=False,
                                      model="m"))
    unchecked = gw.call(Role.VALIDATOR, CallType.INITIAL, "sys", "write tests", InitialSolution, tag="initial")
    assert unchecked.status is AgentStatus.CACHED  # without a check the old answer is still valid
    checked = gw.call(Role.VALIDATOR, CallType.INITIAL, "sys", "write tests", InitialSolution, tag="initial",
                      check=suite_check(SIG))
    assert checked.ok and checked.status is AgentStatus.DONE and len(client.prompts) == 2


# ---------------------------------------------------------------- orchestrator


def test_validator_suite_is_checked_against_the_spec_signature(tmp_path: Path) -> None:
    from quadcode.providers.mock_scenarios import _TESTS, D1_CODE  # the two_sum validator answer

    stringified = [{**t, "args": [json.dumps(a) for a in t["args"]]} if "args" in t and not t.get(
        "expect_exception") else t for t in _TESTS]
    answer = {"language": "python", "code": D1_CODE, "algorithm": "map", "time_complexity": "O(n)",
              "space_complexity": "O(n)", "tests": stringified}
    scen = scenario_with(responses={(Role.VALIDATOR, CallType.INITIAL, WILDCARD): json.dumps(answer)})
    built = build(tmp_path, scenario=scen)
    summary = built.orch.run()
    assert summary.status is RunStatus.UNVERIFIED
    initial = [a for a in summary.availability if a.role is Role.VALIDATOR and a.phase == "initial"]
    assert initial and initial[0].status is AgentStatus.UNAVAILABLE
    assert "do not match the signature" in (initial[0].error or "")
    assert any("do not match the signature" in q.prompt for q in built.provider(Role.VALIDATOR).requests)
    assert not (built.run_dir / "test_suite.json").exists()
