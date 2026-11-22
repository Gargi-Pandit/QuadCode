from __future__ import annotations

import pytest
from pydantic import ValidationError

from quadcode.jsonutil import JSONExtractionError, parse_json_object, parse_model
from quadcode.schemas import ChallengeResponse, Verdict


def test_plain_json() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_fenced_json_with_prose() -> None:
    text = 'Sure! Here you go:\n```json\n{"a": [1, 2]}\n```\nHope that helps.'
    assert parse_json_object(text) == {"a": [1, 2]}


def test_unlabeled_fence() -> None:
    assert parse_json_object('```\n{"b": true}\n```') == {"b": True}


def test_prose_without_fence_and_braces_in_strings() -> None:
    text = 'Answer: {"code": "def f():\\n    return {\\"x\\": 1}", "n": 2} -- done {not json}'
    assert parse_json_object(text) == {"code": 'def f():\n    return {"x": 1}', "n": 2}


def test_skips_invalid_brace_span_before_valid_one() -> None:
    assert parse_json_object('use {braces} like this: {"ok": 1}') == {"ok": 1}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", '{"unterminated": '])
def test_invalid(text: str) -> None:
    with pytest.raises(JSONExtractionError):
        parse_json_object(text)


def test_parse_model() -> None:
    out = parse_model('```json\n{"verdict": "reject", "comment": "fails T03"}\n```', ChallengeResponse)
    assert out.verdict is Verdict.REJECT
    with pytest.raises(ValidationError):
        parse_model('{"verdict": "maybe", "comment": "x"}', ChallengeResponse)
