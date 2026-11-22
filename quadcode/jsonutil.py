"""Extract a JSON object from LLM output that may contain prose or markdown fences."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n(.*?)```", re.DOTALL)


class JSONExtractionError(ValueError):
    """No valid JSON object could be found in the text."""


def _balanced_objects(text: str) -> Iterator[str]:
    """Yield every top-level balanced ``{...}`` span, honoring JSON string escapes.

    Spans nested inside an earlier span are never yielded: when a whole answer is malformed, a small
    object inside it (e.g. ``{}`` in a code string) is not the answer.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            return
        yield text[start : end + 1]
        start = text.find("{", end + 1)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _candidates(text: str) -> Iterator[str]:
    # Reasoning models may prefix their answer with a <think>...</think> trace; it is not the answer.
    text = _THINK_RE.sub("", text)
    stripped = text.strip()
    yield stripped
    for match in _FENCE_RE.finditer(text):
        yield match.group(1).strip()
    yield from _balanced_objects(text)


def parse_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object found in ``text``.

    Order (after removing <think>...</think> traces): whole text, fenced blocks, then balanced brace
    spans scanning left to right.
    """
    first_error: tuple[str, json.JSONDecodeError] | None = None
    for candidate in _candidates(text):
        if not candidate.startswith("{"):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as err:
            if first_error is None:
                first_error = (candidate, err)
            continue
        if isinstance(value, dict):
            return value
    if first_error is None:
        raise JSONExtractionError("no JSON object found in model output")
    candidate, err = first_error
    near = (candidate[max(0, err.pos - 50): err.pos] + "<<HERE>>" + candidate[err.pos: err.pos + 30])
    near = near.replace("\n", " ")
    raise JSONExtractionError(f"the JSON object in the model output is not valid JSON: {err.msg} at line "
                              f"{err.lineno} column {err.colno}, at <<HERE>> in: {near}")


def parse_model(text: str, model: type[T]) -> T:
    """Extract a JSON object and validate it into ``model``.

    Raises ``JSONExtractionError`` or ``pydantic.ValidationError``.
    """
    return model.model_validate(parse_json_object(text))
