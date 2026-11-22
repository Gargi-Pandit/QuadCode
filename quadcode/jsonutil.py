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
    """Yield every top-level balanced ``{...}`` span, honoring JSON string escapes."""
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
        start = text.find("{", start + 1)


def _candidates(text: str) -> Iterator[str]:
    stripped = text.strip()
    yield stripped
    for match in _FENCE_RE.finditer(text):
        yield match.group(1).strip()
    yield from _balanced_objects(text)


def parse_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object found in ``text``.

    Order: whole text, fenced blocks, then balanced brace spans scanning left to right.
    """
    for candidate in _candidates(text):
        if not candidate.startswith("{"):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise JSONExtractionError("no JSON object found in model output")


def parse_model(text: str, model: type[T]) -> T:
    """Extract a JSON object and validate it into ``model``.

    Raises ``JSONExtractionError`` or ``pydantic.ValidationError``.
    """
    return model.model_validate(parse_json_object(text))
