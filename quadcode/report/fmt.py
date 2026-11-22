"""Formatting helpers shared by the report sections."""

from __future__ import annotations

import json
from typing import Any

from quadcode.report.data import RunRecords
from quadcode.report.doc import Block, Note
from quadcode.schemas import RunStatus

MEASURED = "measured"
THEORETICAL = "theoretical, as claimed by the agent"


def seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def num(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def one_line(text: str | None, limit: int | None = None) -> str:
    flat = " ".join((text or "").split())
    if limit is not None and len(flat) > limit:
        return flat[: limit - 3] + "..."
    return flat


def clip(text: str, limit: int, where: str) -> str:
    """Shorten ``text`` for display, saying where the full value is kept."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [{len(text) - limit} more characters; full value in {where}]"


def value_text(value: Any, limit: int, where: str) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(value)
    return clip(text, limit, where)


def listing(items: list[str] | tuple[str, ...]) -> str:
    return "; ".join(i for i in items if i) or "-"


def stop_text(rec: RunRecords) -> str:
    if rec.summary is None:
        return "no summary.json: the run did not finish writing its summary (it may still be running or crashed)"
    return f"status {rec.summary.status.value}; {rec.summary.stop_reason or 'no stop reason recorded'}"


def not_reached(rec: RunRecords, what: str) -> list[Block]:
    """The standard statement for a part of the run that has no records."""
    return [Note(f"Not reached: {what}. Nothing about it was recorded, so nothing is shown "
                 f"({stop_text(rec)}).", "warning")]


def run_finished_ok(rec: RunRecords) -> bool:
    return rec.summary is not None and rec.summary.status is RunStatus.COMPLETED
