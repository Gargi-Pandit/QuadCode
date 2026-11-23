"""Shared pieces of the dashboard tests: mock runs and a byte-level fingerprint of a directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from quadcode.providers.mock import WILDCARD
from quadcode.schemas import CallType, Role

from .orch_helpers import build, scenario_with

APP = Path(__file__).resolve().parent.parent / "quadcode" / "ui" / "app.py"

HOSTILE_REVIEW: dict[str, Any] = {"items": [{
    "target": "G1", "strengths": ["<script>alert(1)</script>"], "weaknesses": ["a | b", "costs $x$ and :red[y]"],
    "bugs": ["```\n</code></pre><img src=x onerror=alert(2)>"], "suggestions": ["# heading\n- list"],
    "justification": "[x](javascript:alert(3))"}]}


def mock_run(root: Path, run_name: str = "run", hostile: bool = False, **kwargs: Any) -> Path:
    """A finished two_sum mock run under ``root/run_name`` (subprocess sandbox, fake clock)."""
    if hostile:
        kwargs["scenario"] = scenario_with({(Role.REVIEWER, CallType.REVIEW, WILDCARD): json.dumps(HOSTILE_REVIEW)})
    built = build(root, run_name=run_name, **kwargs)
    built.orch.run()
    return built.run_dir


def digest(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root``, by relative path."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
