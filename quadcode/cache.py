"""On-disk response cache keyed by sha256(provider + model + prompt + params).

Only successful, schema-valid responses are stored, so a resumed run never repeats
a completed call but always retries failed ones.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    tokens_in: int
    tokens_out: int
    tokens_estimated: bool
    model: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(provider: str, model: str, system: str, prompt: str, params: Mapping[str, Any]) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "system": system,
        "prompt": prompt,
        "params": dict(params),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


class DiskCache:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            return CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError):
            return None

    def put(self, key: str, entry: CacheEntry) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(entry.model_dump_json())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
