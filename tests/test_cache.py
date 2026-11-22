from __future__ import annotations

from pathlib import Path

import pytest

from quadcode.cache import CacheEntry, DiskCache, cache_key

BASE = {"provider": "p", "model": "m", "system": "s", "prompt": "q", "params": {"max_tokens": 10, "temperature": 0.2}}
ENTRY = CacheEntry(text='{"a":1}', tokens_in=3, tokens_out=2, tokens_estimated=False, model="m")


def _key(**over: object) -> str:
    d = {**BASE, **over}
    return cache_key(d["provider"], d["model"], d["system"], d["prompt"], d["params"])  # type: ignore[arg-type]


def test_key_is_stable_and_order_independent() -> None:
    assert _key() == _key()
    assert _key(params={"temperature": 0.2, "max_tokens": 10}) == _key()
    assert len(_key()) == 64


@pytest.mark.parametrize("field,value", [
    ("provider", "p2"), ("model", "m2"), ("system", "s2"), ("prompt", "q2"),
    ("params", {"max_tokens": 11, "temperature": 0.2}), ("params", {"max_tokens": 10, "temperature": 0.3}),
])
def test_key_sensitive_to_every_input(field: str, value: object) -> None:
    assert _key(**{field: value}) != _key()


def test_hit_and_miss(tmp_path: Path) -> None:
    c = DiskCache(tmp_path)
    k = _key()
    assert c.get(k) is None
    c.put(k, ENTRY)
    assert c.get(k) == ENTRY
    assert (tmp_path / k[:2] / f"{k}.json").is_file()
    assert not list(tmp_path.rglob(".tmp-*"))  # atomic write leaves no temp files


def test_corrupted_entry_is_a_miss(tmp_path: Path) -> None:
    c = DiskCache(tmp_path)
    k = _key()
    c.put(k, ENTRY)
    (tmp_path / k[:2] / f"{k}.json").write_text("{garbage")
    assert c.get(k) is None


def test_disabled_cache(tmp_path: Path) -> None:
    c = DiskCache(tmp_path, enabled=False)
    c.put(_key(), ENTRY)
    assert c.get(_key()) is None
    assert not any(tmp_path.iterdir())
