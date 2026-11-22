"""``quadcode models`` and ``quadcode smoke`` against a faked network (httpx transport patched)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from quadcode.cli import main
from quadcode.runlog import read_jsonl

from .conftest import BASE_CONFIG

KEY = "nv-test-SECRET-abcdef123456"
Route = Callable[[httpx.Request], httpx.Response]


def _config(tmp_path: Path) -> Path:
    data: dict[str, Any] = json.loads(json.dumps(BASE_CONFIG))
    data["providers"] = {
        "mock": data["providers"]["mock"],
        "compat": {"kind": "openai_compat", "base_url": "https://compat.test/v1", "api_key_env": "QC_COMPAT_KEY",
                   "rpm": 600, "tpm": 1_000_000, "timeout_s": 5, "max_retries": 3},
        "g": {"kind": "gemini", "api_key_env": "QC_G_KEY", "rpm": 600, "tpm": 1_000_000, "timeout_s": 5},
    }
    data["agents"] = {
        "generator": {"provider": "g", "model": "gemma-test", "inline_system": True},
        "reviewer": {"provider": "compat", "model": "reviewer-model"},
        "optimizer": {"provider": "compat", "model": "optimizer-model",
                      "request_options": {"reasoning_effort": "low"}},
        "validator": {"provider": "mock", "model": "mock-validator-v1"},
    }
    data["run"] = {"runs_dir": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    (tmp_path / "test.env").write_text(f"QC_COMPAT_KEY={KEY}\nQC_G_KEY=g-key\n")
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QC_COMPAT_KEY", raising=False)
    monkeypatch.delenv("QC_G_KEY", raising=False)


def _fake_network(monkeypatch: pytest.MonkeyPatch, route: Route) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handle(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        request.read()
        seen.append(request)
        return route(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    return seen


def _cli(cfg: Path, *args: str, env: str = "test.env") -> list[str]:
    return ["--config", str(cfg), "--env-file", str(cfg.parent / env), *args]


def _chat(text: str, usage: tuple[int, int] = (12, 4)) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}})


def test_smoke_one_request_per_real_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.url.host == "generativelanguage.googleapis.com":
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]},
                                                             "finishReason": "STOP"}],
                                             "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 5}})
        return _chat('```json\n{"ok": true}\n```')

    seen = _fake_network(monkeypatch, route)
    assert main(_cli(_config(tmp_path), "smoke")) == 0
    out = capsys.readouterr().out
    assert len(seen) == 3  # the mock validator slot is skipped
    assert "skipped validator: provider 'mock' is a mock" in out
    assert "g/gemma-test" in out and "compat/reviewer-model" in out and "compat/optimizer-model" in out
    assert out.count(" ok ") >= 3 and "provider" in out
    bodies = [json.loads(r.content) for r in seen]
    assert any(b.get("reasoning_effort") == "low" for b in bodies)
    gemini_body = next(b for b in bodies if "contents" in b)
    assert "systemInstruction" not in gemini_body  # inline_system honored
    [smoke_dir] = (tmp_path / "runs").iterdir()
    records = read_jsonl(smoke_dir / "calls.jsonl")
    assert {r["call_type"] for r in records} == {"smoke"}
    assert all(r["status"] == "ok" and r["tokens_estimated"] is False for r in records)
    assert not (tmp_path / "cache").exists()  # cache off


def test_smoke_failure_is_unavailable_and_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    seen = _fake_network(monkeypatch, lambda r: httpx.Response(503, json={"error": {"message": f"down {KEY}"}}))
    assert main(_cli(_config(tmp_path), "smoke", "--roles", "reviewer")) == 1
    out = capsys.readouterr().out
    assert len(seen) == 1  # max_retries forced to 0
    assert "UNAVAILABLE" in out and "TransientError: HTTP 503" in out
    assert KEY not in out
    [smoke_dir] = (tmp_path / "runs").iterdir()
    assert KEY not in (smoke_dir / "calls.jsonl").read_text()


def test_smoke_checks_each_fallback_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config(tmp_path)
    data = yaml.safe_load(cfg.read_text())
    data["agents"]["reviewer"]["fallbacks"] = [{"provider": "g", "model": "reviewer-backup"}]
    cfg.write_text(yaml.safe_dump(data))

    def route(request: httpx.Request) -> httpx.Response:
        if request.url.host == "generativelanguage.googleapis.com":
            return httpx.Response(429, json={"error": {"message": "Quota exceeded: requests per day"}})
        return _chat('{"ok": true}')

    seen = _fake_network(monkeypatch, route)
    assert main(_cli(cfg, "smoke", "--roles", "reviewer")) == 1
    out = capsys.readouterr().out
    assert len(seen) == 2  # first choice and its one fallback, one request each
    assert "first choice" in out and "fallback 1" in out and "g/reviewer-backup" in out
    assert "daily quota exhausted" in out
    [smoke_dir] = (tmp_path / "runs").iterdir()
    records = read_jsonl(smoke_dir / "calls.jsonl")
    assert [(r["model"], r.get("choice", 0), r["status"]) for r in records] == [
        ("reviewer-model", 0, "ok"), ("reviewer-backup", 1, "provider_error")]


def test_smoke_unexpected_reply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    _fake_network(monkeypatch, lambda r: _chat('{"ok": false}'))
    assert main(_cli(_config(tmp_path), "smoke", "--roles", "optimizer")) == 1
    assert "unexpected reply" in capsys.readouterr().out


def test_smoke_missing_key_sends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    seen = _fake_network(monkeypatch, lambda r: _chat('{"ok": true}'))
    cfg = _config(tmp_path)
    (tmp_path / "partial.env").write_text(f"QC_COMPAT_KEY={KEY}\n")
    assert main(_cli(cfg, "smoke", env="partial.env")) == 2
    assert "missing API key: provider 'g' needs QC_G_KEY" in capsys.readouterr().err
    assert seen == []
    # a role whose key is present can still be checked alone
    assert main(_cli(cfg, "smoke", "--roles", "reviewer", env="partial.env")) == 0
    assert len(seen) == 1


def test_smoke_rejects_unknown_role(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli(_config(tmp_path), "smoke", "--roles", "judge")) == 2
    assert "unknown role 'judge'" in capsys.readouterr().err


def test_smoke_all_mock_has_nothing_to_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli(_config(tmp_path), "smoke", "--roles", "validator")) == 2
    assert "nothing to check" in capsys.readouterr().err


def test_models_lists_and_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    seen = _fake_network(monkeypatch, lambda r: httpx.Response(200, json={"data": [
        {"id": "deepseek-ai/deepseek-v4"}, {"id": "qwen/qwen3-coder"}, {"id": "deepseek-ai/deepseek-r1"}]}))
    assert main(_cli(_config(tmp_path), "models", "compat", "--contains", "DeepSeek")) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v4"]
    assert "2 of 3 models" in captured.err
    assert [r.method for r in seen] == ["GET"] and seen[0].url.path == "/v1/models"
    assert not (tmp_path / "runs").exists()


def test_models_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config(tmp_path)
    assert main(_cli(cfg, "models", "nope")) == 2
    assert "no provider block 'nope'" in capsys.readouterr().err
    _fake_network(monkeypatch, lambda r: httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}}))
    assert main(_cli(cfg, "models", "compat")) == 1
    err = capsys.readouterr().err
    assert "HTTP 401" in err and KEY not in err
    (tmp_path / "empty.env").write_text("")
    assert main(_cli(cfg, "models", "compat", env="empty.env")) == 2
    assert "QC_COMPAT_KEY" in capsys.readouterr().err


def test_models_on_mock_provider(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_cli(_config(tmp_path), "models", "mock")) == 1
    assert "cannot list its models" in capsys.readouterr().err
