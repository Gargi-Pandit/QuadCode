from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import yaml

from quadcode.cli import main
from quadcode.config import ConfigError, Secrets, load_config, load_secrets, parse_config
from quadcode.schemas import Role

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_config_is_valid() -> None:
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert set(cfg.agents) == set(Role)
    rows = cfg.describe_agents()
    assert [r["label"] for r in rows] == ["G1", "C1", "P1", "D1"]
    assert all(r["model"] for r in rows)


def test_valid_dict(config_dict: dict[str, Any]) -> None:
    cfg = parse_config(config_dict)
    assert cfg.provider_for(Role.REVIEWER).kind == "mock"
    assert cfg.agent(Role.OPTIMIZER).model == "mock-optimizer-v1"


def test_missing_role(config_dict: dict[str, Any]) -> None:
    del config_dict["agents"]["validator"]
    with pytest.raises(ConfigError, match="missing agent slots"):
        parse_config(config_dict)


def test_unknown_provider(config_dict: dict[str, Any]) -> None:
    config_dict["agents"]["reviewer"]["provider"] = "nope"
    with pytest.raises(ConfigError, match="unknown provider"):
        parse_config(config_dict)


def test_duplicate_models_rejected_unless_allowed(config_dict: dict[str, Any]) -> None:
    config_dict["agents"]["reviewer"]["model"] = "mock-generator-v1"
    with pytest.raises(ConfigError, match="both list model 'mock-generator-v1'"):
        parse_config(config_dict)
    config_dict["allow_duplicate_models"] = True
    assert parse_config(config_dict).agent(Role.REVIEWER).model == "mock-generator-v1"


@pytest.mark.parametrize("rounds", [0, 6])
def test_rounds_out_of_range(config_dict: dict[str, Any], rounds: int) -> None:
    config_dict["run"]["rounds"] = rounds
    with pytest.raises(ConfigError):
        parse_config(config_dict)


def test_missing_max_tokens(config_dict: dict[str, Any]) -> None:
    del config_dict["max_tokens"]["challenge"]
    with pytest.raises(ConfigError, match="max_tokens"):
        parse_config(config_dict)


def test_real_provider_requires_key_env(config_dict: dict[str, Any]) -> None:
    config_dict["providers"]["or"] = {"kind": "openai_compat", "base_url": "https://x", "rpm": 1, "tpm": 1}
    with pytest.raises(ConfigError, match="api_key_env"):
        parse_config(config_dict)


def test_openai_compat_requires_base_url(config_dict: dict[str, Any]) -> None:
    config_dict["providers"]["or"] = {"kind": "openai_compat", "api_key_env": "K", "rpm": 1, "tpm": 1}
    with pytest.raises(ConfigError, match="base_url"):
        parse_config(config_dict)


def test_unknown_field_rejected(config_dict: dict[str, Any]) -> None:
    config_dict["providers"]["mock"]["api_key"] = "inline-keys-not-allowed"
    with pytest.raises(ConfigError):
        parse_config(config_dict)


def test_load_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_secrets_loaded_only_for_referenced_names(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    config_dict["providers"]["or"] = {
        "kind": "openai_compat", "base_url": "https://x", "api_key_env": "OR_KEY", "rpm": 1, "tpm": 1}
    cfg = parse_config(config_dict)
    env_file = tmp_path / ".env"
    env_file.write_text("OR_KEY=from-file-secret\nUNRELATED=zzz\n")
    secrets = load_secrets(cfg, env_file=env_file, environ={})
    assert secrets.get("OR_KEY") == "from-file-secret"
    assert secrets.get("UNRELATED") is None
    secrets_env = load_secrets(cfg, env_file=env_file, environ={"OR_KEY": "from-env-secret"})
    assert secrets_env.get("OR_KEY") == "from-env-secret"


def test_secrets_never_exposed() -> None:
    s = Secrets({"K": "sk-very-secret-value"})
    assert "sk-very-secret-value" not in repr(s)
    assert "sk-very-secret-value" not in str(s)
    assert s.redact("error: bad key sk-very-secret-value") == "error: bad key <redacted:K>"
    with pytest.raises(TypeError):
        pickle.dumps(s)


def test_cli_check_config(tmp_path: Path, capsys: pytest.CaptureFixture[str], config_dict: dict[str, Any]) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config_dict))
    assert main(["--config", str(path), "--env-file", str(tmp_path / "none"), "check-config"]) == 0
    out = capsys.readouterr().out
    for r in Role:
        assert f"mock-{r.value}-v1" in out
    bad = tmp_path / "bad.yaml"
    config_dict["run"]["rounds"] = 9
    bad.write_text(yaml.safe_dump(config_dict))
    assert main(["--config", str(bad), "check-config"]) == 2


def test_sandbox_defaults_when_block_missing(config_dict: dict[str, Any]) -> None:
    config_dict.pop("sandbox", None)
    cfg = parse_config(config_dict)
    assert cfg.sandbox.backend == "auto"
    assert cfg.sandbox.docker_image == "python:3.12-slim"


def test_sandbox_rejects_unknown_backend(config_dict: dict[str, Any]) -> None:
    config_dict["sandbox"] = {"backend": "chroot"}
    with pytest.raises(ConfigError):
        parse_config(config_dict)


def test_repo_config_sandbox_block() -> None:
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.sandbox.test_timeout_s == 5
    assert cfg.sandbox.benchmark_repeats == 7
