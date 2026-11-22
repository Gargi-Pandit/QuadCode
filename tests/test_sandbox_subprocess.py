"""Sandbox backends at the process level, plus backend selection.

These tests replace the runner with tiny probe scripts so each property (environment,
kill on timeout, output cap) is checked directly.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

from quadcode.config import SandboxConfig
from quadcode.sandbox import (
    DOCKER_PATH,
    RUNNER_FILE,
    SANDBOX_ENV,
    SUBPROCESS_WARNING,
    DockerSandbox,
    DockerStatus,
    SandboxUnavailable,
    SubprocessSandbox,
    docker_status,
    select_sandbox,
)


def test_environment_is_stripped(subprocess_sandbox: SubprocessSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-should-never-leak")
    probe = "import json, os, sys\nprint(json.dumps(dict(os.environ)))\nprint(json.dumps(sys.flags.isolated))"
    res = subprocess_sandbox.execute({RUNNER_FILE: probe}, timeout_s=10)
    assert res.exit_code == 0
    env_line, isolated = res.stdout.splitlines()
    env = json.loads(env_line)
    assert "TEST_API_KEY" not in env
    assert "sk-should-never-leak" not in res.stdout
    assert set(env) <= {"PYTHONHASHSEED", "LANG", "LC_CTYPE"}
    assert json.loads(isolated) == 1


def test_wall_clock_timeout_kills_process_group(subprocess_sandbox: SubprocessSandbox) -> None:
    # The child would outlive a plain kill of the parent; the whole group must die.
    probe = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print('started', flush=True)\n"
        "time.sleep(30)\n"
    )
    t0 = time.perf_counter()
    res = subprocess_sandbox.execute({RUNNER_FILE: probe}, timeout_s=1.0)
    assert res.timed_out
    assert time.perf_counter() - t0 < 10  # readers returned, so no process kept the pipes open
    assert "started" in res.stdout


def test_output_cap_truncates_and_kills() -> None:
    cfg = SandboxConfig(backend="subprocess", max_output_bytes=10_000)
    probe = "import sys\nfor i in range(10**7):\n    sys.stdout.write('{\"event\": \"x\"}\\n')\n"
    res = SubprocessSandbox(cfg).execute({RUNNER_FILE: probe}, timeout_s=20)
    assert res.output_truncated
    assert len(res.stdout.encode()) <= 10_000
    assert res.stdout.endswith("\n")  # partial last line dropped


def test_stderr_keeps_only_tail(subprocess_sandbox: SubprocessSandbox) -> None:
    probe = "import sys\nsys.stderr.write('x' * 100000 + 'END')\n"
    res = subprocess_sandbox.execute({RUNNER_FILE: probe}, timeout_s=10)
    assert res.stderr_tail.endswith("END")
    assert len(res.stderr_tail) <= 4000


def test_workdir_is_removed_and_names_validated(subprocess_sandbox: SubprocessSandbox) -> None:
    probe = "import os\nprint(os.getcwd())"
    res = subprocess_sandbox.execute({RUNNER_FILE: probe}, timeout_s=10)
    workdir = Path(res.stdout.strip())
    assert workdir.name.startswith("quadcode-sbx-")
    assert not workdir.exists()
    with pytest.raises(ValueError):
        subprocess_sandbox.execute({"../escape.py": "x"}, timeout_s=5)
    assert not (Path(tempfile.gettempdir()) / "escape.py").exists()


def test_subprocess_rlimits_follow_config() -> None:
    cfg = SandboxConfig(memory_mb=256, wall_timeout_s=30)
    limits = SubprocessSandbox(cfg).rlimits()
    assert limits["as_bytes"] == 256 * 1024 * 1024
    assert limits["cpu_s"] == 33  # wall 30s + 2s start-up slack + 1
    assert SubprocessSandbox(cfg).isolation_warning == SUBPROCESS_WARNING


def test_docker_command_isolation_flags(tmp_path: Path) -> None:
    cfg = SandboxConfig(memory_mb=300, cpus=0.5, pids_limit=32, docker_image="python:3.12-slim")
    cmd = DockerSandbox(cfg).command(tmp_path, "quadcode-test")
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--memory 300m", "--memory-swap 300m", "--cpus 0.5", "--pids-limit 32",
                 "--user 65534:65534", f"{tmp_path}:/work:ro", "--rm", "--name quadcode-test"):
        assert flag in joined
    # No --env flags; the runner is launched via `env -i` so image/Docker variables are dropped.
    assert not [part for part in cmd if part in ("--env", "-e", "--env-file")]
    image_at = cmd.index("python:3.12-slim")
    assert cmd[image_at - 2:image_at] == ["--entrypoint", "env"]
    after_image = cmd[image_at + 1:]
    python_at = after_image.index("python")
    assert after_image[:python_at] == ["-i", f"PATH={DOCKER_PATH}",
                                       *(f"{k}={v}" for k, v in SANDBOX_ENV.items())]
    assert DockerSandbox(cfg).isolation_warning is None
    assert "as_bytes" not in DockerSandbox(cfg).rlimits()


def test_docker_status_without_cli() -> None:
    status = docker_status("python:3.12-slim", docker="quadcode-no-such-docker-binary")
    assert not status.usable
    assert "not found" in status.reason


def _status(usable: bool) -> DockerStatus:
    return DockerStatus(usable, "fake reason")


def test_select_subprocess_never_probes_docker() -> None:
    def boom(image: str) -> DockerStatus:
        raise AssertionError("docker must not be probed")

    sandbox, reason = select_sandbox(SandboxConfig(backend="subprocess"), status_fn=boom)
    assert sandbox.name == "subprocess"
    assert "config" in reason


def test_select_auto_prefers_usable_docker() -> None:
    sandbox, _ = select_sandbox(SandboxConfig(backend="auto"), status_fn=lambda image: _status(True))
    assert sandbox.name == "docker"


def test_select_auto_falls_back_with_reason() -> None:
    sandbox, reason = select_sandbox(SandboxConfig(backend="auto"), status_fn=lambda image: _status(False))
    assert sandbox.name == "subprocess"
    assert "falling back" in reason and "fake reason" in reason
    assert sandbox.isolation_warning


def test_select_forced_docker_does_not_degrade() -> None:
    with pytest.raises(SandboxUnavailable, match="fake reason"):
        select_sandbox(SandboxConfig(backend="docker"), status_fn=lambda image: _status(False))
