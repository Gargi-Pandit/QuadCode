"""Docker sandbox, end to end. Skipped unless the daemon answers and the image is present.

Enable with: docker pull python:3.12-slim
"""

from __future__ import annotations

import json
import subprocess
from functools import cache
from typing import Any

import pytest

from quadcode.config import SandboxConfig
from quadcode.providers import mock_scenarios
from quadcode.sandbox import DOCKER_PATH, RUNNER_FILE, SANDBOX_ENV, DockerSandbox, DockerStatus, docker_status
from quadcode.schemas import SuiteRun, TestCase, TestStatus
from quadcode.testing import run_suite

from .conftest import SAMPLES

pytestmark = pytest.mark.docker

DOCKER_CFG = SandboxConfig(backend="docker", test_timeout_s=2.0, wall_timeout_s=30.0, test_repeats=2,
                           memory_mb=256)


@cache
def _status() -> DockerStatus:
    return docker_status(DOCKER_CFG.docker_image)


@pytest.fixture
def docker_sandbox() -> DockerSandbox:
    status = _status()
    if not status.usable:
        pytest.skip(f"docker sandbox unavailable: {status.reason}")
    return DockerSandbox(DOCKER_CFG)


def _run(code: str, tests: list[TestCase], sandbox: DockerSandbox, cfg: SandboxConfig = DOCKER_CFG,
         entry: str = "two_sum") -> SuiteRun:
    return run_suite(code=code, entry_point=entry, language="python", tests=tests, sandbox=sandbox,
                     cfg=cfg, version="V")


def _case(id_: str = "X", **kw: Any) -> TestCase:
    return TestCase.model_validate({"id": id_, "category": "adversarial", **kw})


def test_correct_solution_passes_in_container(docker_sandbox: DockerSandbox, two_sum_tests: list[TestCase]) -> None:
    run = _run(mock_scenarios.G1_CODE, two_sum_tests, docker_sandbox)
    assert run.backend == "docker" and run.isolation_warning is None
    assert run.all_passed, [(r.test_id, r.status, r.error) for r in run.results]
    assert run.sandbox_env["python"].startswith("3.12")


def test_network_is_blocked(docker_sandbox: DockerSandbox) -> None:
    code = ("import socket\n\ndef probe():\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=1)\n    return 'connected'\n")
    run = _run(code, [_case(expect_exception="OSError")], docker_sandbox, entry="probe")
    assert run.results[0].status == TestStatus.PASS, run.results[0]


def test_filesystem_is_read_only_except_tmp(docker_sandbox: DockerSandbox) -> None:
    code = ("def probe(path):\n"
            "    with open(path, 'w') as fh:\n        fh.write('x')\n    return 'written'\n")
    tests = [_case("work", args=["/work/evil.py"], expect_exception="OSError"),
             _case("root", args=["/evil"], expect_exception="OSError"),
             _case("tmp", args=["/tmp/ok.txt"], expected="written")]
    run = _run(code, tests, docker_sandbox, entry="probe")
    assert {r.test_id: r.status for r in run.results} == {
        "work": TestStatus.PASS, "root": TestStatus.PASS, "tmp": TestStatus.PASS}


def test_environment_has_no_keys(docker_sandbox: DockerSandbox, two_sum_tests: list[TestCase],
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")
    run = _run((SAMPLES / "bad_env_probe.py").read_text(encoding="utf-8"), two_sum_tests, docker_sandbox)
    assert run.all_passed


def test_container_environment_is_exactly_the_fixed_set(docker_sandbox: DockerSandbox,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing from the host, the image's ENV lines (e.g. GPG_KEY) or Docker (HOSTNAME, HOME)."""
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")
    probe = "import json, os\nprint(json.dumps(dict(os.environ)))\n"
    res = docker_sandbox.execute({RUNNER_FILE: probe}, timeout_s=30)
    assert res.exit_code == 0, res.stderr_tail
    assert json.loads(res.stdout) == {**SANDBOX_ENV, "PATH": DOCKER_PATH}


def test_memory_cap_stops_bomb(docker_sandbox: DockerSandbox) -> None:
    run = _run((SAMPLES / "bad_memory_bomb.py").read_text(encoding="utf-8"),
               [_case("A", args=[[1], 1]), _case("B", args=[[1], 1])], docker_sandbox)
    first = run.results[0]
    # Either Python sees MemoryError or the cgroup OOM killer ends the container.
    assert first.status == TestStatus.ERROR
    assert "MemoryError" in (first.error or "") or "memory limit" in (first.error or "")


def test_wall_timeout_removes_container(docker_sandbox: DockerSandbox) -> None:
    cfg = DOCKER_CFG.model_copy(update={"test_timeout_s": 60.0, "wall_timeout_s": 2.0})
    code = "def two_sum(nums, target):\n    while True:\n        pass\n"
    run = _run(code, [_case(args=[[1], 1])], DockerSandbox(cfg), cfg)
    assert run.timed_out and run.results[0].status == TestStatus.TIMEOUT
    leftover = subprocess.run(["docker", "ps", "-aq", "--filter", "name=quadcode-"],
                              capture_output=True, text=True, check=False)
    assert leftover.stdout.strip() == ""
