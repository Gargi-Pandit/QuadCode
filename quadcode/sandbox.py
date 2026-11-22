"""Sandboxes that execute the in-sandbox runner on untrusted generated code.

Two backends share one interface:
- ``DockerSandbox``: no network, read-only root and workdir, memory/CPU/pids caps,
  unprivileged user, all capabilities dropped. Preferred.
- ``SubprocessSandbox``: fallback when Docker is unusable. Stripped environment,
  temp working dir, rlimits, wall-clock kill of the whole process group. It has NO
  network or filesystem isolation, and every result it produces says so.

API keys never reach either. Both start the interpreter with a fixed environment
(``SANDBOX_ENV``); the container additionally gets a fixed ``PATH`` so ``python`` can
be found. Variables baked into the image or added by Docker (``GPG_KEY``,
``HOSTNAME``, ``HOME``...) are dropped by launching through ``env -i``.
"""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal, Protocol

from quadcode import sandbox_runner
from quadcode.config import SandboxConfig

RUNNER_FILE = "runner.py"
SUBPROCESS_WARNING = (
    "Weaker isolation: Docker was not used, so generated code ran as a local subprocess with "
    "resource limits and an empty environment, but WITHOUT network or filesystem isolation."
)
_STDERR_TAIL = 4_000
_PYTHON_FLAGS = ["-I", "-S", "-B"]
# The complete environment generated code sees. Nothing is inherited from the host or image.
SANDBOX_ENV: dict[str, str] = {"PYTHONHASHSEED": "0", "LANG": "C.UTF-8"}
DOCKER_PATH = "/usr/local/bin:/usr/bin:/bin"
_DOCKER_PROBE_TIMEOUT_S = 10.0
# Extra wall-clock time beyond sandbox.wall_timeout_s for interpreter/container start-up.
WALL_SLACK_S = 2.0


def cpu_limit_s(cfg: SandboxConfig) -> int:
    """RLIMIT_CPU backstop: above the wall-clock limit so the wall clock normally fires first."""
    return math.ceil(cfg.wall_timeout_s + WALL_SLACK_S) + 1


@dataclass(frozen=True)
class ExecResult:
    """What came back from one sandbox process."""

    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr_tail: str
    wall_time_s: float
    output_truncated: bool


class Sandbox(Protocol):
    name: Literal["docker", "subprocess"]
    isolation_warning: str | None

    def execute(self, files: Mapping[str, str], timeout_s: float) -> ExecResult:
        """Run the runner over ``files`` (name -> text) placed in a fresh working dir."""
        ...

    def rlimits(self) -> dict[str, int]:
        """Limits the runner applies to itself before loading the solution."""
        ...


class SandboxUnavailable(RuntimeError):
    """The requested backend cannot be used on this machine."""


def runner_source() -> str:
    return Path(sandbox_runner.__file__).read_text(encoding="utf-8")


# ------------------------------------------------------------ process plumbing


class _CappedReader(threading.Thread):
    """Drain a pipe into memory, keeping at most ``limit`` bytes (head, or tail)."""

    def __init__(self, stream: IO[bytes], limit: int, keep_tail: bool, on_overflow: Callable[[], None] | None) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.limit = limit
        self.keep_tail = keep_tail
        self.on_overflow = on_overflow
        self.data = bytearray()
        self.overflowed = False

    def run(self) -> None:
        while True:
            chunk = self.stream.read(65536)
            if not chunk:
                return
            self.data.extend(chunk)
            if len(self.data) > self.limit:
                if self.keep_tail:
                    del self.data[: len(self.data) - self.limit]
                else:
                    del self.data[self.limit :]
                    if not self.overflowed and self.on_overflow is not None:
                        self.on_overflow()
                self.overflowed = True


def _run_process(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: str | None,
    timeout_s: float,
    max_output_bytes: int,
    new_session: bool,
    kill: Callable[[subprocess.Popen[bytes]], None],
) -> ExecResult:
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env),
        cwd=cwd,
        start_new_session=new_session,
    )
    assert proc.stdout is not None and proc.stderr is not None
    out = _CappedReader(proc.stdout, max_output_bytes, keep_tail=False, on_overflow=lambda: kill(proc))
    err = _CappedReader(proc.stderr, _STDERR_TAIL, keep_tail=True, on_overflow=None)
    out.start()
    err.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill(proc)
        proc.wait()
    out.join()
    err.join()
    wall = time.perf_counter() - t0
    stdout = bytes(out.data).decode("utf-8", errors="replace")
    if out.overflowed:
        # Drop the partial last line; only complete protocol lines are trustworthy.
        stdout = stdout[: stdout.rfind("\n") + 1]
    return ExecResult(
        exit_code=proc.returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr_tail=bytes(err.data).decode("utf-8", errors="replace"),
        wall_time_s=wall,
        output_truncated=out.overflowed,
    )


def _write_workdir(files: Mapping[str, str], world_readable: bool) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="quadcode-sbx-"))
    for name, text in files.items():
        if "/" in name or name.startswith("."):
            raise ValueError(f"invalid sandbox file name: {name!r}")
        (workdir / name).write_text(text, encoding="utf-8")
        if world_readable:
            (workdir / name).chmod(0o644)
    if world_readable:
        workdir.chmod(0o755)
    return workdir


# ------------------------------------------------------------ subprocess backend


class SubprocessSandbox:
    """Fallback: local subprocess with rlimits and a stripped environment."""

    name: Literal["subprocess"] = "subprocess"
    isolation_warning: str | None = SUBPROCESS_WARNING

    def __init__(self, cfg: SandboxConfig, python: str | None = None) -> None:
        if os.name != "posix":
            raise SandboxUnavailable("the subprocess sandbox needs a POSIX system (rlimits, process groups)")
        self.cfg = cfg
        self.python = python or sys.executable

    def rlimits(self) -> dict[str, int]:
        return {
            "as_bytes": self.cfg.memory_mb * 1024 * 1024,
            "cpu_s": cpu_limit_s(self.cfg),
            "fsize_bytes": 16 * 1024 * 1024,
            "nofile": 64,
        }

    def execute(self, files: Mapping[str, str], timeout_s: float) -> ExecResult:
        workdir = _write_workdir(files, world_readable=False)
        try:
            return _run_process(
                [self.python, *_PYTHON_FLAGS, str(workdir / RUNNER_FILE), str(workdir)],
                env=dict(SANDBOX_ENV),
                cwd=str(workdir),
                timeout_s=timeout_s,
                max_output_bytes=self.cfg.max_output_bytes,
                new_session=True,
                kill=_kill_process_group,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ------------------------------------------------------------ docker backend

# Variables the docker CLI itself may need to reach the daemon. Nothing else from the
# orchestrator environment is passed to the CLI, and nothing at all to the container.
_DOCKER_CLI_ENV = ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
                   "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY", "XDG_RUNTIME_DIR")


def _docker_cli_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _DOCKER_CLI_ENV if k in os.environ}


class DockerSandbox:
    """Preferred: throwaway container with no network and a read-only filesystem."""

    name: Literal["docker"] = "docker"
    isolation_warning: str | None = None

    def __init__(self, cfg: SandboxConfig, docker: str = "docker") -> None:
        self.cfg = cfg
        self.docker = docker

    def rlimits(self) -> dict[str, int]:
        # Memory is capped by the container cgroup; RLIMIT_AS would only add noise.
        return {
            "cpu_s": cpu_limit_s(self.cfg),
            "fsize_bytes": 16 * 1024 * 1024,
            "nofile": 64,
        }

    def command(self, workdir: Path, container_name: str) -> list[str]:
        mem = f"{self.cfg.memory_mb}m"
        return [
            self.docker, "run", "--rm",
            "--name", container_name,
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m",
            "--memory", mem,
            "--memory-swap", mem,
            "--cpus", str(self.cfg.cpus),
            "--pids-limit", str(self.cfg.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            "--volume", f"{workdir}:/work:ro",
            "--workdir", "/tmp",
            # `env -i` discards the image's ENV lines and Docker's own variables, so the
            # runner starts with exactly SANDBOX_ENV (+ PATH to locate python).
            "--entrypoint", "env",
            self.cfg.docker_image,
            "-i", f"PATH={DOCKER_PATH}", *(f"{k}={v}" for k, v in SANDBOX_ENV.items()),
            "python", *_PYTHON_FLAGS, f"/work/{RUNNER_FILE}", "/work",
        ]

    def execute(self, files: Mapping[str, str], timeout_s: float) -> ExecResult:
        workdir = _write_workdir(files, world_readable=True)
        container = f"quadcode-{uuid.uuid4().hex[:12]}"

        def kill(proc: subprocess.Popen[bytes]) -> None:
            # Killing the CLI alone would leave the container running.
            subprocess.run([self.docker, "rm", "-f", container], env=_docker_cli_env(),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=_DOCKER_PROBE_TIMEOUT_S, check=False)
            proc.kill()

        try:
            return _run_process(
                self.command(workdir, container),
                env=_docker_cli_env(),
                cwd=None,
                timeout_s=timeout_s,
                max_output_bytes=self.cfg.max_output_bytes,
                new_session=False,
                kill=kill,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


# ------------------------------------------------------------ selection


@dataclass(frozen=True)
class DockerStatus:
    usable: bool
    reason: str


def docker_status(image: str, docker: str = "docker") -> DockerStatus:
    """Check (without pulling anything) whether Docker can run ``image`` right now."""
    if shutil.which(docker) is None:
        return DockerStatus(False, "docker CLI not found on PATH")
    env = _docker_cli_env()
    try:
        info = subprocess.run([docker, "version", "--format", "{{.Server.Version}}"], env=env,
                              capture_output=True, text=True, timeout=_DOCKER_PROBE_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DockerStatus(False, f"docker daemon did not answer: {exc}")
    if info.returncode != 0 or not info.stdout.strip():
        return DockerStatus(False, "docker daemon is not reachable")
    try:
        img = subprocess.run([docker, "image", "inspect", image], env=env, capture_output=True,
                             timeout=_DOCKER_PROBE_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DockerStatus(False, f"docker image check failed: {exc}")
    if img.returncode != 0:
        return DockerStatus(False, f"image {image} is not present locally (run: docker pull {image})")
    return DockerStatus(True, f"docker daemon {info.stdout.strip()} with image {image}")


def select_sandbox(
    cfg: SandboxConfig,
    status_fn: Callable[[str], DockerStatus] = docker_status,
) -> tuple[Sandbox, str]:
    """Pick the backend named in config. Returns (sandbox, human-readable reason).

    ``auto`` falls back to the subprocess sandbox (with its warning); an explicit
    ``docker`` raises ``SandboxUnavailable`` instead of silently degrading.
    """
    if cfg.backend == "subprocess":
        return SubprocessSandbox(cfg), "subprocess backend selected in config"
    status = status_fn(cfg.docker_image)
    if status.usable:
        return DockerSandbox(cfg), status.reason
    if cfg.backend == "docker":
        raise SandboxUnavailable(f"docker backend requested but unusable: {status.reason}")
    return SubprocessSandbox(cfg), f"falling back to subprocess: {status.reason}"
