"""Fail-closed isolated Python executor for MBPP item tests."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    """The host cannot provide the required isolated execution boundary."""


@dataclass(frozen=True)
class SandboxResult:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    runtime_ms: float


class SecurePythonSandbox:
    """Use a disposable network-disabled Linux namespace when available."""

    def __init__(self, timeout_sec: float = 10.0, memory_bytes: int = 1_073_741_824):
        self.timeout_sec = timeout_sec
        self.memory_bytes = memory_bytes
        if os.name != "posix":
            raise SandboxUnavailable("MBPP secure executor requires a POSIX sandbox host")
        self.unshare = shutil.which("unshare")
        if not self.unshare:
            raise SandboxUnavailable("unshare is unavailable; refusing unsandboxed code execution")

    def _limits(self):
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (int(self.timeout_sec), int(self.timeout_sec) + 1))
        resource.setrlimit(resource.RLIMIT_AS, (self.memory_bytes, self.memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        os.setsid()

    def execute(self, code: str) -> SandboxResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="henri_mbpp_") as temp_dir:
            work = Path(temp_dir)
            script = work / "candidate.py"
            script.write_text(code, encoding="utf-8")
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONNOUSERSITE": "1",
                "PYTHONHASHSEED": "0",
                "HOME": str(work),
            }
            command = [
                self.unshare,
                "--user",
                "--map-root-user",
                "--net",
                "--pid",
                "--fork",
                "--mount-proc",
                sys.executable,
                "-I",
                str(script),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=work,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec + 2.0,
                    preexec_fn=self._limits,
                    check=False,
                )
                status = "PASS" if completed.returncode == 0 else "FAIL"
                return SandboxResult(
                    status=status,
                    returncode=completed.returncode,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxResult(
                    status="EXECUTION_ERROR",
                    returncode=None,
                    stdout=exc.stdout or "",
                    stderr=(exc.stderr or "") + "\nTIMEOUT",
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
            except OSError as exc:
                return SandboxResult(
                    status="EXECUTION_ERROR",
                    returncode=None,
                    stdout="",
                    stderr=f"SANDBOX_START_ERROR: {exc}",
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
