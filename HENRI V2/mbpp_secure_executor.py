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


def _launcher_failure(stderr: str) -> bool:
    """True when the launcher (unshare) failed to start the child process.

    A launcher failure is a sandbox infrastructure failure, never a candidate
    task outcome. Candidate tracebacks do not contain the launcher marker.
    """
    lowered = (stderr or "").lower()
    return "unshare" in lowered and (
        "failed" in lowered or "operation not permitted" in lowered
    )


class SecurePythonSandbox:
    """Use a disposable network-disabled Linux namespace when available."""

    def __init__(
        self,
        timeout_sec: float = 10.0,
        memory_bytes: int = 1_073_741_824,
        mode: str = "namespace",
    ):
        """Fail-closed isolated Python executor.

        mode="namespace": requires working unshare user/net/pid namespaces.
            A live constructor probe must succeed, else SandboxUnavailable.
        mode="container-rlimit": explicit surrogate boundary. Runs the
            candidate inside the host container with rlimits only.
            Network isolation is NOT provided. Valid only when the caller
            explicitly accepts the container as the isolation boundary.
        """
        if mode not in ("namespace", "container-rlimit"):
            raise ValueError(f"unknown sandbox mode={mode!r}")
        self.timeout_sec = timeout_sec
        self.memory_bytes = memory_bytes
        self.mode = mode
        if os.name != "posix":
            raise SandboxUnavailable("MBPP secure executor requires a POSIX sandbox host")
        self.unshare = shutil.which("unshare") if mode == "namespace" else None
        if mode == "namespace":
            if not self.unshare:
                raise SandboxUnavailable("unshare is unavailable; refusing unsandboxed code execution")
            self._probe_namespace()

    def _probe_namespace(self) -> None:
        """Prove namespace isolation works before any item runs."""
        probe = [
            self.unshare,
            "--user",
            "--map-root-user",
            "--net",
            "--pid",
            "--fork",
            "--mount-proc",
            sys.executable,
            "-I",
            "-c",
            "pass",
        ]
        try:
            completed = subprocess.run(
                probe,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(f"namespace isolation probe failed: {exc}") from exc
        if completed.returncode != 0:
            raise SandboxUnavailable(
                f"namespace isolation probe failed rc={completed.returncode}: {completed.stderr.strip()}"
            )

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
            if self.mode == "namespace":
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
            else:
                # container-rlimit: explicit surrogate boundary. The host
                # container is the isolation boundary; rlimits bound resource
                # use. Network isolation is NOT provided.
                command = [sys.executable, "-I", str(script)]
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
                if completed.returncode != 0 and _launcher_failure(completed.stderr):
                    return SandboxResult(
                        status="EXECUTION_ERROR",
                        returncode=completed.returncode,
                        stdout=completed.stdout,
                        stderr=f"SANDBOX_LAUNCHER_FAILURE: {completed.stderr.strip()}",
                        runtime_ms=(time.perf_counter() - started) * 1000.0,
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
