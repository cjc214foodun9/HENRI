r"""
Project HENRI V2 -- M3 Isolated Code-Execution Harness (henri_sandbox_harness.py)
=================================================================================
SciCode / Terminal-Bench style evaluation harness: executes *generated* Python
against *published* unit tests inside an isolated executor with a 300 s default
timeout, and returns a typed, comparable result record.

Two execution modes (and one explicit delegation mode)
------------------------------------------------------
1. ``namespace``  -- REAL isolation.  The constructor performs a PROBE that
   actually spawns one namespace-isolated subprocess (``unshare --mount --pid
   --fork --mount-proc --net`` + ``sys.executable``).  Isolation is only
   considered PROVEN when the child reports ``os.getpid() == 1`` inside a fresh
   PID namespace -- i.e. the check verifies the namespace, it does not merely
   verify that a command exited 0.  If the probe cannot spawn, or the child does
   not observe PID 1, the constructor raises :class:`SandboxUnavailable` so the
   caller is BLOCKED rather than silently downgraded.

   PROBE DEFECT GUARD: the probe must never use an *external binary* as its
   check (an old init-script probe used ``echo SUCCESS``; ``/bin/echo`` is an
   external binary whose lookup fails inside a restricted namespace, so the
   probe failed even when the namespace worked).  This probe uses
   ``sys.executable`` directly, and :data:`NAMESPACE_PROBE_SOURCE` contains no
   ``/bin/echo``/``echo`` external call.  A regression test asserts this.

2. ``container-rlimit`` -- an EXPLICIT SURROGATE, never presented as isolation.
   Boundary = ``resource`` rlimits (RLIMIT_AS / RLIMIT_CPU / RLIMIT_NPROC /
   RLIMIT_FSIZE / RLIMIT_CORE) applied in a POSIX ``preexec_fn``, ``setsid``
   (``start_new_session=True``) so the whole tree can be signalled, plus a hard
   wall-clock timeout enforced by the launcher.  There is **NO netns** and no
   mount/PID/UTS namespace: it limits resources, it does not isolate namespaces.
   Every result therefore carries ``surrogate=True`` and ``isolated=False``.
   Where the platform lacks a facility (e.g. Windows has no ``resource``
   module) the shortfall is recorded in ``result.limits`` -- never faked.  On
   Windows a best-effort Job Object (memory cap + kill-on-close) is used
   instead, and whether it was applied is recorded too.

3. ``live-repl`` -- explicit delegation to the live
   :class:`henri_universal_repl.HENRIUniversalREPL` (which enforces its own
   internal 10 s ceiling).  Recorded as a surrogate with the REPL's own limit
   in ``result.limits``.

Live-code reuse
---------------
Execution logic is not duplicated where the repo already provides it:

* ``mbpp_secure_executor.py`` is the established prior art (the only tracked file
  with the ``sandbox-mode`` / ``container-rlimit`` / ``SandboxUnavailable`` /
  ``unshare`` / ``setrlimit`` surface).  This harness reuses its two-mode design,
  its ``--user --map-root-user --net --pid --fork --mount-proc <python> -I``
  launcher flag set (so the PROBE proves exactly the flag set the RUN uses), its
  ``_launcher_failure`` taxonomy and ``SANDBOX_LAUNCHER_FAILURE`` /
  ``SANDBOX_START_ERROR`` stderr markers, its rlimit boundaries
  (CPU / AS / FSIZE / NOFILE / NPROC), its ``setsid`` process-group discipline and
  its minimal child env (``PYTHONNOUSERSITE``, ``PYTHONHASHSEED``).
  :attr:`SandboxResult.legacy_status` bridges to its PASS/FAIL/EXECUTION_ERROR
  vocabulary.
* :func:`henri_code_sanitizer.clean_generated_code` strips markdown fences before
  compilation.
* The wave/veto channel of the live ``HENRIUniversalREPL``
  (``qFHRRUniversalTextTransducer.transduce_text`` + the Sagnac delta +
  ``DualChannelREPLVeto.evaluate_execution``) is reused to compute
  ``sagnac_delta`` / ``repl_q_score`` / ``repl_vetoed`` for every result, and
  ``engine='live-repl'`` delegates execution to
  ``HENRIUniversalREPL.execute_python_repl`` outright.

Torch is imported lazily: the harness still works (telemetry reported as
unavailable) when it is missing.

Status contract
---------------
``status`` in ``{'PASSED','FAILED','EXECUTION_ERROR','TIMEOUT','VETOED'}`` and
precedence is documented in :func:`classify_status`.  INVARIANT: anything that
prevents the candidate from *starting* (launcher failure, missing interpreter,
unreadable/syntax-broken program, missing dependency) maps to
``EXECUTION_ERROR`` (or raises) -- never to a task ``FAILED``.  A ``FAILED``
means "the program ran and did not satisfy the published tests".

Exit-code preservation
----------------------
The candidate is launched directly (no shell pipeline, no output filter in the
process chain), so ``result.returncode`` is the producer's own exit code.
Output truncation happens after collection and never touches ``returncode``.

Arithmetic invariant
--------------------
``SuiteReport.attempted == passed + failed``; ``passed`` counts ``PASSED`` and
``failed`` counts every non-passing status (``FAILED``, ``EXECUTION_ERROR``,
``TIMEOUT``, ``VETOED``).  Per-status counts are also reported so the arithmetic
is auditable.  :meth:`SuiteReport.assert_arithmetic` raises on violation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "STATUSES",
    "DEFAULT_TIMEOUT_S",
    "EXECUTION_MODES",
    "SURROGATE_MODE",
    "MODE_BOUNDARIES",
    "SandboxError",
    "SandboxUnavailable",
    "SandboxResult",
    "SuiteReport",
    "NamespaceProbe",
    "SandboxHarness",
    "classify_status",
    "clean_code_for_exec",
    "PROBE_STATS",
    "NAMESPACE_PROBE_SOURCE",
]

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

STATUS_PASSED = "PASSED"
STATUS_FAILED = "FAILED"
STATUS_EXECUTION_ERROR = "EXECUTION_ERROR"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_VETOED = "VETOED"

STATUSES: Tuple[str, ...] = (
    STATUS_PASSED,
    STATUS_FAILED,
    STATUS_EXECUTION_ERROR,
    STATUS_TIMEOUT,
    STATUS_VETOED,
)
_NONPASS_STATUSES = (STATUS_FAILED, STATUS_EXECUTION_ERROR, STATUS_TIMEOUT, STATUS_VETOED)

#: Bridge to the prior art's three-value vocabulary (mbpp_secure_executor.py),
#: which uses PASS / FAIL / EXECUTION_ERROR. TIMEOUT and VETOED are refinements
#: that the prior art collapsed into EXECUTION_ERROR; here they stay distinct,
#: and both are non-passing for the arithmetic invariant.
LEGACY_STATUS_MAP: Dict[str, str] = {
    STATUS_PASSED: "PASS",
    STATUS_FAILED: "FAIL",
    STATUS_EXECUTION_ERROR: "EXECUTION_ERROR",
    STATUS_TIMEOUT: "EXECUTION_ERROR",
    STATUS_VETOED: "EXECUTION_ERROR",
}

DEFAULT_TIMEOUT_S: float = 300.0
SURROGATE_MODE = "container-rlimit"
EXECUTION_MODES: Tuple[str, ...] = ("namespace", SURROGATE_MODE, "live-repl")

# Child exit-code protocol (contract between the driver and the launcher).
EXIT_OK = 0
EXIT_TEST_FAILED = 3
EXIT_EXEC_ERROR = 4

#: Absolute truth about what each mode does and does not provide.  The surrogate
#: entry states the limitation explicitly (rlimits only, NO netns).
MODE_BOUNDARIES: Dict[str, Dict[str, Any]] = {
    "namespace": {
        "isolated": True,
        "surrogate": False,
        "netns": True,
        "pidns": True,
        "mountns": True,
        "rlimits": False,
        "proven_by": "constructor probe spawning one namespace-isolated subprocess",
        "note": "real boundary: unshare namespaces; unavailable => SandboxUnavailable",
    },
    SURROGATE_MODE: {
        "isolated": False,
        "surrogate": True,
        "netns": False,
        "pidns": False,
        "mountns": False,
        "rlimits": True,
        "proven_by": "resource limits + setsid + per-call timeout",
        "note": "SURROGATE BOUNDARY: rlimits only, NO netns; not isolation",
    },
    "live-repl": {
        "isolated": False,
        "surrogate": True,
        "netns": False,
        "pidns": False,
        "mountns": False,
        "rlimits": False,
        "proven_by": "henri_universal_repl.HENRIUniversalREPL.execute_python_repl",
        "note": "delegated execution; REPL internal 10s ceiling; not isolation",
    },
}

#: The probe body.  Runs INSIDE the namespace.  Uses sys.executable (never an
#: external binary such as /bin/echo) and proves the PID namespace by reporting
#: getpid(), which must be 1 for the probe to count as proof.
NAMESPACE_PROBE_SOURCE = "import os,sys; sys.stdout.write('HENRI_NS_PROBE pid=%d\\n' % os.getpid())"
NAMESPACE_PROBE_MARKER = "HENRI_NS_PROBE"

PROBE_STATS: Dict[str, int] = {"spawns": 0, "attempts": 0, "proven": 0}

_SYNTAX_RE = re.compile(r"\b(SyntaxError|IndentationError|TabError)\b")
_ENV_RE = re.compile(r"\b(ModuleNotFoundError|ImportError)\b: ")

# --- launcher-failure taxonomy, reused verbatim from mbpp_secure_executor.py ---
#: stderr prefixes. A launcher failure is a sandbox infrastructure failure, never
#: a candidate task outcome.
SANDBOX_LAUNCHER_FAILURE = "SANDBOX_LAUNCHER_FAILURE"
SANDBOX_START_ERROR = "SANDBOX_START_ERROR"


def _launcher_failure(stderr: str) -> bool:
    """True when the launcher (unshare) failed to start the child process.

    Reused from ``mbpp_secure_executor._launcher_failure`` (the established prior
    art) so both executors share one taxonomy: candidate tracebacks do not carry
    the launcher marker, so this can never misfire on model output.
    """
    lowered = (stderr or "").lower()
    return "unshare" in lowered and ("failed" in lowered or "operation not permitted" in lowered)


class SandboxError(RuntimeError):
    """Base class for harness failures."""


class SandboxUnavailable(SandboxError):
    """Typed failure: the requested isolation mode cannot be provided.

    Raised by the constructor when ``mode='namespace'`` and the namespace probe
    cannot spawn a proven namespace-isolated subprocess.  The caller is BLOCKED
    (loudly), never silently downgraded to the surrogate boundary.
    """

    def __init__(
        self,
        mode: str,
        reason: str,
        *,
        attempts: Sequence[Dict[str, Any]] = (),
        probe_spawns: int = 0,
    ) -> None:
        self.mode = mode
        self.reason = reason
        self.attempts = list(attempts)
        self.probe_spawns = probe_spawns
        super().__init__(f"[SandboxUnavailable:{mode}] {reason}")


# ---------------------------------------------------------------------------
# Code sanitisation (reuse live sanitizer; local fallback keeps import-safe)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)\n?```", re.DOTALL)
#: Language tags the live sanitizer's regex does not recognise (```py,
#: ```python3, ```text, ...).  They are normalised to bare fences so the live
#: sanitizer stays the single fence-extraction authority.
_FENCE_TAG_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+#.-]+)[ \t]*\r?\n")


def _fallback_clean(text: str) -> str:
    if not text:
        return ""
    matches = _FENCE_RE.findall(text)
    if matches:
        return matches[0].strip()
    return text.replace("```python", "").replace("```", "").strip()


try:  # pragma: no cover - exercised implicitly
    from henri_code_sanitizer import clean_generated_code as _live_clean_generated_code

    _SANITIZER_SOURCE = "henri_code_sanitizer.clean_generated_code"
except Exception:  # pragma: no cover
    _live_clean_generated_code = None
    _SANITIZER_SOURCE = "henri_sandbox_harness._fallback_clean"


def clean_code_for_exec(text: str) -> str:
    """Strip markdown code fences before compilation.

    Raw model output routinely carries `````python`` fences, which produce a
    100% SyntaxError rate if compiled directly.  Delegates to the live
    ``henri_code_sanitizer.clean_generated_code``; language tags that live
    sanitizer does not match (``py``, ``python3``, ``text``, ...) are normalised
    to bare fences first so it remains the single extraction authority.
    """
    if not text:
        return ""
    normalised = _FENCE_TAG_RE.sub("```\n", text)
    if _live_clean_generated_code is not None:
        return _live_clean_generated_code(normalised)
    return _fallback_clean(normalised)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

_REQUIRED_CONTRACT_KEYS = (
    "status",
    "returncode",
    "stdout",
    "stderr",
    "elapsed_ms",
    "mode",
    "isolated",
)


@dataclass(frozen=True)
class SandboxResult:
    """Typed result of one evaluation.

    Required contract keys: ``status, returncode, stdout, stderr, elapsed_ms,
    mode, isolated`` (see :meth:`as_contract`).  Additional provenance fields
    (``surrogate``, ``limits``, ``launcher_error``, ...) are additive and always
    populated -- ``mode`` is recorded on EVERY result, and ``isolated`` is True
    only when real isolation was proven.
    """

    status: str
    returncode: Optional[int]
    stdout: str
    stderr: str
    elapsed_ms: float
    mode: str
    isolated: bool
    # --- additive provenance ------------------------------------------------
    surrogate: bool = True
    killed: bool = False
    timed_out: bool = False
    launcher_error: Optional[str] = None
    returncode_preserved: bool = True
    limits: Dict[str, Any] = field(default_factory=dict)
    status_reason: str = ""
    tests_provided: bool = False
    code_sha256: str = ""
    repl_telemetry: bool = False
    sagnac_delta: Optional[float] = None
    repl_q_score: Optional[float] = None
    repl_vetoed: Optional[bool] = None
    workspace: Optional[str] = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise SandboxError(f"invalid status {self.status!r}; must be one of {STATUSES}")
        if self.mode not in EXECUTION_MODES:
            raise SandboxError(f"invalid mode {self.mode!r}; must be one of {EXECUTION_MODES}")
        if self.status == STATUS_TIMEOUT and not self.killed:
            raise SandboxError("TIMEOUT requires killed=True (we must have killed the tree)")
        if self.status == STATUS_PASSED and self.returncode != EXIT_OK:
            raise SandboxError("PASSED requires returncode 0")
        if (
            self.status == STATUS_EXECUTION_ERROR
            and self.returncode is None
            and not self.launcher_error
        ):
            raise SandboxError(
                "EXECUTION_ERROR with no returncode must record launcher_error "
                "(a launcher failure is infrastructure evidence, not a silent gap)"
            )

    def as_contract(self) -> Dict[str, Any]:
        """Return exactly the seven required contract keys."""
        return {
            "status": self.status,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_ms": self.elapsed_ms,
            "mode": self.mode,
            "isolated": self.isolated,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_ms": self.elapsed_ms,
            "mode": self.mode,
            "isolated": self.isolated,
            "surrogate": self.surrogate,
            "killed": self.killed,
            "timed_out": self.timed_out,
            "launcher_error": self.launcher_error,
            "returncode_preserved": self.returncode_preserved,
            "limits": dict(self.limits),
            "status_reason": self.status_reason,
            "tests_provided": self.tests_provided,
            "code_sha256": self.code_sha256,
            "repl_telemetry": self.repl_telemetry,
            "sagnac_delta": self.sagnac_delta,
            "repl_q_score": self.repl_q_score,
            "repl_vetoed": self.repl_vetoed,
            "label": self.label,
        }

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASSED

    # -- interop with the prior art's vocabulary (mbpp_secure_executor) ------
    @property
    def legacy_status(self) -> str:
        """PASS/FAIL/EXECUTION_ERROR vocabulary used by mbpp_secure_executor.py."""
        return LEGACY_STATUS_MAP[self.status]

    @property
    def runtime_ms(self) -> float:
        """Alias for ``elapsed_ms`` (prior-art field name)."""
        return self.elapsed_ms


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

def classify_status(
    returncode: Optional[int],
    stderr: str,
    *,
    killed: bool = False,
    launcher_error: Optional[str] = None,
    vetoed: bool = False,
) -> Tuple[str, str]:
    """Map a raw execution outcome to ``(status, reason)``.

    Precedence (documented, tested):
      1. ``VETOED``           - policy refused execution (never spawned).
      2. ``EXECUTION_ERROR``  - the launcher could not start the subprocess, or
                                the program could not be compiled/loaded
                                (SyntaxError/IndentationError/ImportError):
                                the candidate never got a fair run => NEVER FAILED.
      3. ``TIMEOUT``          - we killed it at the wall-clock limit.
      4. ``EXECUTION_ERROR``  - child reported the exec-error protocol code.
      5. ``FAILED``           - child reported the test-failure protocol code.
      6. ``PASSED``           - returncode 0 (all published tests satisfied).
      7. ``FAILED``           - anything else: the program ran and did not pass.
    """
    if vetoed:
        return STATUS_VETOED, "pre-execution policy veto"
    if launcher_error:
        return STATUS_EXECUTION_ERROR, (
            f"{SANDBOX_START_ERROR}: launcher could not start subprocess: {launcher_error}"
        )
    if killed:
        return STATUS_TIMEOUT, "wall-clock timeout exceeded; process tree killed"
    if _launcher_failure(stderr or ""):
        # Shared taxonomy with mbpp_secure_executor: a launcher (unshare) failure
        # is a sandbox infrastructure failure, NEVER a candidate task outcome.
        return STATUS_EXECUTION_ERROR, (
            f"{SANDBOX_LAUNCHER_FAILURE}: namespace launcher failed to start the child"
        )
    if returncode == EXIT_OK:
        return STATUS_PASSED, "returncode 0"
    if returncode == EXIT_EXEC_ERROR:
        return STATUS_EXECUTION_ERROR, "child reported execution/compile error (exit 4)"
    if returncode == EXIT_TEST_FAILED:
        return STATUS_FAILED, "published tests failed (exit 3)"
    if _SYNTAX_RE.search(stderr or ""):
        kind = _SYNTAX_RE.search(stderr or "").group(1)
        return STATUS_EXECUTION_ERROR, f"{kind} in candidate/tests (never a task FAIL)"
    if _ENV_RE.search(stderr or ""):
        return STATUS_EXECUTION_ERROR, "unresolved import at load time => environment/launch class"
    if returncode is None:
        return STATUS_EXECUTION_ERROR, "no returncode and no launcher error recorded"
    return STATUS_FAILED, f"program ran and exited {returncode} without satisfying tests"


# ---------------------------------------------------------------------------
# Namespace probe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NamespaceProbe:
    """Evidence that the namespace probe did (or did not) prove isolation."""

    available: bool
    reason: str
    command: Tuple[str, ...] = ()
    isolation_proof: str = ""
    attempts: Tuple[Dict[str, Any], ...] = ()
    spawns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "command": list(self.command),
            "isolation_proof": self.isolation_proof,
            "attempts": [dict(a) for a in self.attempts],
            "spawns": self.spawns,
        }


_PROBE_CACHE: Dict[Tuple[Any, ...], NamespaceProbe] = {}


def _namespace_flags() -> List[str]:
    """Namespace flag set, aligned with the prior art (mbpp_secure_executor.py).

    ``--user --map-root-user`` is what makes unprivileged namespace creation
    possible in the first place, so it must be part of the PROBE flag set as
    well as the run flag set -- otherwise the probe proves a namespace the
    executor cannot actually enter.
    """
    return ["--user", "--map-root-user", "--net", "--pid", "--fork", "--mount-proc"]


def _probe_candidates(allow_wsl: bool) -> List[Tuple[str, List[str]]]:
    py = sys.executable
    direct = shutil.which("unshare") or "/usr/bin/unshare"
    flags = _namespace_flags()
    cands: List[Tuple[str, List[str]]] = [
        (
            "unshare-direct",
            [direct] + flags + [py, "-I", "-c", NAMESPACE_PROBE_SOURCE],
        )
    ]
    bash = shutil.which("bash")
    if bash:
        # Same sys.executable check, but resolved by a POSIX shell (relevant on
        # hosts where unshare only exists inside a bash environment).
        cands.append(
            (
                "unshare-via-bash",
                [
                    bash,
                    "-c",
                    'exec unshare '
                    + " ".join(flags)
                    + ' "$1" -I -c "$2"',
                    "henri-ns-probe",
                    py,
                    NAMESPACE_PROBE_SOURCE,
                ],
            )
        )
    if allow_wsl:
        wsl = shutil.which("wsl") or shutil.which("wsl.exe")
        if wsl:
            cands.append(
                (
                    "unshare-via-wsl",
                    [wsl, "-e", "unshare"] + flags + ["python3", "-I", "-c", NAMESPACE_PROBE_SOURCE],
                )
            )
    return cands


def probe_namespace(
    *,
    timeout_s: float = 20.0,
    allow_wsl: bool = False,
    force: bool = False,
    cache: bool = True,
) -> NamespaceProbe:
    """ACTUALLY spawn one namespace-isolated subprocess to prove isolation.

    Returns a :class:`NamespaceProbe` whose ``available`` is True only when a
    candidate spawn succeeded, the marker was observed, and the child reported
    ``getpid() == 1`` (proof the PID namespace was really entered).
    """
    key = (sys.executable, allow_wsl, tuple(_namespace_flags()))
    if cache and not force and key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    attempts: List[Dict[str, Any]] = []
    spawns_before = PROBE_STATS["spawns"]
    result: Optional[NamespaceProbe] = None

    for label, cmd in _probe_candidates(allow_wsl):
        attempt: Dict[str, Any] = {"label": label, "command": list(cmd)}
        PROBE_STATS["attempts"] += 1
        try:
            PROBE_STATS["spawns"] += 1
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
            )
            attempt["returncode"] = proc.returncode
            attempt["stdout"] = proc.stdout
            attempt["stderr"] = (proc.stderr or "").strip()
            out = proc.stdout or ""
            ok_marker = NAMESPACE_PROBE_MARKER in out
            pid_val: Optional[int] = None
            m = re.search(re.escape(NAMESPACE_PROBE_MARKER) + r" pid=(\d+)", out)
            if m:
                pid_val = int(m.group(1))
            attempt["child_pid"] = pid_val
            if ok_marker and pid_val == 1 and proc.returncode == 0:
                attempt["proven"] = True
                PROBE_STATS["proven"] += 1
                result = NamespaceProbe(
                    available=True,
                    reason="namespace probe spawned an isolated subprocess with getpid()==1",
                    command=tuple(cmd),
                    isolation_proof=f"child reported {NAMESPACE_PROBE_MARKER} pid=1 (fresh PID namespace)",
                    attempts=tuple(attempts + [attempt]),
                    spawns=PROBE_STATS["spawns"] - spawns_before,
                )
                break
            if ok_marker and pid_val != 1:
                attempt["proven"] = False
                attempt["failure"] = (
                    f"marker seen but child pid={pid_val} (expected 1): namespace not proven"
                )
            elif proc.returncode != 0:
                attempt["failure"] = (
                    f"probe exit {proc.returncode}: {(proc.stderr or '').strip().splitlines()[-1] if (proc.stderr or '').strip() else 'no stderr'}"
                )
            else:
                attempt["failure"] = f"probe emitted no {NAMESPACE_PROBE_MARKER} marker"
        except FileNotFoundError as exc:
            attempt["returncode"] = None
            attempt["error"] = f"FileNotFoundError: {exc}"
            attempt["failure"] = f"cannot spawn {cmd[0]!r}: {exc}"
        except PermissionError as exc:
            attempt["returncode"] = None
            attempt["error"] = f"PermissionError: {exc}"
            attempt["failure"] = f"cannot spawn {cmd[0]!r}: {exc}"
        except subprocess.TimeoutExpired:
            attempt["returncode"] = None
            attempt["error"] = f"TimeoutExpired after {timeout_s}s"
            attempt["failure"] = f"probe spawn timed out after {timeout_s}s"
        except OSError as exc:
            attempt["returncode"] = None
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempt["failure"] = f"OSError spawning probe: {exc}"
        attempts.append(attempt)

    if result is None:
        summaries = []
        for a in attempts:
            rc = a.get("returncode")
            detail = a.get("failure") or a.get("error") or "unknown"
            summaries.append(f"{a['label']} rc={rc}: {detail}")
        reason = "namespace isolation NOT PROVEN: " + " | ".join(summaries)
        result = NamespaceProbe(
            available=False,
            reason=reason,
            command=tuple(_probe_candidates(allow_wsl)[0][1]) if attempts else (),
            attempts=tuple(attempts),
            spawns=PROBE_STATS["spawns"] - spawns_before,
        )

    if cache:
        _PROBE_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Windows Job Object limiter (best effort; presence is recorded, never assumed)
# ---------------------------------------------------------------------------

class _WindowsJobLimiter:
    """Best-effort hard resource boundary via a Win32 Job Object.

    Applied on top of the surrogate mode where ``resource`` is unavailable.
    Everything is guarded: any failure yields ``applied=False`` with a reason,
    and the reason is surfaced in ``result.limits``.
    """

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _LIFT_PROCESS_MEMORY = 0x00000100
    _LIFT_JOB_MEMORY = 0x00000200
    _LIFT_ACTIVE_PROCESS = 0x00000008
    _LIFT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    _LIFT_KILL_ON_JOB_CLOSE = 0x00002000

    def __init__(self, mem_limit_mb: Optional[int] = None, active_process_limit: Optional[int] = None):
        self.mem_limit_mb = mem_limit_mb
        self.active_process_limit = active_process_limit
        self.applied = False
        self.reason = "not attempted"
        self._handle = None
        self._k32 = None

    def create(self) -> bool:
        try:
            import ctypes
            from ctypes import wintypes

            class _IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in
                            ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                             "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class _BASIC(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class _EXT(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _BASIC),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = ctypes.c_void_p
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
            handle = k32.CreateJobObjectW(None, None)
            if not handle:
                self.reason = f"CreateJobObjectW failed err={ctypes.get_last_error()}"
                return False

            info = _EXT()
            flags = self._LIFT_KILL_ON_JOB_CLOSE | self._LIFT_DIE_ON_UNHANDLED_EXCEPTION
            if self.mem_limit_mb:
                info.ProcessMemoryLimit = int(self.mem_limit_mb) * 1024 * 1024
                flags |= self._LIFT_PROCESS_MEMORY
            if self.active_process_limit:
                info.BasicLimitInformation.ActiveProcessLimit = int(self.active_process_limit)
                flags |= self._LIFT_ACTIVE_PROCESS
            info.BasicLimitInformation.LimitFlags = flags
            if not k32.SetInformationJobObject(
                ctypes.c_void_p(handle), self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info),
            ):
                self.reason = f"SetInformationJobObject failed err={ctypes.get_last_error()}"
                k32.CloseHandle(ctypes.c_void_p(handle))
                return False
            self._handle = handle
            self._k32 = k32
            self.applied = True
            self.reason = "job object created (memory cap + kill-on-close)"
            return True
        except Exception as exc:  # pragma: no cover - platform dependent
            self.reason = f"{type(exc).__name__}: {exc}"
            return False

    def assign(self, pid: int) -> bool:
        if not self._handle or not self._k32:
            return False
        try:
            import ctypes

            PROCESS_SET_QUOTA = 0x0100
            PROCESS_TERMINATE = 0x0001
            h = self._k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, int(pid))
            if not h:
                self.reason = f"OpenProcess failed err={ctypes.get_last_error()}"
                return False
            ok = bool(self._k32.AssignProcessToJobObject(ctypes.c_void_p(self._handle), ctypes.c_void_p(h)))
            self._k32.CloseHandle(ctypes.c_void_p(h))
            if not ok:
                self.reason = f"AssignProcessToJobObject failed err={ctypes.get_last_error()}"
                return False
            self.reason = "job object assigned to child"
            return True
        except Exception as exc:  # pragma: no cover
            self.reason = f"{type(exc).__name__}: {exc}"
            return False

    def close(self) -> None:
        if self._handle and self._k32:
            try:
                self._k32.CloseHandle.__call__  # type: ignore[attr-defined]
                self._k32.CloseHandle(self._handle)
            except Exception:
                pass
            self._handle = None


# ---------------------------------------------------------------------------
# Child driver (executed INSIDE the sandbox)
# ---------------------------------------------------------------------------

_DRIVER_TEMPLATE = r'''
import os, sys, types, unittest, traceback

SOLUTION = {solution!r}
TESTS = {tests!r}
WORKSPACE = {workspace!r}
EXIT_TEST_FAILED = {exit_test_failed}
EXIT_EXEC_ERROR = {exit_exec_error}

sys.path.insert(0, WORKSPACE)


def _bail(message, code):
    sys.stderr.write(message if message.endswith("\n") else message + "\n")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(code)


# ---- stage 1: load the candidate solution ---------------------------------
try:
    with open(SOLUTION, "r", encoding="utf-8", errors="replace") as fh:
        sol_src = fh.read()
except OSError as exc:
    _bail("HENRI_DRIVER: cannot read solution source: %r" % (exc,), EXIT_EXEC_ERROR)

try:
    sol_code = compile(sol_src, SOLUTION, "exec")
except SyntaxError:
    _bail(traceback.format_exc(), EXIT_EXEC_ERROR)

sol_mod = types.ModuleType("solution")
sol_mod.__file__ = SOLUTION
sys.modules["solution"] = sol_mod
for _p in os.path.dirname(SOLUTION), WORKSPACE:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    exec(sol_code, sol_mod.__dict__)
except SystemExit as exc:
    code = exc.code
    code = code if isinstance(code, int) else 1
    del sys.modules["solution"]
    _bail("HENRI_DRIVER: candidate called sys.exit(%r) during load" % (exc.code,), EXIT_EXEC_ERROR if code else code)
except BaseException:
    _bail(traceback.format_exc(), EXIT_EXEC_ERROR)

# ---- stage 2: run the published unit tests --------------------------------
try:
    with open(TESTS, "r", encoding="utf-8", errors="replace") as fh:
        test_src = fh.read()
except OSError as exc:
    _bail("HENRI_DRIVER: cannot read published tests: %r" % (exc,), EXIT_EXEC_ERROR)

try:
    test_code = compile(test_src, TESTS, "exec")
except SyntaxError:
    _bail(traceback.format_exc(), EXIT_EXEC_ERROR)

test_mod = types.ModuleType("published_tests")
test_mod.__file__ = TESTS
test_mod.__dict__.update(sol_mod.__dict__)
test_mod.__dict__["__name__"] = "published_tests"
sys.modules["published_tests"] = test_mod

try:
    exec(test_code, test_mod.__dict__)
except SystemExit as exc:
    code = exc.code
    code = code if isinstance(code, int) else 1
    _bail("HENRI_DRIVER: published tests called sys.exit(%r)" % (exc.code,), EXIT_TEST_FAILED if code else code)
except BaseException:
    sys.stderr.write(traceback.format_exc())
    _bail("HENRI_DRIVER: published test raised", EXIT_TEST_FAILED)

# unittest.TestCase classes are supported in addition to plain asserts.
suite = unittest.TestLoader().loadTestsFromModule(test_mod)
if suite.countTestCases():
    _res = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
    if not _res.wasSuccessful():
        _bail("HENRI_DRIVER: unittest suite failed", EXIT_TEST_FAILED)

sys.stdout.flush()
sys.stderr.flush()
sys.exit(0)
'''


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@dataclass
class SuiteReport:
    """Batch report with the audited arithmetic invariant."""

    results: List[SandboxResult] = field(default_factory=list)

    @property
    def attempted(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status != STATUS_PASSED)

    @property
    def status_counts(self) -> Dict[str, int]:
        counts = {s: 0 for s in STATUSES}
        for r in self.results:
            counts[r.status] += 1
        return counts

    def arithmetic_ok(self) -> bool:
        return self.attempted == self.passed + self.failed

    def assert_arithmetic(self) -> None:
        if not self.arithmetic_ok():
            raise SandboxError(
                f"arithmetic violation: attempted={self.attempted} "
                f"!= passed={self.passed} + failed={self.failed}"
            )
        if self.attempted != sum(self.status_counts.values()):
            raise SandboxError("arithmetic violation: status counts do not sum to attempted")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempted": self.attempted,
            "passed": self.passed,
            "failed": self.failed,
            "status_counts": self.status_counts,
            "arithmetic_ok": self.arithmetic_ok(),
            "results": [r.to_dict() for r in self.results],
        }


class SandboxHarness:
    """Execute generated Python against published unit tests, isolated, timed.

    Parameters
    ----------
    mode:
        ``'auto'`` (default) tries ``'namespace'`` and, if the probe fails,
        explicitly downgrades to the ``'container-rlimit'`` SURROGATE while
        recording ``downgrade_reason`` (never silent: the surrogate flag appears
        on every result).  ``'namespace'`` raises :class:`SandboxUnavailable`
        in the CONSTRUCTOR when the probe cannot prove isolation -- the caller
        is blocked.  ``'container-rlimit'`` / ``'live-repl'`` select the
        surrogate / delegated engines directly.
    timeout_s:
        Hard wall-clock ceiling per evaluation.  Default :data:`DEFAULT_TIMEOUT_S`
        = 300 s.
    """

    def __init__(
        self,
        mode: str = "auto",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        *,
        allow_surrogate_fallback: bool = True,
        allow_wsl: bool = False,
        probe_cache: bool = True,
        force_probe: bool = False,
        engine: str = "harness",
        python_executable: Optional[str] = None,
        mem_limit_mb: Optional[int] = 1024,
        cpu_limit_s: Optional[float] = None,
        active_process_limit: Optional[int] = None,
        max_output_chars: int = 200_000,
        child_env_extra: Optional[Dict[str, str]] = None,
        forbidden_patterns: Sequence[str] = (),
        veto_policy: str = "static",
        enable_repl_telemetry: bool = True,
        repl_d_model: int = 1024,
        keep_workspace: bool = False,
    ) -> None:
        if mode not in ("auto",) + EXECUTION_MODES:
            raise SandboxError(f"unknown mode {mode!r}; expected 'auto' or one of {EXECUTION_MODES}")
        if engine not in ("harness", "live-repl"):
            raise SandboxError(f"unknown engine {engine!r}")
        if veto_policy not in ("static", "repl", "none"):
            raise SandboxError(f"unknown veto_policy {veto_policy!r}")

        self.requested_mode = mode
        self.timeout_s = float(timeout_s)
        self.allow_surrogate_fallback = allow_surrogate_fallback
        self.allow_wsl = allow_wsl
        self.probe_cache = probe_cache
        self.python_executable = python_executable or sys.executable
        self.mem_limit_mb = mem_limit_mb
        self.cpu_limit_s = cpu_limit_s
        self.active_process_limit = active_process_limit
        self.max_output_chars = int(max_output_chars)
        self.child_env_extra = dict(child_env_extra or {})
        self.forbidden_patterns = tuple(forbidden_patterns)
        self.veto_policy = veto_policy
        self.enable_repl_telemetry = enable_repl_telemetry
        self.repl_d_model = repl_d_model
        self.keep_workspace = keep_workspace

        self.probe: Optional[NamespaceProbe] = None
        self.downgrade_reason: Optional[str] = None
        self.unshare_path: Optional[str] = None
        self._repl = None
        self._repl_import_error: Optional[str] = None

        if engine == "live-repl":
            self.mode = "live-repl"
            self.isolated = False
            self.boundary = dict(MODE_BOUNDARIES["live-repl"])
            return

        if mode == "auto":
            probe = probe_namespace(allow_wsl=allow_wsl, force=force_probe, cache=probe_cache)
            self.probe = probe
            if probe.available:
                self.mode = "namespace"
                self.isolated = True
                self.boundary = dict(MODE_BOUNDARIES["namespace"])
            else:
                if not allow_surrogate_fallback:
                    raise SandboxUnavailable(
                        "namespace", probe.reason, attempts=probe.attempts, probe_spawns=probe.spawns
                    )
                self.mode = SURROGATE_MODE
                self.isolated = False
                self.downgrade_reason = probe.reason
                self.boundary = dict(MODE_BOUNDARIES[SURROGATE_MODE])
        elif mode == "namespace":
            probe = probe_namespace(allow_wsl=allow_wsl, force=force_probe, cache=probe_cache)
            self.probe = probe
            if not probe.available:
                reason = probe.reason
                if os.name != "posix":
                    reason = (
                        "namespace mode requires a POSIX sandbox host "
                        f"(os.name={os.name!r}); " + reason
                    )
                raise SandboxUnavailable(
                    "namespace", reason, attempts=probe.attempts, probe_spawns=probe.spawns
                )
            self.mode = "namespace"
            self.isolated = True
            self.boundary = dict(MODE_BOUNDARIES["namespace"])
            self.unshare_path = shutil.which("unshare") or "/usr/bin/unshare"
        else:
            self.mode = mode
            self.isolated = bool(MODE_BOUNDARIES[mode]["isolated"])
            self.boundary = dict(MODE_BOUNDARIES[mode])

    # -- run command construction -------------------------------------------
    def _wrap_namespace(self, inner: List[str]) -> List[str]:
        """Wrap the child command in the launcher used by the prior art.

        ``unshare --user --map-root-user --net --pid --fork --mount-proc
        <python> -I <script>`` -- exactly the flag set the constructor probe
        proved, so probe success implies run capability.
        """
        exe = self.unshare_path or shutil.which("unshare") or "/usr/bin/unshare"
        return [exe] + _namespace_flags() + inner

    # -- introspection -------------------------------------------------------
    @property
    def surrogate(self) -> bool:
        """True when the active boundary is an EXPLICIT surrogate (no real isolation)."""
        return bool(self.boundary.get("surrogate"))

    def describe(self) -> Dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "mode": self.mode,
            "isolated": self.isolated,
            "surrogate": bool(self.boundary.get("surrogate")),
            "boundary": dict(self.boundary),
            "timeout_s": self.timeout_s,
            "python_executable": self.python_executable,
            "probe": self.probe.to_dict() if self.probe else None,
            "downgrade_reason": self.downgrade_reason,
            "sanitizer": _SANITIZER_SOURCE,
        }

    # -- environment ---------------------------------------------------------
    def _child_env(self) -> Dict[str, str]:
        """Minimal child environment, aligned with mbpp_secure_executor's env.

        Windows-only variables are kept because the interpreter cannot start
        without ``SystemRoot``; POSIX keeps PATH/HOME/TMPDIR.
        """
        keep = ("PATH", "SystemRoot", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "PATHEXT",
                "WINDIR", "COMSPEC", "LANG", "LC_ALL", "USERPROFILE")
        env = {k: os.environ[k] for k in keep if k in os.environ}
        env["PYTHONNOUSERSITE"] = "1"   # prior art: ignore user site-packages
        env["PYTHONHASHSEED"] = "0"     # deterministic candidate runs
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["HENRI_SANDBOX_MODE"] = self.mode
        env.update(self.child_env_extra)
        return env

    def _preexec(self):
        """POSIX preexec: rlimits via the ``resource`` module (surrogate boundary)."""
        if not _HAS_RESOURCE:
            return None
        rlimit_specs = self._rlimit_specs()

        def _apply():  # pragma: no cover - POSIX only
            import resource as _res

            for name, (soft, hard) in rlimit_specs.items():
                try:
                    _res.setrlimit(getattr(_res, name), (soft, hard))
                except Exception:
                    pass

        return _apply

    def _rlimit_specs(self) -> Dict[str, Tuple[int, int]]:
        """rlimit set, reusing the prior art's boundaries (mbpp_secure_executor).

        RLIMIT_CPU / RLIMIT_AS / RLIMIT_FSIZE / RLIMIT_NOFILE / RLIMIT_NPROC are
        the surrogate boundary. These bound resource use only: NO netns, no
        mount/PID/UTS namespace.
        """
        specs: Dict[str, Tuple[int, int]] = {}
        if self.mem_limit_mb:
            b = int(self.mem_limit_mb) * 1024 * 1024
            specs["RLIMIT_AS"] = (b, b)
        cpu = int(self.cpu_limit_s if self.cpu_limit_s else (self.timeout_s + 5))
        specs["RLIMIT_CPU"] = (cpu, cpu + 5)
        specs["RLIMIT_CORE"] = (0, 0)
        specs["RLIMIT_FSIZE"] = (8 * 1024 * 1024, 8 * 1024 * 1024)
        specs["RLIMIT_NOFILE"] = (64, 64)
        if self.active_process_limit:
            specs["RLIMIT_NPROC"] = (int(self.active_process_limit), int(self.active_process_limit))
        return specs

    # -- launcher ------------------------------------------------------------
    def _spawn(
        self, argv: Sequence[str], cwd: str, timeout_s: float, extra_limits: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Launch ``argv`` with the mode boundary applied; never raise for a
        child-side failure -- launcher failures are returned as data so the
        caller can map them to EXECUTION_ERROR (never FAIL)."""
        posix = os.name == "posix"
        limits: Dict[str, Any] = {"rlimits_applied": False, "setsid": False, "job_object": False}
        if extra_limits:
            limits.update(extra_limits)
        kwargs: Dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL,
            "cwd": cwd,
            "env": self._child_env(),
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        pre = self._preexec()
        if posix:
            kwargs["start_new_session"] = True  # setsid -> whole tree signalable
            limits["setsid"] = True
            if pre is not None:
                kwargs["preexec_fn"] = pre
                limits["rlimits_applied"] = True
                limits["rlimit_specs"] = {k: list(v) for k, v in self._rlimit_specs().items()}
            else:
                limits["rlimits_reason"] = "resource module unavailable on this platform"
        else:
            limits["setsid"] = False
            limits["rlimits_reason"] = "resource module unavailable on this platform (Windows)"
            cf = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            cn = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if cf or cn:
                kwargs["creationflags"] = cf | cn
                limits["creationflags"] = kwargs["creationflags"]

        job = None
        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(list(argv), **kwargs)
        except (OSError, ValueError) as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "elapsed_ms": elapsed,
                "killed": False,
                "timed_out": False,
                "launcher_error": f"{type(exc).__name__}: {exc}",
                "limits": limits,
                "argv": list(argv),
            }

        if not posix:
            job = _WindowsJobLimiter(self.mem_limit_mb, self.active_process_limit)
            if job.create():
                limits["job_object"] = bool(job.assign(proc.pid))
                limits["job_object_reason"] = job.reason
            else:
                limits["job_object_reason"] = job.reason

        killed = False
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            killed = True
            self._kill_tree(proc)
            try:
                stdout, stderr = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            returncode = None
        finally:
            if job is not None:
                job.close()

        elapsed = (time.perf_counter() - t0) * 1000.0
        stdout = stdout or ""
        stderr = stderr or ""
        return {
            "returncode": returncode,
            "stdout": stdout[: self.max_output_chars],
            "stderr": stderr[: self.max_output_chars],
            "stdout_truncated": len(stdout) > self.max_output_chars,
            "stderr_truncated": len(stderr) > self.max_output_chars,
            "elapsed_ms": elapsed,
            "killed": killed,
            "timed_out": timed_out,
            "launcher_error": None,
            "limits": limits,
            "argv": list(argv),
        }

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                import signal

                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
            else:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if proc.poll() is None:
                    proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # -- live REPL reuse -----------------------------------------------------
    def _load_live_repl(self):
        if self._repl is not None or self._repl_import_error is not None:
            return self._repl
        try:
            from henri_universal_repl import HENRIUniversalREPL

            self._repl = HENRIUniversalREPL(d_model=self.repl_d_model, device="cpu")
        except Exception as exc:  # torch missing / heavy import failed
            self._repl_import_error = f"{type(exc).__name__}: {exc}"
            self._repl = None
        return self._repl

    def _live_repl_telemetry(self, code: str, spawn: Dict[str, Any]) -> Dict[str, Any]:
        """Reuse the live REPL's wave/veto channel instead of re-deriving it."""
        tel: Dict[str, Any] = {
            "repl_telemetry": False,
            "sagnac_delta": None,
            "repl_q_score": None,
            "repl_vetoed": None,
            "repl_error": self._repl_import_error,
        }
        if not self.enable_repl_telemetry:
            return tel
        repl = self._load_live_repl()
        if repl is None:
            return tel
        try:
            import torch

            rc = spawn.get("returncode")
            stdout = spawn.get("stdout", "")
            stderr = spawn.get("stderr", "")
            payload = stdout if rc == 0 else stderr
            w_code = repl.transducer.transduce_text(code)
            w_out = repl.transducer.transduce_text(payload)
            sagnac_delta = 1.0 - (0.5 * (1.0 + torch.dot(w_code, w_out).item()))
            is_vetoed, q_score = repl.veto_engine.evaluate_execution(
                code, -1 if rc is None else rc, stdout, stderr, sagnac_delta
            )
            tel.update(
                {
                    "repl_telemetry": True,
                    "sagnac_delta": float(sagnac_delta),
                    "repl_q_score": (None if q_score == -float("inf") else float(q_score)),
                    "repl_vetoed": bool(is_vetoed),
                    "repl_error": None,
                }
            )
        except Exception as exc:  # telemetry must never break evaluation
            tel["repl_error"] = f"{type(exc).__name__}: {exc}"
        return tel

    # -- evaluation ----------------------------------------------------------
    def evaluate(
        self,
        code: str,
        tests: Optional[str] = None,
        *,
        timeout_s: Optional[float] = None,
        mode: Optional[str] = None,
        label: str = "",
        keep_workspace: Optional[bool] = None,
        force_probe: bool = False,
    ) -> SandboxResult:
        """Run ``code`` (optionally against published ``tests``) and return a typed result."""
        import hashlib

        eff_mode = mode or self.mode
        if eff_mode != self.mode:
            raise SandboxError(
                f"per-call mode override {eff_mode!r} != harness mode {self.mode!r} is not supported; "
                "construct a separate SandboxHarness instead"
            )
        eff_timeout = float(timeout_s) if timeout_s is not None else self.timeout_s
        keep = self.keep_workspace if keep_workspace is None else keep_workspace

        original_code = code or ""
        clean = clean_code_for_exec(original_code)
        clean_tests = clean_code_for_exec(tests) if tests is not None else None
        code_hash = hashlib.sha256(clean.encode("utf-8", "replace")).hexdigest()[:16]

        # (1) pre-execution policy veto -- no spawn at all.
        if self.forbidden_patterns:
            for pat in self.forbidden_patterns:
                if re.search(pat, clean):
                    return self._build_result(
                        status=STATUS_VETOED,
                        returncode=None,
                        stdout="",
                        stderr=f"policy veto: pattern {pat!r} matched",
                        elapsed_ms=0.0,
                        killed=False,
                        timed_out=False,
                        launcher_error=None,
                        limits={"spawned": False},
                        status_reason=f"policy veto: forbidden pattern {pat!r} matched",
                        tests_provided=tests is not None,
                        code_sha256=code_hash,
                        label=label,
                        telemetry={"repl_telemetry": False, "sagnac_delta": None,
                                   "repl_q_score": None, "repl_vetoed": None, "repl_error": None},
                        workspace=None,
                    )

        # (2) engine delegation: the live REPL owns the execution.
        if eff_mode == "live-repl":
            repl = self._load_live_repl()
            if repl is None:
                return self._build_result(
                    status=STATUS_EXECUTION_ERROR,
                    returncode=None,
                    stdout="",
                    stderr="",
                    elapsed_ms=0.0,
                    killed=False,
                    timed_out=False,
                    launcher_error=f"live REPL unavailable: {self._repl_import_error}",
                    limits={"spawned": False, "repl_internal_timeout_s": 10.0},
                    status_reason="live-repl engine could not be imported",
                    tests_provided=tests is not None,
                    code_sha256=code_hash,
                    label=label,
                    telemetry={"repl_telemetry": False, "sagnac_delta": None,
                               "repl_q_score": None, "repl_vetoed": None, "repl_error": self._repl_import_error},
                    workspace=None,
                )
            t0 = time.perf_counter()
            raw = repl.execute_python_repl(clean)
            spawn = {
                "returncode": raw.get("returncode"),
                "stdout": (raw.get("stdout") or "")[: self.max_output_chars],
                "stderr": (raw.get("stderr") or "")[: self.max_output_chars],
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                "killed": raw.get("returncode") == 124,
                "timed_out": raw.get("returncode") == 124,
                "launcher_error": None,
                "limits": {"repl_internal_timeout_s": 10.0, "spawned": True},
            }
            vetoed = bool(raw.get("is_vetoed")) and self.veto_policy == "repl"
            status, reason = classify_status(
                spawn["returncode"], spawn["stderr"], killed=spawn["killed"], vetoed=vetoed
            )
            return self._build_result(
                status=status,
                returncode=spawn["returncode"],
                stdout=spawn["stdout"],
                stderr=spawn["stderr"],
                elapsed_ms=spawn["elapsed_ms"],
                killed=spawn["killed"],
                timed_out=spawn["timed_out"],
                launcher_error=None,
                limits=spawn["limits"],
                status_reason=reason,
                tests_provided=tests is not None,
                code_sha256=code_hash,
                label=label,
                telemetry={
                    "repl_telemetry": True,
                    "sagnac_delta": raw.get("sagnac_delta"),
                    "repl_q_score": None if raw.get("q_score") == -float("inf") else raw.get("q_score"),
                    "repl_vetoed": bool(raw.get("is_vetoed")),
                    "repl_error": None,
                },
                workspace=None,
            )

        # (3) harness engine: own subprocess under the mode boundary.
        workspace = tempfile.mkdtemp(prefix="henri_sandbox_")
        try:
            solution_path = os.path.join(workspace, "solution.py")
            with open(solution_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(clean)

            if clean_tests is None:
                target = [self.python_executable, "-I", "-u", solution_path]
                run_cwd = workspace
            else:
                tests_path = os.path.join(workspace, "published_tests.py")
                with open(tests_path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(clean_tests)
                driver_path = os.path.join(workspace, "henri_driver.py")
                with open(driver_path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(
                        _DRIVER_TEMPLATE.format(
                            solution=solution_path,
                            tests=tests_path,
                            workspace=workspace,
                            exit_test_failed=EXIT_TEST_FAILED,
                            exit_exec_error=EXIT_EXEC_ERROR,
                        )
                    )
                target = [self.python_executable, "-I", "-u", driver_path]
                run_cwd = workspace

            if self.mode == "namespace":
                argv = self._wrap_namespace(target)
                extra_limits = {
                    "launcher": "unshare",
                    "namespace_flags": _namespace_flags(),
                    "argv_inner": list(target),
                }
            else:
                argv = target
                extra_limits = {
                    "launcher": "direct",
                    "surrogate_boundary": "rlimits + setsid + wall-clock timeout; NO netns",
                    "network_isolation": False,
                }

            spawn = self._spawn(argv, run_cwd, eff_timeout, extra_limits=extra_limits)
            telemetry = self._live_repl_telemetry(clean, spawn)

            vetoed = bool(telemetry.get("repl_vetoed")) and self.veto_policy == "repl"
            status, reason = classify_status(
                spawn["returncode"],
                spawn["stderr"],
                killed=spawn["killed"],
                launcher_error=spawn["launcher_error"],
                vetoed=vetoed,
            )
            if telemetry.get("repl_vetoed") and self.veto_policy == "repl":
                reason = reason + " (live REPL Sagnac veto applied via veto_policy='repl')"
            return self._build_result(
                status=status,
                returncode=spawn["returncode"],
                stdout=spawn["stdout"],
                stderr=spawn["stderr"],
                elapsed_ms=spawn["elapsed_ms"],
                killed=spawn["killed"],
                timed_out=spawn["timed_out"],
                launcher_error=spawn["launcher_error"],
                limits=spawn["limits"],
                status_reason=reason,
                tests_provided=tests is not None,
                code_sha256=code_hash,
                label=label,
                telemetry=telemetry,
                workspace=workspace if keep else None,
            )
        finally:
            if not keep:
                shutil.rmtree(workspace, ignore_errors=True)

    def _build_result(
        self,
        *,
        status: str,
        returncode: Optional[int],
        stdout: str,
        stderr: str,
        elapsed_ms: float,
        killed: bool,
        timed_out: bool,
        launcher_error: Optional[str],
        limits: Dict[str, Any],
        status_reason: str,
        tests_provided: bool,
        code_sha256: str,
        label: str,
        telemetry: Dict[str, Any],
        workspace: Optional[str],
    ) -> SandboxResult:
        return SandboxResult(
            status=status,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_ms=elapsed_ms,
            mode=self.mode,
            isolated=bool(self.isolated and status != STATUS_VETOED),
            surrogate=bool(self.boundary.get("surrogate")),
            killed=killed,
            timed_out=timed_out,
            launcher_error=launcher_error,
            returncode_preserved=True,
            limits=dict(limits),
            status_reason=status_reason,
            tests_provided=tests_provided,
            code_sha256=code_sha256,
            repl_telemetry=bool(telemetry.get("repl_telemetry")),
            sagnac_delta=telemetry.get("sagnac_delta"),
            repl_q_score=telemetry.get("repl_q_score"),
            repl_vetoed=telemetry.get("repl_vetoed"),
            workspace=workspace,
            label=label,
        )

    # -- batch ---------------------------------------------------------------
    def evaluate_suite(self, cases: Iterable[Dict[str, Any]]) -> SuiteReport:
        """Evaluate ``[{'code':..., 'tests':..., 'label':..., 'timeout_s':...}, ...]``."""
        report = SuiteReport()
        for case in cases:
            result = self.evaluate(
                case.get("code", ""),
                case.get("tests"),
                timeout_s=case.get("timeout_s"),
                label=case.get("label", ""),
                keep_workspace=case.get("keep_workspace"),
            )
            report.results.append(result)
        report.assert_arithmetic()
        return report

    def evaluate_against_published_tests(self, cases: Iterable[Dict[str, Any]]) -> SuiteReport:
        """Alias for :meth:`evaluate_suite` (SciCode-style naming)."""
        return self.evaluate_suite(cases)


try:  # POSIX only; absent on Windows
    import resource as _resource  # noqa: F401

    _HAS_RESOURCE = True
except Exception:  # pragma: no cover - Windows
    _resource = None
    _HAS_RESOURCE = False


# ---------------------------------------------------------------------------
# CLI: emit the live probe measurement (this is the auditable artifact)
# ---------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="HENRI M3 sandbox harness / namespace probe")
    ap.add_argument("--mode", default="auto", choices=["auto"] + list(EXECUTION_MODES))
    ap.add_argument("--probe-only", action="store_true", help="run only the namespace probe and report")
    ap.add_argument("--allow-wsl", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-telemetry", action="store_true")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.probe_only:
        probe = probe_namespace(allow_wsl=args.allow_wsl, force=True, cache=False)
        payload = {
            "platform": sys.platform,
            "python": sys.executable,
            "has_resource_module": _HAS_RESOURCE,
            "probe": probe.to_dict(),
            "verdict": "namespace AVAILABLE" if probe.available else "SandboxUnavailable",
        }
        print(json.dumps(payload, indent=2))
        return 0 if probe.available else 2

    try:
        harness = SandboxHarness(
            mode=args.mode,
            timeout_s=args.timeout,
            allow_wsl=args.allow_wsl,
            probe_cache=not args.no_cache,
            force_probe=True,
            enable_repl_telemetry=not args.no_telemetry,
        )
    except SandboxUnavailable as exc:
        print(json.dumps({"error": "SandboxUnavailable", "reason": exc.reason,
                          "attempts": exc.attempts}, indent=2))
        return 2

    print(json.dumps(harness.describe(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
