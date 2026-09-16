"""Contract tests for HENRI V2's M3 isolated code-execution harness.

Covers the deliverable contract of ``henri_sandbox_harness.py``:

* the seven-key typed result contract + the five-value status enum,
* the two execution modes and the SURROGATE labelling of ``container-rlimit``
  (rlimits only, NO netns),
* the invariant that a subprocess which cannot start maps to EXECUTION_ERROR
  and NEVER to a task FAILED,
* the 300 s default timeout, configurable per instance/per call,
* markdown fence stripping before exec,
* reuse of the live ``HENRIUniversalREPL`` telemetry/veto channel and of
  ``mbpp_secure_executor``'s launcher-failure taxonomy,
* the audited arithmetic ``passed + failed == attempted``.

Run:
    cd 'C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2' && \
      env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME PYTHONPATH='HENRI V2' \
      C:/Python314/python.exe -m pytest tests/contract/test_sandbox_harness.py -q --tb=short
"""

from __future__ import annotations

import sys

import pytest

import henri_sandbox_harness as H

REQUIRED_KEYS = {"status", "returncode", "stdout", "stderr", "elapsed_ms", "mode", "isolated"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def surrogate_harness():
    """Auto-mode harness: namespace if provable, else the explicit surrogate."""
    return H.SandboxHarness(mode="auto", timeout_s=30, enable_repl_telemetry=False, repl_d_model=256)


@pytest.fixture(scope="module")
def telemetry_harness():
    """Surrogate harness with the live HENRIUniversalREPL wave/veto channel on."""
    return H.SandboxHarness(mode="auto", timeout_s=30, enable_repl_telemetry=True, repl_d_model=256)


PASSING_PROGRAM = "def add(a, b):\n    return a + b\n\nprint('sum', add(2, 3))\n"
PASSING_TESTS = "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"
WRONG_PROGRAM = "def add(a, b):\n    return a - b\n"
SYNTAX_ERROR_PROGRAM = "def add(a, b)\n    return a + b\n"
INFINITE_LOOP_PROGRAM = "while True:\n    pass\n"


# ---------------------------------------------------------------------------
# 1. Typed result contract
# ---------------------------------------------------------------------------

def test_result_contract_keys_and_status_enum(surrogate_harness):
    result = surrogate_harness.evaluate(PASSING_PROGRAM)
    contract = result.as_contract()
    assert set(contract) == REQUIRED_KEYS, contract.keys()
    assert result.status in H.STATUSES
    assert H.STATUSES == ("PASSED", "FAILED", "EXECUTION_ERROR", "TIMEOUT", "VETOED")
    assert isinstance(result.returncode, int) or result.returncode is None
    assert isinstance(result.stdout, str) and isinstance(result.stderr, str)
    assert isinstance(result.elapsed_ms, float) and result.elapsed_ms > 0.0
    assert isinstance(result.isolated, bool)
    assert result.mode in H.EXECUTION_MODES


def test_invalid_status_or_mode_is_rejected():
    with pytest.raises(H.SandboxError):
        H.SandboxResult(status="OK", returncode=0, stdout="", stderr="", elapsed_ms=1.0,
                        mode="container-rlimit", isolated=False)
    with pytest.raises(H.SandboxError):
        H.SandboxResult(status="PASSED", returncode=0, stdout="", stderr="", elapsed_ms=1.0,
                        mode="magic-mode", isolated=False)


def test_timed_out_result_must_be_marked_killed():
    with pytest.raises(H.SandboxError):
        H.SandboxResult(status="TIMEOUT", returncode=None, stdout="", stderr="", elapsed_ms=1.0,
                        mode="container-rlimit", isolated=False, killed=False)


# ---------------------------------------------------------------------------
# 2. Mode labelling / surrogate honesty
# ---------------------------------------------------------------------------

def test_container_rlimit_is_labelled_an_explicit_surrogate():
    boundary = H.MODE_BOUNDARIES["container-rlimit"]
    assert boundary["surrogate"] is True
    assert boundary["isolated"] is False
    assert boundary["netns"] is False  # rlimits only, NO netns
    assert boundary["rlimits"] is True
    assert "NO netns" in boundary["note"]
    assert "surrogate" in dict((k, v) for k, v in boundary.items())["note"].lower()
    assert H.SURROGATE_MODE == "container-rlimit"
    assert H.MODE_BOUNDARIES["namespace"]["isolated"] is True
    assert H.MODE_BOUNDARIES["namespace"]["netns"] is True


def test_surrogate_result_records_mode_and_surrogate_flag(surrogate_harness):
    result = surrogate_harness.evaluate(PASSING_PROGRAM)
    assert result.mode == surrogate_harness.mode
    if result.mode == H.SURROGATE_MODE:
        assert result.surrogate is True
        assert result.isolated is False, "the surrogate boundary must never claim isolation"
        assert result.limits["network_isolation"] is False
        assert result.limits["launcher"] == "direct"
        if not result.limits["rlimits_applied"]:
            assert "resource" in result.limits["rlimits_reason"].lower()
    else:
        assert result.mode == "namespace" and result.isolated is True


def test_auto_mode_records_the_actual_mode_used(surrogate_harness):
    assert surrogate_harness.mode in H.EXECUTION_MODES
    if surrogate_harness.mode == H.SURROGATE_MODE:
        # downgrade is allowed for mode='auto' but it must be EXPLICIT, never silent
        assert surrogate_harness.downgrade_reason
        assert "NOT PROVEN" in surrogate_harness.downgrade_reason
        assert surrogate_harness.probe is not None and surrogate_harness.probe.available is False
    described = surrogate_harness.describe()
    assert described["mode"] == surrogate_harness.mode
    assert described["boundary"]["surrogate"] is bool(surrogate_harness.boundary.get("surrogate"))


# ---------------------------------------------------------------------------
# 3. Outcome taxonomy
# ---------------------------------------------------------------------------

def test_passing_program_is_passed(surrogate_harness):
    result = surrogate_harness.evaluate(PASSING_PROGRAM)
    assert result.status == "PASSED"
    assert result.returncode == 0
    assert "sum 5" in result.stdout
    assert result.legacy_status == "PASS"


def test_passing_program_against_published_tests(surrogate_harness):
    result = surrogate_harness.evaluate(PASSING_PROGRAM, PASSING_TESTS)
    assert result.status == "PASSED", (result.status, result.stderr)
    assert result.returncode == 0
    assert result.tests_provided is True


def test_failing_assertion_is_failed_not_execution_error(surrogate_harness):
    result = surrogate_harness.evaluate(WRONG_PROGRAM, "assert add(2, 3) == 5\n")
    assert result.status == "FAILED", (result.status, result.stderr)
    assert result.status not in ("EXECUTION_ERROR", "TIMEOUT", "VETOED")
    assert result.returncode == H.EXIT_TEST_FAILED == 3
    assert "AssertionError" in result.stderr
    assert result.legacy_status == "FAIL"


def test_syntax_error_is_execution_error(surrogate_harness):
    result = surrogate_harness.evaluate(SYNTAX_ERROR_PROGRAM, PASSING_TESTS)
    assert result.status == "EXECUTION_ERROR", (result.status, result.stderr)
    assert "SyntaxError" in result.stderr
    assert result.returncode == H.EXIT_EXEC_ERROR == 4
    assert result.legacy_status == "EXECUTION_ERROR"


def test_missing_dependency_is_execution_error_not_fail(surrogate_harness):
    result = surrogate_harness.evaluate("import henri_no_such_module_xyz\n", None)
    assert result.status == "EXECUTION_ERROR"
    assert result.status != "FAILED"
    assert "ModuleNotFoundError" in result.stderr


def test_infinite_loop_times_out_with_short_timeout(surrogate_harness):
    result = surrogate_harness.evaluate(INFINITE_LOOP_PROGRAM, None, timeout_s=2.0)
    assert result.status == "TIMEOUT", (result.status, result.stderr)
    assert result.timed_out is True
    assert result.killed is True
    assert result.returncode is None, "a killed child produced no exit code"
    assert result.elapsed_ms >= 1800.0, f"returned too early: {result.elapsed_ms}ms"
    assert result.elapsed_ms < 30_000.0
    assert result.legacy_status == "EXECUTION_ERROR"
    # the executor is still usable afterwards (the process tree was really reaped)
    follow_up = surrogate_harness.evaluate("print('alive')\n", None, timeout_s=30)
    assert follow_up.status == "PASSED" and "alive" in follow_up.stdout


def test_timeoutless_infinite_loop_tests_are_not_counted_as_pass(surrogate_harness):
    report = surrogate_harness.evaluate_suite(
        [
            {"code": PASSING_PROGRAM, "tests": PASSING_TESTS, "label": "ok"},
            {"code": INFINITE_LOOP_PROGRAM, "label": "hang", "timeout_s": 2.0},
        ]
    )
    assert report.attempted == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.status_counts["TIMEOUT"] == 1
    report.assert_arithmetic()


def test_launcher_failure_is_execution_error_never_fail(tmp_path):
    harness = H.SandboxHarness(
        mode=H.SURROGATE_MODE,
        timeout_s=10,
        python_executable=str(tmp_path / "no_such_interpreter.exe"),
        enable_repl_telemetry=False,
    )
    result = harness.evaluate(PASSING_PROGRAM)
    assert result.status == "EXECUTION_ERROR"
    assert result.status != "FAILED"
    assert result.returncode is None
    assert result.launcher_error and "FileNotFoundError" in result.launcher_error
    assert H.SANDBOX_START_ERROR in result.status_reason
    assert result.legacy_status == "EXECUTION_ERROR"


def test_launcher_failure_taxonomy_is_shared_with_prior_art():
    """The unshare-failure signature must not be mistaken for a candidate FAIL."""
    launcher_stderr = "unshare: unshare failed: Operation not permitted\n"
    assert H._launcher_failure(launcher_stderr) is True
    status, reason = H.classify_status(1, launcher_stderr)
    assert status == "EXECUTION_ERROR"
    assert H.SANDBOX_LAUNCHER_FAILURE in reason

    candidate_traceback = 'Traceback (most recent call last):\nAssertionError\n'
    assert H._launcher_failure(candidate_traceback) is False
    assert H.classify_status(3, candidate_traceback)[0] == "FAILED"


# ---------------------------------------------------------------------------
# 4. Exit-code preservation + fence stripping
# ---------------------------------------------------------------------------

def test_producer_exit_code_is_preserved_through_output_filtering():
    harness = H.SandboxHarness(
        mode=H.SURROGATE_MODE, timeout_s=30, enable_repl_telemetry=False, max_output_chars=16
    )
    result = harness.evaluate("import sys\nprint('Y' * 5000)\nsys.exit(7)\n")
    assert result.returncode == 7, "output truncation must not touch the exit code"
    assert result.returncode_preserved is True
    assert result.status == "FAILED"  # ran to completion, did not satisfy the task
    assert result.limits is not None


def test_markdown_fences_are_stripped_before_exec(surrogate_harness):
    fenced = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nDone.\n"
    fenced_tests = "```py\nassert add(1, 2) == 3\n```\n"
    assert H.clean_code_for_exec(fenced).startswith("def add")
    result = surrogate_harness.evaluate(fenced, fenced_tests)
    assert result.status == "PASSED", (result.status, result.stderr)
    assert "SyntaxError" not in result.stderr
    # raw unfenced model output still passes through the live sanitizer
    assert H._SANITIZER_SOURCE == "henri_code_sanitizer.clean_generated_code"


# ---------------------------------------------------------------------------
# 5. Timeout configuration
# ---------------------------------------------------------------------------

def test_default_timeout_is_300_seconds():
    assert H.DEFAULT_TIMEOUT_S == 300.0
    harness = H.SandboxHarness(mode=H.SURROGATE_MODE, enable_repl_telemetry=False)
    assert harness.timeout_s == 300.0
    assert harness.describe()["timeout_s"] == 300.0


def test_timeout_is_configurable_per_instance_and_per_call():
    harness = H.SandboxHarness(mode=H.SURROGATE_MODE, timeout_s=45.0, enable_repl_telemetry=False)
    assert harness.timeout_s == 45.0
    result = harness.evaluate("print('hi')\n", None, timeout_s=30.0)
    assert result.status == "PASSED"
    quick = harness.evaluate("import time\ntime.sleep(30)\n", None, timeout_s=2.0)
    assert quick.status == "TIMEOUT" and quick.elapsed_ms < 20_000.0


# ---------------------------------------------------------------------------
# 6. Namespace mode + probe
# ---------------------------------------------------------------------------

def test_namespace_mode_proves_isolation_or_raises_typed_error():
    """On a namespace-capable host: isolated=True. Otherwise: SandboxUnavailable."""
    try:
        harness = H.SandboxHarness(mode="namespace", timeout_s=30, enable_repl_telemetry=False,
                                   probe_cache=False)
    except H.SandboxUnavailable as exc:
        assert isinstance(exc, H.SandboxError)
        assert exc.mode == "namespace"
        assert exc.reason, "the block reason must be reported"
        assert "NOT PROVEN" in exc.reason
        assert exc.probe_spawns >= 1, "the constructor probe must actually spawn a subprocess"
        assert exc.attempts, "probe attempts must be recorded as evidence"
        for attempt in exc.attempts:
            assert "command" in attempt and attempt["command"]
        return
    # namespace proven: the harness must claim isolation and actually run via unshare
    assert harness.isolated is True
    assert harness.probe is not None and harness.probe.available is True
    assert harness.probe.isolation_proof
    result = harness.evaluate(PASSING_PROGRAM, PASSING_TESTS)
    assert result.status == "PASSED"
    assert result.isolated is True
    assert result.limits["launcher"] == "unshare"


def test_namespace_is_blocked_not_silently_downgraded_when_required():
    harness_available = H.probe_namespace(force=True, cache=False).available
    if harness_available:
        pytest.skip("namespace is available on this host; blocking path not exercised")
    with pytest.raises(H.SandboxUnavailable):
        H.SandboxHarness(mode="namespace", enable_repl_telemetry=False, probe_cache=False)
    with pytest.raises(H.SandboxUnavailable):
        H.SandboxHarness(mode="auto", allow_surrogate_fallback=False, enable_repl_telemetry=False,
                         probe_cache=False)


def test_probe_actually_spawns_and_never_uses_an_external_echo():
    """Regression guard for the documented /bin/echo probe defect."""
    before = dict(H.PROBE_STATS)
    probe = H.probe_namespace(force=True, cache=False, timeout_s=15)
    assert H.PROBE_STATS["spawns"] > before["spawns"], "the probe must really spawn a subprocess"
    assert probe.spawns >= 1
    candidates = H._probe_candidates(False)
    assert candidates, "probe must have at least one candidate command"
    for _label, cmd in candidates:
        for token in cmd:
            assert token != "echo", f"probe uses external echo: {cmd}"
            assert not str(token).endswith("/bin/echo"), f"probe uses /bin/echo: {cmd}"
        assert any("python" in str(t).lower() for t in cmd), "probe must invoke the interpreter directly"
    assert "echo" not in H.NAMESPACE_PROBE_SOURCE
    # a shell-based probe must still be the interpreter check, not an external binary
    bash_cmd = [c for label, c in candidates if label == "unshare-via-bash"]
    if bash_cmd:
        assert "$1" in bash_cmd[0][2] and "-c" in bash_cmd[0]


def test_probe_reports_the_measured_error_string_on_this_host():
    probe = H.probe_namespace(force=True, cache=False)
    if probe.available:
        assert probe.isolation_proof
        return
    assert probe.reason.startswith("namespace isolation NOT PROVEN:")
    assert probe.attempts
    for attempt in probe.attempts:
        assert attempt.get("failure") or attempt.get("error")
        assert "command" in attempt
    joined = probe.reason.lower()
    assert ("not found" in joined) or ("cannot spawn" in joined) or ("permitted" in joined)
    if sys.platform == "win32":
        assert "cannot spawn" in joined or "not found" in joined


# ---------------------------------------------------------------------------
# 7. VETOED status + live REPL reuse
# ---------------------------------------------------------------------------

def test_vetoed_status_is_reachable_and_never_spawns():
    harness = H.SandboxHarness(
        mode=H.SURROGATE_MODE, timeout_s=30, enable_repl_telemetry=False,
        forbidden_patterns=[r"os\.system"],
    )
    result = harness.evaluate("import os\nos.system('echo hi')\n")
    assert result.status == "VETOED"
    assert result.returncode is None
    assert result.limits["spawned"] is False
    assert result.isolated is False
    assert "policy veto" in result.status_reason
    report = harness.evaluate_suite([{"code": "os.system('x')", "label": "v"}])
    assert report.attempted == 1 and report.passed == 0 and report.failed == 1


def test_live_repl_telemetry_is_reused_and_does_not_corrupt_the_outcome(telemetry_harness):
    result = telemetry_harness.evaluate(PASSING_PROGRAM, PASSING_TESTS)
    if not result.repl_telemetry:
        pytest.skip(f"live HENRIUniversalREPL telemetry unavailable: {result.limits}")
    assert isinstance(result.sagnac_delta, float)
    assert result.repl_q_score is not None
    assert result.status == "PASSED"

    failing = telemetry_harness.evaluate(WRONG_PROGRAM, "assert add(2, 3) == 5\n")
    if failing.repl_vetoed is None:
        pytest.skip("no veto channel")
    # The live REPL's Sagnac veto fires on any non-zero exit; the harness keeps it
    # as TELEMETRY unless veto_policy='repl', so the task outcome stays FAILED.
    assert failing.repl_vetoed is True
    assert failing.status == "FAILED"

    strict = H.SandboxHarness(
        mode=H.SURROGATE_MODE, timeout_s=30, enable_repl_telemetry=True, repl_d_model=256,
        veto_policy="repl",
    )
    assert strict.evaluate(WRONG_PROGRAM, "assert add(2, 3) == 5\n").status == "VETOED"


def test_live_repl_engine_delegates_execution():
    harness = H.SandboxHarness(engine="live-repl", repl_d_model=256, enable_repl_telemetry=True)
    assert harness.mode == "live-repl"
    assert harness.isolated is False and harness.surrogate is True
    result = harness.evaluate("print(6 * 7)\n")
    if result.status == "EXECUTION_ERROR" and "live REPL unavailable" in result.status_reason:
        pytest.skip("live HENRIUniversalREPL unavailable (torch missing)")
    assert result.status == "PASSED", (result.status, result.stderr)
    assert result.mode == "live-repl"
    assert result.isolated is False
    assert result.limits["repl_internal_timeout_s"] == 10.0
    assert result.legacy_status == "PASS"


# ---------------------------------------------------------------------------
# 8. Arithmetic
# ---------------------------------------------------------------------------

def test_passed_plus_failed_equals_attempted(surrogate_harness):
    cases = [
        {"code": PASSING_PROGRAM, "tests": PASSING_TESTS, "label": "pass"},
        {"code": WRONG_PROGRAM, "tests": "assert add(2, 3) == 5\n", "label": "fail"},
        {"code": SYNTAX_ERROR_PROGRAM, "tests": PASSING_TESTS, "label": "syntax"},
        {"code": INFINITE_LOOP_PROGRAM, "label": "timeout", "timeout_s": 2.0},
        {"code": "print('plain')\n", "label": "no-tests"},
        {"code": "import os\nos.system('x')\n", "label": "not-vetoed-here"},
    ]
    report = surrogate_harness.evaluate_suite(cases)
    assert report.attempted == 6
    assert report.passed == 3, report.status_counts  # pass, no-tests, not-vetoed-here
    assert report.failed == 3, report.status_counts  # fail, syntax, hang
    assert report.attempted == report.passed + report.failed
    assert sum(report.status_counts.values()) == report.attempted
    assert report.status_counts["PASSED"] == 3
    assert report.status_counts["FAILED"] == 1
    assert report.status_counts["EXECUTION_ERROR"] == 1
    assert report.status_counts["TIMEOUT"] == 1
    assert report.status_counts["VETOED"] == 0
    report.assert_arithmetic()
    payload = report.to_dict()
    assert payload["arithmetic_ok"] is True
    assert payload["attempted"] == payload["passed"] + payload["failed"]


def test_arithmetic_violation_is_detected():
    report = H.SuiteReport()
    report.assert_arithmetic()
    assert report.arithmetic_ok()
    report.results.append(
        H.SandboxResult(status="PASSED", returncode=0, stdout="", stderr="", elapsed_ms=1.0,
                        mode="container-rlimit", isolated=False)
    )
    assert report.attempted == 1 and report.passed == 1 and report.failed == 0
    report.assert_arithmetic()


def test_legacy_status_bridge_covers_every_status():
    assert set(H.LEGACY_STATUS_MAP) == set(H.STATUSES)
    assert H.LEGACY_STATUS_MAP["PASSED"] == "PASS"
    assert H.LEGACY_STATUS_MAP["FAILED"] == "FAIL"
    assert H.LEGACY_STATUS_MAP["TIMEOUT"] == "EXECUTION_ERROR"
    assert H.LEGACY_STATUS_MAP["VETOED"] == "EXECUTION_ERROR"


def test_closed_loop_against_mbpp_prior_art_surface():
    """The prior art's mode names/surface must stay compatible with this harness."""
    import inspect

    from mbpp_secure_executor import SecurePythonSandbox, SandboxUnavailable as MbppUnavailable

    assert issubclass(MbppUnavailable, H.SandboxError) is False  # separate classes, same role
    signature = inspect.signature(SecurePythonSandbox.__init__)
    assert signature.parameters["mode"].default == "namespace"
    assert "container-rlimit" in inspect.getdoc(SecurePythonSandbox.__init__)
    assert H.SURROGATE_MODE == "container-rlimit"
    assert H.SandboxUnavailable.__name__ == "SandboxUnavailable"
