"""Contract tests for the MBPP fail-closed sandbox executor (pure logic, no POSIX execution)."""

import os

import pytest

from mbpp_secure_executor import SandboxUnavailable, SecurePythonSandbox, _launcher_failure


def test_launcher_failure_detects_unshare_eprm():
    assert _launcher_failure("unshare: unshare failed: Operation not permitted\n")


def test_launcher_failure_ignores_candidate_traceback():
    assert not _launcher_failure("Traceback (most recent call last):\nAssertionError: 3 != 4\n")


def test_launcher_failure_empty_stderr():
    assert not _launcher_failure("")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        SecurePythonSandbox(mode="bogus")


@pytest.mark.skipif(os.name == "posix", reason="expects non-POSIX host (local Windows)")
def test_non_posix_raises_unavailable():
    with pytest.raises(SandboxUnavailable):
        SecurePythonSandbox(mode="namespace")
