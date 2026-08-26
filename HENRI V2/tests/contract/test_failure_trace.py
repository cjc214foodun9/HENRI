"""Contract tests: P1 five-step failure trace (instrumentation-only).

Run on the Vast CUDA target with /venv/main/bin/python:
    PYTHONPATH="$(pwd)" /venv/main/bin/python -m pytest tests/contract/test_failure_trace.py -q -p no:cacheprovider
"""
import inspect
import textwrap

import pytest

from failure_trace import (
    DEFAULT_K,
    FLAG,
    FailureTraceDisabledError,
    FailureTraceWindow,
    resolve_valence,
)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    return True


def test_c1_valence_resolution():
    """C1: nu = 0.0 on progress, -1.0 on stall/zero (spec: nu=-1.0)."""
    assert resolve_valence(0.01) == 0.0
    assert resolve_valence(0.0) == -1.0
    assert resolve_valence(-0.5) == -1.0


def test_c2_sliding_window_resolves_at_k(flag_on):
    """C2: window resolves only at k=5 records; slides thereafter."""
    w = FailureTraceWindow(k=DEFAULT_K)
    for i in range(4):
        out = w.observe(i, f"a{i}", 0.0)
        assert out["status"] == "PENDING"
    out = w.observe(4, "a4", 0.0)
    assert out["status"] == "RESOLVED"
    assert out["window_delta"] == 0.0
    assert out["nu"] == -1.0
    assert out["window_len"] == 5
    # slide: add a +1 delta -> window sum 1 -> nu 0.0
    out = w.observe(5, "a5", 1.0)
    assert out["status"] == "RESOLVED"
    assert out["nu"] == 0.0
    assert out["window_delta"] == 1.0


def test_c3_stall_fires_on_nonpositive_sum(flag_on):
    """C3: a window summing <= 0 resolves nu=-1 (stall detection)."""
    w = FailureTraceWindow(k=DEFAULT_K)
    for i, d in enumerate([0.1, 0.1, -0.3, -0.1, -0.2]):
        w.observe(i, f"a{i}", d)
    out = w.observe(5, "a5", 0.0)  # window [0.1,-0.3,-0.1,-0.2,0.0] sum -0.5
    assert out["status"] == "RESOLVED"
    assert out["nu"] == -1.0
    assert w.summary()["stall_windows"] == 1


def test_c4_default_off_raises(monkeypatch):
    """C4: without HENRI_FAILURE_TRACE=1, construction fails closed."""
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(FailureTraceDisabledError):
        FailureTraceWindow()


def test_c5_reset_clears_window(flag_on):
    """C5: explicit episode boundary resets the window (T0 pattern)."""
    w = FailureTraceWindow(k=DEFAULT_K)
    for i in range(5):
        w.observe(i, f"a{i}", 0.0)
    assert w.summary()["windows_resolved"] == 1
    w.reset("ep2")
    assert w.summary()["window_len"] == 0
    out = w.observe(0, "b0", 1.0)
    assert out["status"] == "PENDING"


def test_c6_zero_trainable_static_audit():
    """C6: no torch, Parameter, optimizer, or backward in the carrier."""
    src = textwrap.dedent(inspect.getsource(FailureTraceWindow))
    for forbidden in ("Parameter(", "optimizer", ".backward("):
        assert forbidden not in src, f"forbidden token {forbidden} in carrier"
    assert "import torch" not in textwrap.dedent(inspect.getsource(
        __import__("failure_trace")))
