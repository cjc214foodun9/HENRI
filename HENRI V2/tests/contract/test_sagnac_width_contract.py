"""Sagnac WIDTH CONTRACT + NAME-BINDING regression.

WHY THIS FILE EXISTS
    Two measured defects in one session, both invisible to ordinary checks:

    1. WIDTH. `dual_channel_sagnac_veto` multiplies two waves elementwise. Production
       passed a 65536-wide complex candidate against 512-wide real refs, so it raised
       on EVERY call. The caller's broad `except` left `hard_vetoed` at its
       initialised False, so an UNAVAILABLE gate was indistinguishable from a
       PERMISSIVE one. Measured: 60/60 payloads were `{"error": "RuntimeError"}`.
       The contract now raises a NAMED type and the caller records a distinct
       `gate_status` with NO `hard_vetoed` key.

    2. NAME BINDING. The companion import patch FAILED ("Found 2 matches") while the
       handler patch SUCCEEDED, leaving `SagnacGateUnavailable` used-but-unbound. The
       handler would have raised NameError the instant the veto raised -- converting a
       recorded UNAVAILABLE into a crash. `py_compile` and `ast.parse` BOTH passed on
       that tree, because neither resolves names. The last test here is the guard.

INVARIANTS
    W1  the named type exists and derives RuntimeError
    W2  mismatched widths RAISE it when the diagnostic bridge is OFF (default)
    W3  the bridge, when ON, computes and records itself as diagnostic-only
    W4  matched widths keep the EXACT 3-tuple contract and record no bridge
    W5  hard_vetoed reaches BOTH values at matched width (a one-sided gate is not a gate)
    W6  binding: every name the runner's veto handler uses is bound by an import
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER = ROOT / "production_arc_run.py"
DIM = 512
BIG = 65536


@pytest.fixture(scope="module")
def planner():
    from sagnac_mcts_planner import SagnacMCTSPlanner
    return SagnacMCTSPlanner(d_model=DIM, k_blocks=64, tau_veto=0.35, device="cpu")


def _unit(dim: int = DIM, idx: int = 0) -> torch.Tensor:
    v = torch.zeros(dim)
    v[idx] = 1.0
    return v


# ------------------------------------------------------------------ W1
def test_gate_unavailable_type_is_runtimeerror():
    from sagnac_mcts_planner import SagnacGateUnavailable
    assert issubclass(SagnacGateUnavailable, RuntimeError), (
        "SagnacGateUnavailable must derive RuntimeError so existing broad handlers "
        "still catch it, while callers can distinguish it by type")


# ------------------------------------------------------------------ W2 / W3
def test_width_mismatch_raises_when_bridge_off(planner, monkeypatch):
    """W2. The pre-registered default: the gate reports 'did not run'."""
    monkeypatch.delenv("HENRI_SAGNAC_WIDTH_BRIDGE", raising=False)
    from sagnac_mcts_planner import SagnacGateUnavailable
    big = torch.randn(BIG, generator=torch.Generator().manual_seed(0)).to(torch.complex64)
    small = torch.randn(DIM, generator=torch.Generator().manual_seed(1))
    with pytest.raises(SagnacGateUnavailable) as ei:
        planner.dual_channel_sagnac_veto(big, small, small, epsilon_hard=planner.tau_veto)
    msg = str(ei.value)
    assert "width mismatch" in msg and "65536" in msg and "512" in msg, (
        f"the raise must name the actual widths so the cause is visible; got {msg[:160]}")


def test_bridge_on_computes_and_marks_itself_diagnostic(planner, monkeypatch):
    """W3. The bridge is a DIAGNOSTIC. It must say so in its own record."""
    monkeypatch.setenv("HENRI_SAGNAC_WIDTH_BRIDGE", "1")
    big = torch.randn(BIG, generator=torch.Generator().manual_seed(2)).to(torch.complex64)
    small = torch.randn(DIM, generator=torch.Generator().manual_seed(3))
    r = planner.dual_channel_sagnac_veto(big, small, small, epsilon_hard=planner.tau_veto)
    assert isinstance(r, tuple) and len(r) == 3, (
        "the bridge must not change the return arity; callers depend on 3 values")
    b = planner.last_sagnac_bridge
    assert isinstance(b, dict), "the bridge must record itself on the instance"
    assert b.get("diagnostic_only") is True, (
        "the bridge must declare itself diagnostic-only, because pooling a field wave "
        "to a grid wave's width does NOT make the two comparable")
    assert b.get("pool_factor") == BIG // DIM


def test_bridge_declines_when_widths_are_not_divisible(planner, monkeypatch):
    """W3b. A non-divisible ratio must NOT be silently truncated into a fake match."""
    monkeypatch.setenv("HENRI_SAGNAC_WIDTH_BRIDGE", "1")
    from sagnac_mcts_planner import SagnacGateUnavailable
    odd = torch.randn(500, generator=torch.Generator().manual_seed(4)).to(torch.complex64)
    small = torch.randn(DIM, generator=torch.Generator().manual_seed(5))
    with pytest.raises(SagnacGateUnavailable):
        planner.dual_channel_sagnac_veto(odd, small, small, epsilon_hard=planner.tau_veto)


# ------------------------------------------------------------------ W4 / W5
def test_matched_width_contract_is_unchanged(planner, monkeypatch):
    """W4. Default-off must be byte-equivalent in behaviour for matched widths."""
    monkeypatch.delenv("HENRI_SAGNAC_WIDTH_BRIDGE", raising=False)
    u = _unit()
    r = planner.dual_channel_sagnac_veto(u, u, u, epsilon_hard=planner.tau_veto)
    assert isinstance(r, tuple) and len(r) == 3
    assert planner.last_sagnac_bridge is None, (
        "no bridge may be recorded for a matched-width call")
    assert abs(float(r[0])) < 1e-6, f"identical waves gave delta {r[0]}"
    assert r[2] is False, "identical waves must not be vetoed"


def test_hard_veto_reaches_both_values(planner, monkeypatch):
    """W5. A gate that only ever reads one way is not a gate.

    Uses a deterministic pair: a unit vector against ITSELF (perfect agreement) and
    against its NEGATION (perfect disagreement). Both must be reachable, or the
    threshold is not a threshold.
    """
    monkeypatch.delenv("HENRI_SAGNAC_WIDTH_BRIDGE", raising=False)
    u = _unit()
    r_agree = planner.dual_channel_sagnac_veto(u, u, u, epsilon_hard=planner.tau_veto)
    r_oppose = planner.dual_channel_sagnac_veto(-u, u, u, epsilon_hard=planner.tau_veto)
    assert r_agree[2] is False, f"perfect agreement was vetoed (delta {r_agree[0]})"
    assert r_oppose[2] is True, f"perfect disagreement was NOT vetoed (delta {r_oppose[0]})"
    assert float(r_oppose[0]) > float(r_agree[0]), "stress did not order correctly"


# ------------------------------------------------------------------ W6
def test_runner_binds_every_name_its_veto_handler_uses():
    """W6. THE guard for the defect this session introduced.

    The handler uses `SagnacGateUnavailable` and `os`. Both must be bound by an
    import in the SAME file. `py_compile` and `ast.parse` pass on an unbound name, so
    a syntax check cannot catch this -- only resolving the binding can.

    MEASURED defect: the isinstance patch applied while its import patch failed
    (`Found 2 matches` for `from sagnac_mcts_planner import SagnacMCTSPlanner`, which
    appears twice). The handler then raised NameError exactly when the gate raised,
    turning a recorded UNAVAILABLE status into a crash.
    """
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    bound: set = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            bound.update(a.asname or a.name for a in n.names)
        elif isinstance(n, ast.Import):
            bound.update((a.asname or a.name).split(".")[0] for a in n.names)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for name in ("SagnacGateUnavailable", "os"):
        assert name in used, f"{name} is expected in the runner's veto handler"
        assert name in bound, (
            f"{name} is USED but never BOUND in production_arc_run.py. The handler "
            f"will raise NameError the moment the veto raises, converting a recorded "
            f"UNAVAILABLE status into a crash. Bound: {sorted(bound)}")
