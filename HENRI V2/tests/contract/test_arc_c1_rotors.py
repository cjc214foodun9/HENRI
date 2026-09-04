"""Contract tests — Carrier C1 factorized SO(8) rotor action generators.

Directive: Carrier_C1_Master_Directive_SO8_Rotor_Action_Generators.md
(SHA-256 2554c3fc4f2169bc5324219f91b839653134ce99a35972518e7fcc70ee728814).
Prereg: docs/spec/c1_rotor_generators_preregistration.md (sealed before code).

Covers: Cayley orthogonality (C1_GATE_ORTHOGONALITY), per-row norm
preservation WITHOUT global renormalization (live [num_blocks, 8] boundary),
displacement separation (C1_GATE_DISPLACEMENT corrected per-row-RMS metric),
seed determinism, skew-symmetry, zero-angle identity, default-OFF G-series
launcher differential (sealed seam correction #3), and launcher routing.
Core math tests run on CPU; full-scale CUDA orthogonality is gated by CUDA
availability (remote gate owns the verdict).
"""

import os
import pathlib
import sys

import numpy as np
import pytest
import torch

TESTS = pathlib.Path(__file__).resolve()
ROOT = TESTS.parents[2]  # <repo>/HENRI V2
VERIF = ROOT / "experiments" / "verification"
for _p in (str(ROOT), str(VERIF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arc_c1_rotor_engine import (  # noqa: E402
    C1_FLAG,
    FactorizedSO8ActionGenerators,
    require_c1_flag,
)

SEED = 20260930
N_BLOCKS = 8192
D_BLOCK = 8
NUM_ACTIONS = 7


def _unit_rows(n_blocks=N_BLOCKS, d_block=D_BLOCK, seed=7):
    g = torch.Generator().manual_seed(seed)
    psi = torch.randn(n_blocks, d_block, generator=g)
    return psi / psi.norm(p=2, dim=-1, keepdim=True)


def _engine(seed=SEED, num_actions=NUM_ACTIONS):
    return FactorizedSO8ActionGenerators(
        num_actions=num_actions, d_block=D_BLOCK, seed=seed)


# ---------------------------------------------------------------------------
# Fail-closed flag
# ---------------------------------------------------------------------------

def test_flag_fail_closed():
    os.environ.pop(C1_FLAG, None)
    with pytest.raises(RuntimeError):
        require_c1_flag()
    os.environ[C1_FLAG] = "1"
    try:
        require_c1_flag()  # must not raise
    finally:
        os.environ.pop(C1_FLAG, None)


def test_flag_string_pinned():
    assert C1_FLAG == "HENRI_C1_SO8_ROTORS"


# ---------------------------------------------------------------------------
# Cayley orthogonality (C1_GATE_ORTHOGONALITY <= 1e-6)
# ---------------------------------------------------------------------------

def test_orthogonality_all_actions():
    eng = _engine()
    for a in range(NUM_ACTIONS):
        R = eng.get_rotor(a)
        err = (R.t() @ R - torch.eye(D_BLOCK)).norm(p="fro").item()
        assert err <= 1e-6, f"action {a} orth err {err} > 1e-6"


def test_rotor_determinant_positive():
    eng = _engine()
    for a in range(NUM_ACTIONS):
        det = torch.det(eng.get_rotor(a)).item()
        assert abs(det - 1.0) < 1e-4, f"action {a} det {det} != 1 (SO not O)"


def test_actions_are_distinct():
    eng = _engine()
    Rs = [eng.get_rotor(a) for a in range(NUM_ACTIONS)]
    for a in range(NUM_ACTIONS):
        for b in range(a + 1, NUM_ACTIONS):
            sep = (Rs[a] - Rs[b]).norm().item()
            assert sep > 1e-3, f"rotors {a},{b} collapsed: sep {sep}"


# ---------------------------------------------------------------------------
# Skew symmetry and Cayley identity
# ---------------------------------------------------------------------------

def test_skew_symmetric_generator():
    eng = _engine()
    for a in range(NUM_ACTIONS):
        A = eng.get_skew(a)
        assert torch.allclose(A, -A.t(), atol=1e-6)
        assert torch.allclose(torch.diag(A), torch.zeros(D_BLOCK), atol=1e-7)


def test_zero_bivector_is_identity():
    eng = _engine()
    with torch.no_grad():
        eng.bivectors.zero_()
    for a in range(NUM_ACTIONS):
        R = eng.get_rotor(a)
        assert torch.allclose(R, torch.eye(D_BLOCK), atol=1e-6)


# ---------------------------------------------------------------------------
# Norm preservation WITHOUT global renormalization (live boundary)
# ---------------------------------------------------------------------------

def test_per_row_norm_preserved_no_global_renorm():
    eng = _engine()
    psi = _unit_rows()
    out = eng.rotate(psi, 0)
    assert out.shape == psi.shape
    # per-row unit norm preserved exactly by an orthogonal rotor
    row_norm_err = (out.norm(p=2, dim=-1) - 1.0).abs().max().item()
    assert row_norm_err <= 1e-5
    # no global renormalization applied: total norm change is the same bound
    total_err = abs(out.norm().item() - psi.norm().item())
    assert total_err <= 1e-3


def test_flat_input_rejected():
    eng = _engine()
    flat = torch.randn(1, N_BLOCKS * D_BLOCK)
    with pytest.raises(ValueError):
        eng.rotate(flat, 0)


def test_norm_drift_over_8_steps():
    eng = _engine()
    psi = _unit_rows()
    x = psi.clone()
    with torch.no_grad():
        for _ in range(8):
            x = eng.rotate(x, 0)
    drift = (x.norm(p=2, dim=-1) - 1.0).abs().max().item()
    assert drift <= 1e-5


# ---------------------------------------------------------------------------
# Displacement separation (C1_GATE_DISPLACEMENT, corrected per-row-RMS)
# ---------------------------------------------------------------------------

def test_displacement_gate_fixture():
    eng = _engine()
    psi = _unit_rows(seed=SEED)  # frozen prereg fixture
    m = eng.displacement_metrics(psi)
    assert m["min_pairwise_sep"] >= 0.05, m
    assert m["min_per_action_disp"] >= 0.02, m


def test_displacement_zero_when_identity():
    eng = _engine()
    with torch.no_grad():
        eng.bivectors.zero_()
    psi = _unit_rows()
    m = eng.displacement_metrics(psi)
    assert m["min_pairwise_sep"] < 1e-4
    assert m["min_per_action_disp"] < 1e-4


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_seed_determinism():
    e1 = _engine(seed=SEED)
    e2 = _engine(seed=SEED)
    assert torch.equal(e1.bivectors, e2.bivectors)
    e3 = _engine(seed=SEED + 1)
    assert not torch.equal(e1.bivectors, e3.bivectors)


def test_param_footprint():
    eng = _engine()
    assert eng.bivectors.shape == (NUM_ACTIONS, 28)
    assert eng.bivectors.numel() == NUM_ACTIONS * 28  # 196 floats / 784 B fp32


# ---------------------------------------------------------------------------
# G-series launcher seam (default-OFF differential, sealed correction #3)
# ---------------------------------------------------------------------------

LAUNCHER = VERIF / "arc_g7_calibrated_engine.py"


def test_launcher_flag_wiring_present():
    src = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    assert C1_FLAG in src
    assert "C1RotorSteeringEngine" in src
    assert "require_c1_flag" in src


def test_launcher_default_path_is_g7():
    """Flags absent -> G7CalibratedAffordanceEngine remains the default."""
    src = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    assert "engine_cls = G7CalibratedAffordanceEngine" in src
    assert "use_c1" in src


def test_efe_planner_has_no_c1_seam():
    """The EFEPlanner score_actions hook was reverted at the seam pivot."""
    src = (ROOT / "efe_planner.py").read_text(encoding="utf-8", errors="replace")
    assert "c1_rotor" not in src


# ---------------------------------------------------------------------------
# Runner wiring source audit
# ---------------------------------------------------------------------------

def test_engines_never_import_runner():
    """C1 modules must stay standalone (remote-importable, no runner dep)."""
    for name in ("arc_c1_rotor_engine.py", "arc_c1_steering_engine.py"):
        src = pathlib.Path(VERIF / name).read_text(
            encoding="utf-8", errors="replace")
        assert "production_arc_run" not in src


# ---------------------------------------------------------------------------
# Full-scale CUDA orthogonality (remote gate preflight, skipped on CPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_full_scale_orthog():
    eng = _engine().cuda()
    psi = torch.randn(N_BLOCKS, D_BLOCK, device="cuda")
    psi = psi / psi.norm(p=2, dim=-1, keepdim=True)
    for a in range(NUM_ACTIONS):
        R = eng.get_rotor(a)
        err = (R.t() @ R - torch.eye(D_BLOCK, device="cuda")).norm().item()
        assert err <= 1e-6
        out = eng.rotate(psi, a)
        row_err = (out.norm(p=2, dim=-1) - 1.0).abs().max().item()
        assert row_err <= 1e-5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
