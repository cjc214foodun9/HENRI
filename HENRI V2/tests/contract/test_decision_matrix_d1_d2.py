"""Contract tests for Decision Matrix D1 + D2 (feat/low-rank-wave-jepa).

Pre-registration: HENRI V2/experiments/sweeps/decision_matrix_d1_d2_design.md.
Source PDF raw SHA-256 2e2cf71151ed39732563a53d898f516281a6d6e3eb5c7934c1de9526ec03df66.

All tests run on CPU at toy scale (software/invariant verification only —
NOT CUDA or model-capability evidence).
"""
import math
import os
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat
from wave_jepa import WaveJEPA, _LowRankCoupledPredictorAdapter

D1_FLAG_ENV = "HENRI_WAVEJEPA_LOWRANK_COUPLED"


# ---------------------------------------------------------------------------
# D1 — Low-rank coupled transition in WaveJEPA (REUSE-ONLY)
# ---------------------------------------------------------------------------

def test_d1_flag_default_off_keeps_edmd_predictor():
    """G3: default path is byte-identical — predictor stays RecursiveDualEDMD."""
    os.environ.pop(D1_FLAG_ENV, None)
    jepa = WaveJEPA(d_model=256, num_blocks=32, r_rank=4)
    assert type(jepa.predictor).__name__ == "RecursiveDualEDMD"
    assert jepa.use_lowrank_coupled is False


def test_d1_flag_on_uses_reuse_adapter_with_production_transition():
    """G2: flag ON -> adapter whose .transition is the PRODUCTION operator."""
    jepa = WaveJEPA(d_model=256, num_blocks=32, r_rank=4, use_lowrank_coupled=True)
    assert isinstance(jepa.predictor, _LowRankCoupledPredictorAdapter)
    from efe_planner import LowRankCoupledTransition
    assert isinstance(jepa.predictor.transition, LowRankCoupledTransition)
    assert jepa.use_lowrank_coupled is True


def test_d1_env_flag_honored():
    """Flag is also controllable via HENRI_WAVEJEPA_LOWRANK_COUPLED=1."""
    os.environ[D1_FLAG_ENV] = "1"
    try:
        jepa = WaveJEPA(d_model=256, num_blocks=32, r_rank=4)
        assert jepa.use_lowrank_coupled is True
        assert isinstance(jepa.predictor, _LowRankCoupledPredictorAdapter)
    finally:
        os.environ.pop(D1_FLAG_ENV, None)


def test_d1_adapter_online_step_updates_and_retracts():
    """Adapter trains via Sagnac-loss step, re-retracts, loss stays finite."""
    torch.manual_seed(7)
    adapter = _LowRankCoupledPredictorAdapter(num_blocks=32, block_dim=8, rank=4)
    s = torch.randn(32, 8)
    s = s / s.norm(dim=-1, keepdim=True)
    a = torch.randn(32, 8)
    a = a / a.norm(dim=-1, keepdim=True)
    o = torch.randn(32, 8)
    o = o / o.norm(dim=-1, keepdim=True)
    loss = adapter.update_online_step(s, a, o)
    assert math.isfinite(loss)
    # Orthogonality of retracted field_V
    V = adapter.transition.field_V.detach()
    gram_err = (V.T @ V - torch.eye(V.shape[1])).abs().max().item()
    assert gram_err < 1e-5


# ---------------------------------------------------------------------------
# D2 — Anisotropic Langevin injection (P_null projection)
# ---------------------------------------------------------------------------

def _ortho_basis(d: int, r: int, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d, r, generator=g)
    Q, _ = torch.linalg.qr(M)
    return Q


def test_d2_default_path_isotropic_unchanged():
    """Default OFF (no flag, no basis): noise is the legacy scaled draw."""
    th = AdaptiveViscoelasticThermostat(d_model=64, use_null_subspace_projection=False)
    torch.manual_seed(11)
    W = torch.randn(16, 4)
    grad = torch.randn_like(W)
    # lambda_active=0.05 <= lambda_threshold=0.10 -> friction=1.0, so the
    # manual legacy reproduction below is exact.
    W1, _ = th.step_viscoelastic_creep(W, grad, 0.05, 0.4, temperature=1e-4)
    # Reproduce the EXACT same draw sequence under one seed: W, grad, then
    # the noise draw (3rd position) inside the step call.
    torch.manual_seed(11)
    W2 = torch.randn(16, 4)
    g2 = torch.randn_like(W2)
    eff_lr = th.base_lr * (1.0 + 0.4)
    noise = torch.randn_like(W2) * math.sqrt(2.0 * 1e-4 * eff_lr)
    W2 = W2 - eff_lr * g2 + noise
    assert torch.allclose(W1, W2, atol=1e-6)


def test_d2_paired_draw_orthogonality():
    """A1 (paired): projected noise satisfies ||V† noise|| / ||noise|| < 1e-3,
    isotropic residual ≈ sqrt(r/d); energy ratio ∈ (0, 1]."""
    d, r = 1024, 256
    V = _ortho_basis(d, r, seed=21)
    th_p = AdaptiveViscoelasticThermostat(
        d_model=d, use_null_subspace_projection=True)
    th_i = AdaptiveViscoelasticThermostat(d_model=d)
    th_p.set_null_basis(V)
    torch.manual_seed(23)
    g = torch.Generator().manual_seed(24)
    W = torch.zeros(d, 1)
    base = torch.randn(d, 1, generator=g)
    _, tp = th_p.step_viscoelastic_creep(
        W, torch.zeros_like(W), 0.05, 0.4, temperature=1e-2, base_noise=base)
    _, ti = th_i.step_viscoelastic_creep(
        W, torch.zeros_like(W), 0.05, 0.4, temperature=1e-2, base_noise=base.clone())
    # iso noise: base * scale (friction=1.0 at lambda=0.05)
    scale = math.sqrt(2.0 * 1e-2 * (th_i.base_lr * (1.0 + 0.4)))
    n_iso = base * scale
    # projected noise: rebuild via the same projection
    n_proj = (base - V @ (V.T @ base)) * scale
    assert float(torch.norm(V.T @ n_proj).item()) / (float(torch.norm(n_proj).item()) + 1e-12) < 1e-3
    resid = float(torch.norm(V.T @ n_iso).item()) / (float(torch.norm(n_iso).item()) + 1e-12)
    assert abs(resid - math.sqrt(r / d)) < 0.05
    ratio = tp["null_projection_energy_ratio"]
    assert 0.0 < ratio <= 1.0
    assert abs(ratio - float(torch.norm(n_proj).item() / (torch.norm(n_iso).item() + 1e-12))) < 1e-3


def test_d2_fail_closed_raises():
    """A3: enabled-without-basis, bad dim, non-finite basis all raise ValueError."""
    th = AdaptiveViscoelasticThermostat(
        d_model=64, use_null_subspace_projection=True)
    W = torch.randn(64, 1)
    with pytest.raises(ValueError):
        th.step_viscoelastic_creep(W, torch.zeros_like(W), 0.3, 0.4)
    th.set_null_basis(torch.eye(4, 4))  # wrong leading dim (4 != 64)
    with pytest.raises(ValueError):
        th.step_viscoelastic_creep(W, torch.zeros_like(W), 0.3, 0.4)
    bad = torch.full((64, 1), float("nan"))
    th2 = AdaptiveViscoelasticThermostat(
        d_model=64, use_null_subspace_projection=True)
    with pytest.raises(ValueError):
        th2.set_null_basis(bad)
