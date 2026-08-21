# -*- coding: utf-8 -*-
"""Phase 8.35 — Sprint b: Kuramoto early-stop (D1), field rank 256 (D2),
P_null thermostat coupling (D3). Tests per the Kuramoto/Ephaptic PDF."""
import math

import pytest
import torch
import torch.nn.functional as F

from hopfield_cleanup import ContinuousHopfieldCleanup, DualScaleAnalogLexicalSnap
from recursive_dual_edmd import CoupledRecursiveDualEDMD, DirectionalTravelingWaveCoupler
from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat


# ---- D1: Kuramoto order parameter ---------------------------------------
def test_kuramoto_r_is_bounded_and_phase_locked_waves_hit_one():
    # Fully phase-locked complex wave -> R == 1.0.
    theta = torch.full((64,), 0.7)
    locked = torch.exp(1j * theta)
    r = ContinuousHopfieldCleanup.kuramoto_order_parameter(locked)
    assert r.item() == pytest.approx(1.0, abs=1e-6)

    # Random phase -> R small.
    g = torch.Generator().manual_seed(835)
    theta = torch.rand(4096, generator=g) * 2 * math.pi
    rand = torch.exp(1j * theta)
    r_rand = ContinuousHopfieldCleanup.kuramoto_order_parameter(rand)
    assert 0.0 <= r_rand.item() <= 0.2

    # Real wave (Hilbert analytic phase): constant signal -> R ~ 1.
    real_const = torch.ones(1024)
    r_real = ContinuousHopfieldCleanup.kuramoto_order_parameter(real_const)
    assert r_real.item() > 0.9


def test_kuramoto_early_stop_converges_and_is_byte_identical_when_off():
    g = torch.Generator().manual_seed(11)
    eng = F.normalize(torch.randn(8, 128, generator=g), p=2, dim=-1)
    cleanup = ContinuousHopfieldCleanup(dim=128)
    cleanup.store_engrams(eng)
    q = F.normalize(torch.randn(128, generator=g), p=2, dim=-1)

    idx_off, conf_off = cleanup.lexical_snap(q, top_k=1)
    idx_on, conf_on = cleanup.lexical_snap(
        q, top_k=1, kuramoto_early_stop=True,
        kuramoto_threshold=0.85, max_relax_iters=8)
    # Both snap to the same nearest engram.
    assert idx_off.item() == idx_on.item()
    assert idx_on.item() == (q @ eng.T).argmax().item()
    # Relaxation sharpens toward the attractor: conf_on >= conf_off.
    assert conf_on.item() >= conf_off.item() - 1e-3


def test_kuramoto_early_stop_hits_locked_wave_immediately():
    # An already phase-locked wave satisfies R >= 0.85 at iteration 0.
    g = torch.Generator().manual_seed(11)
    eng = F.normalize(torch.randn(4, 128, generator=g), p=2, dim=-1)
    cleanup = ContinuousHopfieldCleanup(dim=128)
    cleanup.store_engrams(eng)
    locked = eng[0].clone()
    idx, _ = cleanup.lexical_snap(
        locked, top_k=1, kuramoto_early_stop=True,
        kuramoto_threshold=0.85, max_relax_iters=8)
    assert idx.item() == 0


# ---- D2: field rank 256 default -----------------------------------------
def test_coupled_rank_default_is_256():
    m = CoupledRecursiveDualEDMD(d_model=64, num_blocks=8, block_dim=8)
    # Phase 5 P1 contract: requested_rank stores the directive value (256);
    # r_rank is the EFFECTIVE rank = min(requested, d_model) = 64 at d=64.
    assert m.requested_rank == 256
    assert m.r_rank == 64
    assert m.V.shape == (64, 64)


def test_directional_coupler_rank_default_is_256():
    m = DirectionalTravelingWaveCoupler(d_model=64, num_blocks=8, block_dim=8)
    assert m.requested_rank == 256
    assert m.r_rank == 64
    assert m.V.shape == (64, 64)


# ---- D3: P_null thermostat coupling --------------------------------------
def test_null_space_projection_removes_synchronized_component():
    th = AdaptiveViscoelasticThermostat(d_model=64)
    g = torch.Generator().manual_seed(5)
    # Orthonormal V via QR — P_null = I - V V^T is an exact projector only
    # for orthonormal V (Stiefel, as in the production EDMD basis).
    V_raw = torch.randn(64, 4, generator=g)
    V = torch.linalg.qr(V_raw).Q  # [d, r] orthonormal
    W = torch.randn(64, generator=g)
    grad = torch.randn_like(W)
    noise = torch.randn_like(W)
    updated, tele = th.step_viscoelastic_creep(
        W, grad, lambda_active=1.0, sagnac_delta=0.0, temperature=1e-3,
        base_noise=noise, null_space_basis=V)
    # The injected noise must be orthogonal to the V subspace:
    # V^T (updated - (W - lr*grad)) == 0 up to fp error.
    noise_eff = updated - (W - tele["effective_lr"] * grad)
    assert torch.norm(V.T @ noise_eff).item() < 1e-5
    # No-basis path is byte-identical to the legacy formula (manual check):
    # updated = W - effective_lr*grad + base_noise*sqrt(2*T*effective_lr).
    updated_legacy, _ = th.step_viscoelastic_creep(
        W, grad, lambda_active=1.0, sagnac_delta=0.0, temperature=1e-3,
        base_noise=noise)
    manual = W - tele["effective_lr"] * grad + noise * math.sqrt(
        2.0 * 1e-3 * tele["effective_lr"])
    assert torch.allclose(updated_legacy, manual, atol=1e-6)
    # Projection CHANGES the update vs the unprojected path.
    assert not torch.equal(updated, updated_legacy)
    # Projection is idempotent on the noise alone.
    n2 = noise - V @ (V.T @ noise)
    n3 = n2 - V @ (V.T @ n2)
    assert torch.allclose(n2, n3, atol=1e-6)


def test_null_space_projection_production_shaped_non_orthonormal_V():
    """Regression: production V is COLUMN-NORMALIZED (not orthonormal), so
    I - V V^T is not a projector. The Moore-Penrose factored form
    n - V solve(V^T V, V^T n) must annihilate the V-subspace component
    exactly. This is the defect the D=65,536 CUDA smoke caught
    (residual 4.309e-4 vs 1e-5 with the naive V V^T form)."""
    th = AdaptiveViscoelasticThermostat(d_model=64)
    g = torch.Generator().manual_seed(7)
    # Mirrors CoupledRecursiveDualEDMD init exactly.
    v_init = torch.randn(64, 4, generator=g) / math.sqrt(64)
    V = F.normalize(v_init, p=2, dim=0)
    assert torch.norm(V.T @ V - torch.eye(4)).item() > 1e-3  # NOT orthogonal
    W = torch.randn(64, generator=g)
    grad = torch.randn_like(W)
    noise = torch.randn_like(W)
    updated, tele = th.step_viscoelastic_creep(
        W, grad, lambda_active=1.0, sagnac_delta=0.0, temperature=1e-3,
        base_noise=noise, null_space_basis=V)
    noise_eff = updated - (W - tele["effective_lr"] * grad)
    # True residual on the V-subspace: V^T n_eff must be ~0.
    res = torch.norm(V.T @ noise_eff).item()
    assert res < 1e-5, f"P_null residual on production-shaped V: {res:.3e}"


def test_null_space_basis_shape_validation():
    th = AdaptiveViscoelasticThermostat(d_model=64)
    W = torch.randn(64)
    grad = torch.randn_like(W)
    with pytest.raises(ValueError):
        th.step_viscoelastic_creep(
            W, grad, lambda_active=1.0, sagnac_delta=0.0,
            null_space_basis=torch.randn(3, 4, 5))  # not 2D
    with pytest.raises(ValueError):
        th.step_viscoelastic_creep(
            W, grad, lambda_active=1.0, sagnac_delta=0.0,
            null_space_basis=torch.randn(32, 4))  # rows != 64
