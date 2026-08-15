"""Phase 8.11 contract tests — native complex wave space transition (default-OFF).

Gates (pre-registered, phase811_native_complex_transition_design.md):
- G1a: identity — forward(s, zero-phase action) cos-sim 1.0 vs s at egress.
- G1b: native-complex exactness (accept) — closed-form angle-residual fit
       across 32 synthetic diagonal-phase pairs; held-out REAL-egress Sagnac
       <= 0.05 within <= 3 fit calls (small dims on CPU).
- G3:  egress contract — real [B, 8] per-block unit, float32, finite.
- G4:  default-OFF — flag OFF -> legacy LowRankCoupledTransition; flag ON
       -> NativeComplexWaveTransition.
- G6:  mutual exclusion — use_diagonal_transition + use_complex_transition
       raises ValueError (intended boundary).
- G7:  fail-closed — use_complex_transition + learnable_actions raises
       ValueError (intended boundary).
- WIRE: select_action / train_transition_step run on the complex branch and
       return finite values; field_channel_wave/load_field_channel_wave
       round-trip the complex action-phase buffer.

Small dims on CPU (num_blocks=64, d=512) for speed; CUDA matrix separately
(phase811_native_complex_cuda_check.py at D=65,536 on RTX 5090).
"""

import math

import pytest
import torch

from complex_phase_transition import NativeComplexWaveTransition


NB = 64
BD = 8
D = NB * BD
NA = 4


@pytest.fixture
def adapter():
    return NativeComplexWaveTransition(
        dimension=D, num_actions=NA, device="cpu",
        num_blocks=NB, block_dim=BD)


def _unit_wave(seed: int = 0, num_blocks: int = NB, block_dim: int = BD) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(num_blocks, block_dim, generator=g)
    return w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)


def _zero_action(num_blocks: int = NB, block_dim: int = BD) -> torch.Tensor:
    return torch.zeros(num_blocks, block_dim)


def _diag_phase_triples(n: int, seed: int, num_blocks: int = NB,
                        block_dim: int = BD, span: float = 0.6):
    """Real unit waves s_i and exact diagonal-phase targets
    n_i = per_block_normalize(cos(acos(s_i) + delta)) with ONE SHARED delta
    (single action rotation; the closed-form fit must recover it and
    generalize across unseen states)."""
    g = torch.Generator().manual_seed(seed)
    states, nexts = [], []
    delta = ((torch.rand(D, generator=g) - 0.5) * 2.0 * span)
    for i in range(n):
        s = _unit_wave(seed + i, num_blocks, block_dim)
        alpha = torch.acos(s.reshape(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        n = torch.cos(alpha + delta).reshape(num_blocks, block_dim)
        n = n / (torch.norm(n, p=2, dim=-1, keepdim=True) + 1e-9)
        states.append(s)
        nexts.append(n)
    return torch.stack(states), torch.stack(nexts)


def _phasor_pairs(n: int, delta_seed: int, alpha_seed: int, span: float = 0.6):
    """NATIVE-DOMAIN trajectory pairs (per pre-registration G1): per-element
    unit-modulus phasors z_{t+1} = z_t * exp(j*delta) with ONE SHARED delta
    (the single action rotation). delta is fixed by delta_seed; the states
    (alpha) vary by alpha_seed. Returns list of (z_t, z_n) [D] complex."""
    g = torch.Generator().manual_seed(delta_seed)
    delta = (torch.rand(D, generator=g) - 0.5) * 2.0 * span
    g2 = torch.Generator().manual_seed(alpha_seed)
    pairs = []
    for _ in range(n):
        alpha = (torch.rand(D, generator=g2) - 0.5) * 2.0 * math.pi
        z_t = torch.polar(torch.ones(D), alpha)
        z_n = z_t * torch.polar(torch.ones(D), delta)
        pairs.append((z_t, z_n))
    return pairs


def test_g1a_identity_forward(adapter):
    s = _unit_wave(1)
    out = adapter.forward(s, _zero_action())
    assert out.shape == (NB, BD)
    assert out.dtype == torch.float32
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    cos = torch.nn.functional.cosine_similarity(
        out.reshape(1, -1), s.reshape(1, -1), dim=-1)
    assert cos.item() > 0.9999, f"G1a identity FAIL: cos={cos.item():.6f}"


def test_g1b_accept_closed_form_exactness(adapter):
    """G1b accept gate (mechanism, per pre-registration): native-domain
    per-element unit-modulus phasor pairs z_{t+1} = z_t * exp(j*delta);
    <= 3 closed-form angle-residual calls; held-out REAL-egress Sagnac
    <= 0.05 across 32 pairs (d=512 local)."""
    n_fit, n_eval = 32, 32
    fit_pairs = _phasor_pairs(n_fit, delta_seed=100, alpha_seed=200)
    eval_pairs = _phasor_pairs(n_eval, delta_seed=100, alpha_seed=900)
    s0 = adapter.project_to_real_egress(eval_pairs[0][0]).reshape(-1)
    o0 = adapter.project_to_real_egress(eval_pairs[0][1]).reshape(-1)
    pre = float(1.0 - torch.dot(s0, o0) / (torch.norm(s0) * torch.norm(o0)).clamp(min=1e-12))
    for _ in range(3):
        for zt, zn in fit_pairs:
            adapter.update_phase_complex(zt, zn, 0, lr=1.0)
    sags = []
    for zt, zn in eval_pairs:
        p = adapter.project_to_real_egress(
            adapter.forward_complex(zt, 0)).reshape(-1)
        o = adapter.project_to_real_egress(zn).reshape(-1)
        sags.append(float(1.0 - torch.dot(p, o) /
                          (torch.norm(p) * torch.norm(o)).clamp(min=1e-12)))
    post = sum(sags) / len(sags)
    assert post <= 0.05, f"G1b accept FAIL: held-out Sagnac={post:.6f} (pre={pre:.6f})"


def test_g2_real_lift_transfer_boundary(adapter):
    """G2 boundary (pre-registered EXPECTED FAIL): production REAL
    L2-normalized waves lift (acos) -> rotate -> egress. The lossy lift means
    the closed-form fit cannot improve held-out real Sagnac by 0.02.
    Kill gate = post >= pre - 0.02 (asserts the honest boundary; NOT a
    mechanism kill — next lever is a complex-native encoder at ingress)."""
    n_fit, n_eval = 32, 16
    s_fit, n_fit_t = _diag_phase_triples(n_fit, seed=100)
    s_eval, n_eval_t = _diag_phase_triples(n_eval, seed=900)
    a = _unit_wave(7)  # stable nonzero action (fingerprint -> index 0)
    pre = float(1.0 - torch.nn.functional.cosine_similarity(
        adapter.forward(s_eval[0], a).reshape(1, -1),
        n_eval_t[0].reshape(1, -1), dim=-1))
    for _ in range(3):
        adapter.fit_batch(s_fit, a.repeat(n_fit, 1, 1), n_fit_t,
                          iters=1, lr=1.0)
    sags = []
    for i in range(n_eval):
        p = adapter.forward(s_eval[i], a).reshape(-1)
        o = n_eval_t[i].reshape(-1)
        sags.append(float(1.0 - torch.dot(p, o) /
                          (torch.norm(p) * torch.norm(o)).clamp(min=1e-12)))
    post = sum(sags) / len(sags)
    # kill fires (gate records) when NO improvement >= 0.02
    assert post >= pre - 0.02, \
        f"G2 boundary UNEXPECTED improvement: pre={pre:.6f} post={post:.6f}"


def test_g3_egress_contract(adapter):
    s = _unit_wave(3)
    a = _unit_wave(4)
    out = adapter.forward(s, a)
    assert out.shape == (NB, BD)
    assert out.dtype == torch.float32
    assert bool(torch.isfinite(out).all())
    assert bool((out >= -1.0 - 1e-6).all()) and bool((out <= 1.0 + 1e-6).all())
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), \
        "G3 egress: per-block L2 norm != 1.0"


def test_g4_default_off_and_on(adapter):
    from efe_planner import EFEPlanner, LowRankCoupledTransition
    p_off = EFEPlanner(num_blocks=NB, d_model=D, num_actions=NA)
    assert isinstance(p_off.transition, LowRankCoupledTransition)
    assert not p_off._use_complex_transition
    p_on = EFEPlanner(num_blocks=NB, d_model=D, num_actions=NA,
                      use_complex_transition=True)
    assert isinstance(p_on.transition, NativeComplexWaveTransition)
    assert p_on._use_complex_transition


def test_g6_mutual_exclusion(adapter):
    from efe_planner import EFEPlanner
    with pytest.raises(ValueError, match="mutually"):
        EFEPlanner(num_blocks=NB, d_model=D, num_actions=NA,
                   use_diagonal_transition=True, use_complex_transition=True)


def test_g7_fail_closed_learnable_actions(adapter):
    from efe_planner import EFEPlanner
    with pytest.raises(ValueError, match="learnable_actions=False"):
        EFEPlanner(num_blocks=NB, d_model=D, num_actions=NA,
                   use_complex_transition=True, learnable_actions=True)


def test_wire_select_action_and_train_step(adapter):
    from efe_planner import EFEPlanner
    p_on = EFEPlanner(num_blocks=NB, d_model=D, num_actions=NA,
                      use_complex_transition=True)
    s = _unit_wave(11)
    cands = [(i, _unit_wave(30 + i)) for i in range(NA)]
    boundary = _unit_wave(99).unsqueeze(0)
    action_id, predicted, table, chosen = p_on.select_action(
        s, cands, boundary_axioms=boundary)
    assert len(table) == NA
    assert bool(torch.isfinite(predicted).all())
    a4 = _unit_wave(12)
    out4 = p_on.transition(s, a4)
    pre = p_on.train_transition_step(s, a4, out4, lr=0.05)
    assert bool(torch.isfinite(torch.tensor(pre)))


def test_wire_field_channel_roundtrip(adapter):
    from efe_planner import EFEPlanner
    p_on = EFEPlanner(num_blocks=NB, d_model=D, num_actions=NA,
                      use_complex_transition=True)
    wave = p_on.field_channel_wave()
    assert wave.numel() == NA * D
    p_on.load_field_channel_wave(wave)
    assert bool(torch.isfinite(p_on.transition.action_phases).all())
