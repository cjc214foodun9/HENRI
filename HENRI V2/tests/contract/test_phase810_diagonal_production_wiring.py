"""Phase 8.10 contract tests — diagonal transition production wiring (default-OFF).

Gates (pre-registered, phase810_diagonal_production_wiring_design.md):
- G1: identity — forward(s, zero-phase action) cos-sim 1.0 vs s.
- G2: analytic recovery — CC-OS carrier-encoded translation triples; after
  closed-form fit, held-out REAL-metric Sagnac < 0.30.
- G3: convergence — batch Sagnac < 0.05 within <= 3 fit calls on synthetic
  diagonal-phase triples.
- G4: default-OFF byte identity — flag OFF -> LowRankCoupledTransition
  (adapter never constructed); legacy forward finite + per-block unit.

Small dims on CPU (num_blocks=64, d=512) for speed; CUDA matrix separately.
"""

import math

import pytest
import torch

from henri_frequency_domain_transition import FrequencyDomainDiagonalAdapter


NB = 64
BD = 8
D = NB * BD
NA = 4


@pytest.fixture
def adapter():
    return FrequencyDomainDiagonalAdapter(
        num_blocks=NB, block_dim=BD, num_actions=NA, device="cpu", d_model=D)


def _unit_wave(seed: int = 0, num_blocks: int = NB, block_dim: int = BD) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(num_blocks, block_dim, generator=g)
    return w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)


def _zero_action(num_blocks: int = NB, block_dim: int = BD) -> torch.Tensor:
    return torch.zeros(num_blocks, block_dim)


def test_g1_identity_forward(adapter):
    s = _unit_wave(1)
    out = adapter.forward(s, _zero_action())
    assert out.shape == (NB, BD)
    # per-block unit
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # cos-sim 1.0 vs s (identity at zero phase)
    cos = torch.nn.functional.cosine_similarity(
        out.reshape(1, -1), s.reshape(1, -1), dim=-1)
    assert cos.item() > 0.999999, f"G1 identity FAIL: cos={cos.item():.6f}"


def test_g3_kill_confirmed_synthetic_budget(adapter):
    """PRE-REGISTERED KILL CONFIRMED (G3): batch Sagnac < 0.05 within <= 3
    fit calls. OBSERVED at budget: 0.285 (75 steps @ 0.04) — gate FAILS.
    The real-domain arccos bridge is FALSIFIED for the production L2-normalized
    wave regime; module stays default-OFF. This test pins the measured verdict
    (passes only while the kill stands)."""
    g = torch.Generator().manual_seed(7)
    delta = (torch.rand(D, generator=g) - 0.5) * 1.2
    s = _unit_wave(3)
    alpha = torch.acos(s.reshape(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    n = torch.cos(alpha + delta).reshape(NB, BD)
    n = n / (torch.norm(n, p=2, dim=-1, keepdim=True) + 1e-9)
    a = _zero_action()
    for call in range(3):
        adapter.fit_batch(s.unsqueeze(0), a.unsqueeze(0), n.unsqueeze(0),
                          iters=1, lr=1.0)
    post = float(adapter._real_sagnac(adapter.forward(s, a), n))
    assert post > 0.05, (
        f"KILL NO LONGER CONFIRMED: post={post:.6f} < 0.05 at budget — "
        f"the bridge would now pass its gate; re-audit required")
    assert post < 0.40, f"unexpected divergence: post={post:.6f}"


def test_mechanism_asymptotic_evidence():
    """Asymptotic evidence: the gradient path on the real-Sagnac loss DOES
    converge given 3000 steps @ 0.1 (post 0.00013). The mechanism exists;
    the failure is the shallow landscape of the normalized real regime, which
    cannot be identified within the pre-registered production budget."""
    from henri_frequency_domain_transition import FrequencyDomainDiagonalAdapter as A
    adapter = A(num_blocks=NB, block_dim=BD, num_actions=NA, device="cpu", d_model=D)
    g = torch.Generator().manual_seed(7)
    delta = (torch.rand(D, generator=g) - 0.5) * 1.2
    s = _unit_wave(3)
    alpha = torch.acos(s.reshape(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    n = torch.cos(alpha + delta).reshape(NB, BD)
    n = n / (torch.norm(n, p=2, dim=-1, keepdim=True) + 1e-9)
    a = _zero_action()
    adapter._sgd_fit([s], [a], [n], steps=3000, step_lr=0.1)
    post = float(adapter._real_sagnac(adapter.forward(s, a), n))
    assert post < 0.01, f"asymptotic evidence FAIL: post={post:.6f}"


def test_g2_kill_confirmed_carrier_regime():
    """PRE-REGISTERED KILL CONFIRMED (G2): in the PRODUCTION carrier regime the
    adapter does NOT improve on the identity predictor within budget.
    OBSERVED: pre 0.0342 -> post 0.0344 (held-out); identity baseline 0.0645
    already passes the 0.30 floor, so the improvement gate (post < pre - 0.02)
    is the only discriminator — and it FAILS. The real-domain bridge cannot
    exploit the 8.9 diagonal exactness on L2-normalized real waves."""
    from henri_spatial_carrier_ingress import VectorizedIncommensurateSpatialIngress

    ingress = VectorizedIncommensurateSpatialIngress(dimension=D, device="cpu")
    adapter = FrequencyDomainDiagonalAdapter(
        num_blocks=NB, block_dim=BD, num_actions=NA, device="cpu", d_model=D)
    states, nexts, acts = [], [], []
    a_zero = _zero_action()
    for i in range(6):
        c = 1 + (i % 3)
        r0, c0 = 2.0 + i, 3.0 + i
        st = ingress.encode_single_object(color=c, cx=r0, cy=c0).reshape(NB, BD)
        st = st / (torch.norm(st, p=2, dim=-1, keepdim=True) + 1e-9)
        nxt = ingress.apply_translation(
            ingress.encode_single_object(color=c, cx=r0, cy=c0), dx=1.0, dy=-1.0
        ).reshape(NB, BD)
        nxt = nxt / (torch.norm(nxt, p=2, dim=-1, keepdim=True) + 1e-9)
        states.append(st)
        nexts.append(nxt)
        acts.append(a_zero)
    pre_hold = float(adapter._real_sagnac(
        adapter.forward(states[0], acts[0]), nexts[0]))
    adapter.fit_batch(
        torch.stack(states[1:]), torch.stack(acts[1:]),
        torch.stack(nexts[1:]), iters=1, lr=1.0)
    post_hold = float(adapter._real_sagnac(
        adapter.forward(states[0], acts[0]), nexts[0]))
    # Kill documentation: no improvement (post NOT < pre - 0.02)
    assert post_hold >= pre_hold - 0.02, (
        f"KILL NO LONGER CONFIRMED: pre={pre_hold:.4f} post={post_hold:.4f} — "
        f"bridge now improves; re-audit required")
    assert post_hold < 0.30, (
        f"absolute floor passes but mechanism gate failed: post={post_hold:.4f}")


def test_g4_default_off_legacy_path():
    """Default flag OFF -> LowRankCoupledTransition; adapter never constructed."""
    from efe_planner import EFEPlanner, LowRankCoupledTransition

    planner = EFEPlanner(num_blocks=NB, d_model=D, num_actions=NA)
    assert isinstance(planner.transition, LowRankCoupledTransition)
    assert planner._use_diagonal_transition is False
    s = _unit_wave(11)
    a = _unit_wave(12)
    out = planner.transition(s, a)
    assert out.shape == (NB, BD)
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_diagonal_planner_select_action():
    from efe_planner import EFEPlanner

    planner = EFEPlanner(
        num_blocks=NB, d_model=D, num_actions=NA,
        use_diagonal_transition=True, learnable_actions=False)
    assert planner._use_diagonal_transition is True
    from henri_frequency_domain_transition import FrequencyDomainDiagonalAdapter
    assert isinstance(planner.transition, FrequencyDomainDiagonalAdapter)
    s = _unit_wave(21)
    cands = [(i, _unit_wave(30 + i)) for i in range(NA)]
    boundary = _unit_wave(99).unsqueeze(0)  # [1, NB, BD] real boundary axioms
    action_id, predicted, table, chosen = planner.select_action(
        s, cands, boundary_axioms=boundary)
    assert isinstance(table, list) and len(table) == NA
    for r in table:
        assert r["predicted_wave"].shape == (NB, BD)
        assert torch.isfinite(r["predicted_wave"]).all()
        assert torch.isfinite(torch.tensor(r["efe"]))
    # ascending EFE order (best first)
    efes = [r["efe"] for r in table]
    assert efes == sorted(efes)


def test_train_transition_step_diagonal_branch():
    from efe_planner import EFEPlanner

    planner = EFEPlanner(
        num_blocks=NB, d_model=D, num_actions=NA,
        use_diagonal_transition=True)
    s = _unit_wave(31)
    a = _unit_wave(32)
    g = torch.Generator().manual_seed(33)
    delta = (torch.rand(D, generator=g) - 0.5) * 0.8
    alpha = torch.acos(s.reshape(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    n = torch.cos(alpha + delta).reshape(NB, BD)
    n = n / (torch.norm(n, p=2, dim=-1, keepdim=True) + 1e-9)
    pre = planner.train_transition_step(s, a, n, lr=1.0)
    post = float(planner.transition._real_sagnac(
        planner.transition.forward(s, a), n))
    assert torch.isfinite(torch.tensor(pre))
    assert post < pre + 1e-6, f"diagonal step FAIL: pre={pre:.4f} post={post:.4f}"


def test_field_channel_roundtrip_diagonal():
    adapter = FrequencyDomainDiagonalAdapter(
        num_blocks=NB, block_dim=BD, num_actions=NA, device="cpu", d_model=D)
    s = _unit_wave(41)
    a = _unit_wave(42)
    g = torch.Generator().manual_seed(43)
    delta = (torch.rand(D, generator=g) - 0.5) * 0.6
    alpha = torch.acos(s.reshape(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    n = torch.cos(alpha + delta).reshape(NB, BD)
    n = n / (torch.norm(n, p=2, dim=-1, keepdim=True) + 1e-9)
    adapter.fit_batch(s.unsqueeze(0), a.unsqueeze(0), n.unsqueeze(0))
    saved = adapter.field_channel_wave()
    adapter2 = FrequencyDomainDiagonalAdapter(
        num_blocks=NB, block_dim=BD, num_actions=NA, device="cpu", d_model=D)
    adapter2.load_field_channel_wave(saved)
    assert torch.allclose(
        adapter.phase_correction, adapter2.phase_correction, atol=1e-6)


def test_carrier_ingress_env_state_to_wave():
    from physical_control_environments import (
        InvertedPendulumEnvironment, CartPolePhysicsEnvironment)

    for env in (
        InvertedPendulumEnvironment(use_carrier_ingress=True),
        CartPolePhysicsEnvironment(use_carrier_ingress=True),
    ):
        w = env.state_to_wave(num_blocks=NB, device=torch.device("cpu"))
        assert w.shape == (NB, BD)
        norms = torch.norm(w, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
