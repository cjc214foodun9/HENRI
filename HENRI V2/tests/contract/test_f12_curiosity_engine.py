"""Contract tests for Carrier F12 sub-goal curiosity engine.

Covers: default-OFF guard, skew-symmetric rollout orthogonality, intrinsic
surprise, count-based novelty, dense intrinsic valence + creep engagement,
Tier-2 curiosity selection, deterministic frontier hashing, and the C7
pre-flight synthetic in-sample engagement kill.

Directive HENRI-DIR-2026-08-F11-POSTMORTEM-SUBGOAL-INGRESS (f19107fd).
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "HENRI V2" / "experiments" / "verification"))

os.environ.pop("HENRI_F12_CURIOSITY", None)

import arc_f12_curiosity_engine as f12


# ---------------------------------------------------------------------------
# C1 — default-OFF guard
# ---------------------------------------------------------------------------
def test_c1_default_off_guard():
    with pytest.raises(RuntimeError):
        f12.require_f12_enabled()
    f12.require_f12_enabled(_force_enabled=True)


# ---------------------------------------------------------------------------
# C2 — skew-symmetric rollout operators are orthogonal (norm-preserving)
# ---------------------------------------------------------------------------
def test_c2_rollout_orthogonality():
    torch.manual_seed(0)
    engine = f12.CuriosityEngine(D=64, n_actions=8, seed=0)
    for a in range(8):
        O = engine.expD[a]
        # exp(skew) is orthogonal: O^T O == I
        err = torch.linalg.matrix_norm(O.T @ O - torch.eye(64)).item()
        assert err < 1e-4, f"action {a} orthogonality error {err}"
    # norm preservation on a random wave
    psi = F.normalize(torch.randn(64), dim=-1)
    for a in range(8):
        pred = engine.rollout(psi, a)
        assert torch.allclose(torch.linalg.vector_norm(pred),
                              torch.linalg.vector_norm(psi), atol=1e-4)


# ---------------------------------------------------------------------------
# C3 — intrinsic prediction surprise
# ---------------------------------------------------------------------------
def test_c3_intrinsic_surprise():
    torch.manual_seed(1)
    engine = f12.CuriosityEngine(D=64, n_actions=8, seed=1)
    psi = F.normalize(torch.randn(64), dim=-1)
    # identical -> |cos| = 1 -> surprise 0
    assert engine.surprise(psi, psi) == pytest.approx(0.0, abs=1e-5)
    # orthogonal -> |cos| = 0 -> surprise 1
    orth = F.normalize(torch.randn(64), dim=-1)
    orth = F.normalize(orth - (orth @ psi) * psi, dim=-1)  # Gram-Schmidt
    assert engine.surprise(psi, orth) == pytest.approx(1.0, abs=1e-4)
    # bounded [0, 1]
    s = engine.surprise(psi, F.normalize(torch.randn(64), dim=-1))
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# C4 — count-based novelty hash frontier
# ---------------------------------------------------------------------------
def test_c4_novelty_frontier():
    torch.manual_seed(2)
    engine = f12.CuriosityEngine(D=64, n_actions=8, seed=2)
    psi_a = F.normalize(torch.randn(64), dim=-1)
    psi_b = F.normalize(torch.randn(64), dim=-1)
    h_a = engine.hash_wave(psi_a)
    h_b = engine.hash_wave(psi_b)
    assert h_a != h_b  # distinct waves -> distinct hashes (overwhelmingly likely)
    # first visit: novelty 1/sqrt(0+1) = 1.0
    assert engine.novelty(h_a) == pytest.approx(1.0)
    engine.visit(h_a)  # Tier 4: increment AFTER reward (D7)
    assert engine.novelty(h_a) == pytest.approx(1.0 / (2.0 ** 0.5))
    assert engine.frontier[h_a] == 1
    # hash determinism
    assert engine.hash_wave(psi_a) == h_a


# ---------------------------------------------------------------------------
# C5 — dense intrinsic valence + Hebbian creep engagement
# ---------------------------------------------------------------------------
def test_c5_dense_valence_and_creep():
    torch.manual_seed(3)
    engine = f12.CuriosityEngine(D=64, n_actions=8, seed=3)
    psi = F.normalize(torch.randn(64), dim=-1)
    # predicted next state far from actual -> surprise > 0; novel state -> novelty > 0
    pred = engine.rollout(psi, 2)
    actual = F.normalize(torch.randn(64), dim=-1)
    if f12.abs_cos(pred, actual) > 0.5:
        actual = F.normalize(pred - 2.0 * actual, dim=-1)
    r_ext = 0.0  # no exteroceptive signal (sparse-reward regime)
    nu = engine.compute_valence_intrinsic(r_ext, pred, actual, engine.hash_wave(actual))
    assert nu > 0.0  # dense: non-zero even with r_ext = 0
    before = engine.memory.M[2].clone()
    engine.creep(2, nu, psi)
    assert not torch.equal(engine.memory.M[2], before)  # creep moved the row
    assert torch.allclose(torch.linalg.vector_norm(engine.memory.M[2]),
                          torch.tensor(1.0), atol=1e-5)  # L2-normalized


# ---------------------------------------------------------------------------
# C6 — Tier-2 intrinsic action selection (directive formula: argmax of
#      lambda_cur*(1 - |cos(rollout(a), goal)|) + lambda_nov*novelty)
# ---------------------------------------------------------------------------
def test_c6_intrinsic_selection():
    torch.manual_seed(4)
    engine = f12.CuriosityEngine(D=64, n_actions=8, seed=4)
    psi = F.normalize(torch.randn(64), dim=-1)
    goal = engine.rollout(psi, 1)
    # lambda_nov=0: selection = argmax_a (1 - |cos(rollout(a), goal)|).
    # Action 1's rollout IS the goal (score 0); action 6 differs -> higher score.
    assert f12.abs_cos(engine.rollout(psi, 6), goal) < 0.999  # distinct rotations
    sel = engine.select_intrinsic(psi, goal, candidates=[1, 6], lambda_nov=0.0)
    assert sel == 6
    # lambda_cur=0: novelty preference — pre-visit action 5, unvisited 6 wins.
    psi2 = F.normalize(torch.randn(64), dim=-1)
    engine.visit(engine.hash_wave(engine.rollout(psi2, 5)))  # pre-visit action 5
    sel2 = engine.select_intrinsic(psi2, goal, candidates=[5, 6], lambda_cur=0.0)
    assert sel2 == 6


# ---------------------------------------------------------------------------
# C7 — pre-flight kill: synthetic in-sample engagement (K2 rule)
# ---------------------------------------------------------------------------
def test_c7_synthetic_in_sample_engagement():
    torch.manual_seed(5)
    engine = f12.CuriosityEngine(D=64, n_actions=8, seed=5)
    psi = F.normalize(torch.randn(64), dim=-1)
    creeps = 0
    # constructed stream of novel states: every step must produce delta_nu > 0
    for i in range(10):
        actual = F.normalize(torch.randn(64), dim=-1)
        pred = engine.rollout(psi, i % 8)
        if f12.abs_cos(pred, actual) > 0.5:
            actual = F.normalize(pred - 2.0 * actual, dim=-1)
        nu = engine.compute_valence_intrinsic(0.0, pred, actual, engine.hash_wave(actual))
        assert nu > 0.0
        engine.creep(i % 8, nu, psi)
        engine.visit(engine.hash_wave(actual))
        creeps += 1
        psi = actual
    assert creeps == 10  # dense engagement, not dormant


# ---------------------------------------------------------------------------
# C8 — rollout determinism (sealed-carrier reproducibility)
# ---------------------------------------------------------------------------
def test_c8_rollout_determinism():
    torch.manual_seed(6)
    e1 = f12.CuriosityEngine(D=64, n_actions=8, seed=42)
    e2 = f12.CuriosityEngine(D=64, n_actions=8, seed=42)
    psi = F.normalize(torch.randn(64), dim=-1)
    for a in range(8):
        assert torch.equal(e1.rollout(psi, a), e2.rollout(psi, a))
        assert torch.equal(e1.expD[a], e2.expD[a])
    # different seed -> different operators
    e3 = f12.CuriosityEngine(D=64, n_actions=8, seed=43)
    assert not torch.equal(e1.expD[0], e3.expD[0])


# ---------------------------------------------------------------------------
# C9 — hash_wave is safe on grad-requiring tensors (live ingress path)
# ---------------------------------------------------------------------------
def test_c9_hash_wave_on_grad_tensor():
    torch.manual_seed(7)
    engine = f12.CuriosityEngine(D=64, n_actions=8, seed=7)
    # Live path: psi from ingress() requires grad; hash must not raise.
    x = torch.randn(1, 64, requires_grad=True)
    psi = F.normalize(x, dim=-1)
    assert psi.requires_grad
    h = engine.hash_wave(psi)
    assert isinstance(h, int)
    assert 0 <= h < 2 ** 32
