"""Contract tests for Carrier F11 closed-loop reward plasticity engine.

Covers: default-OFF guard, valence computation, Tier-2 Hebbian creep geometry,
Tier-3 Langevin escape, R-hat tracking, valence-weighted selection, and the
C7 synthetic in-sample descent pre-flight kill.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "HENRI V2" / "experiments" / "verification"))

os.environ.pop("HENRI_F11_PLASTICITY", None)

import arc_f11_plasticity_engine as f11


# ---------------------------------------------------------------------------
# C1 — default-OFF guard
# ---------------------------------------------------------------------------
def test_c1_default_off_guard():
    with pytest.raises(RuntimeError):
        f11.require_f11_enabled()
    f11.require_f11_enabled(_force_enabled=True)  # explicit opt-in passes


# ---------------------------------------------------------------------------
# C2 — M matrix geometry
# ---------------------------------------------------------------------------
def test_c2_m_matrix_geometry():
    torch.manual_seed(0)
    m = f11.ActionPrototypeMemory(n_actions=8, D=64)
    assert m.M.shape == (8, 64)
    assert torch.all(m.M == 0.0)  # zero init: R-hat = 0 until first valence event
    psi = F.normalize(torch.randn(1, 64), dim=-1)
    m.creep(3, +1.0, psi)
    row = m.M[3]
    assert torch.isfinite(row).all()
    assert torch.allclose(torch.linalg.vector_norm(row), torch.tensor(1.0), atol=1e-5)  # L2-normalized
    assert m.rhat[3] > 0.0  # reward estimate updated


# ---------------------------------------------------------------------------
# C3 — valence computation (D2 mapping)
# ---------------------------------------------------------------------------
def test_c3_valence_computation():
    # level-up: +1.0
    assert f11.compute_valence(prev_levels=0, cur_levels=1, was_reset=False) == pytest.approx(1.0)
    # no change: 0.0
    assert f11.compute_valence(prev_levels=2, cur_levels=2, was_reset=False) == pytest.approx(0.0)
    # reset penalty: -0.5
    assert f11.compute_valence(prev_levels=1, cur_levels=1, was_reset=True) == pytest.approx(-0.5)
    # level-up + reset on same step: +0.5
    assert f11.compute_valence(prev_levels=1, cur_levels=2, was_reset=True) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# C4 — Tier-2 creep directionality
# ---------------------------------------------------------------------------
def test_c4_creep_directionality():
    torch.manual_seed(1)
    m = f11.ActionPrototypeMemory(n_actions=8, D=64)
    psi_a = F.normalize(torch.randn(1, 64), dim=-1)
    psi_b = F.normalize(torch.randn(1, 64), dim=-1)
    cos_before = F.cosine_similarity(m.M[2].unsqueeze(0), psi_a).item()
    for _ in range(20):
        m.creep(2, +0.05, psi_a)  # positive valence: pull toward psi_a
    cos_after_pos = F.cosine_similarity(m.M[2].unsqueeze(0), psi_a).item()
    assert cos_after_pos > cos_before + 0.05
    for _ in range(20):
        m.creep(2, -0.05, psi_b)  # negative valence: push away from psi_b
    cos_after_neg = F.cosine_similarity(m.M[2].unsqueeze(0), psi_b).item()
    assert cos_after_neg < 0.2  # pushed well away


# ---------------------------------------------------------------------------
# C5 — R-hat moving average
# ---------------------------------------------------------------------------
def test_c5_rhat_tracking():
    torch.manual_seed(2)
    m = f11.ActionPrototypeMemory(n_actions=8, D=64, rhat_decay=0.9)
    psi = F.normalize(torch.randn(1, 64), dim=-1)
    m.update_rhat(5, +1.0)
    assert m.rhat[5] == pytest.approx(0.1)  # first update: (1-0.9)*1.0
    m.update_rhat(5, +1.0)
    assert m.rhat[5] == pytest.approx(0.19)  # 0.9*0.1 + 0.1*1.0
    # creep also refreshes rhat from the prototype cosine
    m.creep(5, +1.0, psi)
    assert m.rhat[5] > 0.0


# ---------------------------------------------------------------------------
# C6 — Tier-3 anisotropic Langevin escape
# ---------------------------------------------------------------------------
def test_c6_langevin_escape():
    f11_engine = f11.F11PlasticityEngine(D=64, n_actions=8, seed=3)
    # no negative streak -> base temperature
    assert f11_engine.active_temperature(streak=0, delta_nu=0.0) == pytest.approx(f11_engine.t_base)
    # negative streak -> elevated temperature
    t_hot = f11_engine.active_temperature(streak=5, delta_nu=-1.0)
    assert t_hot > f11_engine.t_base + 0.2
    assert t_hot <= f11_engine.t_base + f11_engine.kappa  # capped at kappa*1.0


# ---------------------------------------------------------------------------
# C7 — pre-flight kill: synthetic in-sample descent (rewarded action wins)
# ---------------------------------------------------------------------------
def test_c7_synthetic_in_sample_descent():
    torch.manual_seed(4)
    engine = f11.F11PlasticityEngine(D=64, n_actions=8, seed=4)
    # two distinguishable wave states; action 1 always rewarded, action 2 always penalized
    psi_good = F.normalize(torch.randn(64), dim=-1)
    psi_bad = F.normalize(torch.randn(64), dim=-1)
    # ensure separation
    if F.cosine_similarity(psi_good, psi_bad, dim=0).item() > 0.3:
        psi_bad = F.normalize(psi_good - 2.0 * psi_bad, dim=-1)
    # 30 episodes of creep: action 1 positive valence on psi_good
    for _ in range(30):
        engine.memory.creep(1, +0.05, psi_good.unsqueeze(0))
        engine.memory.creep(2, -0.05, psi_bad.unsqueeze(0))
    # rewarded action prototype must have pulled toward psi_good
    cos_1 = F.cosine_similarity(engine.memory.M[1].unsqueeze(0), psi_good.unsqueeze(0)).item()
    cos_2 = F.cosine_similarity(engine.memory.M[2].unsqueeze(0), psi_good.unsqueeze(0)).item()
    assert cos_1 > 0.5
    assert cos_1 > cos_2 + 0.3
    # R-hat favors the rewarded action
    assert engine.memory.rhat[1] > engine.memory.rhat[2]


# ---------------------------------------------------------------------------
# C8 — valence-weighted selection prefers rewarded action
# ---------------------------------------------------------------------------
def test_c8_valence_weighted_selection():
    torch.manual_seed(5)
    engine = f11.F11PlasticityEngine(D=64, n_actions=8, seed=5, lambda_reward=2.0)
    psi = F.normalize(torch.randn(1, 64), dim=-1)
    # give action 4 a strong positive reward history
    for _ in range(25):
        engine.memory.creep(4, +0.05, psi)
    # selection over candidates {4, 6} with identical Sagnac to goal: reward term must win
    sagnac = {4: 0.30, 6: 0.30}
    rhat = {4: engine.memory.rhat[4], 6: engine.memory.rhat[6]}
    chosen = engine.select_valence_weighted(sagnac, rhat, candidates=[4, 6])
    assert chosen == 4
