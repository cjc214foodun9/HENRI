"""Contract tests for Carrier F13 hierarchical goal-steering engine.

Covers: default-OFF guard, waypoint geodesic interpolation endpoints,
directional J scoring + Sagnac penalty, signed goal-convergence valence,
beam-search first-action commitment, synthetic favorable-stream pre-flight
kill (K2 rule), determinism, and grad-tensor safety on the steering path.

Directive HENRI-DIR-2026-08-F12-POSTMORTEM-HIERARCHICAL-STEERING (d02eca2c).
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "HENRI V2" / "experiments" / "verification"))

os.environ.pop("HENRI_F13_STEERING", None)

import arc_f13_steering_engine as f13


# ---------------------------------------------------------------------------
# C1 — default-OFF guard
# ---------------------------------------------------------------------------
def test_c1_default_off_guard():
    with pytest.raises(RuntimeError):
        f13.require_f13_enabled()
    f13.require_f13_enabled(_force_enabled=True)


# ---------------------------------------------------------------------------
# C2 — waypoint geodesic interpolation (directive Tier 1 formula)
# ---------------------------------------------------------------------------
def test_c2_waypoint_interpolation():
    torch.manual_seed(0)
    engine = f13.SteeringEngine(D=64, n_actions=8, seed=0)
    psi_t = F.normalize(torch.randn(64), dim=-1)
    psi_goal = F.normalize(torch.randn(64), dim=-1)

    wp0 = engine.waypoint(psi_t, psi_goal, tau=0.0)
    assert torch.allclose(wp0, F.normalize(psi_t, dim=-1), atol=1e-5)

    wp1 = engine.waypoint(psi_t, psi_goal, tau=1.0)
    assert torch.allclose(wp1, F.normalize(psi_goal, dim=-1), atol=1e-5)

    wp = engine.waypoint(psi_t, psi_goal, tau=0.25)
    assert torch.allclose(torch.linalg.vector_norm(wp), torch.tensor(1.0), atol=1e-5)
    # alignment with goal strictly increases with tau (monotone steering)
    sim_lo = f13.abs_cos(engine.waypoint(psi_t, psi_goal, 0.1), psi_goal)
    sim_hi = f13.abs_cos(engine.waypoint(psi_t, psi_goal, 0.9), psi_goal)
    assert sim_hi > sim_lo


# ---------------------------------------------------------------------------
# C3 — directional J scoring: aligned macro-path beats misaligned; Sagnac
#      penalty lowers J (directive Tier 2 formula)
# ---------------------------------------------------------------------------
def test_c3_directional_j_scoring():
    torch.manual_seed(1)
    engine = f13.SteeringEngine(D=64, n_actions=8, seed=1)
    psi = F.normalize(torch.randn(64), dim=-1)
    waypoint = engine.rollout(psi, 1)  # action 1's one-step rollout IS the waypoint

    # horizon=1: beam must commit action 1 (J = 1 - 0 = 1, maximal)
    action, _ = engine.beam_search(psi, waypoint, candidates=[0, 1, 2, 3, 4, 5, 6, 7],
                                   horizon=1, beam=8, alpha=0.05)
    assert action == 1

    # penalty is zero on a perfectly aligned path (coherence loss = 0)
    aligned = [waypoint]
    assert engine.score_path(aligned, waypoint, alpha=0.05) == pytest.approx(1.0, abs=1e-4)
    # on a MISALIGNED path the Sagnac penalty strictly lowers J
    misaligned = [F.normalize(torch.randn(64), dim=-1)]
    j_no_pen = engine.score_path(misaligned, waypoint, alpha=0.0)
    j_pen = engine.score_path(misaligned, waypoint, alpha=0.05)
    assert j_pen < j_no_pen


# ---------------------------------------------------------------------------
# C4 — signed goal-convergence valence (directive Tier 4 literal)
# ---------------------------------------------------------------------------
def test_c4_signed_goal_valence():
    torch.manual_seed(2)
    engine = f13.SteeringEngine(D=64, n_actions=8, seed=2)
    waypoint = F.normalize(torch.randn(64), dim=-1)
    psi_t = F.normalize(torch.randn(64), dim=-1)

    # moving toward the waypoint -> positive valence
    psi_next = F.normalize(0.5 * psi_t + 0.5 * waypoint, dim=-1)
    dnu_pos = engine.valence_delta(psi_next, psi_t, waypoint)
    assert dnu_pos > 0.0

    # moving away from the waypoint -> negative valence
    v = F.normalize(torch.randn(64), dim=-1)
    v = F.normalize(v - (v @ waypoint) * waypoint, dim=-1)  # orthogonal to waypoint
    psi_away = F.normalize(waypoint - 0.5 * v, dim=-1)
    dnu_neg = engine.valence_delta(psi_away, waypoint, waypoint)
    assert dnu_neg < 0.0


# ---------------------------------------------------------------------------
# C5 — signed valence drives Hebbian creep (memory row moves, L2-normalized)
# ---------------------------------------------------------------------------
def test_c5_signed_valence_creep():
    torch.manual_seed(3)
    engine = f13.SteeringEngine(D=64, n_actions=8, seed=3)
    psi = F.normalize(torch.randn(64), dim=-1)
    before = engine.memory.M[2].clone()
    engine.creep(2, +0.5, psi)
    after_pos = engine.memory.M[2].clone()
    assert not torch.equal(after_pos, before)
    assert torch.allclose(torch.linalg.vector_norm(after_pos), torch.tensor(1.0), atol=1e-5)

    # zero valence -> no movement
    engine.creep(2, 0.0, psi)
    assert torch.equal(engine.memory.M[2], after_pos)


# ---------------------------------------------------------------------------
# C6 — beam search commits the FIRST action of the best macro-path
# ---------------------------------------------------------------------------
def test_c6_beam_search_first_action():
    torch.manual_seed(4)
    engine = f13.SteeringEngine(D=64, n_actions=8, seed=4)
    psi = F.normalize(torch.randn(64), dim=-1)
    waypoint = engine.rollout(psi, 3)
    action, info = engine.beam_search(psi, waypoint, candidates=list(range(8)),
                                      horizon=3, beam=4, alpha=0.05)
    assert action in range(8)
    assert info["horizon"] == 3
    assert info["beam"] == 4
    assert len(info["actions"]) == 3  # committed macro-path length == horizon
    assert info["actions"][0] == action  # Tier 3: commit a_1*


# ---------------------------------------------------------------------------
# C7 — pre-flight kill: synthetic favorable stream must yield mean dnu > 0
#      (K2 rule; no live run without this pass). Construction is provably
#      monotone: psi_k = normalize((1-t_k)*psi_0 + t_k*waypoint) with psi_0
#      orthogonal to the waypoint; |cos(psi_k, waypoint)| = t_k /
#      sqrt((1-t_k)^2 + t_k^2) is strictly increasing in t_k in [0,1).
# ---------------------------------------------------------------------------
def test_c7_synthetic_favorable_stream():
    torch.manual_seed(5)
    engine = f13.SteeringEngine(D=64, n_actions=8, seed=5)
    waypoint = F.normalize(torch.randn(64), dim=-1)
    v = F.normalize(torch.randn(64), dim=-1)
    psi0 = F.normalize(v - (v @ waypoint) * waypoint, dim=-1)  # orthogonal start
    assert abs(psi0 @ waypoint) < 1e-5
    prev = psi0
    dnus = []
    for k in range(1, 21):
        t = k / 20.0
        psi_k = F.normalize((1.0 - t) * psi0 + t * waypoint, dim=-1)
        dnus.append(engine.valence_delta(psi_k, prev, waypoint))
        prev = psi_k
    assert all(d > 0.0 for d in dnus)
    assert float(torch.tensor(dnus).mean()) > 0.0


# ---------------------------------------------------------------------------
# C8 — determinism (sealed-carrier reproducibility)
# ---------------------------------------------------------------------------
def test_c8_determinism():
    torch.manual_seed(6)
    e1 = f13.SteeringEngine(D=64, n_actions=8, seed=42)
    e2 = f13.SteeringEngine(D=64, n_actions=8, seed=42)
    psi = F.normalize(torch.randn(64), dim=-1)
    goal = F.normalize(torch.randn(64), dim=-1)
    wp = e1.waypoint(psi, goal, 0.25)
    for a in range(8):
        assert torch.equal(e1.expD[a], e2.expD[a])
        assert torch.equal(e1.rollout(psi, a), e2.rollout(psi, a))
    a1, i1 = e1.beam_search(psi, wp, list(range(8)), horizon=3, beam=4, alpha=0.05)
    a2, i2 = e2.beam_search(psi, wp, list(range(8)), horizon=3, beam=4, alpha=0.05)
    assert a1 == a2
    assert i1["actions"] == i2["actions"]
    e3 = f13.SteeringEngine(D=64, n_actions=8, seed=43)
    assert not torch.equal(e1.expD[0], e3.expD[0])


# ---------------------------------------------------------------------------
# C10 — live gauntlet data-path boundary: SinglePassHorizon consumes the
#       BATCHED [1, num_blocks, 8] ingress wave; the flat [64] engine wave
#       raises. F13 run 1 fail-closed at step 0 on this exact shape
#       (receipt 8eb1e6ad… preserved as evidence); this test locks the fix.
# ---------------------------------------------------------------------------
def test_c10_horizon_batched_boundary():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "HENRI V2" / "experiments" / "verification"))
    from arc_f10_live_engine import PatchIngress, SinglePassHorizon
    torch.manual_seed(20260911)
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=20260911)
    horizon = SinglePassHorizon(d=64, rank=32, K=8, num_blocks=8, seed=20260911)
    raw = torch.randn(4096)
    psi_b = ingress(raw.unsqueeze(0))
    assert tuple(psi_b.shape) == (1, 8, 8)
    roll = horizon(psi_b)  # batched: must not raise
    assert tuple(roll[0, 0].shape) == (8, 8)
    with pytest.raises(RuntimeError):
        horizon(psi_b[0])  # flat [64]: the run-1 live error shape '[8, 64]' invalid


# ---------------------------------------------------------------------------
# C11 — vectorized beam search equivalence: the batched tensor beam must
#       select the SAME macro-path as the naive per-path loop. F13 run 2
#       (PID 851590) hit 38.98 ms/step with the Python-loop beam (G1 FAIL,
#       receipt bd25fec4… preserved as evidence); the vectorized form is the
#       directive's prescribed single-pass batch computation.
# ---------------------------------------------------------------------------
def _naive_beam(engine, psi, waypoint, candidates, horizon, beam, alpha):
    paths = [([], psi.reshape(-1).float(), 0.0, 0.0)]  # (acts, state, ssum, jp)
    for _ in range(horizon):
        expanded = []
        for acts, state, ssum, _jp in paths:
            for a in candidates:
                nxt = engine.rollout(state, a)
                sk = float(engine.sagnac_to(nxt, waypoint).item())
                jp = f13.abs_cos(nxt, waypoint).item() - alpha * (ssum + sk)
                expanded.append((acts + [a], nxt, ssum + sk, jp))
        expanded.sort(key=lambda x: x[3], reverse=True)
        paths = expanded[:beam]
    best = max(paths, key=lambda x: x[3])
    return best[0][0], [int(a) for a in best[0]]


def test_c11_vectorized_beam_equivalence():
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        engine = f13.SteeringEngine(D=64, n_actions=8, seed=seed)
        psi = F.normalize(torch.randn(64), dim=-1)
        goal = F.normalize(torch.randn(64), dim=-1)
        wp = engine.waypoint(psi, goal, 0.25)
        a_v, info_v = engine.beam_search(psi, wp, list(range(8)), horizon=3, beam=4, alpha=0.05)
        a_n, acts_n = _naive_beam(engine, psi, wp, list(range(8)), 3, 4, 0.05)
        assert a_v == a_n, f"seed {seed}: action {a_v} != {a_n}"
        assert info_v["actions"] == acts_n, f"seed {seed}: path {info_v['actions']} != {acts_n}"
        assert isinstance(info_v["actions"], list)
        assert all(isinstance(x, int) for x in info_v["actions"])


# ---------------------------------------------------------------------------
# C9 — steering path is safe on grad-requiring tensors (live ingress path;
#      no numpy/hash on the steering boundary)
# ---------------------------------------------------------------------------
def test_c9_steering_on_grad_tensor():
    torch.manual_seed(7)
    engine = f13.SteeringEngine(D=64, n_actions=8, seed=7)
    x = torch.randn(1, 64, requires_grad=True)
    psi = F.normalize(x, dim=-1)
    assert psi.requires_grad
    goal = F.normalize(torch.randn(64, requires_grad=True), dim=-1)
    wp = engine.waypoint(psi[0], goal[0], 0.25)  # must not raise
    action, _ = engine.beam_search(psi[0], wp, [0, 1, 2, 3], horizon=2, beam=2, alpha=0.05)
    assert action in range(4)
    dnu = engine.valence_delta(psi[0], psi[0], wp)  # must not raise
    assert isinstance(dnu, float)
