"""Contract tests for Carrier G1 topological gauge-wave scattering engine.

Directive: HENRI-DIR-2026-09-V3-TOPOLOGICAL-GAUGE-WAVE-ORDER
(818968573ada74e28353af6d2779390ac09a203cecfacd495183ac2f06c6e0b0, 23,536 B, 352 lines)

C1  flag gate (default-OFF)
C2  moving/blocked partition (displacement threshold 0.05)
C3  affordance classifier AUC >= 0.85 on separable synthetic bank
C4  scattering identity: pi=0 -> psi; pi=1 -> T_free psi
C5  homotopy beam prunes blocked actions (J ~ 0)
C6  free-motion generators fit ONLY on moving transitions (blocked-only -> I)
C7  online affordance update moves prediction toward observed
C8  G4 single-pass scattered consistency ~ 0 in both regimes
C9  waypoint advancement uses the ACTUAL post-action frame (G3 discipline)
C10 verdict precedence (latency -> G2 -> G3 -> G4)
C11 directive-defect control: state-independent W => AUC ~ 0.5 => PG1 kill
C12 no-affordance-engagement detection
"""
from __future__ import annotations

import math
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

try:
    from arc_g1_topological_engine import (
        G1Engine,
        compute_auc,
        compile_free_generators_capped,
        fit_affordance_classifiers,
        require_flag,
        scatter_prediction,
        MOVING_THRESH,
        PG1_MIN_AUC,
        G1_LATENCY_MS,
        G2_MIN_SOLVED,
        G3_MIN_DELTA_NU,
        G4_MAX_AFFORDANCE,
        TAU_SHARP,
    )
except Exception as exc:  # pragma: no cover - import isolation
    raise AssertionError(f"G1 engine import failed: {exc!r}")

D = 64
SEED = 20260924
N_OPEN = 20
N_BLOCK = 20


def _unit(n, d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), p=2, dim=-1)


def _skew_rot(theta=0.25, seed=1):
    """Deterministic SO(64) rotation with spectral norm theta."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, D, generator=g)
    S = 0.5 * (A - A.T)
    n = torch.linalg.matrix_norm(S, ord=2)
    return torch.linalg.matrix_exp(S * (theta / n))


def _bank(seed=7, n_actions=2, n_open=N_OPEN, n_block=N_BLOCK):
    """Per-action mixed open/blocked clusters.

    For each action a: states near center e_open_a move (nxt = T_a psi);
    states near e_block_a are blocked (nxt = psi). Returns
    (psi, nxt, onehot, centers) with centers [D, 2*n_actions].
    """
    g = torch.Generator().manual_seed(seed)
    centers, _ = torch.linalg.qr(torch.randn(D, n_actions * 2, generator=g))
    psi_list, nxt_list, onehot_list = [], [], []
    for a in range(n_actions):
        e_open = centers[:, a]
        e_block = centers[:, n_actions + a]
        xo = F.normalize(e_open.unsqueeze(0) + 0.2 * torch.randn(n_open, D, generator=g), dim=-1)
        xb = F.normalize(e_block.unsqueeze(0) + 0.2 * torch.randn(n_block, D, generator=g), dim=-1)
        T = _skew_rot(theta=0.30, seed=seed + a)
        yo = F.normalize(xo @ T.T, dim=-1)
        yb = xb.clone()  # blocked: zero displacement
        psi_list += [xo, xb]
        nxt_list += [yo, yb]
        oh = torch.zeros(n_open + n_block, n_actions)
        oh[:, a] = 1.0
        onehot_list.append(oh)
    psi = torch.cat(psi_list)
    nxt = torch.cat(nxt_list)
    onehot = torch.cat(onehot_list)
    return psi, nxt, onehot, centers


def _engine(seed=SEED):
    psi, nxt, onehot, centers = _bank()
    comp = compile_free_generators_capped(psi, nxt, onehot, omega_bound=math.pi / 32.0)
    W, b = fit_affordance_classifiers(psi, onehot, comp["is_moving"])
    return G1Engine(
        generators=comp["generators"], transitions=comp["transitions"],
        t_pow=comp["t_pow"], recon=comp["recon"], W_contact=W, b_contact=b,
        action_names=["0", "1"], n_actions=2, seed=seed,
        horizon=8, device="cpu", omega_bound=math.pi / 32.0,
        waypoints=[F.normalize(_unit(1, seed=3)[0], dim=-1)],
        waypoint_advance_thresh=0.60, langevin_temp=0.50,
        eta_affordance=0.10, moving_thresh=MOVING_THRESH,
    ), centers


def _pi(psi, W, b):
    """Directive §1.2 gate: sigmoid((<Psi, W_a Psi> - theta)/tau_sharp)."""
    return torch.sigmoid(torch.einsum("bd,ade,be->ba", psi, W, psi) / TAU_SHARP + b.unsqueeze(0))


def test_c1_flag_gate():
    os.environ.pop("HENRI_G1_TOPOLOGICAL", None)
    try:
        require_flag()
        raise AssertionError("flag gate must refuse without env")
    except RuntimeError:
        pass
    os.environ["HENRI_G1_TOPOLOGICAL"] = "1"
    require_flag()  # must not raise
    os.environ.pop("HENRI_G1_TOPOLOGICAL", None)


def test_c2_moving_blocked_partition():
    psi, nxt, onehot, _ = _bank()
    disp = torch.norm(nxt - psi, p=2, dim=-1)
    is_moving = (disp > MOVING_THRESH).float()
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        assert int(is_moving[mask].sum()) == N_OPEN, f"action {a} moving count"
        assert int((1.0 - is_moving[mask]).sum()) == N_BLOCK, f"action {a} blocked count"


def test_c3_affordance_auc():
    psi, nxt, onehot, _ = _bank()
    comp = compile_free_generators_capped(psi, nxt, onehot, omega_bound=math.pi / 32.0)
    W, b = fit_affordance_classifiers(psi, onehot, comp["is_moving"])
    pi = _pi(psi, W, b)
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        auc = compute_auc(pi[mask, a], comp["is_moving"][mask])
        assert auc >= PG1_MIN_AUC, f"action {a} AUC {auc:.4f} < {PG1_MIN_AUC}"


def test_c4_scattering_identity():
    eng, _ = _engine()
    psi = F.normalize(_unit(1, seed=5)[0], dim=-1)
    pred0 = scatter_prediction(psi, eng.transitions[0], 0.0)
    pred1 = scatter_prediction(psi, eng.transitions[0], 1.0)
    assert torch.allclose(pred0, psi, atol=1e-5), "pi=0 must preserve state"
    assert torch.allclose(pred1, psi @ eng.transitions[0].T, atol=1e-5), "pi=1 must apply T_free"


def test_c5_beam_prunes_blocked():
    eng, centers = _engine()
    # State in action-0's OPEN cluster (passable) -> pi_0 ~ 1.
    e_open0 = centers[:, 0]
    psi = F.normalize(e_open0, dim=-1)
    wp = F.normalize(psi @ eng.transitions[0].T, dim=-1)  # waypoint = moving outcome
    js = eng.score_all_actions(psi, waypoint=wp)
    # Action 1's gate for a state far from its clusters: pi ~ sigmoid(b_1) = 0.5,
    # so J_1 <= 1 * 0.5^8 ~ 0.0039 -> effectively pruned.
    assert js["0"] > js["1"], f"moving action must outscore blocked: {js}"
    assert js["1"] < 1e-2, f"blocked action must be pruned: {js['1']:.4f}"


def test_c6_free_generators_moving_only():
    psi, nxt, onehot, _ = _bank()
    comp = compile_free_generators_capped(psi, nxt, onehot, omega_bound=math.pi / 32.0)
    # Action 0 was fit on moving rows only -> orthogonality preserved + recon high.
    T0 = comp["transitions"][0]
    err = torch.linalg.matrix_norm(T0 @ T0.T - torch.eye(D), ord=2)
    assert err < 1e-4, f"T_free must stay orthogonal, err {err:.2e}"
    assert comp["recon"][0] >= 0.85, f"moving-only recon {comp['recon'][0]:.4f} < 0.85"
    # Blocked-only action (zero moving rows) yields identity.
    psi2 = torch.cat([psi, _unit(5, seed=21)])
    nxt2 = torch.cat([nxt, psi2[-5:].clone()])  # appended rows all blocked
    oh2 = torch.zeros(psi2.shape[0], 3)
    oh2[: psi.shape[0], :2] = onehot
    oh2[psi.shape[0]:, 2] = 1.0
    comp2 = compile_free_generators_capped(psi2, nxt2, oh2, omega_bound=math.pi / 32.0)
    assert torch.allclose(comp2["transitions"][2], torch.eye(D), atol=1e-5), \
        "blocked-only action must yield T = I"


def test_c7_online_affordance_update():
    eng, _ = _engine()
    psi = F.normalize(_unit(1, seed=8)[0], dim=-1)
    pi_before = eng.predict_affordance(psi)[0, 0].item()
    # Simulate unexpected collision: predicted pass, observed blocked.
    eng.update_online_affordance(psi, 0, psi, eta=0.10)
    pi_after = eng.predict_affordance(psi)[0, 0].item()
    assert pi_after < pi_before, f"collision must lower pi: {pi_before:.4f} -> {pi_after:.4f}"
    assert eng.affordance_updates == 1


def test_c8_g4_scattered_consistency():
    eng, centers = _engine()
    # Moving regime: state in action-0's open cluster, actual = T_free psi.
    e_open0 = centers[:, 0]
    psi_m = F.normalize(e_open0, dim=-1)
    psi_next_m = F.normalize(psi_m @ eng.transitions[0].T, dim=-1)
    d_m = eng.g4_single_pass(psi_m, 0, psi_next_m)
    assert d_m < 1e-4, f"moving-regime Delta {d_m:.6f} must be ~0"
    # Blocked regime: state in action-1's blocked cluster, actual = psi.
    # The gate saturates to pi ~ 0.005 (not exactly 0) on the stochastic
    # fixture, so Delta ~ 1e-4..1e-3 — still ~200x inside the real G4 gate
    # (<= 0.05). Assert <= 1e-2 (5x margin) for honest stochastic tolerance.
    e_block1 = centers[:, 3]
    psi_b = F.normalize(e_block1, dim=-1)
    d_b = eng.g4_single_pass(psi_b, 1, psi_b)
    assert d_b < 1e-2, f"blocked-regime Delta {d_b:.6f} must be small (<1e-2)"


def test_c9_waypoint_advance_actual_frame():
    eng, _ = _engine()
    wp = eng.waypoints[0]
    psi_far = F.normalize(_unit(1, seed=11)[0], dim=-1)
    c = float((psi_far * wp).sum().abs().clamp(0.0, 1.0).item())
    assert c < 0.60, "fixture: far state must not advance"
    assert eng.advance_waypoint_index(psi_far, wp, 0) == 0, "far frame must not advance"
    psi_actual = wp.clone()  # actual post-action frame reaches the waypoint
    assert eng.advance_waypoint_index(psi_actual, wp, 0) == 1, "actual frame must advance"


def test_c10_verdict_precedence():
    eng, _ = _engine()
    v = eng._decide_verdict(
        mean_latency=3.0, solved=2, mean_delta_nu=0.02, g4_mean=0.01,
        steps_done=100, updates=100)
    assert v == "G1_GATE_G1_FAILED", v
    v = eng._decide_verdict(
        mean_latency=1.0, solved=0, mean_delta_nu=0.02, g4_mean=0.01,
        steps_done=100, updates=100)
    assert v == "G1_GATE_G2_FAILED", v
    v = eng._decide_verdict(
        mean_latency=1.0, solved=1, mean_delta_nu=0.001, g4_mean=0.01,
        steps_done=100, updates=100)
    assert v == "G1_GATE_G3_FAILED", v
    v = eng._decide_verdict(
        mean_latency=1.0, solved=1, mean_delta_nu=0.02, g4_mean=0.10,
        steps_done=100, updates=100)
    assert v == "G1_GATE_G4_FAILED", v
    v = eng._decide_verdict(
        mean_latency=1.0, solved=1, mean_delta_nu=0.02, g4_mean=0.01,
        steps_done=100, updates=100)
    assert v == "G1_PASS", v


def test_c11_state_independent_w_defect():
    """Directive reference W = I*(mean-0.5) is state-independent: AUC ~ 0.5."""
    psi, nxt, onehot, _ = _bank()
    y = (torch.norm(nxt - psi, p=2, dim=-1) > MOVING_THRESH).float()
    mean_y = y[onehot[:, 0].bool()].mean()
    W_defect = torch.eye(D) * (mean_y - 0.5)
    b_defect = torch.tensor(0.0)
    pi = torch.sigmoid(
        torch.einsum("bd,de,be->b", psi, W_defect, psi) + b_defect
    )
    auc = compute_auc(pi[onehot[:, 0].bool()], y[onehot[:, 0].bool()])
    assert auc < PG1_MIN_AUC, f"defect control AUC {auc:.4f} must fail PG1"
    assert abs(auc - 0.5) < 0.02, f"defect control must be ~0.5, got {auc:.4f}"


def test_c12_no_affordance_engagement():
    eng, _ = _engine()
    v = eng._decide_verdict(
        mean_latency=1.0, solved=1, mean_delta_nu=0.02, g4_mean=0.01,
        steps_done=10, updates=0)
    assert v == "G1_NO_AFFORDANCE_ENGAGEMENT", v
