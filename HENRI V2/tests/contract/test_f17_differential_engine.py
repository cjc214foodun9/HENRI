"""Carrier F17 contract tests — Candidate-Differential Killing-Form Steering engine.

C1  Killing-form identity: gamma_a == <Psi_goal, D_a Psi_t>; sign split for
    aligned (+Omega) vs opposed (-Omega) generators
C2  candidate-differential: aligned generator boosted (scale > 1), opposed
    damped (scale < 1)
C3  tanh bound: scale in [1 - kappa, 1 + kappa] for any gamma
C4  exp(D~) orthogonal (||M^T M - I||_F small)
C5  rank-break anti-lock: goal-direct J varies across actions (non-degenerate
    goal); degenerate goal (Psi_goal == Psi_t) reproduces the F15/F16 lock
    (warped J identical to static J for every action)
C6  steering gain: goal-aligned generator boosted toward goal; opposed damped
C7  Lyapunov damping: mu=0 -> damped == raw; mu>0 -> damped >= raw and J drops
C8  beam determinism: same seed -> same action
C9  bank/PG1 fail-closed: constant-trajectory bank -> psi0 == goal -> kill
C10 flag fail-closed: require_f17_enabled raises without HENRI_F17_DIFFERENTIAL=1
C11 no-bank fail-closed: run_gauntlet without bank -> F17_BLOCKED_NO_TRAJECTORY_BANK
C12 valence semantics: positive toward goal, negative away
C13 module-constant guard (C13 lesson): all gate constants exist at import time
C14 substrate-constructor reachability (C14 lesson): PatchIngress +
    SinglePassHorizon + DifferentialLieEngine construct on the live path
C15 engagement-gate sensitivity: gamma std > 0 on a non-degenerate pair,
    exactly 0 for goal == state; _verdict maps no-engagement -> F17_FALSIFIED_NO_ENGAGEMENT
    and engaged-failure -> F17_GATE_G3_FAILED
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../unified-vla/HENRI V2
ENGINE_DIR = REPO_ROOT / "experiments" / "verification"
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from arc_f10_live_engine import PatchIngress, SinglePassHorizon  # noqa: E402
from arc_f15_trajectory_engine import _bridge_to_d64, pg1_pass, resolve_trajectory_goal  # noqa: E402
from arc_f17_differential_engine import (  # noqa: E402
    DEFAULT_BEAM,
    DEFAULT_BETA_SAGNAC,
    DEFAULT_ENVS,
    DEFAULT_ETA_FAST,
    DEFAULT_HORIZON,
    DEFAULT_KAPPA_DIFF,
    DEFAULT_MU_DAMP,
    DEFAULT_SEED,
    ENGAGEMENT_MIN_GAMMA_STD,
    G3_MIN_DNU,
    LATENCY_BUDGET_MS,
    MAX_INITIAL_OVERLAP,
    SAGNAC_TAU_F17,
    DifferentialLieEngine,
    _verdict,
    killing_coeffs,
    omega_goal,
    require_f17_enabled,
    run_gauntlet,
)

D = 64


def _ingress(seed=7):
    return PatchIngress(in_dim=4096, d=D, num_blocks=8, p=32, seed=seed)


def _engine(seed=7, **kw):
    kw.setdefault("D", D)
    kw.setdefault("n_actions", 8)
    return DifferentialLieEngine(seed=seed, **kw)


def _orthogonal_pair(seed=3):
    g = torch.Generator().manual_seed(seed)
    x = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
    y = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
    y = y - (y @ x) * x
    y = F.normalize(y, p=2, dim=-1)
    return x, y


def abs_cos_impl(a, b):
    return F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float(), dim=-1).clamp(0.0, 1.0).item()


# ---------------------------------------------------------------- C1
def test_c1_killing_form_identity():
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)  # (state, goal)
    gen = torch.stack([0.5 * om, -0.5 * om])  # aligned, opposed
    gams = killing_coeffs(gen, om)
    # identity: gamma_a = <Psi_goal, D_a Psi_t>
    expected = torch.stack([y @ (gen[0] @ x), y @ (gen[1] @ x)])
    assert torch.allclose(gams, expected, atol=1e-4), (gams, expected)
    assert gams[0] > 0.0, "aligned generator must have positive gamma"
    assert gams[1] < 0.0, "opposed generator must have negative gamma"


# ---------------------------------------------------------------- C2
def test_c2_candidate_differential_scales():
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    gen = torch.stack([0.5 * om, -0.5 * om])
    gams = killing_coeffs(gen, om)
    scale = 1.0 + DEFAULT_KAPPA_DIFF * torch.tanh(gams)
    assert scale[0] > 1.0, "aligned must be amplified"
    assert scale[1] < 1.0, "opposed must be damped"
    # gamma differs across candidates -> rank invariance broken
    assert float(gams.std().item()) > 1e-3


# ---------------------------------------------------------------- C3
def test_c3_tanh_bound():
    g = torch.Generator().manual_seed(5)
    gams = torch.randn(64, generator=g) * 10.0
    scale = 1.0 + DEFAULT_KAPPA_DIFF * torch.tanh(gams)
    assert float(scale.min().item()) >= 1.0 - DEFAULT_KAPPA_DIFF - 1e-6
    assert float(scale.max().item()) <= 1.0 + DEFAULT_KAPPA_DIFF + 1e-6


# ---------------------------------------------------------------- C4
def test_c4_warped_exp_orthogonal():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    ops = eng.warped_ops([0, 1, 2], om)
    for m in ops:
        err = float((m.T @ m - torch.eye(D)).abs().max().item())
        assert err < 1e-3, "exp(D~) must be orthogonal (max err {})".format(err)


# ---------------------------------------------------------------- C5
def test_c5_rank_break_antilock():
    eng = _engine(kappa_diff=DEFAULT_KAPPA_DIFF)
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)  # (state, goal)
    js = [eng.score_action(x, y, om, a, horizon=1) for a in range(8)]
    std = float(np.std(js))
    assert std > 1e-3, "goal-direct J must vary across actions (rank break), J={}".format(js)
    # warping must actually re-rank vs the static objective (kappa=0)
    js_static = [eng.score_action(x, y, om, a, horizon=1, kappa_diff=0.0) for a in range(8)]
    moved = sum(1 for w, s in zip(js, js_static) if abs(w - s) > 1e-9)
    assert moved >= 1, "warping must change candidate scores (J={}, static={})".format(js, js_static)
    # F15/F16 lock reproduction: goal == state -> Omega == 0 -> gamma == 0 ->
    # warped J identical to static J for every action
    om0 = omega_goal(x, x)
    js0 = [eng.score_action(x, x, om0, a, horizon=1) for a in range(8)]
    js0_static = [eng.score_action(x, x, om0, a, horizon=1, kappa_diff=0.0) for a in range(8)]
    assert all(abs(w - s) < 1e-9 for w, s in zip(js0, js0_static)), \
        "degenerate goal (Omega=0) must leave the static ranking untouched"


# ---------------------------------------------------------------- C6
def test_c6_steering_gain():
    e0 = torch.zeros(D)
    e0[0] = 1.0
    e1 = torch.zeros(D)
    e1[1] = 1.0
    om = omega_goal(e0, e1)  # (state, goal)
    base = 0.1 * om
    gamma = killing_coeffs(base[None], om)[0]
    scale = 1.0 + DEFAULT_KAPPA_DIFF * torch.tanh(gamma)
    static_align = abs_cos_impl(torch.linalg.matrix_exp(base) @ e0, e1)
    warped_align = abs_cos_impl(torch.linalg.matrix_exp(scale * base) @ e0, e1)
    assert warped_align > static_align + 1e-4, \
        "aligned generator must be accelerated: {} vs {}".format(warped_align, static_align)
    # opposed generator is damped: less progress toward goal than static
    opp = -0.1 * om
    gamma_opp = killing_coeffs(opp[None], om)[0]
    scale_opp = 1.0 + DEFAULT_KAPPA_DIFF * torch.tanh(gamma_opp)
    opp_align = abs_cos_impl(torch.linalg.matrix_exp(scale_opp * opp) @ e0, e1)
    opp_static = abs_cos_impl(torch.linalg.matrix_exp(opp) @ e0, e1)
    assert opp_align < opp_static + 1e-6, \
        "opposed generator must be damped: {} vs {}".format(opp_align, opp_static)


# ---------------------------------------------------------------- C7
def test_c7_lyapunov_damping():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    j_raw = eng.score_action(x, y, om, 0, horizon=2, mu_damp=0.0)
    j_damped = eng.score_action(x, y, om, 0, horizon=2, mu_damp=DEFAULT_MU_DAMP)
    assert j_damped <= j_raw + 1e-9, "Lyapunov damping must not increase J"
    assert j_damped < j_raw - 1e-6, "damping must actually penalize generator norm"


# ---------------------------------------------------------------- C8
def test_c8_beam_determinism():
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)  # (state, goal)
    e1 = _engine(seed=11)
    e2 = _engine(seed=11)
    a1, _ = e1.beam_search(x, y, om, horizon=2)
    a2, _ = e2.beam_search(x, y, om, horizon=2)
    assert a1 == a2


# ---------------------------------------------------------------- C9
def _synthetic_bank(tmp_path, env_ids=("e1-aaaa",), rows=40, degenerate=False, seed=0):
    n_env = len(env_ids)
    N = n_env * rows
    rng = np.random.default_rng(seed)
    waves = rng.standard_normal((N, 65536)).astype(np.float32)
    if degenerate:
        waves[:] = waves[0]  # constant trajectory -> psi0 == goal
    npz = tmp_path / "bank.npz"
    np.savez(npz, psi=waves)
    jsonl = tmp_path / "bank.jsonl"
    with open(jsonl, "w") as f:
        for i in range(N):
            f.write(json.dumps({"env": env_ids[i // rows], "step": i % rows}) + "\n")
    return str(npz), str(jsonl)


def test_c9_degenerate_bank_pg1_kill(tmp_path):
    npz, jsonl = _synthetic_bank(tmp_path, degenerate=True)
    ing = _ingress()
    goal, _meta = resolve_trajectory_goal(npz, jsonl, "e1-aaaa", device="cpu", ingress=ing)
    # psi0 = the bank's OWN first row through the real bridge+ingress path
    waves = np.load(npz)["psi"]
    first = torch.from_numpy(np.asarray(waves[0])).float()
    pooled = _bridge_to_d64(first, "cpu")
    psi0 = ing(pooled.unsqueeze(0))[0].detach()
    ok, overlap = pg1_pass(psi0, goal)
    assert not ok, "degenerate bank must fail PG1 (overlap {:.4f})".format(overlap)
    assert overlap > 0.90


# ---------------------------------------------------------------- C10
def test_c10_flag_fail_closed():
    import os
    had = os.environ.pop("HENRI_F17_DIFFERENTIAL", None)
    try:
        with pytest.raises(RuntimeError):
            require_f17_enabled()
    finally:
        if had is not None:
            os.environ["HENRI_F17_DIFFERENTIAL"] = had
    require_f17_enabled(_force_enabled=True)


# ---------------------------------------------------------------- C11
def test_c11_no_bank_fail_closed(tmp_path):
    receipt = run_gauntlet(
        env_names=["e1-aaaa"], steps_per_env=5, seed=1,
        trajectory_bank=None, trajectory_jsonl=None,
        out_dir=str(tmp_path), receipt_out=str(tmp_path / "r.json"),
        _force_enabled=True,
    )
    assert receipt["verdict"] == "F17_BLOCKED_NO_TRAJECTORY_BANK"
    assert all(v is False for v in receipt["gates"].values())


# ---------------------------------------------------------------- C12
def test_c12_valence_semantics():
    eng = _engine()
    e0 = torch.zeros(D)
    e0[0] = 1.0
    e1 = torch.zeros(D)
    e1[1] = 1.0
    mid = F.normalize(e0 + 0.3 * e1, p=2, dim=-1)
    dnu_pos = eng.valence_delta_goal(mid, e0, e1)     # moving toward goal
    dnu_neg = eng.valence_delta_goal(e0, mid, e1)     # moving away
    assert dnu_pos > 0.0, dnu_pos
    assert dnu_neg < 0.0, dnu_neg


# ---------------------------------------------------------------- C13
def test_c13_module_constants_guard():
    assert LATENCY_BUDGET_MS == 5.0
    assert SAGNAC_TAU_F17 == 0.050
    assert G3_MIN_DNU == 0.0200
    assert MAX_INITIAL_OVERLAP == 0.90
    assert DEFAULT_KAPPA_DIFF == 0.75
    assert DEFAULT_MU_DAMP == 0.15
    assert DEFAULT_BETA_SAGNAC == 0.05
    assert DEFAULT_HORIZON == 8
    assert DEFAULT_BEAM == 8
    assert DEFAULT_SEED == 20260916
    assert DEFAULT_ETA_FAST == 0.05
    assert ENGAGEMENT_MIN_GAMMA_STD == 1e-6
    assert len(DEFAULT_ENVS) == 12


# ---------------------------------------------------------------- C14
def test_c14_substrate_constructor_reachable():
    ing = _ingress(seed=20260916)
    hor = SinglePassHorizon(d=D, rank=32, K=8, num_blocks=8, seed=20260916)
    eng = DifferentialLieEngine(
        D=D, n_actions=8, seed=20260916, horizon=8, beam=8,
        kappa_diff=DEFAULT_KAPPA_DIFF, mu_damp=DEFAULT_MU_DAMP,
        beta_sagnac=0.05, eta_fast=0.05,
    )
    x = F.normalize(torch.randn(D), p=2, dim=-1)
    y = F.normalize(torch.randn(D), p=2, dim=-1)
    om = eng.omega(x, y)
    sel, j = eng.beam_search(x, y, om, horizon=2)
    assert 0 <= sel < 8
    assert math.isfinite(j)
    roll = hor(x.unsqueeze(0))
    assert roll.shape[0] == 1


# ---------------------------------------------------------------- C15
def test_c15_engagement_gate_sensitivity():
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    eng = _engine()
    gams = eng.gamma_all(range(8), om)
    assert float(gams.std().item()) > ENGAGEMENT_MIN_GAMMA_STD, \
        "non-degenerate pair must engage the Killing warp"
    # degenerate: goal == state -> Omega == 0 -> gamma == 0 exactly
    gams0 = eng.gamma_all(range(8), omega_goal(x, x))
    assert float(gams0.std().item()) == 0.0
    assert float(gams0.abs().max().item()) < 1e-12
    # verdict wiring: engaged G3-only failure -> G3 verdict; G2-failure takes
    # precedence (F16 verdict pattern: iterate G2, G3, G4 in directive order);
    # no-engagement -> FALSIFIED_NO_ENGAGEMENT
    gates_g3 = {"PG1": True, "G1": True, "G2": True, "G3": False, "G4": True}
    v_engaged = _verdict(gates_g3, {"killing_gamma_std_mean": 0.05})
    assert v_engaged == "F17_GATE_G3_FAILED", v_engaged
    gates_g2 = {"PG1": True, "G1": True, "G2": False, "G3": False, "G4": False}
    assert _verdict(gates_g2, {"killing_gamma_std_mean": 0.05}) == "F17_GATE_G2_FAILED"
    v_noeng = _verdict(gates_g2, {"killing_gamma_std_mean": 0.0})
    assert v_noeng == "F17_FALSIFIED_NO_ENGAGEMENT", v_noeng
    v_none = _verdict(gates_g2, {"killing_gamma_std_mean": None})
    assert v_none == "F17_FALSIFIED_NO_ENGAGEMENT", v_none
