"""Carrier F16 contract tests — Active Lie Generator Warping engine.

C1  Omega_goal skew-symmetric (Omega + Omega^T == 0); zero for parallel vectors
C2  warped generator D~ = D + alpha*Omega is skew
C3  exp(D~) orthogonal (||M^T M - I||_F small)
C4  geodesic rotation law: |<exp(alpha*Omega)e0, e1>| == sin(alpha), monotone
C5  rank-break anti-lock: goal-direct J varies across actions (non-degenerate
    goal); degenerate goal (Psi_goal == Psi_t) reproduces the F15 lock (J uniform)
C6  steering gain: goal-aligned generator achieves higher goal alignment under
    warping than static (exponential phase acceleration, directive §1.2)
C7  damping: gamma=0 -> damped == raw; gamma>0 -> damped >= raw
C8  beam determinism: same seed -> same action
C9  bank/PG1 fail-closed: constant-trajectory bank -> psi0 == goal -> kill
C10 flag fail-closed: require_f16_enabled raises without HENRI_F16_WARPED=1
C11 no-bank fail-closed: run_gauntlet without bank -> F16_BLOCKED_NO_TRAJECTORY_BANK
C12 valence semantics: positive toward goal, negative away
C13 module-constant guard (C13 lesson): all gate constants exist at import time
C14 substrate-constructor reachability (C14 lesson): PatchIngress +
    SinglePassHorizon + WarpedLieEngine construct on the live path
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
from arc_f16_warped_engine import (  # noqa: E402
    DEFAULT_ALPHA_STEER,
    DEFAULT_BEAM,
    DEFAULT_ENVS,
    DEFAULT_GAMMA_DAMP,
    DEFAULT_HORIZON,
    DEFAULT_SEED,
    G3_MIN_DNU,
    LATENCY_BUDGET_MS,
    MAX_INITIAL_OVERLAP,
    SAGNAC_TAU_F16,
    WarpedLieEngine,
    omega_goal,
    require_f16_enabled,
    run_gauntlet,
)

D = 64


def _ingress(seed=7):
    return PatchIngress(in_dim=4096, d=D, num_blocks=8, p=32, seed=seed)


def _engine(seed=7, **kw):
    kw.setdefault("D", D)
    kw.setdefault("n_actions", 8)
    return WarpedLieEngine(seed=seed, **kw)


def _orthogonal_pair(seed=3):
    g = torch.Generator().manual_seed(seed)
    x = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
    y = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
    # orthogonalize y against x
    y = y - (y @ x) * x
    y = F.normalize(y, p=2, dim=-1)
    return x, y


# ---------------------------------------------------------------- C1
def test_c1_omega_skew_and_degenerate():
    x, y = _orthogonal_pair()
    om = omega_goal(y, x)
    assert float((om + om.T).abs().max().item()) < 1e-5, "Omega must be skew"
    assert float(om.norm().item()) > 1e-3, "Omega nonzero for orthogonal pair"
    om0 = omega_goal(x, x)
    assert float(om0.norm().item()) < 1e-6, "Omega vanishes for parallel vectors"


# ---------------------------------------------------------------- C2
def test_c2_warped_generator_skew():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(y, x)
    for a in range(3):
        dt = eng.D_a[a] + eng.alpha_steer * om
        assert float((dt + dt.T).abs().max().item()) < 1e-5, "D~ must be skew"


# ---------------------------------------------------------------- C3
def test_c3_warped_exp_orthogonal():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(y, x)
    ops = eng.warped_ops([0, 1, 2], om)
    for m in ops:
        err = float((m.T @ m - torch.eye(D)).abs().max().item())
        assert err < 1e-3, "exp(D~) must be orthogonal (max err {})".format(err)


# ---------------------------------------------------------------- C4
def test_c4_geodesic_rotation_law():
    e0 = torch.zeros(D)
    e0[0] = 1.0
    e1 = torch.zeros(D)
    e1[1] = 1.0
    om = omega_goal(e0, e1)  # (state, goal) — rotates e0 -> e1
    vals = []
    for alpha in (0.0, 0.35, 1.0):
        M = torch.linalg.matrix_exp(alpha * om)
        out = F.normalize(M @ e0, p=2, dim=-1)
        vals.append(float(abs_cos_impl(out, e1)))
    assert vals[0] < 1e-6, "alpha=0 -> no rotation"
    assert vals[0] < vals[1] < vals[2], "monotone in alpha: {}".format(vals)
    assert abs(vals[1] - math.sin(0.35)) < 1e-3, "sin-law: {}".format(vals[1])


def abs_cos_impl(a, b):
    return F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float(), dim=-1).clamp(0.0, 1.0).item()


# ---------------------------------------------------------------- C5
def test_c5_rank_break_antilock():
    eng = _engine(alpha_steer=DEFAULT_ALPHA_STEER)
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)  # (state, goal)
    js = [eng.score_action(x, y, om, a, horizon=1) for a in range(8)]
    std = float(np.std(js))
    assert std > 1e-3, "goal-direct J must vary across actions (rank break), J={}".format(js)
    # warping must actually re-rank vs the static objective: at least one
    # action's score moves (not a uniform rescale)
    js_static = [eng.score_action(x, y, om, a, horizon=1, alpha_steer=0.0) for a in range(8)]
    moved = sum(1 for w, s in zip(js, js_static) if abs(w - s) > 1e-9)
    assert moved >= 1, "warping must change candidate scores (J={}, static={})".format(js, js_static)
    # F15 lock reproduction: goal == state -> Omega == 0 -> the goal channel
    # is rank-inert (warped J identical to static J for every action)
    om0 = omega_goal(x, x)
    js0 = [eng.score_action(x, x, om0, a, horizon=1) for a in range(8)]
    js0_static = [eng.score_action(x, x, om0, a, horizon=1, alpha_steer=0.0) for a in range(8)]
    assert all(abs(w - s) < 1e-9 for w, s in zip(js0, js0_static)), \
        "degenerate goal (Omega=0) must leave the static ranking untouched"


# ---------------------------------------------------------------- C6
def test_c6_steering_gain():
    e0 = torch.zeros(D)
    e0[0] = 1.0
    e1 = torch.zeros(D)
    e1[1] = 1.0
    om = omega_goal(e0, e1)  # (state, goal)
    eng = _engine()
    # goal-aligned generator: base D = 0.1 * Omega (rotates e0 -> e1)
    base = 0.1 * om
    static_align = abs_cos_impl(torch.linalg.matrix_exp(base) @ e0, e1)
    warped_align = abs_cos_impl(torch.linalg.matrix_exp(base + eng.alpha_steer * om) @ e0, e1)
    assert warped_align > static_align + 1e-4, \
        "warping must accelerate goal-aligned rotation: {} vs {}".format(warped_align, static_align)
    # exponential phase acceleration: warped angle = 0.1 + 0.35
    assert abs(warped_align - math.sin(0.45)) < 1e-3, warped_align


# ---------------------------------------------------------------- C7
def test_c7_damping_nonnegative():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(y, x)
    j_raw = eng.score_action(x, y, om, 0, horizon=2, gamma=0.0)
    j_damped = eng.score_action(x, y, om, 0, horizon=2, gamma=0.1)
    # damping adds a non-negative term to the Sagnac penalty -> J_damped <= J_raw
    assert j_damped <= j_raw + 1e-9, "damped objective must be <= raw objective"
    assert j_damped < j_raw - 1e-6, "damping must actually penalize dispersion (nonzero displacement)"


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
    # (constant trajectory -> first row == terminal row -> psi0 == goal)
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
    had = os.environ.pop("HENRI_F16_WARPED", None)
    try:
        with pytest.raises(RuntimeError):
            require_f16_enabled()
    finally:
        if had is not None:
            os.environ["HENRI_F16_WARPED"] = had
    require_f16_enabled(_force_enabled=True)


# ---------------------------------------------------------------- C11
def test_c11_no_bank_fail_closed(tmp_path):
    receipt = run_gauntlet(
        env_names=["e1-aaaa"], steps_per_env=5, seed=1,
        trajectory_bank=None, trajectory_jsonl=None,
        out_dir=str(tmp_path), receipt_out=str(tmp_path / "r.json"),
        _force_enabled=True,
    )
    assert receipt["verdict"] == "F16_BLOCKED_NO_TRAJECTORY_BANK"
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
    assert SAGNAC_TAU_F16 == 0.050
    assert G3_MIN_DNU == 0.0200
    assert MAX_INITIAL_OVERLAP == 0.90
    assert DEFAULT_ALPHA_STEER == 0.35
    assert DEFAULT_GAMMA_DAMP == 0.10
    assert DEFAULT_HORIZON == 8
    assert DEFAULT_BEAM == 8
    assert DEFAULT_SEED == 20260915
    assert len(DEFAULT_ENVS) == 12


# ---------------------------------------------------------------- C14
def test_c14_substrate_constructor_reachable():
    # C14 lesson: harness defects invisible to unit tests (stale shadowed
    # imports, deleted constants) are caught by constructing the exact live
    # substrate chain in the test path.
    ing = _ingress(seed=20260915)
    hor = SinglePassHorizon(d=D, rank=32, K=8, num_blocks=8, seed=20260915)
    eng = WarpedLieEngine(
        D=D, n_actions=8, seed=20260915, horizon=8, beam=8,
        alpha_steer=DEFAULT_ALPHA_STEER, gamma_damp=DEFAULT_GAMMA_DAMP,
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
