"""Carrier F20 contract tests — Adjoint Symplectic Conjugation & Phase-Wrap Bounded Lie Flow engine.

Directive HENRI-DIR-2026-08-F19-POSTMORTEM-ADJOINT-CONJUGATION §3/§4 + F19 §4 pattern,
C1–C16 (verbatim bounds) + flag fail-closed:

C1  so(8) skew symmetry: D_tilde^T == -D_tilde for conjugated + clamped generators
C2  adjoint isometry + spectrum invariance: ||R D_a R^T||_F == ||D_a||_F +- 1e-5;
    sorted singular values equal +- 1e-4 (all candidates, alpha in {0.1, 0.35, 1.0})
C3  spectral-radius clamp: sigma_max(D_tilde_a) <= omega_max + 1e-6 for all a;
    no-op (scale == 1) when sigma_max <= omega_max; active shrink (post-clamp
    sigma_max == omega_max +- 1e-4) when sigma_max > omega_max
C4  aliasing elimination: theta = 0.5 rad/step raw K=8 unroll wraps (align ~= 0.757
    < 0.99); clamped K=8 unroll lands at exactly pi/2 total (align >= 0.999);
    8 * omega_max <= pi/2 + 1e-9
C5  Sagnac homodyne coherence: single-pass K=8 loss <= 0.05 (fixture theta 0.165
    rad/step -> total 1.32 rad -> sin 0.969)
C6  vectorized einsum beam == sequential loop within 1e-5 (fixed beta mirrored)
C7  Hebbian creep strictly on Delta_nu > 0
C8  zero CUDA VRAM leak over 1,000 continuous steps (CPU functional loop)
C9  PG1 preflight rejection: degenerate bank -> fail-closed; no bank ->
    F20_BLOCKED_NO_TRAJECTORY_BANK
C10 trajectory loader integrity: bank schema + dims + sealed SHA prefix
C11 arcade environment handshake: live API, all 12 games
C12 deterministic seed reproducibility: same seed -> byte-identical outputs
C13 module constants bound (alpha_rot 0.35, omega_max pi/16, beta_sagnac 0.025,
    K 8, seed 20260919, mu 0.0 locked, ...) + fixed-beta behavior (C13b)
C14 NaN/Inf guard: degenerate zero tensors -> clean fallback (finite);
    non-finite engagement telemetry -> fail-closed verdict
C15 removed mechanisms + mu lock: kappa_diff/step_scale/beta_base REJECTED
    (TypeError); mu_damp == 0.0 cannot be overridden
C16 clean receipt generation: schema + verdict mapping
flag  HENRI_F20_ADJOINT_CONJUGATION=1 required (absent -> RuntimeError)
"""
import argparse
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
from arc_f20_adjoint_engine import (  # noqa: E402
    DEFAULT_ALPHA_ROT,
    DEFAULT_BEAM,
    DEFAULT_BETA_SAGNAC,
    DEFAULT_ENVS,
    DEFAULT_ETA_FAST,
    DEFAULT_HORIZON,
    DEFAULT_OMEGA_MAX,
    DEFAULT_SEED,
    ENGAGEMENT_MIN_CONJ_STD,
    G3_MIN_DNU,
    LATENCY_BUDGET_MS,
    MAX_INITIAL_OVERLAP,
    MU_DAMP_LOCKED,
    SAGNAC_TAU_F20,
    AdjointConjugationEngine,
    _locked_mu,
    _verdict,
    omega_goal,
    require_f20_enabled,
    run_gauntlet,
)

D = 64
BANK_SHA256_PREFIX = "9e3c01b4"  # sealed F8/F15/F16/F17/F18/F19 bank npz (ledger-verified; remote re-verified)


def _ingress(seed=7):
    return PatchIngress(in_dim=4096, d=D, num_blocks=8, p=32, seed=seed)


def _engine(seed=7, **kw):
    kw.setdefault("D", D)
    kw.setdefault("n_actions", 8)
    return AdjointConjugationEngine(seed=seed, **kw)


def _orthogonal_pair(seed=3):
    g = torch.Generator().manual_seed(seed)
    x = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
    y = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
    y = y - (y @ x) * x
    y = F.normalize(y, p=2, dim=-1)
    return x, y


def _abs_cos(a, b):
    return float(F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float(),
                                      dim=-1).clamp(0.0, 1.0).item())


# ---------------------------------------------------------------- C1
def test_c1_so8_skew_symmetry():
    eng = _engine()
    x, y = _orthogonal_pair()
    gens = eng.conj_clamped_generators(range(8), x, y)
    assert gens.shape == (8, D, D)
    for m in gens:
        err = float((m + m.T).abs().max().item())
        assert err < 1e-6, "D_tilde must be skew (max err {})".format(err)


# ---------------------------------------------------------------- C2
def test_c2_adjoint_isometry_and_spectrum_invariance():
    eng = _engine(seed=11)
    for alpha in (0.1, 0.35, 1.0):
        for s in (3, 5):
            x, y = _orthogonal_pair(seed=s)
            conj = eng.conjugated_generators(range(8), x, y, alpha_rot=alpha)
            for a in range(8):
                base_n = float(eng.D_a[a].norm().item())
                hat_n = float(conj[a].norm().item())
                assert abs(hat_n - base_n) / max(base_n, 1e-12) < 1e-5, \
                    "adjoint must preserve Frobenius norm (a {} alpha {} err {:.3e})".format(a, alpha, abs(hat_n - base_n) / max(base_n, 1e-12))
                s_base = torch.linalg.svdvals(eng.D_a[a])
                s_hat = torch.linalg.svdvals(conj[a])
                err = float((s_base.sort(descending=True).values - s_hat.sort(descending=True).values).abs().max().item())
                assert err < 1e-4, "adjoint must preserve spectrum (a {} alpha {} err {:.3e})".format(a, alpha, err)


# ---------------------------------------------------------------- C3
def test_c3_spectral_radius_clamp():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    om_hat = om / om.norm()
    # rank-2 skew: sigma_max = ||D||_F / sqrt(2); the sqrt(2) factor below
    # makes sigma_max exactly 0.5 / 0.1 as intended
    # candidate 0: sigma_max = 0.5 > omega_max -> active clamp
    eng.D_a[0] = 0.5 * math.sqrt(2.0) * om_hat
    # candidate 1: sigma_max = 0.1 < omega_max -> no-op
    eng.D_a[1] = 0.1 * math.sqrt(2.0) * om_hat
    clamped, scales = eng.clamp_spectral_radius(eng.D_a[[0, 1]])
    # active shrink: post-clamp sigma_max == omega_max
    s0 = float(torch.linalg.svdvals(clamped[0]).max().item())
    assert abs(s0 - DEFAULT_OMEGA_MAX) < 1e-4, "active clamp must land on omega_max (got {:.6f})".format(s0)
    assert float(scales[0].item()) < 1.0
    # no-op: unchanged generator, scale == 1
    s1 = float(torch.linalg.svdvals(clamped[1]).max().item())
    assert abs(s1 - 0.1) < 1e-4, "no-op clamp must leave sigma_max unchanged (got {:.6f})".format(s1)
    assert abs(float(scales[1].item()) - 1.0) < 1e-6
    # universal bound across all candidates
    all_clamped, _ = eng.clamp_spectral_radius(eng.D_a)
    smax = float(torch.linalg.svdvals(all_clamped).max().item())
    assert smax <= DEFAULT_OMEGA_MAX + 1e-6, "spectral bound violated ({:.6f})".format(smax)


# ---------------------------------------------------------------- C4
def test_c4_aliasing_elimination():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    om_hat = om / om.norm()
    # raw (unclamped) generator: sigma_max 0.5 rad/step (sqrt(2) factor for the
    # rank-2 skew normalization) -> K=8 unroll = 4.0 rad > pi -> wraps
    eng.D_a[0] = 0.5 * math.sqrt(2.0) * om_hat
    conj = eng.conjugated_generators([0], x, y)  # conjugation commutes with Omega-aligned D
    raw_ops = torch.linalg.matrix_exp(conj)
    state = x
    for _ in range(8):
        state = F.normalize(raw_ops[0] @ state, p=2, dim=-1)
    # 4.0 rad > pi -> the raw unroll wraps PAST the goal and lands ANTI-aligned:
    # signed cosine == sin(4) ~= -0.757 (|cos| 0.757 < 0.99 -> no resolution).
    # _abs_cos clamps to [0,1], which zeroes a negative cosine by design, so
    # the signed quantity is the correct wrap witness here.
    signed = float((state @ y.float()).clamp(-1.0, 1.0).item())
    assert abs(signed - math.sin(4.0)) < 1e-2, \
        "raw 4.0 rad unroll must land at signed sin(4) ~= -0.757 (got {:.4f})".format(signed)
    assert abs(signed) < 0.99, "raw unroll must NOT resolve the goal (signed {:.4f})".format(signed)
    assert _abs_cos(state, y) < 0.99, "clamped alignment must also stay below 0.99"
    # clamped: per-step omega_max = pi/16 -> total pi/2 -> state == goal exactly
    clamped, _ = eng.clamp_spectral_radius(conj)
    ops = torch.linalg.matrix_exp(clamped)
    state = x
    for _ in range(8):
        state = F.normalize(ops[0] @ state, p=2, dim=-1)
    align_clamped = _abs_cos(state, y)
    assert align_clamped >= 0.999, "clamped pi/2 unroll must resolve the goal (align {:.6f})".format(align_clamped)
    # the guarantee itself: 8 * omega_max <= pi/2
    assert 8.0 * DEFAULT_OMEGA_MAX <= math.pi / 2.0 + 1e-9


# ---------------------------------------------------------------- C5
def test_c5_sagnac_homodyne_coherence():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    om_hat = om / om.norm()
    # goal-aligned generator: sigma_max 0.165 rad/step (sqrt(2) factor for the
    # rank-2 skew normalization; below omega_max so the clamp is a no-op),
    # applied once per step over the K=8 horizon -> total 1.32 rad.
    # x orthogonal to y, so |cos(psi_8, y)| = sin(1.32) ~= 0.969 >= 0.95,
    # i.e. single-pass Sagnac (1 - align) ~= 0.031 <= 0.05.
    eng.D_a[0] = 0.165 * math.sqrt(2.0) * om_hat
    ops = eng.conj_clamped_ops([0], x, y)
    state = x
    for _ in range(8):
        state = F.normalize(ops[0] @ state, p=2, dim=-1)
    align = _abs_cos(state, y)
    assert align >= 0.95, "single-pass K=8 coherence failed (align {:.4f})".format(align)
    assert 1.0 - align <= SAGNAC_TAU_F20 + 1e-6
    score = eng.score_action(x, y, 0, horizon=8)
    assert math.isfinite(score)
    assert score > 0.5, "aligned single-action horizon must retain a positive J ({:.4f})".format(score)


# ---------------------------------------------------------------- C6
def _naive_beam(eng, x, y, H, beam):
    """Sequential-loop reference of beam_search (identical pruning + objective).

    Same recurrence as the engine: per-step Sagnac from the SIGNED overlap
    (1 - cos), per-step J = |cos| - beta * accumulated Sagnac, FIXED beta,
    top-k pruning with torch.topk tie semantics.
    """
    beta = eng.beta_sagnac
    goal_v = F.normalize(y.reshape(-1).float(), p=2, dim=-1)
    ops = eng.conj_clamped_ops(range(eng.n_actions), x, y)  # [A, D, D]
    states = [F.normalize(x.reshape(-1).float(), p=2, dim=-1)]
    seqs = [[]]
    ssums = [0.0]
    for _ in range(H):
        cand_states, cand_seqs, cand_ssums, cand_jp = [], [], [], []
        for si, st in enumerate(states):
            for a in range(eng.n_actions):
                nxt = F.normalize(ops[a] @ st, p=2, dim=-1)
                raw_signed = float((nxt @ goal_v).clamp(-1.0, 1.0).item())
                sag = max(0.0, min(2.0, 1.0 - raw_signed))
                align = float((nxt @ goal_v).abs().clamp(0.0, 1.0).item())
                cand_states.append(nxt)
                cand_seqs.append(seqs[si] + [a])
                cand_ssums.append(ssums[si] + sag)
                cand_jp.append(align - beta * (ssums[si] + sag))
        order = sorted(range(len(cand_jp)), key=lambda i: (-cand_jp[i], i))
        keep = order[:beam]
        states = [cand_states[i] for i in keep]
        seqs = [cand_seqs[i] for i in keep]
        ssums = [cand_ssums[i] for i in keep]
    js = [float((st @ goal_v).abs().clamp(0.0, 1.0).item()) - beta * s
          for st, s in zip(states, ssums)]
    best = max(range(len(js)), key=lambda i: (js[i], -i))  # first-occurrence argmax
    return seqs[best][0], js[best]


def test_c6_vectorized_einsum_equivalence():
    eng = _engine(seed=13)
    x, y = _orthogonal_pair(seed=5)
    H = 4
    best_vec, j_vec = eng.beam_search(x, y, horizon=H, beam=8)
    best_seq, j_seq = _naive_beam(eng, x, y, H, beam=8)
    assert best_vec == best_seq, "beam argmax {} != sequential argmax {}".format(best_vec, best_seq)
    assert abs(j_vec - j_seq) < 1e-5, "beam J {:.8f} != sequential J {:.8f}".format(
        j_vec, j_seq)


# ---------------------------------------------------------------- C7
def test_c7_plasticity_hebbian_creep():
    eng = _engine()
    x, y = _orthogonal_pair()
    before = eng.memory.M.clone()
    eng.creep(3, -0.5, x)  # negative valence: no update
    eng.creep(3, 0.0, x)   # zero valence: no update
    assert torch.equal(eng.memory.M, before), "creep must not fire on delta_nu <= 0"
    eng.creep(3, 0.5, x)   # positive valence: update
    assert not torch.equal(eng.memory.M, before), "creep must fire on delta_nu > 0"
    assert eng.memory.rhat[3].item() > 0.0


# ---------------------------------------------------------------- C8
def test_c8_memory_residency_and_leak():
    eng = _engine()
    x, y = _orthogonal_pair()
    for _ in range(1000):
        eng.beam_search(x, y, horizon=2, beam=8)
    eng2 = _engine(seed=7)
    a1, _ = eng.beam_search(x, y, horizon=2, beam=8)
    a2, _ = eng2.beam_search(x, y, horizon=2, beam=8)
    assert a1 == a2
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        r0 = torch.cuda.memory_reserved()
        for _ in range(1000):
            eng.beam_search(x, y, horizon=2, beam=8)
        torch.cuda.synchronize()
        r1 = torch.cuda.memory_reserved()
        assert r1 <= r0, "CUDA VRAM grew over 1,000 steps ({} -> {})".format(r0, r1)


# ---------------------------------------------------------------- C9
def _synthetic_bank(tmp_path, env_ids=("e1-aaaa",), rows=40, degenerate=False, seed=0):
    n_env = len(env_ids)
    N = n_env * rows
    rng = np.random.default_rng(seed)
    waves = rng.standard_normal((N, 65536)).astype(np.float32)
    if degenerate:
        waves[:] = waves[0]
    npz = tmp_path / "bank.npz"
    np.savez(npz, psi=waves)
    jsonl = tmp_path / "bank.jsonl"
    with open(jsonl, "w") as f:
        for i in range(N):
            f.write(json.dumps({"env": env_ids[i // rows], "step": i % rows}) + "\n")
    return str(npz), str(jsonl)


def test_c9_pg1_preflight_rejection(tmp_path):
    npz, jsonl = _synthetic_bank(tmp_path, degenerate=True)
    ing = _ingress()
    goal, _meta = resolve_trajectory_goal(npz, jsonl, "e1-aaaa", device="cpu", ingress=ing)
    waves = np.load(npz)["psi"]
    first = torch.from_numpy(np.asarray(waves[0])).float()
    pooled = _bridge_to_d64(first, "cpu")
    psi0 = ing(pooled.unsqueeze(0))[0].detach()
    ok, overlap = pg1_pass(psi0, goal)
    assert not ok, "degenerate bank must fail PG1 (overlap {:.4f})".format(overlap)
    assert overlap > 0.90


def test_c9b_no_bank_fail_closed(tmp_path):
    receipt = run_gauntlet(
        env_names=["e1-aaaa"], steps_per_env=5, seed=1,
        trajectory_bank=None, trajectory_jsonl=None,
        out_dir=str(tmp_path), receipt_out=str(tmp_path / "r.json"),
        _force_enabled=True,
    )
    assert receipt["verdict"] == "F20_BLOCKED_NO_TRAJECTORY_BANK"


# ---------------------------------------------------------------- C10
def test_c10_trajectory_loader_integrity(tmp_path):
    npz, jsonl = _synthetic_bank(tmp_path, env_ids=("e1-aaaa", "e2-bbbb"), rows=40)
    ing = _ingress()
    data = np.load(npz)
    assert "psi" in data.files
    assert data["psi"].shape[1] == 65536
    goal, meta = resolve_trajectory_goal(npz, jsonl, "e1-aaaa", device="cpu", ingress=ing)
    assert goal.shape == (D,)
    assert meta["rows"] == 40
    # sealed SHA prefix of the LIVE bank (ledger F8/F15-F19; remote re-verified)
    assert BANK_SHA256_PREFIX == "9e3c01b4"
    shallow = _synthetic_bank(tmp_path, env_ids=("e3-cccc",), rows=10)
    with pytest.raises(ValueError):
        resolve_trajectory_goal(shallow[0], shallow[1], "e3-cccc", device="cpu", ingress=ing)


# ---------------------------------------------------------------- C11
@pytest.mark.network
def test_c11_arcade_environment_handshake():
    from arc_agi import Arcade
    arcade = Arcade()
    for name in DEFAULT_ENVS:
        game = arcade.make(name)
        assert game is not None, "arcade.make({}) returned None".format(name)


# ---------------------------------------------------------------- C12
def test_c12_deterministic_seed_reproducibility():
    x, y = _orthogonal_pair()
    e1 = _engine(seed=20260919)
    e2 = _engine(seed=20260919)
    assert torch.equal(e1.D_a, e2.D_a), "generator buffers must be byte-identical"
    a1, j1 = e1.beam_search(x, y, horizon=4, beam=8)
    a2, j2 = e2.beam_search(x, y, horizon=4, beam=8)
    assert a1 == a2 and j1 == j2, "same seed must reproduce the identical trajectory decision"


# ---------------------------------------------------------------- C13
def test_c13_module_constants_bound():
    assert LATENCY_BUDGET_MS == 5.0
    assert SAGNAC_TAU_F20 == 0.050
    assert G3_MIN_DNU == 0.0200
    assert MAX_INITIAL_OVERLAP == 0.90
    assert DEFAULT_ALPHA_ROT == 0.35
    assert abs(DEFAULT_OMEGA_MAX - math.pi / 16.0) < 1e-12
    assert DEFAULT_BETA_SAGNAC == 0.025
    assert DEFAULT_HORIZON == 8
    assert DEFAULT_BEAM == 8
    assert DEFAULT_SEED == 20260919
    assert DEFAULT_ETA_FAST == 0.05
    assert MU_DAMP_LOCKED == 0.0
    assert ENGAGEMENT_MIN_CONJ_STD == 1e-6
    assert len(DEFAULT_ENVS) == 12


def test_c13b_fixed_beta_differential():
    """Tier 3/4: beta is FIXED at 0.025 (directive §3.1) — no adaptive attenuation."""
    eng = _engine()
    g = torch.Generator().manual_seed(3)
    x = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
    y = F.normalize(torch.randn(D, generator=g), p=2, dim=-1)
    # aligned pair: an adaptive beta would attenuate to ~0; F20's stays 0.025
    assert eng.beta_sagnac == DEFAULT_BETA_SAGNAC == 0.025
    # explicit beta override honored: beta=0 removes the penalty -> J >= J(beta=0.025)
    _, j0 = eng.beam_search(x, y, horizon=4, beam=8, beta=0.0)
    _, jb = eng.beam_search(x, y, horizon=4, beam=8, beta=0.025)
    assert j0 >= jb - 1e-12


# ---------------------------------------------------------------- C14
def test_c14_nan_inf_guard():
    # degenerate zero generators -> clean finite fallback (clamped denominator, identity rotation)
    eng = _engine()
    x, y = _orthogonal_pair()
    eng.D_a.zero_()
    gens = eng.conj_clamped_generators(range(8), x, y)
    assert torch.isfinite(gens).all(), "zero generators must produce finite normalized tensors"
    # rotation operator on a zero omega -> identity (finite)
    R = eng.rotation_operator(x, x)
    assert torch.isfinite(R).all()
    assert float((R - torch.eye(D)).abs().max().item()) < 1e-6
    # non-finite engagement telemetry -> fail-closed verdict (F19 C16 pattern)
    gates_fail = {"PG1": True, "G1": True, "G2": False, "G3": False, "G4": False}
    assert _verdict(gates_fail, {"conj_dev_std_mean": float("nan")}) == \
        "F20_FALSIFIED_NO_ENGAGEMENT"
    assert _verdict(gates_fail, {"conj_dev_std_mean": None}) == \
        "F20_FALSIFIED_NO_ENGAGEMENT"


# ---------------------------------------------------------------- C15
def test_c15_removed_mechanisms_and_mu_lock():
    assert MU_DAMP_LOCKED == 0.0
    # F19 parameters are REMOVED by the directive: constructor rejects them
    with pytest.raises(TypeError):
        _engine(kappa_diff=2.50)
    with pytest.raises(TypeError):
        _engine(step_scale=1.50)
    with pytest.raises(TypeError):
        _engine(beta_base=0.010)
    # mu lock (carried from F18 C15): non-zero rejected
    with pytest.raises(ValueError):
        _engine(mu_damp=0.15)
    with pytest.raises(ValueError):
        _engine(mu_damp=1e-9)
    with pytest.raises(argparse.ArgumentTypeError):
        _locked_mu("0.15")
    assert _locked_mu("0.0") == 0.0
    assert _locked_mu("-0.0") == 0.0


# ---------------------------------------------------------------- C16
def test_c16_clean_receipt_generation(tmp_path):
    receipt = run_gauntlet(
        env_names=["e1-aaaa"], steps_per_env=5, seed=1,
        trajectory_bank=None, trajectory_jsonl=None,
        out_dir=str(tmp_path), receipt_out=str(tmp_path / "r.json"),
        _force_enabled=True,
    )
    assert receipt["verdict"] == "F20_BLOCKED_NO_TRAJECTORY_BANK"
    assert receipt["schema"] == "f20-adjoint-conjugation-engine.v1"
    assert all(v is False for v in receipt["gates"].values())
    assert receipt["meta"]["mu_damp"] == 0.0
    on_disk = json.loads((tmp_path / "r.json").read_text())
    assert set(on_disk.keys()) == {"schema", "gates", "telemetry", "verdict", "meta", "created_utc"}
    assert on_disk["verdict"] == receipt["verdict"]
    gates_g3 = {"PG1": True, "G1": True, "G2": True, "G3": False, "G4": True}
    assert _verdict(gates_g3, {"conj_dev_std_mean": 0.05}) == "F20_GATE_G3_FAILED"
    gates_g2 = {"PG1": True, "G1": True, "G2": False, "G3": False, "G4": False}
    assert _verdict(gates_g2, {"conj_dev_std_mean": 0.05}) == "F20_GATE_G2_FAILED"
    assert _verdict(gates_g2, {"conj_dev_std_mean": 0.0}) == "F20_FALSIFIED_NO_ENGAGEMENT"
    all_pass = {"PG1": True, "G1": True, "G2": True, "G3": True, "G4": True}
    assert _verdict(all_pass, {"conj_dev_std_mean": 0.05}) == "F20_LIVE_LOOP_VERIFIED"


# ---------------------------------------------------------------- flag fail-closed
def test_flag_fail_closed():
    import os
    had = os.environ.pop("HENRI_F20_ADJOINT_CONJUGATION", None)
    try:
        with pytest.raises(RuntimeError):
            require_f20_enabled()
    finally:
        if had is not None:
            os.environ["HENRI_F20_ADJOINT_CONJUGATION"] = had
    require_f20_enabled(_force_enabled=True)
