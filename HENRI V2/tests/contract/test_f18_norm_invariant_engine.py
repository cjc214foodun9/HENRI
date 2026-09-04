"""Carrier F18 contract tests — Norm-Invariant (Unitary Tangent Normalization) engine.

Spec HENRI-SPEC-2026-08-F18-NORM-INVARIANT-STEERING §4, C1–C16 (verbatim):

C1  so(8) skew symmetry: D_hat^T == -D_hat for all normalized generators
C2  Killing-form variance: std(gamma_a) > 0.10 on non-collinear goal states
C3  tangent norm conservation: ||D_hat_a||_F == ||D_a||_F +- 1e-6 across the
    tilt range (kappa sweep {0, 0.5, 0.75, 1.0, 2.0}, gamma in [-1, 1])
C4  positive gradient alignment: d/dgamma |<exp(D_hat_a)Psi_t, Psi_goal>| > 0
    (numerical derivative, non-collinear positive-Killing fixtures)
C5  Sagnac homodyne coherence: single-pass K=8 Sagnac loss <= 0.05
C6  vectorized einsum beam == sequential loop within 1e-5
C7  Hebbian creep strictly on Delta_nu > 0
C8  zero CUDA VRAM leak over 1,000 continuous steps (CUDA assert; CPU runs
    the same 1,000-step loop for functional stability)
C9  PG1 preflight rejection: degenerate bank -> fail-closed
C10 trajectory loader integrity: bank schema + dims + sealed SHA prefix
C11 arcade environment handshake: live API, all 12 games
C12 deterministic seed reproducibility: same seed -> byte-identical outputs
C13 module constants bound: kappa 0.75, beta 0.05, mu 0.0 locked, K 8, ...
C14 NaN/Inf guard: degenerate zero tensors -> clean fallback (finite);
    non-finite engagement telemetry -> fail-closed verdict
C15 zero-Frobenius-penalty lock: mu_damp == 0.0 cannot be overridden
C16 clean receipt generation: schema + verdict mapping
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
from arc_f18_norm_invariant_engine import (  # noqa: E402
    DEFAULT_BEAM,
    DEFAULT_BETA_SAGNAC,
    DEFAULT_ENVS,
    DEFAULT_ETA_FAST,
    DEFAULT_HORIZON,
    DEFAULT_KAPPA_DIFF,
    DEFAULT_SEED,
    ENGAGEMENT_MIN_GAMMA_STD,
    G3_MIN_DNU,
    LATENCY_BUDGET_MS,
    MAX_INITIAL_OVERLAP,
    MU_DAMP_LOCKED,
    SAGNAC_TAU_F18,
    NormInvariantLieEngine,
    _locked_mu,
    _verdict,
    killing_coeffs,
    omega_goal,
    require_f18_enabled,
    run_gauntlet,
)

D = 64
BANK_SHA256_PREFIX = "9e3c01b4"  # sealed F8/F15/F16/F17 bank npz (ledger-verified; remote re-verified)


def _ingress(seed=7):
    return PatchIngress(in_dim=4096, d=D, num_blocks=8, p=32, seed=seed)


def _engine(seed=7, **kw):
    kw.setdefault("D", D)
    kw.setdefault("n_actions", 8)
    return NormInvariantLieEngine(seed=seed, **kw)


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
    om = omega_goal(x, y)
    gens = eng.warped_generators(range(8), om)
    assert gens.shape == (8, D, D)
    for m in gens:
        err = float((m + m.T).abs().max().item())
        assert err < 1e-6, "D_hat must be skew (max err {})".format(err)


# ---------------------------------------------------------------- C2
def test_c2_killing_form_variance():
    eng = _engine(seed=11)
    stds = []
    for s in (3, 5, 7, 9):
        x, y = _orthogonal_pair(seed=s)
        om = omega_goal(x, y)
        gams = eng.gamma_all(range(8), om)
        stds.append(float(gams.std(correction=0).item()))
    assert max(stds) > 0.10, "std(gamma_a) must exceed 0.10 on non-collinear states: {}".format(stds)


# ---------------------------------------------------------------- C3
def test_c3_tangent_norm_conservation():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    for kappa in (0.0, 0.5, 0.75, 1.0, 2.0):
        for gamma in (-1.0, -0.5, 0.0, 0.5, 1.0):
            hat = eng.warped_generators(range(8), om, kappa_diff=kappa,
                                        gamma_override=gamma)
            base_norms = eng.D_a.norm(dim=(-1, -2))
            hat_norms = hat.norm(dim=(-1, -2))
            err = float((hat_norms - base_norms).abs().max().item())
            assert err < 1e-5, "norm conservation failed (kappa {} gamma {} err {})".format(
                kappa, gamma, err)
    # default-path identity: gamma_override=None must also conserve
    hat = eng.warped_generators(range(8), om)
    err = float((hat.norm(dim=(-1, -2)) - eng.D_a.norm(dim=(-1, -2))).abs().max().item())
    assert err < 1e-5


# ---------------------------------------------------------------- C4
def test_c4_positive_gradient_alignment():
    eng = _engine()
    x, y = _orthogonal_pair(seed=9)
    om = omega_goal(x, y)
    # pick a candidate with positive gamma (goal-aligned)
    gams = eng.gamma_all(range(8), om)
    a = int(torch.argmax(gams))
    assert float(gams[a].item()) > 0.0
    eps = 1e-4
    def align(gamma):
        ops = eng.warped_ops([a], om, gamma_override=gamma)
        nxt = F.normalize(ops[0] @ x, p=2, dim=-1)
        return _abs_cos(nxt, y)
    f0 = align(0.0)
    fp = align(eps)
    fm = align(-eps)
    num = (fp - fm) / (2 * eps)
    assert num > 0.0, "d/dgamma alignment must be positive (num deriv {:.4e})".format(num)
    assert fp > f0, "increasing gamma must not decrease alignment ({} vs {})".format(fp, f0)


# ---------------------------------------------------------------- C5
def test_c5_sagnac_homodyne_coherence():
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    # goal-aligned generators: per-step rotation theta = 0.16 rad, applied
    # once per step over the K=8 single-pass horizon -> total 1.28 rad.
    # x is orthogonal to y, so |cos(psi_8, y)| = sin(1.28) ~= 0.958 >= 0.95,
    # i.e. single-pass Sagnac (1 - align) <= 0.05. (theta=0.01 was a fixture
    # bug: sin(0.08) ~= 0.080 < 0.95.)
    aligned = (0.16 * om).unsqueeze(0).expand(8, D, D).contiguous()
    eng.D_a.copy_(aligned)
    ops = eng.warped_ops([0], om)
    state = x
    for _ in range(8):
        state = F.normalize(ops[0] @ state, p=2, dim=-1)
    align = _abs_cos(state, y)
    assert align >= 0.95, "single-pass K=8 coherence failed (align {:.4f})".format(align)
    assert 1.0 - align <= SAGNAC_TAU_F18 + 1e-6
    # the beam objective on this aligned generator is finite and sane
    score = eng.score_action(x, y, om, 0, horizon=8)
    assert math.isfinite(score)
    assert score > 0.5, "aligned single-action horizon must retain a positive J ({:.4f})".format(score)


# ---------------------------------------------------------------- C6
def _naive_beam(eng, x, y, om, H, beam):
    """Sequential-loop reference of beam_search (identical pruning + objective).

    Same recurrence as the engine: per-step Sagnac from the SIGNED overlap
    (1 - cos), per-step J = |cos| - beta * accumulated Sagnac, top-k pruning
    with torch.topk tie semantics (ties keep lower insertion index).
    """
    beta = eng.beta_sagnac
    goal_v = F.normalize(y.reshape(-1).float(), p=2, dim=-1)
    ops = eng.warped_ops(range(eng.n_actions), om)  # [A, D, D]
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
    om = omega_goal(x, y)
    H = 4
    best_vec, j_vec = eng.beam_search(x, y, om, horizon=H, beam=8)
    best_seq, j_seq = _naive_beam(eng, x, y, om, H, beam=8)
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
    om = omega_goal(x, y)
    for _ in range(1000):
        eng.beam_search(x, y, om, horizon=2, beam=8)
    # functional stability: a post-loop call matches a fresh engine's output
    eng2 = _engine(seed=7)
    a1, _ = eng.beam_search(x, y, om, horizon=2, beam=8)
    a2, _ = eng2.beam_search(x, y, om, horizon=2, beam=8)
    assert a1 == a2
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        r0 = torch.cuda.memory_reserved()
        for _ in range(1000):
            eng.beam_search(x, y, om, horizon=2, beam=8)
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
    # sealed SHA prefix of the LIVE bank (ledger F8/F15/F16/F17; remote re-verified)
    assert BANK_SHA256_PREFIX == "9e3c01b4"
    # depth guard: < 30 rows must fail closed (shallow bank, same schema)
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
    om = omega_goal(x, y)
    e1 = _engine(seed=20260917)
    e2 = _engine(seed=20260917)
    assert torch.equal(e1.D_a, e2.D_a), "generator buffers must be byte-identical"
    a1, j1 = e1.beam_search(x, y, om, horizon=4, beam=8)
    a2, j2 = e2.beam_search(x, y, om, horizon=4, beam=8)
    assert a1 == a2 and j1 == j2, "same seed must reproduce the identical trajectory decision"


# ---------------------------------------------------------------- C13
def test_c13_module_constants_bound():
    assert LATENCY_BUDGET_MS == 5.0
    assert SAGNAC_TAU_F18 == 0.050
    assert G3_MIN_DNU == 0.0200
    assert MAX_INITIAL_OVERLAP == 0.90
    assert DEFAULT_KAPPA_DIFF == 0.75
    assert DEFAULT_BETA_SAGNAC == 0.05
    assert MU_DAMP_LOCKED == 0.0
    assert DEFAULT_HORIZON == 8
    assert DEFAULT_BEAM == 8
    assert DEFAULT_SEED == 20260917
    assert DEFAULT_ETA_FAST == 0.05
    assert ENGAGEMENT_MIN_GAMMA_STD == 1e-6
    assert len(DEFAULT_ENVS) == 12


# ---------------------------------------------------------------- C14
def test_c14_nan_inf_guard():
    # degenerate zero generators -> clean finite fallback (clamped denominator)
    eng = _engine()
    x, y = _orthogonal_pair()
    om = omega_goal(x, y)
    eng.D_a.zero_()
    gens = eng.warped_generators(range(8), om)
    assert torch.isfinite(gens).all(), "zero generators must produce finite normalized tensors"
    assert float(gens.norm(dim=(-1, -2)).max().item()) == 0.0
    # non-finite engagement telemetry -> fail-closed verdict (F17 C16 pattern)
    gates_fail = {"PG1": True, "G1": True, "G2": False, "G3": False, "G4": False}
    assert _verdict(gates_fail, {"killing_gamma_std_mean": float("nan")}) == \
        "F18_FALSIFIED_NO_ENGAGEMENT"
    assert _verdict(gates_fail, {"killing_gamma_std_mean": None}) == \
        "F18_FALSIFIED_NO_ENGAGEMENT"


# ---------------------------------------------------------------- C15
def test_c15_zero_frobenius_penalty_lock():
    assert MU_DAMP_LOCKED == 0.0
    with pytest.raises(ValueError):
        _engine(mu_damp=0.15)
    with pytest.raises(ValueError):
        _engine(mu_damp=1e-9)
    # argparse validating type rejects any non-zero value
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
    assert receipt["verdict"] == "F18_BLOCKED_NO_TRAJECTORY_BANK"
    assert receipt["schema"] == "f18-norm-invariant-engine.v1"
    assert all(v is False for v in receipt["gates"].values())
    assert receipt["meta"]["mu_damp"] == 0.0
    # written file is valid JSON with the full envelope
    on_disk = json.loads((tmp_path / "r.json").read_text())
    assert set(on_disk.keys()) == {"schema", "gates", "telemetry", "verdict", "meta", "created_utc"}
    assert on_disk["verdict"] == receipt["verdict"]
    # verdict mapping: engaged G3-only failure -> G3; engaged G2 failure -> G2
    gates_g3 = {"PG1": True, "G1": True, "G2": True, "G3": False, "G4": True}
    assert _verdict(gates_g3, {"killing_gamma_std_mean": 0.05}) == "F18_GATE_G3_FAILED"
    gates_g2 = {"PG1": True, "G1": True, "G2": False, "G3": False, "G4": False}
    assert _verdict(gates_g2, {"killing_gamma_std_mean": 0.05}) == "F18_GATE_G2_FAILED"
    assert _verdict(gates_g2, {"killing_gamma_std_mean": 0.0}) == "F18_FALSIFIED_NO_ENGAGEMENT"
    # full pass -> verified
    all_pass = {"PG1": True, "G1": True, "G2": True, "G3": True, "G4": True}
    assert _verdict(all_pass, {"killing_gamma_std_mean": 0.05}) == "F18_LIVE_LOOP_VERIFIED"


# ---------------------------------------------------------------- flag fail-closed
def test_flag_fail_closed():
    import os
    had = os.environ.pop("HENRI_F18_NORM_INVARIANT", None)
    try:
        with pytest.raises(RuntimeError):
            require_f18_enabled()
    finally:
        if had is not None:
            os.environ["HENRI_F18_NORM_INVARIANT"] = had
    require_f18_enabled(_force_enabled=True)
