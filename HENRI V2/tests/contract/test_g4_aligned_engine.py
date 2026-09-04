"""Contract tests for Carrier G4 aligned sparse affordance engine.

Directive: Sprint_Closeout_Synthesis___Carrier_G4_Master_Directive.md
(bb25dfe247..., 18,288 B) + Functional_Consistency_Synthesis.md (1203d7d8...).
Prereg: docs/spec/g4_aligned_affordance_preregistration.md (dcc09dcc..., sealed
#fd47cb46). Branch feat/carrier-g4-aligned-affordance. Seed 20260927.

Covers: C1 fit/score homology (shared functional), PG1 synthetic pre-flight
(min_action_auc >= 0.85 local), PG2 norm drift <= 1e-6, PG3 top-k=64 variance
rule, sign-corrected affordance (C13), C2 zero policy leakage, stratified N=128
subset, ridge fit recovery (prediction orientation), determinism, CLI kill path.
"""
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

TESTS = pathlib.Path(__file__).resolve()
ROOT = TESTS.parents[2]  # .../HENRI V2 (code dir)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

from arc_g4_aligned_engine import (  # noqa: E402
    SEED,
    TOP_K,
    RIDGE_G4,
    PG2_NORM_TOL,
    SUBSET_PER_ACTION,
    aligned_affordance_pi,
    aligned_mean_quadratic,
    calibrate_theta_tau,
    fit_aligned_transitions,
    per_block_displacement_variance,
    select_topk_blocks,
    stratified_subset,
)
from arc_g1_topological_engine import compute_auc  # noqa: E402


def _rotation_blocks(block_dim: int, signal_blocks, angle: float = 0.6,
                     seed: int = 7):
    """{m: T_m [8,8]} orthogonal rotations in a shared 2D plane, per block."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for m in signal_blocks:
        A = torch.randn(block_dim, block_dim, generator=g)
        Q, _ = torch.linalg.qr(A)
        c, s = math.cos(angle), math.sin(angle)
        R = torch.eye(block_dim)
        R[0, 0], R[0, 1] = c, -s
        R[1, 0], R[1, 1] = s, c
        out[m] = Q @ R @ Q.T
    return out


def _synthetic_bank(num_actions=4, num_blocks=512, block_dim=8,
                    rows_per_action=30, signal_blocks_per_action=8,
                    angle=0.6, angle_jitter=0.2, seed=SEED):
    """Synthetic bank with per-action structured top-k signal blocks.

    Moving rows: psi_next = T(angle_i) psi on signal blocks (per-row angle
    jitter => the blocks carry real variance, per the PG3 selection rule),
    identity elsewhere, per-row flat-normalized. Blocked rows: psi + small
    per-block noise. Returns psi_full [N, M, 8], nxt_full, onehot [N, A], y.
    """
    g = torch.Generator().manual_seed(seed)
    N = num_actions * 2 * rows_per_action
    psi_full = torch.randn(N, num_blocks, block_dim, generator=g)
    psi_full = F.normalize(psi_full.reshape(N, -1), p=2, dim=-1) \
        .view(N, num_blocks, block_dim)
    nxt_full = psi_full.clone()
    onehot = torch.zeros(N, num_actions)
    y = torch.zeros(N)
    for a in range(num_actions):
        blocks = list(range(a * signal_blocks_per_action,
                            (a + 1) * signal_blocks_per_action))
        for i in range(rows_per_action):
            row = (a * 2 * rows_per_action) + i
            onehot[row, a] = 1.0
            ai = angle + angle_jitter * float(torch.randn(1, generator=g).item())
            T = _rotation_blocks(block_dim, blocks, ai, seed + a + i)
            for m in blocks:
                nxt_full[row, m] = T[m] @ psi_full[row, m]
            nxt_full[row] = F.normalize(nxt_full[row].reshape(-1), p=2, dim=-1) \
                .view(num_blocks, block_dim)
            y[row] = 1.0
        for i in range(rows_per_action):
            row = (a * 2 * rows_per_action) + rows_per_action + i
            onehot[row, a] = 1.0
            eps = torch.randn(num_blocks, block_dim, generator=g) * 0.01
            nxt_full[row] = F.normalize((nxt_full[row] + eps).reshape(-1),
                                        p=2, dim=-1).view(num_blocks, block_dim)
            y[row] = 0.0
    return psi_full, nxt_full, onehot, y


def _canonical(psi_full):
    flat = psi_full.reshape(psi_full.shape[0], -1)
    return F.normalize(flat, p=2, dim=-1).view(psi_full.shape)


def _fit_action(psi_c, nxt_c, onehot, y, a, top_k=TOP_K, ridge=RIDGE_G4):
    mask = onehot[:, a].bool()
    mov = mask & (y == 1.0)
    var = per_block_displacement_variance(psi_c[mov], nxt_c[mov])
    topk = select_topk_blocks(var, top_k)
    trans = fit_aligned_transitions(psi_c[mov], nxt_c[mov], topk, ridge)
    r_m = aligned_mean_quadratic(psi_c[mov], nxt_c[mov], trans)
    r_b = aligned_mean_quadratic(psi_c[mask & (y == 0.0)],
                                 nxt_c[mask & (y == 0.0)], trans)
    theta, tau = calibrate_theta_tau(r_m, r_b)
    return topk, trans, theta, tau


def test_c1_fit_score_shared_functional():
    """C1: the score functional IS the fit functional; it separates classes."""
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    for a in range(onehot.shape[1]):
        _, trans, _, _ = _fit_action(psi_c, nxt_c, onehot, y, a)
        mask = onehot[:, a].bool()
        r_mov = aligned_mean_quadratic(psi_c[mask & (y == 1.0)],
                                       nxt_c[mask & (y == 1.0)], trans)
        r_blk = aligned_mean_quadratic(psi_c[mask & (y == 0.0)],
                                       nxt_c[mask & (y == 0.0)], trans)
        assert torch.isfinite(r_mov).all() and torch.isfinite(r_blk).all()
        assert float(r_mov.mean()) < float(r_blk.mean()), \
            f"a{a}: moving residual {r_mov.mean():.5f} must be < blocked {r_blk.mean():.5f}"


def test_pg1_synthetic_preflight_auc():
    """PG1 synthetic pre-flight: min_action_auc >= 0.85 on the fixture."""
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    aucs = {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        _, trans, theta, tau = _fit_action(psi_c, nxt_c, onehot, y, a)
        r_all = aligned_mean_quadratic(psi_c, nxt_c, trans)
        pi = aligned_affordance_pi(r_all, theta, tau)
        aucs[a] = compute_auc(pi[mask], y[mask])
    assert min(aucs.values()) >= 0.85, f"preflight AUCs: {aucs}"


def test_pg2_norm_drift():
    psi, _, _, _ = _synthetic_bank(seed=SEED)
    c = _canonical(psi)
    drift = float((c.reshape(c.shape[0], -1).norm(dim=-1) - 1.0).abs().max().item())
    assert drift <= PG2_NORM_TOL, f"norm drift {drift}"


def test_pg3_topk_selection():
    """PG3: per-action top-k=64 by displacement variance; signal blocks dominate."""
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED, rows_per_action=40)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    a = 0
    mask = onehot[:, a].bool()
    mov = mask & (y == 1.0)
    var = per_block_displacement_variance(psi_c[mov], nxt_c[mov])
    topk = select_topk_blocks(var, TOP_K)
    assert topk.shape[0] == TOP_K
    assert set(range(8)).issubset(set(topk.tolist())), \
        f"signal blocks 0-7 must be in top-k: {sorted(topk.tolist())[:16]}"


def test_sign_corrected_affordance():
    """C13: lower residual => HIGHER pi (blocked cannot yield pi->1)."""
    r_mov = torch.tensor([0.02, 0.03])
    r_blk = torch.tensor([0.5, 0.6])
    pi_mov = aligned_affordance_pi(r_mov, theta=0.2, tau=0.1)
    pi_blk = aligned_affordance_pi(r_blk, theta=0.2, tau=0.1)
    assert float(pi_mov.mean()) > 0.8
    assert float(pi_blk.mean()) < 0.2


def test_c2_zero_policy_leakage():
    src = (ROOT / "production_arc_run.py").read_text(encoding="utf-8")
    assert "arc_g3_wave_packet_search" not in src
    g4 = (ROOT / "experiments" / "verification" / "arc_g4_aligned_engine.py") \
        .read_text(encoding="utf-8")
    assert "arc_g3_wave_packet_search" not in g4


def test_stratified_subset_128():
    onehot = torch.zeros(1000, 7)
    for i in range(1000):
        onehot[i, i % 7] = 1.0
    m1 = stratified_subset(onehot, seed=SEED)
    m2 = stratified_subset(onehot, seed=SEED)
    assert torch.equal(m1, m2), "deterministic"
    assert int(m1.sum().item()) == 128
    counts = [int((m1 & onehot[:, a].bool()).sum().item()) for a in range(7)]
    assert all(c >= 18 for c in counts), f"per-action counts: {counts}"
    assert sum(counts) == 128


def test_ridge_fit_recovers_known_rotation():
    """Recovery: prediction error tiny; stored T is the transpose-conjugate
    of the true operator (engine convention: pred = psi @ T.T)."""
    m = 0
    T_true = _rotation_blocks(8, [m], angle=0.5, seed=11)[m]
    g = torch.Generator().manual_seed(5)
    X = F.normalize(torch.randn(64, 8, generator=g), p=2, dim=-1)
    Y = F.normalize(X @ T_true.T, p=2, dim=-1)
    T_hat = fit_aligned_transitions(X.unsqueeze(1), Y.unsqueeze(1),
                                    torch.tensor([m]), RIDGE_G4)[m]
    pred = X @ T_hat.T
    rel_err = float((pred - Y).norm() / Y.norm())
    # Y rows are renormalized after rotation, so the target deviates from a
    # pure linear map by ~1e-3; ridge shrinkage adds ~1e-4. 1e-2 still
    # separates a correct forward operator (0.13% error) from the previous
    # transposed fit (43% error).
    assert rel_err < 1e-2, f"prediction rel err {rel_err}"
    op_err = float((T_hat - T_true).abs().max().item())
    assert op_err < 0.05, f"operator err {op_err}"


def test_engine_update_lagged_causal():
    from arc_g4_aligned_engine import G4AlignedEngine
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED, num_actions=2)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    topk_masks, trans, theta, tau = {}, {}, {}, {}
    for a in range(2):
        topk_masks[a], trans[a], theta[a], tau[a] = _fit_action(
            psi_c, nxt_c, onehot, y, a)
    eng = G4AlignedEngine(
        transitions_g4=trans, topk_masks=topk_masks,
        theta=[theta[0], theta[1]], tau=[tau[0], tau[1]],
        generators=None, transitions=[], t_pow=torch.eye(64), recon={},
        n_actions=2, seed=SEED, device="cpu")
    eng.update_online_affordance(psi_c[0], 0, nxt_c[0])
    assert eng.affordance_updates == 1
    r = eng.affordance_residuals(psi_c[0], 0)
    assert r is not None and torch.isfinite(r).all()
    assert eng.transitions_g4[0] is trans[0], "no transition mutation"


def test_cli_pg1_kill_path(tmp_path):
    """CLI smoke: degenerate 8192-block bank => PG1 kill, receipt written."""
    import numpy as np
    g = torch.Generator().manual_seed(SEED)
    N = 40
    psi = torch.randn(N, 8192, 8, generator=g)
    psi = F.normalize(psi.reshape(N, -1), p=2, dim=-1).view(N, 8192, 8)
    nxt = psi + torch.randn(N, 8192, 8, generator=g) * 0.05
    nxt = F.normalize(nxt.reshape(N, -1), p=2, dim=-1).view(N, 8192, 8)
    onehot = torch.zeros(N, 2)
    onehot[:20, 0] = 1.0
    onehot[20:, 1] = 1.0
    bank = tmp_path / "bank.npz"
    np.savez(bank, psi=psi.reshape(N, -1).numpy(),
             next_wave=nxt.reshape(N, -1).numpy(),
             actions_onehot=onehot.numpy())
    jsonl = tmp_path / "bank.jsonl"
    jsonl.write_text("", encoding="utf-8")
    out = tmp_path / "g4out"
    env = dict(os.environ)
    env["HENRI_G4_ALIGNED_AFFORDANCE"] = "1"
    r = subprocess.run(
        [sys.executable,
         str(ROOT / "experiments" / "verification" / "arc_g4_aligned_engine.py"),
         "--device", "cpu", "--trajectory-bank", str(bank),
         "--trajectory-jsonl", str(jsonl), "--envs", "dummy-env",
         "--out-dir", str(out)],
        capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 1, \
        f"rc={r.returncode}\nstdout={r.stdout[-1200:]}\nstderr={r.stderr[-1200:]}"
    receipt = out / "g4_gates_receipt.json"
    assert receipt.exists(), f"no receipt; stderr={r.stderr[-800:]}"
    data = json.loads(receipt.read_text())
    assert data["verdict"] == "G4_AFFORDANCE_FIT_COLLAPSE"


def test_c3_same_seed_determinism():
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    a = 0
    mask = onehot[:, a].bool()
    mov = mask & (y == 1.0)
    v1 = per_block_displacement_variance(psi_c[mov], nxt_c[mov])
    v2 = per_block_displacement_variance(psi_c[mov], nxt_c[mov])
    t1 = select_topk_blocks(v1, TOP_K)
    t2 = select_topk_blocks(v2, TOP_K)
    assert torch.equal(t1, t2)
    T1 = fit_aligned_transitions(psi_c[mov], nxt_c[mov], t1, RIDGE_G4)
    T2 = fit_aligned_transitions(psi_c[mov], nxt_c[mov], t2, RIDGE_G4)
    for m in t1.tolist():
        assert torch.equal(T1[m], T2[m]), "transition determinism"


def test_require_flag_gate():
    env = dict(os.environ)
    env.pop("HENRI_G4_ALIGNED_AFFORDANCE", None)
    code = (
        "import sys\n"
        "sys.path.insert(0, r'%s')\n"
        "from arc_g4_aligned_engine import require_flag\n"
        "require_flag('HENRI_G4_ALIGNED_AFFORDANCE')\n" % (ROOT / "experiments" / "verification")
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.returncode == 1, "must fail closed without the flag"
