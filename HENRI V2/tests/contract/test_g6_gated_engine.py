"""Contract tests for Carrier G6 piecewise gated pure-support affordance engine.

Directive: Carrier_G6_Master_Directive___G5_Post-Mortem.md
(HENRI-DIR-2026-09-V3-CARRIER-G6-PIECEWISE-GATING, d5ec31cd..., 18,268 B).
Prereg: docs/spec/g6_gated_affordance_preregistration.md (7c371c0f..., sealed
#52782c82). Branch feat/carrier-g6-gated-affordance. Seed 20260929.

Covers: C1 fit/score homology per arm (bridge + full-D), C2 piecewise gating
stability (alpha EXACTLY 0.0 for N_moving >= 40 — the G5 anti-kill; monotone in
N_a within regime 2; route boundaries), C3 W0 gateway (no wave-packet-search
import), PG1 synthetic preflight, PG2 norm drift, PG3 routing determinism +
CPU==CUDA identity, bridge-arm recovery, CLI kill path, flag gate.
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

from arc_g6_gated_engine import (  # noqa: E402
    SEED,
    TOP_K,
    RIDGE_G6,
    PG2_NORM_TOL,
    NU0,
    SPARSE_THRESHOLD,
    DENSE_THRESHOLD,
    BRIDGE_BLOCKS,
    G6GatedAffordanceEngine,
    compute_piecewise_topk_masks,
    pg3_piecewise_determinism,
    piecewise_alpha,
    piecewise_route_decision,
    require_flag,
)
from arc_g5_shrunk_engine import (  # noqa: E402
    bridge_fit_transitions,
    bridge_mean_quadratic,
)
from arc_g4_aligned_engine import (  # noqa: E402
    aligned_affordance_pi,
    aligned_mean_quadratic,
    calibrate_theta_tau,
    fit_aligned_transitions,
    per_block_displacement_variance,
    select_topk_blocks,
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
                    rows_per_action=40, signal_blocks_per_action=8,
                    angle=0.6, angle_jitter=0.2, seed=SEED):
    """Synthetic bank: per-action structured top-k signal blocks.

    Moving rows: psi_next = T(angle_i) psi on signal blocks (per-row angle
    jitter => real variance), identity elsewhere, flat-normalized. Blocked
    rows: small per-block noise. Returns psi_full [N, M, 8], nxt_full, onehot.
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


def _synthetic_bridge_bank(num_actions=2, bridge_blocks=BRIDGE_BLOCKS,
                           block_dim=8, rows_per_action=40, angle=0.6,
                           seed=SEED):
    """Synthetic D=64 bridge bank: psi64/nxt64 [N, bridge_blocks, 8]."""
    g = torch.Generator().manual_seed(seed)
    N = num_actions * 2 * rows_per_action
    psi = torch.randn(N, bridge_blocks, block_dim, generator=g)
    psi = F.normalize(psi.reshape(N, -1), p=2, dim=-1).view(N, bridge_blocks, block_dim)
    nxt = psi.clone()
    onehot = torch.zeros(N, num_actions)
    y = torch.zeros(N)
    for a in range(num_actions):
        blocks = list(range(a * 4, a * 4 + 4))
        for i in range(rows_per_action):
            row = (a * 2 * rows_per_action) + i
            onehot[row, a] = 1.0
            ai = angle + 0.2 * float(torch.randn(1, generator=g).item())
            T = _rotation_blocks(block_dim, blocks, ai, seed + a + i)
            for m in blocks:
                nxt[row, m] = T[m] @ psi[row, m]
            nxt[row] = F.normalize(nxt[row].reshape(-1), p=2, dim=-1) \
                .view(bridge_blocks, block_dim)
            y[row] = 1.0
        for i in range(rows_per_action):
            row = (a * 2 * rows_per_action) + rows_per_action + i
            onehot[row, a] = 1.0
            eps = torch.randn(bridge_blocks, block_dim, generator=g) * 0.05
            nxt[row] = F.normalize((nxt[row] + eps).reshape(-1), p=2, dim=-1) \
                .view(bridge_blocks, block_dim)
            y[row] = 0.0
    return psi, nxt, onehot, y


# --------------------------------------------------------------------------
# C2: PIECEWISE GATING STABILITY (incl. the G5 anti-kill)
# --------------------------------------------------------------------------

def test_c2_alpha_dense_exact_zero():
    """C2 anti-kill: alpha is EXACTLY 0.0 for every dense action (N >= 40)."""
    for n in (40, 41, 56, 100, 361):
        assert piecewise_alpha(n) == 0.0, f"alpha({n}) must be exactly 0.0"
    # Naming: the G5 kill was alpha=0.1176 perturbing a2/a3; G6 must be 0.0.


def test_c2_alpha_mid_formula_and_monotone():
    """C2: mid regime (20..39) uses nu0/(nu0+N*d) and never increases."""
    for n in range(20, 40):
        expected = 64.0 / (64.0 + float(n) * 8.0)
        assert piecewise_alpha(n) == pytest.approx(expected, rel=1e-6), n
    vals = [piecewise_alpha(n) for n in range(20, 40)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_c2_alpha_sparse_zero_unused():
    """C2: sparse actions carry alpha 0.0 (bridge owns them; alpha unused)."""
    for n in (5, 10, 19):
        assert piecewise_alpha(n) == 0.0


def test_piecewise_route_boundaries():
    """Route boundaries: 19->bridge, 20->topk_shrunk, 39->topk_shrunk, 40->pure."""
    assert piecewise_route_decision(3) == "skip"
    assert piecewise_route_decision(10) == "bridge"
    assert piecewise_route_decision(19) == "bridge"
    assert piecewise_route_decision(20) == "topk_shrunk"
    assert piecewise_route_decision(39) == "topk_shrunk"
    assert piecewise_route_decision(40) == "topk_pure"
    assert piecewise_route_decision(56) == "topk_pure"


def test_c2_anti_kill_dense_pure_support_exact():
    """THE G5 anti-kill: dense actions get PURE empirical top-k, exactly.

    With N_moving = 40 (dense), the piecewise mask must be byte-identical to
    select_topk_blocks(per_block_displacement_variance(...)) — zero prior
    perturbation. This directly contradicts G5's alpha=0.1176 support shift.
    """
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED, rows_per_action=40)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    masks = compute_piecewise_topk_masks(psi_c, nxt_c, onehot, y,
                                         nu0=NU0, k=TOP_K)
    assert set(masks.keys()) == {0, 1, 2, 3}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        mov = mask & (y == 1.0)
        assert int(mov.sum().item()) >= DENSE_THRESHOLD
        var = per_block_displacement_variance(psi_c[mov], nxt_c[mov])
        pure = select_topk_blocks(var, TOP_K)
        assert torch.equal(masks[a], pure), \
            f"a{a}: dense mask must equal pure empirical top-k exactly"
        # Signal blocks must survive.
        assert set(range(a * 8, a * 8 + 8)).issubset(set(masks[a].tolist())), \
            f"a{a}: signal blocks must survive pure top-k"


def test_mid_regime_shrunk_keeps_signal():
    """Mid regime (30 moving): shrunk mask still keeps structured signal."""
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED, rows_per_action=30)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    masks = compute_piecewise_topk_masks(psi_c, nxt_c, onehot, y,
                                         nu0=NU0, k=TOP_K)
    for a in range(onehot.shape[1]):
        assert masks[a].shape[0] == TOP_K
        assert set(range(a * 8, a * 8 + 8)).issubset(set(masks[a].tolist())), \
            f"a{a}: signal blocks must survive mid-regime shrinkage"


def test_sparse_actions_route_bridge_not_masks():
    """Sparse action (10 moving): route is 'bridge' (mask unused)."""
    assert piecewise_route_decision(10) == "bridge"
    # The piecewise mask dict may still carry the sparse action (unused), but
    # the route decision is authoritative for the engine's bridge arm.
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED, rows_per_action=10)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    masks = compute_piecewise_topk_masks(psi_c, nxt_c, onehot, y,
                                         nu0=NU0, k=TOP_K)
    for a in masks:
        assert piecewise_route_decision(10) == "bridge"


# --------------------------------------------------------------------------
# C1: PER-ARM FIT/SCORE HOMOLOGY
# --------------------------------------------------------------------------

def test_c1_bridge_arm_fit_score_homology():
    """C1: within the bridge arm, moving residual < blocked residual."""
    psi, nxt, onehot, y = _synthetic_bridge_bank(seed=SEED)
    trans = bridge_fit_transitions(psi.reshape(psi.shape[0], -1),
                                   nxt.reshape(nxt.shape[0], -1),
                                   onehot, y, RIDGE_G6)
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        r_mov = bridge_mean_quadratic(psi[mask & (y == 1.0)].reshape(-1, BRIDGE_BLOCKS * 8),
                                      nxt[mask & (y == 1.0)].reshape(-1, BRIDGE_BLOCKS * 8),
                                      trans[a])
        r_blk = bridge_mean_quadratic(psi[mask & (y == 0.0)].reshape(-1, BRIDGE_BLOCKS * 8),
                                      nxt[mask & (y == 0.0)].reshape(-1, BRIDGE_BLOCKS * 8),
                                      trans[a])
        assert float(r_mov.mean()) < float(r_blk.mean()), \
            f"a{a} bridge: moving {r_mov.mean():.5f} must be < blocked {r_blk.mean():.5f}"


def test_c1_full_arm_fit_score_homology():
    """C1: within the full-D pure arm, moving residual < blocked residual."""
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED, rows_per_action=40)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    masks = compute_piecewise_topk_masks(psi_c, nxt_c, onehot, y,
                                         nu0=NU0, k=TOP_K)
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        mov = mask & (y == 1.0)
        trans = fit_aligned_transitions(psi_c[mov], nxt_c[mov], masks[a], RIDGE_G6)
        r_mov = aligned_mean_quadratic(psi_c[mov], nxt_c[mov], trans)
        r_blk = aligned_mean_quadratic(psi_c[mask & (y == 0.0)],
                                       nxt_c[mask & (y == 0.0)], trans)
        assert float(r_mov.mean()) < float(r_blk.mean()), \
            f"a{a} full: moving {r_mov.mean():.5f} must be < blocked {r_blk.mean():.5f}"


# --------------------------------------------------------------------------
# PG1/PG2/PG3
# --------------------------------------------------------------------------

def test_pg1_synthetic_preflight_auc():
    """PG1 synthetic preflight: min_action_auc >= 0.85 on the fixture."""
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED, rows_per_action=40)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    masks = compute_piecewise_topk_masks(psi_c, nxt_c, onehot, y,
                                         nu0=NU0, k=TOP_K)
    aucs = {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        mov = mask & (y == 1.0)
        trans = fit_aligned_transitions(psi_c[mov], nxt_c[mov], masks[a], RIDGE_G6)
        r_m = aligned_mean_quadratic(psi_c[mov], nxt_c[mov], trans)
        r_b = aligned_mean_quadratic(psi_c[mask & (y == 0.0)],
                                     nxt_c[mask & (y == 0.0)], trans)
        theta, tau = calibrate_theta_tau(r_m, r_b)
        r_all = aligned_mean_quadratic(psi_c, nxt_c, trans)
        pi = aligned_affordance_pi(r_all, theta, tau)
        aucs[a] = compute_auc(pi[mask], y[mask])
    assert min(aucs.values()) >= 0.85, f"preflight AUCs: {aucs}"


def test_pg2_norm_drift():
    psi, _, _, _ = _synthetic_bank(seed=SEED)
    c = _canonical(psi)
    drift = float((c.reshape(c.shape[0], -1).norm(dim=-1) - 1.0).abs().max().item())
    assert drift <= PG2_NORM_TOL, f"norm drift {drift}"


def test_pg3_routing_determinism():
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    m1 = compute_piecewise_topk_masks(psi_c, nxt_c, onehot, y, NU0, TOP_K)
    m2 = compute_piecewise_topk_masks(psi_c, nxt_c, onehot, y, NU0, TOP_K)
    for a in m1:
        assert torch.equal(m1[a], m2[a]), "same-seed determinism"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA required for cross-device identity")
def test_pg3_cpu_cuda_identity():
    psi, nxt, onehot, y = _synthetic_bank(seed=SEED)
    psi_c, nxt_c = _canonical(psi), _canonical(nxt)
    cpu = compute_piecewise_topk_masks(psi_c, nxt_c, onehot, y, NU0, TOP_K)
    cuda = compute_piecewise_topk_masks(psi_c.cuda(), nxt_c.cuda(),
                                        onehot.cuda(), y.cuda(), NU0, TOP_K)
    for a in cpu:
        assert torch.equal(cpu[a], cuda[a].cpu()), f"a{a} CPU != CUDA"


# --------------------------------------------------------------------------
# C3 / ENGINE / CLI / FLAG
# --------------------------------------------------------------------------

def test_c3_w0_gateway_no_wiring():
    src = (ROOT / "experiments" / "verification" / "arc_g6_gated_engine.py") \
        .read_text(encoding="utf-8")
    assert "arc_g3_wave_packet_search" not in src


def test_engine_bridge_arm_routes():
    """Engine routes a low-support action through the bridge arm (fail-closed
    without ingress; finite residual with a real D=64 projection)."""
    from arc_f21_edmd_engine import _bridge_to_d64_batch
    from arc_f10_live_engine import PatchIngress

    psi, nxt, onehot, y = _synthetic_bridge_bank(seed=SEED, num_actions=1)
    psi64 = psi.reshape(psi.shape[0], -1)
    nxt64 = nxt.reshape(nxt.shape[0], -1)
    trans = bridge_fit_transitions(psi64, nxt64, onehot, y, RIDGE_G6)

    eng = G6GatedAffordanceEngine(
        transitions_g4={}, topk_masks={}, theta=[0.5], tau=[1.0],
        bridge_transitions=trans, bridge_route_flags={0: True},
        generators=None, transitions=[], t_pow=torch.eye(64), recon={},
        n_actions=1, seed=SEED, device="cpu", ingress=None)
    r_none = eng.affordance_residuals(psi64[:1].view(1, BRIDGE_BLOCKS, 8), 0,
                                      nxt64[:1].view(1, BRIDGE_BLOCKS, 8))
    assert r_none is None, "bridge arm must fail closed without ingress"

    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=SEED)
    psi_full = torch.randn(1, 8192, 8, generator=torch.Generator().manual_seed(1))
    psi_full = F.normalize(psi_full.reshape(1, -1), p=2, dim=-1).view(1, 8192, 8)
    nxt_full = psi_full.clone()
    eng.ingress = ingress
    r = eng.affordance_residuals(psi_full, 0, nxt_full)
    assert r is not None and torch.isfinite(r).all(), "bridge residual finite"


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
    out = tmp_path / "g6out"
    env = dict(os.environ)
    env["HENRI_G6_GATED_AFFORDANCE"] = "1"
    r = subprocess.run(
        [sys.executable,
         str(ROOT / "experiments" / "verification" / "arc_g6_gated_engine.py"),
         "--device", "cpu", "--trajectory-bank", str(bank),
         "--trajectory-jsonl", str(jsonl), "--envs", "dummy-env",
         "--out-dir", str(out)],
        capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 1, \
        f"rc={r.returncode}\nstdout={r.stdout[-1200:]}\nstderr={r.stderr[-1200:]}"
    receipt = out / "g6_gates_receipt.json"
    assert receipt.exists(), f"no receipt; stderr={r.stderr[-800:]}"
    data = json.loads(receipt.read_text())
    assert data["verdict"] == "G6_AFFORDANCE_FIT_COLLAPSE"


def test_require_flag_gate():
    env = dict(os.environ)
    env.pop("HENRI_G6_GATED_AFFORDANCE", None)
    code = (
        "import sys\n"
        "sys.path.insert(0, r'%s')\n"
        "from arc_g6_gated_engine import require_flag\n"
        "require_flag()\n" % (ROOT / "experiments" / "verification")
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.returncode == 1, "must fail closed without the flag"
