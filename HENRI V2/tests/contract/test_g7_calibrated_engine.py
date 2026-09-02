"""Contract tests for Carrier G7 calibrated stratified affordance engine.

Directive: Carrier_G7_Master_Directive___G6_Post-Mortem.md
(HENRI-DIR-2026-09-V3-CARRIER-G7-CALIBRATION-EXPANSION, 24b7665d..., 18,073 B).
Prereg: docs/spec/g7_calibrated_affordance_preregistration.md (02083db5..., sealed
#a795c151). Branch feat/carrier-g7-calibrated-affordance. Seed 20260930.

Covers: C1 fit/score homology per arm (bridge + full-D) + score monotonicity
(rank-invariance of tau_a), C2 tau bounds + moving-residual-only source + dense
alpha EXACTLY 0.0 (G5/G6 anti-kill) + route boundaries, C3 W0 gateway (no
wave-packet-search import), PG1 synthetic preflight on the N=256 stratified
subset (quantization claim: per-action rows >= 32, grid step <= 1/32), PG2 norm
drift, PG3 routing + top-k + tau CPU==CUDA identity, bridge-arm recovery, CLI
kill path, flag gate.
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

from arc_g7_calibrated_engine import (  # noqa: E402
    SEED,
    TOP_K,
    RIDGE_G7,
    PG2_NORM_TOL,
    PG1_MIN_AUC_G7,
    NU0,
    SPARSE_THRESHOLD,
    DENSE_THRESHOLD,
    BRIDGE_BLOCKS,
    TAU_LO,
    TAU_HI,
    N_SUBSET_G7,
    SUBSET_PER_ACTION_G7,
    G7CalibratedAffordanceEngine,
    calibrated_affordance_score,
    calibrated_temperature,
    pg3_calibrated_determinism,
    stratified_subset_256,
)
from arc_g6_gated_engine import (  # noqa: E402
    compute_piecewise_topk_masks,
    pg3_piecewise_determinism,
    piecewise_alpha,
    piecewise_route_decision,
)
from arc_g5_shrunk_engine import (  # noqa: E402
    bridge_fit_transitions,
    bridge_mean_quadratic,
    require_flag,
)
from arc_g4_aligned_engine import (  # noqa: E402
    aligned_mean_quadratic,
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
            eps = torch.randn(bridge_blocks, block_dim, generator=g) * 0.01
            nxt[row] = F.normalize((nxt[row] + eps).reshape(-1),
                                   p=2, dim=-1).view(bridge_blocks, block_dim)
            y[row] = 0.0
    return psi, nxt, onehot, y


def _run_pipeline(psi_full, nxt_full, onehot, y, seed=SEED):
    """Shared PG1 pipeline: masks -> fits -> tau -> calibrated scores -> AUCs."""
    n_actions = onehot.shape[1]
    masks = compute_piecewise_topk_masks(psi_full, nxt_full, onehot, y, NU0,
                                         TOP_K, sparse=SPARSE_THRESHOLD,
                                         dense=DENSE_THRESHOLD)
    trans, tau_cal = {}, {}
    for a in range(n_actions):
        mov = onehot[:, a].bool() & (y == 1.0)
        if int(mov.sum().item()) < 5:
            trans[a] = {}
            tau_cal[a] = 1.0
            continue
        trans[a] = fit_aligned_transitions(psi_full[mov], nxt_full[mov],
                                           masks[a], RIDGE_G7)
        r_mov = aligned_mean_quadratic(psi_full[mov], nxt_full[mov], trans[a])
        tau_cal[a] = calibrated_temperature(r_mov, TAU_LO, TAU_HI)
    pi = torch.zeros(onehot.shape[0], n_actions)
    for a in range(n_actions):
        if not trans[a]:
            pi[:, a] = 0.5
            continue
        r = aligned_mean_quadratic(psi_full, nxt_full, trans[a])
        pi[:, a] = calibrated_affordance_score(r, tau_cal[a])
    subset = stratified_subset_256(onehot, seed=seed)
    aucs, aucs_sub, n_sub = {}, {}, {}
    for a in range(n_actions):
        mask = onehot[:, a].bool()
        aucs[str(a)] = compute_auc(pi[mask, a], y[mask])
        mask_sub = mask & subset
        n_sub[str(a)] = (int((mask_sub & (y == 1.0)).sum().item()),
                         int((mask_sub & (y == 0.0)).sum().item()))
        aucs_sub[str(a)] = compute_auc(pi[mask_sub, a], y[mask_sub])
    return pi, masks, trans, tau_cal, subset, aucs, aucs_sub, n_sub


# ---------------------------------------------------------------------------
# C1: functional homology + score surface
# ---------------------------------------------------------------------------

def test_c1_score_is_exp_minus_mean_over_tau():
    r = torch.tensor([0.05, 0.5, 2.0])
    tau = 0.5
    got = calibrated_affordance_score(r, tau)
    exp = torch.exp(-r / tau)
    assert torch.allclose(got, exp, atol=1e-6)


def test_c1_score_monotone_rank_invariant_in_tau():
    """exp(-r/tau) is strictly decreasing in r for fixed tau: per-action AUC is
    invariant to tau_a (pre-registered honesty note)."""
    r = torch.tensor([0.1, 0.3, 0.9, 1.7])
    for tau in (0.05, 0.5, 2.0):
        s = calibrated_affordance_score(r, tau)
        # Descending s == ascending r (exp(-r/tau) is monotone decreasing).
        assert torch.equal(torch.argsort(s, descending=True),
                           torch.argsort(r))


def test_c1_fit_score_homology_full():
    """Fit functional == score functional (same mean-quadratic path)."""
    psi_full, nxt_full, onehot, y = _synthetic_bank(seed=SEED)
    psi_full = _canonical(psi_full)
    nxt_full = _canonical(nxt_full)
    a = 0
    mov = onehot[:, a].bool() & (y == 1.0)
    var = per_block_displacement_variance(psi_full[mov], nxt_full[mov])
    topk = select_topk_blocks(var, TOP_K)
    trans = fit_aligned_transitions(psi_full[mov], nxt_full[mov], topk, RIDGE_G7)
    r_mov = aligned_mean_quadratic(psi_full[mov], nxt_full[mov], trans)
    # Score uses the SAME residual values the fit minimized (C1).
    assert torch.isfinite(r_mov).all()
    assert float(r_mov.mean()) < 1e-2


# ---------------------------------------------------------------------------
# C2: temperature bounds + moving-only source; piecewise alpha + routes
# ---------------------------------------------------------------------------

def test_c2_tau_bounds_empty_and_clamps():
    assert calibrated_temperature(torch.tensor([])) == 1.0
    assert calibrated_temperature(None) == 1.0
    assert calibrated_temperature(torch.tensor([0.5, 0.5, 0.6])) == 0.5
    assert calibrated_temperature(torch.tensor([5.0, 6.0])) == TAU_HI
    assert calibrated_temperature(torch.tensor([0.001, 0.002])) == TAU_LO
    assert TAU_LO > 0.0 and TAU_HI >= 2.0


def test_c2_tau_uses_moving_residuals_only():
    """tau_a is the median of the action's MOVING-row residuals (C2)."""
    moving = torch.tensor([0.1, 0.2, 0.3, 0.4])
    # torch.median returns the lower-middle element for even counts (float32).
    assert calibrated_temperature(moving) == pytest.approx(0.2, abs=1e-6)
    # A blocked outlier must not move the median (odd count -> middle 0.3).
    assert calibrated_temperature(torch.cat([moving, torch.tensor([50.0])])) \
        == pytest.approx(0.3, abs=1e-6)


def test_c2_dense_alpha_exact_zero_and_routes():
    for n in (40, 65, 200):
        assert piecewise_alpha(n, NU0) == 0.0
    # Monotone non-increasing within regime 2.
    alphas = [piecewise_alpha(n, NU0) for n in range(20, 40)]
    assert all(a2 <= a1 for a1, a2 in zip(alphas, alphas[1:]))
    assert all(a > 0.0 for a in alphas)
    assert piecewise_route_decision(10) == "bridge"
    assert piecewise_route_decision(19) == "bridge"
    assert piecewise_route_decision(20) == "topk_shrunk"
    assert piecewise_route_decision(39) == "topk_shrunk"
    assert piecewise_route_decision(40) == "topk_pure"
    assert piecewise_route_decision(3) == "skip"


# ---------------------------------------------------------------------------
# PG1: synthetic preflight on the N=256 stratified subset (quantization claim)
# ---------------------------------------------------------------------------

def test_pg1_synthetic_preflight_dense():
    """Dense 4-action bank: global >= 0.9000 and all per-action >= 0.9500."""
    psi_full, nxt_full, onehot, y = _synthetic_bank(seed=SEED)
    psi_full = _canonical(psi_full)
    nxt_full = _canonical(nxt_full)
    pi, masks, trans, tau_cal, subset, aucs, aucs_sub, n_sub = \
        _run_pipeline(psi_full, nxt_full, onehot, y, seed=SEED)
    assert min(aucs_sub.values()) >= PG1_MIN_AUC_G7
    assert all(v >= 0.9500 for v in aucs_sub.values())
    for a in range(onehot.shape[1]):
        assert n_sub[str(a)][0] >= 10 and n_sub[str(a)][1] >= 10


def test_pg1_subset_quantization_claim():
    """Directive §1.2: N=256 => per-action rows >= 32 and grid step <= 1/32.

    With one legitimate outlier the action can still clear 0.9500
    ((n-1)/n >= 0.9500 for n >= 40)."""
    psi_full, nxt_full, onehot, y = _synthetic_bank(num_actions=7, seed=SEED)
    subset = stratified_subset_256(onehot, seed=SEED)
    assert int(subset.sum().item()) == N_SUBSET_G7
    counts = {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        counts[a] = int((mask & subset).sum().item())
        assert counts[a] >= SUBSET_PER_ACTION_G7  # >= 32 (directive bound)
    # Grid step for the per-action subset (balanced n1/n0 case).
    for a in range(onehot.shape[1]):
        assert 1.0 / counts[a] <= 1.0 / 32.0
    # Determinism: same seed -> same mask.
    subset2 = stratified_subset_256(onehot, seed=SEED)
    assert torch.equal(subset, subset2)


def test_pg1_subset_extras_on_largest_actions():
    """4-action bank: 37 rows on all four largest (36 + 1 extra), no overlap."""
    psi_full, nxt_full, onehot, y = _synthetic_bank(num_actions=4, seed=SEED)
    subset = stratified_subset_256(onehot, seed=SEED)
    counts = [int((onehot[:, a].bool() & subset).sum().item())
              for a in range(onehot.shape[1])]
    assert counts == [SUBSET_PER_ACTION_G7 + 1] * 4
    assert int(subset.sum().item()) == 4 * (SUBSET_PER_ACTION_G7 + 1)


# ---------------------------------------------------------------------------
# PG2: norm drift
# ---------------------------------------------------------------------------

def test_pg2_norm_drift_canonical():
    psi_full, nxt_full, onehot, y = _synthetic_bank(seed=SEED)
    psi_full = _canonical(psi_full)
    drift = float((psi_full.reshape(psi_full.shape[0], -1).norm(dim=-1) - 1.0)
                  .abs().max().item())
    assert drift <= PG2_NORM_TOL


# ---------------------------------------------------------------------------
# PG3: CPU == CUDA identity (CUDA branch skipped on CPU-only hosts)
# ---------------------------------------------------------------------------

def test_pg3_cpu_identity():
    psi_full, nxt_full, onehot, y = _synthetic_bank(seed=SEED)
    psi_full = _canonical(psi_full)
    nxt_full = _canonical(nxt_full)
    assert pg3_calibrated_determinism(psi_full, nxt_full, onehot, y, NU0, TOP_K,
                                      TAU_LO, TAU_HI)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA-only cross-device identity")
def test_pg3_cuda_identity():
    psi_full, nxt_full, onehot, y = _synthetic_bank(seed=SEED)
    psi_full = _canonical(psi_full).cuda()
    nxt_full = _canonical(nxt_full).cuda()
    assert pg3_calibrated_determinism(psi_full, nxt_full, onehot.cuda(), y.cuda(),
                                      NU0, TOP_K, TAU_LO, TAU_HI)


# ---------------------------------------------------------------------------
# C3: W0 gateway — no wave-packet planner import in the engine
# ---------------------------------------------------------------------------

def test_c3_w0_gated():
    src = (ROOT / "experiments" / "verification" / "arc_g7_calibrated_engine.py") \
        .read_text(encoding="utf-8")
    assert "arc_g3_wave_packet_search" not in src
    assert "HENRI_G7_CALIBRATED_AFFORDANCE" in src


# ---------------------------------------------------------------------------
# Bridge arm (sparse action recovery) + engine verdicts
# ---------------------------------------------------------------------------

def test_bridge_arm_recovery():
    """Sparse action (< 20 moving) routed through the D=64 bridge recovers AUC
    (G5/G6-verified a4/a6 result; G7 retains the bridge)."""
    psi, nxt, onehot, y = _synthetic_bridge_bank(rows_per_action=8, seed=SEED)
    # 8 moving rows/action -> route == "bridge".
    a = 0
    assert piecewise_route_decision(8) == "bridge"
    mov = onehot[:, a].bool() & (y == 1.0)
    # Live bridge-fit path (G6/G7 main): fit_aligned_transitions on the
    # D=64 bridge blocks; G5's bridge_fit_transitions takes a onehot matrix,
    # not a block list — this is the verified live call shape.
    trans = fit_aligned_transitions(psi[mov].view(-1, BRIDGE_BLOCKS, 8),
                                    nxt[mov].view(-1, BRIDGE_BLOCKS, 8),
                                    torch.arange(BRIDGE_BLOCKS), RIDGE_G7)
    r = bridge_mean_quadratic(psi, nxt, trans)
    tau = calibrated_temperature(
        bridge_mean_quadratic(psi[mov], nxt[mov], trans), TAU_LO, TAU_HI)
    pi = calibrated_affordance_score(r, tau)
    assert compute_auc(pi[onehot[:, a].bool()], y[onehot[:, a].bool()]) >= 0.9500


def test_engine_no_engagement_verdict():
    """steps_done > 0 with zero online updates => NO_AFFORDANCE_ENGAGEMENT."""
    psi, nxt, onehot, y = _synthetic_bridge_bank(seed=SEED)
    eng = G7CalibratedAffordanceEngine(
        transitions_g4={}, topk_masks={}, theta=[0.0, 0.0], tau=[1.0, 1.0],
        bridge_transitions={}, bridge_route_flags={0: True, 1: True},
        generators=[], transitions=[], t_pow=torch.eye(64), recon=None,
        tau_cal=[1.0, 1.0], n_actions=2, seed=SEED, device="cpu")
    v = eng._decide_verdict(1.0, 0, 0.0, 0.01, steps_done=10, updates=0)
    assert v == "G7_NO_AFFORDANCE_ENGAGEMENT"


# ---------------------------------------------------------------------------
# Flag gate (default-OFF) + CLI kill path
# ---------------------------------------------------------------------------

def test_flag_gate():
    os.environ.pop("HENRI_G7_CALIBRATED_AFFORDANCE", None)
    with pytest.raises(SystemExit) as ei:
        require_flag("HENRI_G7_CALIBRATED_AFFORDANCE")
    assert ei.value.code == 1
    os.environ["HENRI_G7_CALIBRATED_AFFORDANCE"] = "1"
    require_flag("HENRI_G7_CALIBRATED_AFFORDANCE")  # no raise


def test_cli_kill_path_flag_absent():
    """CLI with the gate env absent fails closed BEFORE any computation."""
    code = (
        "import sys; sys.path.insert(0, 'experiments/verification'); "
        "from arc_g7_calibrated_engine import main; raise SystemExit(main())"
    )
    env = dict(os.environ)
    env.pop("HENRI_G7_CALIBRATED_AFFORDANCE", None)
    r = subprocess.run(
        [sys.executable, "-c", code,
         "--trajectory-bank", "x.npz", "--trajectory-jsonl", "x.jsonl"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1
    assert "BLOCKED" in (r.stdout + r.stderr)
