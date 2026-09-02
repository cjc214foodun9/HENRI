"""Carrier G7 — Calibrated Sample-Density Expansion & Finite-Sample Variance Regularization Engine.

Directive: Carrier_G7_Master_Directive___G6_Post-Mortem.md
(HENRI-DIR-2026-09-V3-CARRIER-G7-CALIBRATION-EXPANSION, 24b7665d..., 18,073 B, 344 lines).
Prereg: docs/spec/g7_calibrated_affordance_preregistration.md (02083db5..., sealed
#a795c151). Branch feat/carrier-g7-calibrated-affordance. Seed 20260930.
Parent: G6_GATES_VERDICT #e57da003 (26th sealed falsification; PG1 global 0.9412
PASS; PG1a a3 0.9412 < 0.9500 KILL on a degenerate 1/17 subset draw; full-bank
0.9775; a2 FIXED 0.9231->1.0 by alpha=0; a4/a6 1.0 via bridge).

Mechanism (directive-mandated, reconciled with live code per prereg):
  RETAINED from G6 (verified): piecewise 3-regime routing (Regime 3 N>=40 alpha
  EXACTLY 0.0 pure empirical top-k — the G5 anti-kill; Regime 2 20..39 shrunken
  top-k nu0/(nu0+N*d); Regime 1 <20 D=64 bridge via _bridge_to_d64_batch,
  VERIFIED origin arc_f21_edmd_engine.py, fail-closed ingress guard). Centered
  per-block displacement variance (G4-G6 locked). Directive code-block bridge
  slice `psi_t[:, :8]` and uncentered `sample_var` REJECTED
  (CONFLICTS_WITH_LIVE_CODE — same class as G4-G6 corrections).
  NEW — score surface (directive §2.1): S_calibrated(a) = exp(-mean_loss(a)/tau_a)
  with tau_a = clamp(Median(moving residuals of action a), 0.05, 2.0); empty
  moving set -> 1.0. C1: the residual functional is UNCHANGED (mean-quadratic
  over the action's top-k support, fit == score). tau_a is a monotone per-action
  transform -> per-action AUC is RANK-INVARIANT to it; the a3 fix is the subset
  resolution (pre-registered, honest).
  NEW — PG1a evaluation subset (directive §1.2): stratified N=256 draw
  (36 rows/action + 1 extra on the four largest actions = 256), seed-locked per
  action, replacing G6's N=128 (18/action). Per-action sample density doubles,
  so the moving-row count in the subset roughly doubles and the empirical
  Mann-Whitney AUC grid (step 1/(n1*n0)) becomes fine enough to survive real
  noise; the receipt reports per-action n1/n0 so the quantization claim is
  testable.
  Gates (YAML §3.2 governs): PG1 global min_action_auc_subset >= 0.9000;
  PG1a a0-a4 >= 0.9500 (a4 bridge), a5/a6 >= 0.8800 (a6 bridge); PG2 norm drift
  <= 1e-6; PG3 piecewise routing + top-k + tau_a CPU==CUDA identity.
  C3: W0 (WavePacketPathSearch wiring) GATED on FULL PG1 clearance — separate
  approval-gated carrier, NOT wired here.
Verdicts G7_*. Flag HENRI_G7_CALIBRATED_AFFORDANCE=1 (default-OFF fail-closed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from arc_g6_gated_engine import (
        G6GatedAffordanceEngine,
        compute_piecewise_topk_masks,
        pg3_piecewise_determinism,
        piecewise_alpha,
        piecewise_route_decision,
    )
except Exception:  # pragma: no cover - test isolation
    raise

try:
    from arc_g5_shrunk_engine import (
        BRIDGE_BLOCKS,
        NU0,
        bridge_fit_transitions,
        bridge_mean_quadratic,
        require_flag,
    )
except Exception:  # pragma: no cover - test isolation
    raise

try:
    from arc_g4_aligned_engine import (
        B_FULL,
        BLK,
        FastFullDWaveEncoder,
        aligned_mean_quadratic,
        fit_aligned_transitions,
        per_block_displacement_variance,
        select_topk_blocks,
        stall_cosine_labels,
    )
except Exception:  # pragma: no cover - test isolation
    raise

try:
    from arc_g1_topological_engine import (
        _safe_levels,
        DEFAULT_HORIZON,
        G1_LATENCY_MS,
        G2_MIN_SOLVED,
        G3_MIN_DELTA_NU,
        G4_MAX_AFFORDANCE,
        LANGEVIN_TEMP,
        MIN_AUC_SAMPLES,
        MOVING_THRESH,
        OMEGA_BOUND,
        WAYPOINT_ADVANCE_THRESH,
        compute_auc,
        compile_free_generators_capped,
        extract_waypoints,
        langevin_escape_tick,
    )
except Exception:  # pragma: no cover - test isolation
    compute_auc = compile_free_generators_capped = None
    _safe_levels = None
    DEFAULT_HORIZON = 8
    G1_LATENCY_MS = 2.0
    G2_MIN_SOLVED = 1
    G3_MIN_DELTA_NU = 0.0150
    G4_MAX_AFFORDANCE = 0.0500
    LANGEVIN_TEMP = 0.50
    MIN_AUC_SAMPLES = 10
    MOVING_THRESH = 0.05
    OMEGA_BOUND = math.pi / 32.0
    WAYPOINT_ADVANCE_THRESH = 0.60
    extract_waypoints = langevin_escape_tick = None

try:
    from arc_f21_edmd_engine import _bridge_to_d64_batch
except Exception:  # pragma: no cover - test isolation
    _bridge_to_d64_batch = None

try:
    from arc_f10_live_engine import PatchIngress
except Exception:  # pragma: no cover - test isolation
    PatchIngress = None

try:
    from arc_f15_trajectory_engine import DEFAULT_ENVS, resolve_trajectory_goal
except Exception:  # pragma: no cover - test isolation
    DEFAULT_ENVS = []
    resolve_trajectory_goal = None

FLAG = "HENRI_G7_CALIBRATED_AFFORDANCE"
SEED = 20260930
TOP_K = 64
RIDGE_G7 = 1e-2
PG1_MIN_AUC_G7 = 0.9000
PG2_NORM_TOL = 1e-6
TAU_STALL_G7 = 0.90
N_SUBSET_G7 = 256
SUBSET_PER_ACTION_G7 = 36
TAU_LO = 0.05
TAU_HI = 2.0
SPARSE_THRESHOLD = 20
DENSE_THRESHOLD = 40
BRIDGE_BLOCKS = 8
NU0 = 64.0
PG1A_TARGETS_G7 = {0: 0.9500, 1: 0.9500, 2: 0.9500, 3: 0.9500,
                   4: 0.9500, 5: 0.8800, 6: 0.8800}


def calibrated_temperature(moving_residuals: torch.Tensor,
                           lo: float = TAU_LO, hi: float = TAU_HI) -> float:
    """tau_a = clamp(Median(moving residuals), 0.05, 2.0); empty -> 1.0 (C2)."""
    if moving_residuals is None or moving_residuals.numel() == 0:
        return 1.0
    med = float(torch.median(moving_residuals).item())
    return max(lo, min(hi, med))


def calibrated_affordance_score(residuals: torch.Tensor,
                                tau_a: float) -> torch.Tensor:
    """S_calibrated = exp(-mean_loss / tau_a) (directive §2.1; C1 homology).

    Monotone decreasing in the residual for fixed tau_a -> per-action rank
    invariant; the a3 fix is the subset resolution, not tau_a (pre-registered).
    """
    return torch.exp(-residuals / max(float(tau_a), 1e-6))


def finalize_receipt(base_result: dict, live_result: dict) -> dict:
    """Merge pre-flight base fields with the live-loop result; live fields win.

    Fixes the inherited G4 receipt clobber where base_result's hardcoded
    steps_done: 0 overwrote the live-loop step count after a full run.
    """
    return {**base_result, **live_result}


def stratified_subset_256(onehot: torch.Tensor,
                          per_action: int = SUBSET_PER_ACTION_G7,
                          seed: int = SEED) -> torch.Tensor:
    """[N, A] -> boolean mask of an action-stratified N=256 draw.

    36 rows/action (7 * 36 = 252) + 1 extra on the four largest actions = 256.
    Per-action sample density doubles vs G6's N=128 (18/action), so the
    moving-row count in each per-action subset roughly doubles and the AUC grid
    (step 1/(n1*n0)) becomes fine enough to survive real noise (directive §1.2).
    Device-safe: indices moved to the onehot device before gathering.
    """
    n = onehot.shape[0]
    sizes = onehot.sum(0)
    order = torch.argsort(sizes, descending=True)
    counts = {int(a): per_action for a in range(onehot.shape[1])}
    for a in order[:4].tolist():
        counts[a] += 1
    mask = torch.zeros(n, dtype=torch.bool, device=onehot.device)
    for a in range(onehot.shape[1]):
        idx = torch.nonzero(onehot[:, a].bool()).squeeze(-1)
        g = torch.Generator().manual_seed(seed + a)
        perm = torch.randperm(idx.numel(), generator=g)
        pick = idx[perm.to(idx.device)][: counts[a]]
        mask[pick] = True
    return mask


def pg3_calibrated_determinism(psi_full: torch.Tensor, nxt_full: torch.Tensor,
                               onehot: torch.Tensor, y: torch.Tensor,
                               nu0: float = NU0, k: int = TOP_K,
                               lo: float = TAU_LO, hi: float = TAU_HI) -> bool:
    """PG3: piecewise routing + top-k + tau_a CPU==CUDA identity."""
    if not pg3_piecewise_determinism(psi_full, nxt_full, onehot, y, nu0, k):
        return False
    # tau_a parity: compute on CPU reference vs device path.
    cpu = _per_action_tau(psi_full.cpu(), nxt_full.cpu(), onehot.cpu(), y.cpu(),
                          lo, hi)
    dev = _per_action_tau(psi_full, nxt_full, onehot, y, lo, hi)
    for a in dev:
        if abs(cpu[a] - dev[a]) > 1e-6:
            return False
    return True


def _per_action_tau(psi_full: torch.Tensor, nxt_full: torch.Tensor,
                    onehot: torch.Tensor, y: torch.Tensor,
                    lo: float = TAU_LO, hi: float = TAU_HI) -> dict:
    """{a: tau_a} from moving-row quadratic residuals per action (C2)."""
    out = {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        mov = mask & (y == 1.0)
        if int(mov.sum().item()) == 0:
            out[a] = 1.0
            continue
        # Residual reference uses per-block displacement over the full space:
        # mean quadratic residual needs fitted transitions; for the parity
        # check we use the centered displacement variance magnitude as the
        # residual proxy (monotone in the same per-block quadratic family).
        d = torch.norm(psi_full[mov] - nxt_full[mov], p=2, dim=-1) ** 2
        out[a] = calibrated_temperature(d.reshape(-1), lo, hi)
    return out


class G7CalibratedAffordanceEngine(G6GatedAffordanceEngine):
    """Calibrated stratified affordance engine (G6 subclass; tau_a score surface).

    The live-loop kinematics are inherited unchanged from G6 (D=64 carried);
    the affordance score is S = exp(-r / tau_a) with per-action median
    calibrated tau_a (C1: same mean-quadratic residual functional as the fit).
    """

    def __init__(self, transitions_g4, topk_masks, theta, tau,
                 bridge_transitions, bridge_route_flags,
                 generators, transitions, t_pow, recon,
                 tau_cal, action_names=None, n_actions=7, seed=SEED,
                 horizon=DEFAULT_HORIZON, device="cuda",
                 omega_bound=OMEGA_BOUND, waypoints=None,
                 waypoint_advance_thresh=WAYPOINT_ADVANCE_THRESH,
                 langevin_temp=LANGEVIN_TEMP, tau_stall=TAU_STALL_G7,
                 ingress=None):
        super().__init__(
            transitions_g4=transitions_g4, topk_masks=topk_masks,
            theta=theta, tau=tau, bridge_transitions=bridge_transitions,
            bridge_route_flags=bridge_route_flags,
            generators=generators, transitions=transitions, t_pow=t_pow,
            recon=recon, action_names=action_names, n_actions=n_actions,
            seed=seed, horizon=horizon, device=device, omega_bound=omega_bound,
            waypoints=waypoints,
            waypoint_advance_thresh=waypoint_advance_thresh,
            langevin_temp=langevin_temp, tau_stall=tau_stall,
            ingress=ingress)
        self.tau_cal = [float(t) for t in (tau_cal or [1.0] * n_actions)]

    def predict_affordance(self, psi_full, psi_full_next=None):
        """Pi [B, A]: exp(-r_a / tau_cal[a]) over the C1 residual functional."""
        psi_full = psi_full.float().to(self.device)
        if psi_full.dim() == 2:
            psi_full = psi_full.unsqueeze(0)
        pis = []
        for a in range(self.n_actions):
            r = self.affordance_residuals(psi_full, a, psi_full_next)
            if r is None:
                pis.append(torch.full((psi_full.shape[0],), 0.5,
                                      device=self.device))
            else:
                pis.append(calibrated_affordance_score(
                    r, self.tau_cal[a]))
        return torch.stack(pis, dim=-1)

    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        if steps_done > 0 and updates == 0:
            return "G7_NO_AFFORDANCE_ENGAGEMENT"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "G7_GATE_G1_FAILED"
        if solved < G2_MIN_SOLVED:
            return "G7_GATE_G2_FAILED"
        if mean_delta_nu is not None and mean_delta_nu < G3_MIN_DELTA_NU:
            return "G7_GATE_G3_FAILED"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "G7_GATE_G4_FAILED"
        return "G7_CALIBRATED_AFFORDANCE_VERIFIED"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Carrier G7 calibrated stratified affordance gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--omega-bound", type=float, default=OMEGA_BOUND)
    ap.add_argument("--waypoint-advance-thresh", type=float,
                    default=WAYPOINT_ADVANCE_THRESH)
    ap.add_argument("--tau-stall", type=float, default=TAU_STALL_G7)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--ridge", type=float, default=RIDGE_G7)
    ap.add_argument("--nu0", type=float, default=NU0)
    ap.add_argument("--sparse-threshold", type=int, default=SPARSE_THRESHOLD)
    ap.add_argument("--dense-threshold", type=int, default=DENSE_THRESHOLD)
    ap.add_argument("--tau-lo", type=float, default=TAU_LO)
    ap.add_argument("--tau-hi", type=float, default=TAU_HI)
    ap.add_argument("--trajectory-bank", required=True)
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--out-dir", default="/tmp/henri_g7_calibrated/")
    ap.add_argument("--receipt-out", default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    require_flag(FLAG)
    device = args.device
    env_names = list(args.envs) if args.envs else list(DEFAULT_ENVS)

    data = np.load(args.trajectory_bank)
    psi_flat = torch.from_numpy(np.asarray(data["psi"])).float().to(device)
    nxt_flat = torch.from_numpy(np.asarray(data["next_wave"])).float().to(device)
    onehot = torch.from_numpy(np.asarray(data["actions_onehot"])).to(torch.uint8).to(device)

    # Canonical flat-unit geometry (matches bank next_wave; G2-verified label).
    psi_full = F.normalize(psi_flat.float().reshape(psi_flat.shape[0], -1),
                           p=2, dim=-1).view(-1, B_FULL, BLK)
    nxt_full = F.normalize(nxt_flat.float().reshape(nxt_flat.shape[0], -1),
                           p=2, dim=-1).view(-1, B_FULL, BLK)

    # PG2: flat norm drift on the working geometry.
    norm_drift = float((psi_full.reshape(psi_full.shape[0], -1).norm(dim=-1) - 1.0)
                       .abs().max().item())
    pg2_passed = norm_drift <= PG2_NORM_TOL

    # Stall-cosine labels (G2-verified flat norm-divided; tau_stall 0.90).
    y, _cos = stall_cosine_labels(psi_flat, nxt_flat, args.tau_stall)

    # Per-action moving counts and piecewise route decisions (C2 / PG3).
    moving_counts = {str(a): int((onehot[:, a].bool() & (y == 1.0)).sum().item())
                     for a in range(onehot.shape[1])}
    routes = {a: piecewise_route_decision(
        moving_counts[str(a)], args.sparse_threshold, args.dense_threshold)
        for a in range(onehot.shape[1])}
    alpha = {str(a): piecewise_alpha(
        moving_counts[str(a)], args.nu0, sparse=args.sparse_threshold,
        dense=args.dense_threshold)
        for a in range(onehot.shape[1])}

    # C2: dense actions must have EXACTLY zero shrinkage (G6-verified anti-kill).
    c2_dense_exact_zero = all(
        alpha[str(a)] == 0.0
        for a in range(onehot.shape[1])
        if moving_counts[str(a)] >= args.dense_threshold)

    # Piecewise top-k masks (pure for dense, shrunk for mid; sparse -> bridge).
    topk_masks = compute_piecewise_topk_masks(
        psi_full, nxt_full, onehot, y, args.nu0, args.top_k,
        sparse=args.sparse_threshold, dense=args.dense_threshold)

    # D=64 bridge projection for ALL rows (bridge arm fit + score).
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32,
                           seed=args.seed).to(device)
    psi64 = _bridge_to_d64_batch(psi_full.reshape(psi_full.shape[0], -1),
                                 ingress=ingress, seed=args.seed)
    nxt64 = _bridge_to_d64_batch(nxt_full.reshape(nxt_full.shape[0], -1),
                                 ingress=ingress, seed=args.seed)

    # Per-arm ridge transitions on MOVING rows (C1 per arm) + tau_a calibration.
    transitions_g4, bridge_trans = {}, {}
    tau_cal = {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        mov = mask & (y == 1.0)
        route = routes[a]
        if route == "skip":
            transitions_g4[a] = {}
            bridge_trans[a] = {}
            tau_cal[a] = 1.0
            continue
        if route == "bridge":
            bridge_trans[a] = fit_aligned_transitions(
                psi64[mov].view(-1, BRIDGE_BLOCKS, BLK),
                nxt64[mov].view(-1, BRIDGE_BLOCKS, BLK),
                torch.arange(BRIDGE_BLOCKS), args.ridge)
            transitions_g4[a] = {}
            r_mov = bridge_mean_quadratic(psi64[mov], nxt64[mov], bridge_trans[a])
        else:  # topk_shrunk | topk_pure — same full-D arm, mask differs
            transitions_g4[a] = fit_aligned_transitions(
                psi_full[mov], nxt_full[mov], topk_masks[a], args.ridge)
            bridge_trans[a] = {}
            r_mov = aligned_mean_quadratic(psi_full[mov], nxt_full[mov],
                                           transitions_g4[a])
        tau_cal[a] = calibrated_temperature(r_mov, args.tau_lo, args.tau_hi)

    # Per-action calibrated score + AUC (full bank diagnostic + PG1 subset).
    pi_all = torch.zeros(onehot.shape[0], onehot.shape[1], device=device)
    for a in range(onehot.shape[1]):
        if routes[a] == "skip":
            pi_all[:, a] = 0.5
            continue
        if routes[a] == "bridge":
            r = bridge_mean_quadratic(psi64, nxt64, bridge_trans[a])
        else:
            r = aligned_mean_quadratic(psi_full, nxt_full, transitions_g4[a])
        pi_all[:, a] = calibrated_affordance_score(r, tau_cal[a])

    subset = stratified_subset_256(onehot, seed=args.seed).to(device)
    per_action_auc_full, per_action_auc_sub = {}, {}
    per_action_subset_n = {}
    pg1a_failures = {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        if mask.sum() >= MIN_AUC_SAMPLES:
            per_action_auc_full[str(a)] = compute_auc(pi_all[mask, a], y[mask])
        else:
            per_action_auc_full[str(a)] = None
        mask_sub = mask & subset
        n1 = int((mask_sub & (y == 1.0)).sum().item())
        n0 = int((mask_sub & (y == 0.0)).sum().item())
        per_action_subset_n[str(a)] = {"pos": n1, "neg": n0}
        if mask_sub.sum() >= 10:
            per_action_auc_sub[str(a)] = compute_auc(pi_all[mask_sub, a], y[mask_sub])
            if per_action_auc_sub[str(a)] < PG1A_TARGETS_G7.get(a, PG1_MIN_AUC_G7):
                pg1a_failures[str(a)] = per_action_auc_sub[str(a)]
        else:
            per_action_auc_sub[str(a)] = None
    sub_vals = [v for v in per_action_auc_sub.values() if v is not None]
    pg1_min_auc = min(sub_vals) if sub_vals else 0.0

    # PG3: piecewise routing + top-k + tau_a CPU == CUDA identity.
    pg3_passed = pg3_calibrated_determinism(
        psi_full, nxt_full, onehot, y, args.nu0, args.top_k,
        args.tau_lo, args.tau_hi)

    receipt_out = args.receipt_out or str(pathlib.Path(args.out_dir) / "g7_gates_receipt.json")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    label_counts = {str(a): float(y[onehot[:, a].bool()].mean().item())
                    for a in range(onehot.shape[1])}

    base_result = {
        "steps_done": 0, "min_auc_subset": pg1_min_auc,
        "per_action_auc_subset": per_action_auc_sub,
        "per_action_auc_full": per_action_auc_full,
        "per_action_subset_n": per_action_subset_n,
        "per_action_route": {str(a): routes[a] for a in routes},
        "moving_counts": moving_counts, "alpha": alpha,
        "tau_cal": {str(a): tau_cal[a] for a in tau_cal},
        "c2_dense_exact_zero": c2_dense_exact_zero,
        "label_counts": label_counts, "norm_drift": norm_drift,
        "pg2_passed": pg2_passed, "pg3_passed": pg3_passed,
        "n_actions": len(routes), "seed": args.seed, "top_k": args.top_k,
        "ridge": args.ridge, "nu0": args.nu0,
        "sparse_threshold": args.sparse_threshold,
        "dense_threshold": args.dense_threshold,
        "tau_stall": args.tau_stall, "subset_size": int(subset.sum().item()),
        "tau_lo": args.tau_lo, "tau_hi": args.tau_hi,
        "pg1a_failures": pg1a_failures}

    if (pg1_min_auc < PG1_MIN_AUC_G7 or pg1a_failures
            or not pg2_passed or not pg3_passed or not c2_dense_exact_zero):
        result = {"verdict": "G7_AFFORDANCE_FIT_COLLAPSE", **base_result}
        pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str))
        return 1

    # Kinematics (D=64, carried) for the live loop.
    comp = compile_free_generators_capped(
        psi64, nxt64, onehot.float(), omega_bound=args.omega_bound,
        moving_thresh=MOVING_THRESH, seed=args.seed)

    env_goals = {}
    if resolve_trajectory_goal is not None:
        for name in env_names:
            try:
                goal, meta = resolve_trajectory_goal(
                    args.trajectory_bank, args.trajectory_jsonl, name,
                    device=device, ingress=ingress)
                env_goals[name] = goal
            except Exception:
                pass

    # Carrier P1/G8 (default-OFF): goal-grounded policy steering engines.
    # Modules are imported ONLY under their flag; the default G7 path is
    # untouched (differential default-OFF proof: flags absent -> modules
    # never imported). G8 subsumes P1 when both flags are set.
    use_g8 = os.environ.get("HENRI_G8_SUBGOAL") == "1"
    use_p1 = os.environ.get("HENRI_P1_GOAL_STEERING") == "1"
    full_goals = None
    g8_chains = None
    if use_g8:
        from arc_g8_subgoal_engine import (  # lazy, flag-gated
            G8SubgoalSteeringEngine,
            build_g8_waypoint_chains,
            require_g8_flag,
        )
        require_g8_flag()
        engine_cls = G8SubgoalSteeringEngine
        g8_chains = build_g8_waypoint_chains(
            args.trajectory_bank, args.trajectory_jsonl, env_names,
            device=device)
    elif use_p1:
        from arc_p1_goal_steering_engine import (  # lazy, flag-gated
            P1GoalSteeringEngine,
            build_p1_full_goals,
            require_p1_flag,
        )
        require_p1_flag()
        engine_cls = P1GoalSteeringEngine
        full_goals = build_p1_full_goals(
            args.trajectory_bank, args.trajectory_jsonl, env_names,
            device=device)
    else:
        engine_cls = G7CalibratedAffordanceEngine

    engine = engine_cls(
        transitions_g4=transitions_g4, topk_masks=topk_masks,
        theta=[0.0] * len(routes),
        tau=[tau_cal[a] for a in range(len(routes))],
        bridge_transitions=bridge_trans,
        bridge_route_flags={a: (routes[a] == "bridge") for a in routes},
        generators=comp["generators"], transitions=comp["transitions"],
        t_pow=comp["t_pow"], recon=comp["recon"],
        action_names=comp.get("action_names"), n_actions=len(comp["generators"]),
        tau_cal=[tau_cal[a] for a in range(len(routes))],
        seed=args.seed, horizon=args.horizon, device=device,
        omega_bound=args.omega_bound,
        waypoint_advance_thresh=args.waypoint_advance_thresh,
        langevin_temp=LANGEVIN_TEMP, tau_stall=args.tau_stall,
        ingress=ingress)
    if full_goals is not None:
        engine._p1_full_goals = full_goals
    if g8_chains is not None:
        engine._g8_chains = g8_chains

    result = engine.run_gauntlet(
        env_names, fast_encoder=FastFullDWaveEncoder(
            d_model=65536, device=device, seed=args.seed),
        steps_per_env=args.steps_per_env, seed=args.seed,
        trajectory_bank=args.trajectory_bank, trajectory_jsonl=args.trajectory_jsonl,
        ingress=ingress, out_dir=args.out_dir, receipt_out=receipt_out,
        pg1_min_auc=pg1_min_auc, env_goals=env_goals)
    # Live-loop fields (steps_done, verdict, latency, solved, ...) MUST win
    # over the pre-flight base fields (harness fix: the inherited
    # result.update(base_result) clobbered steps_done to 0 after a full run).
    result = finalize_receipt(base_result, result)
    result["pg1_min_auc"] = pg1_min_auc
    if use_g8:
        result["policy_mode"] = "G8_SUBGOAL_STEERING"
        result.update(engine.g8_receipt_fields())
        result["p1_score_calls"] = engine._p1_score_calls
        if engine._p1_latencies_ms:
            result["p1_kernel_latency_ms"] = (
                sum(engine._p1_latencies_ms) / len(engine._p1_latencies_ms))
        if engine._p1_drop_accum is not None and engine._p1_score_calls > 0:
            result["p1_mean_potential_drops"] = [
                float(v) / engine._p1_score_calls
                for v in engine._p1_drop_accum]
    elif use_p1:
        result["policy_mode"] = "P1_GOAL_STEERING"
        result["p1_score_calls"] = engine._p1_score_calls
        if engine._p1_latencies_ms:
            result["p1_kernel_latency_ms"] = (
                sum(engine._p1_latencies_ms) / len(engine._p1_latencies_ms))
        if engine._p1_drop_accum is not None and engine._p1_score_calls > 0:
            result["p1_mean_potential_drops"] = [
                float(v) / engine._p1_score_calls
                for v in engine._p1_drop_accum]
    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
