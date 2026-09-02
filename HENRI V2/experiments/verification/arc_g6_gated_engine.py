"""Carrier G6 — Piecewise Gated Subspace Selection & Pure-Support Preservation Engine.

Directive: Carrier_G6_Master_Directive___G5_Post-Mortem.md
(HENRI-DIR-2026-09-V3-CARRIER-G6-PIECEWISE-GATING, d5ec31cd..., 18,268 B, 341 lines).
Prereg: docs/spec/g6_gated_affordance_preregistration.md (7c371c0f..., sealed
#52782c82). Branch feat/carrier-g6-gated-affordance. Seed 20260929.
Parent: G5_GATES_VERDICT #eeed5b17 (25th sealed falsification; PG1 global 0.9231
PASS — first in chain; PG1a a2/a3 0.9231 < 0.9500 KILL; a4/a6 1.0 via bridge).

Mechanism (directive-mandated, reconciled with live code per prereg):
  Piecewise hard-gated support selection (Ranking Inversion Theorem, §1.1):
    Regime 1 (N_moving < 20):  D=64 bridge arm (_bridge_to_d64_batch, VERIFIED
                               origin arc_f21_edmd_engine.py) — sparse actions.
    Regime 2 (20..39):         D=65,536 shrunken top-k, alpha = nu0/(nu0+N*d).
    Regime 3 (N_moving >= 40): D=65,536 PURE empirical top-k, alpha EXACTLY 0.0
                               (zero prior contamination — C2; the G5 kill axis).
  Variance statistic: CENTERED per-block displacement variance
    (per_block_displacement_variance, G4/G5-verified). Directive's code-block
    uncentered mean-of-squares REJECTED (same class as G4 centered correction).
  Bridge arm: `_bridge_to_d64_batch` (PatchIngress; fail-closed when ingress is
    None). Directive's `psi_t[:, :8]` slice REJECTED (not the verified bridge).
  Label: flat norm-divided cosine on raw as-captured waves (G2 C13-locked),
    tau_stall 0.90. Bank pinned trajectories_production_run_f3v2.npz
    (9e3c01b4...). 12 envs x 150, seed 20260929, top-k 64, ridge 0.01.
  Per-arm C1 homology: fit functional == score functional within each arm.
  PG1 (binding): min_action_auc >= 0.8800 on the N=128 action-stratified subset;
    PG1a per-action targets a0-a4 >= 0.9500, a5/a6 >= 0.8800. PG2: flat norm
    drift <= 1e-6. PG3: piecewise routing + top-k CPU == CUDA tensor identity.
Verdicts G6_*. Flag HENRI_G6_GATED_AFFORDANCE=1 (default-OFF fail-closed).
W0 (WavePacketPathSearch wiring) is GATED on a FULL PG1 pass — NOT wired here (C3).
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
    from arc_g5_shrunk_engine import (
        BRIDGE_BLOCKS,
        NU0,
        PG1A_TARGETS,
        PG2_NORM_TOL,
        RIDGE_G5,
        SEED as SEED_G5,
        TOP_K,
        TAU_STALL_G5,
        G5ShrunkAffordanceEngine,
        bridge_fit_transitions,
        bridge_mean_quadratic,
        require_flag,
        shrunken_variance,
    )
except Exception:  # pragma: no cover - test isolation
    raise

try:
    from arc_g4_aligned_engine import (
        B_FULL,
        BLK,
        MIN_AUC_SAMPLES,
        FastFullDWaveEncoder,
        aligned_affordance_pi,
        aligned_mean_quadratic,
        calibrate_theta_tau,
        compute_auc,
        fit_aligned_transitions,
        per_block_displacement_variance,
        select_topk_blocks,
        stall_cosine_labels,
        stratified_subset,
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
        MOVING_THRESH,
        OMEGA_BOUND,
        WAYPOINT_ADVANCE_THRESH,
        _bridge_to_d64_single,
        advance_waypoint_index,
        compile_free_generators_capped,
        extract_waypoints,
        langevin_escape_tick,
    )
except Exception:  # pragma: no cover - test isolation
    _safe_levels = None
    DEFAULT_HORIZON = 8
    G1_LATENCY_MS = 2.0
    G2_MIN_SOLVED = 1
    G3_MIN_DELTA_NU = 0.0150
    G4_MAX_AFFORDANCE = 0.0500
    LANGEVIN_TEMP = 0.50
    MOVING_THRESH = 0.05
    OMEGA_BOUND = math.pi / 32.0
    WAYPOINT_ADVANCE_THRESH = 0.60
    _bridge_to_d64_single = advance_waypoint_index = None
    compile_free_generators_capped = extract_waypoints = langevin_escape_tick = None

# VERIFIED bridge origin: arc_f21_edmd_engine.py:79 (G4/G5-verified; G4's import
# from arc_f21_1_vectorized_engine is a latent dead import).
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

FLAG = "HENRI_G6_GATED_AFFORDANCE"
SEED = 20260929
TOP_K = 64
RIDGE_G6 = 1e-2
PG1_MIN_AUC = 0.8800
PG2_NORM_TOL = 1e-6
TAU_STALL_G6 = 0.90
N_SUBSET = 128
SUBSET_PER_ACTION = 18
SPARSE_THRESHOLD = 20
DENSE_THRESHOLD = 40
NU0 = 64.0
BRIDGE_BLOCKS = 8
PG1A_TARGETS = {0: 0.9500, 1: 0.9500, 2: 0.9500, 3: 0.9500,
                4: 0.9500, 5: 0.8800, 6: 0.8800}


def piecewise_alpha(n_moving: float, nu0: float = NU0, d: int = BLK,
                    sparse: int = SPARSE_THRESHOLD,
                    dense: int = DENSE_THRESHOLD) -> float:
    """Piecewise shrinkage coefficient (C2).

    Regime 3 (n_moving >= dense): EXACTLY 0.0 — zero prior contamination.
    Regime 2 (sparse <= n_moving < dense): nu0 / (nu0 + n_moving * d).
    Regime 1 (n_moving < sparse): 0.0 (bridge arm owns the action; alpha unused).
    """
    if n_moving >= dense:
        return 0.0
    if n_moving >= sparse:
        return float(nu0 / (nu0 + float(n_moving) * d))
    return 0.0


def piecewise_route_decision(n_moving: int, sparse: int = SPARSE_THRESHOLD,
                             dense: int = DENSE_THRESHOLD,
                             min_fit: int = 5) -> str:
    """'skip' | 'bridge' | 'topk_shrunk' | 'topk_pure' by moving-sample count."""
    if n_moving < min_fit:
        return "skip"
    if n_moving < sparse:
        return "bridge"
    if n_moving < dense:
        return "topk_shrunk"
    return "topk_pure"


def compute_piecewise_topk_masks(psi_full: torch.Tensor, nxt_full: torch.Tensor,
                                 onehot: torch.Tensor, y: torch.Tensor,
                                 nu0: float = NU0, k: int = TOP_K,
                                 min_mov: int = 5,
                                 sparse: int = SPARSE_THRESHOLD,
                                 dense: int = DENSE_THRESHOLD) -> dict:
    """Per-action piecewise top-k masks {a: [k] tensor}.

    Dense (>= dense): PURE empirical variance top-k (alpha == 0.0 exactly —
    the G5 anti-kill). Mid (sparse..dense): shrunken with the pooled prior.
    Sparse (< sparse): omitted (bridge arm owns the action).
    """
    n_actions = onehot.shape[1]
    vars_a: dict = {}
    for a in range(n_actions):
        mask = onehot[:, a].bool()
        mov = mask & (y == 1.0)
        if int(mov.sum().item()) >= min_mov:
            vars_a[a] = per_block_displacement_variance(
                psi_full[mov], nxt_full[mov])
    if not vars_a:
        return {}
    prior = torch.stack(list(vars_a.values()), dim=0).mean(dim=0)
    masks = {}
    for a, var in vars_a.items():
        mask = onehot[:, a].bool()
        n_mov = int((mask & (y == 1.0)).sum().item())
        alpha = piecewise_alpha(n_mov, nu0, sparse=sparse, dense=dense)
        if alpha == 0.0:
            effective = var  # pure empirical — zero prior perturbation (C2)
        else:
            effective = shrunken_variance(var, prior, alpha)
        masks[a] = select_topk_blocks(effective, k)
    return masks


def pg3_piecewise_determinism(psi_full: torch.Tensor, nxt_full: torch.Tensor,
                              onehot: torch.Tensor, y: torch.Tensor,
                              nu0: float = NU0, k: int = TOP_K) -> bool:
    """PG3: piecewise top-k indices identical on CPU and CUDA (when present)."""
    cpu = compute_piecewise_topk_masks(psi_full.cpu(), nxt_full.cpu(),
                                       onehot.cpu(), y.cpu(), nu0, k)
    ref = compute_piecewise_topk_masks(psi_full, nxt_full, onehot, y, nu0, k)
    for a in ref:
        if not torch.equal(cpu[a], ref[a].cpu()):
            return False
    return True


class G6GatedAffordanceEngine(G5ShrunkAffordanceEngine):
    """Piecewise-gated affordance engine (G5 subclass; routes encoded in masks)."""

    def __init__(self, transitions_g4, topk_masks, theta, tau,
                 bridge_transitions, bridge_route_flags,
                 generators, transitions, t_pow, recon,
                 action_names=None, n_actions=7, seed=SEED,
                 horizon=DEFAULT_HORIZON, device="cuda",
                 omega_bound=OMEGA_BOUND, waypoints=None,
                 waypoint_advance_thresh=WAYPOINT_ADVANCE_THRESH,
                 langevin_temp=LANGEVIN_TEMP, tau_stall=TAU_STALL_G6,
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
        # Piecewise thresholds recorded for telemetry (C2).
        self.sparse_threshold = SPARSE_THRESHOLD
        self.dense_threshold = DENSE_THRESHOLD

    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        if steps_done > 0 and updates == 0:
            return "G6_NO_AFFORDANCE_ENGAGEMENT"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "G6_GATE_G1_FAILED"
        if solved < G2_MIN_SOLVED:
            return "G6_GATE_G2_FAILED"
        if mean_delta_nu is not None and mean_delta_nu < G3_MIN_DELTA_NU:
            return "G6_GATE_G3_FAILED"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "G6_GATE_G4_FAILED"
        return "G6_GATED_AFFORDANCE_VERIFIED"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Carrier G6 piecewise gated affordance gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--omega-bound", type=float, default=OMEGA_BOUND)
    ap.add_argument("--waypoint-advance-thresh", type=float,
                    default=WAYPOINT_ADVANCE_THRESH)
    ap.add_argument("--tau-stall", type=float, default=TAU_STALL_G6)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--ridge", type=float, default=RIDGE_G6)
    ap.add_argument("--nu0", type=float, default=NU0)
    ap.add_argument("--sparse-threshold", type=int, default=SPARSE_THRESHOLD)
    ap.add_argument("--dense-threshold", type=int, default=DENSE_THRESHOLD)
    ap.add_argument("--trajectory-bank", required=True)
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--out-dir", default="/tmp/henri_g6_gated/")
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

    # Per-action moving counts and piecewise route decisions (PG3 / C2).
    moving_counts = {str(a): int((onehot[:, a].bool() & (y == 1.0)).sum().item())
                     for a in range(onehot.shape[1])}
    routes = {a: piecewise_route_decision(
        moving_counts[str(a)], args.sparse_threshold, args.dense_threshold)
        for a in range(onehot.shape[1])}
    alpha = {str(a): piecewise_alpha(
        moving_counts[str(a)], args.nu0, sparse=args.sparse_threshold,
        dense=args.dense_threshold)
        for a in range(onehot.shape[1])}

    # C2: dense actions must have EXACTLY zero shrinkage.
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

    # Per-arm ridge transitions + theta/tau on MOVING rows (C1 per arm).
    transitions_g4, bridge_trans = {}, {}
    theta, tau = {}, {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        mov = mask & (y == 1.0)
        route = routes[a]
        if route == "skip":
            transitions_g4[a] = {}
            bridge_trans[a] = {}
            theta[a], tau[a] = 0.0, 1.0
            continue
        if route == "bridge":
            bridge_trans[a] = fit_aligned_transitions(
                psi64[mov].view(-1, BRIDGE_BLOCKS, BLK),
                nxt64[mov].view(-1, BRIDGE_BLOCKS, BLK),
                torch.arange(BRIDGE_BLOCKS), args.ridge)
            transitions_g4[a] = {}
            r_mov = bridge_mean_quadratic(psi64[mov], nxt64[mov], bridge_trans[a])
            r_blk = bridge_mean_quadratic(psi64[mask & (y == 0.0)],
                                          nxt64[mask & (y == 0.0)],
                                          bridge_trans[a])
        else:  # topk_shrunk | topk_pure — same full-D arm, mask differs
            transitions_g4[a] = fit_aligned_transitions(
                psi_full[mov], nxt_full[mov], topk_masks[a], args.ridge)
            bridge_trans[a] = {}
            r_mov = aligned_mean_quadratic(psi_full[mov], nxt_full[mov],
                                           transitions_g4[a])
            r_blk = aligned_mean_quadratic(psi_full[mask & (y == 0.0)],
                                           nxt_full[mask & (y == 0.0)],
                                           transitions_g4[a])
        theta[a], tau[a] = calibrate_theta_tau(r_mov, r_blk)

    # Per-action affordance pi + AUC (full bank diagnostic + PG1 subset).
    pi_all = torch.zeros(onehot.shape[0], onehot.shape[1], device=device)
    for a in range(onehot.shape[1]):
        if routes[a] == "skip":
            pi_all[:, a] = 0.5
            continue
        if routes[a] == "bridge":
            r = bridge_mean_quadratic(psi64, nxt64, bridge_trans[a])
        else:
            r = aligned_mean_quadratic(psi_full, nxt_full, transitions_g4[a])
        pi_all[:, a] = aligned_affordance_pi(r, theta[a], tau[a])

    subset = stratified_subset(onehot, seed=args.seed).to(device)
    per_action_auc_full, per_action_auc_sub = {}, {}
    pg1a_failures = {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        if mask.sum() >= MIN_AUC_SAMPLES:
            per_action_auc_full[str(a)] = compute_auc(pi_all[mask, a], y[mask])
        else:
            per_action_auc_full[str(a)] = None
        mask_sub = mask & subset
        if mask_sub.sum() >= 10:
            per_action_auc_sub[str(a)] = compute_auc(pi_all[mask_sub, a], y[mask_sub])
            if per_action_auc_sub[str(a)] < PG1A_TARGETS.get(a, PG1_MIN_AUC):
                pg1a_failures[str(a)] = per_action_auc_sub[str(a)]
        else:
            per_action_auc_sub[str(a)] = None
    sub_vals = [v for v in per_action_auc_sub.values() if v is not None]
    pg1_min_auc = min(sub_vals) if sub_vals else 0.0

    # PG3: piecewise routing + top-k CPU == CUDA identity.
    pg3_passed = pg3_piecewise_determinism(psi_full, nxt_full, onehot, y,
                                           args.nu0, args.top_k)

    receipt_out = args.receipt_out or str(pathlib.Path(args.out_dir) / "g6_gates_receipt.json")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    label_counts = {str(a): float(y[onehot[:, a].bool()].mean().item())
                    for a in range(onehot.shape[1])}

    base_result = {
        "steps_done": 0, "min_auc_subset": pg1_min_auc,
        "per_action_auc_subset": per_action_auc_sub,
        "per_action_auc_full": per_action_auc_full,
        "per_action_route": {str(a): routes[a] for a in routes},
        "moving_counts": moving_counts, "alpha": alpha,
        "c2_dense_exact_zero": c2_dense_exact_zero,
        "label_counts": label_counts, "norm_drift": norm_drift,
        "pg2_passed": pg2_passed, "pg3_passed": pg3_passed,
        "n_actions": len(routes), "seed": args.seed, "top_k": args.top_k,
        "ridge": args.ridge, "nu0": args.nu0,
        "sparse_threshold": args.sparse_threshold,
        "dense_threshold": args.dense_threshold,
        "tau_stall": args.tau_stall, "subset_size": int(subset.sum().item()),
        "pg1a_failures": pg1a_failures}

    if (pg1_min_auc < PG1_MIN_AUC or pg1a_failures
            or not pg2_passed or not pg3_passed or not c2_dense_exact_zero):
        result = {"verdict": "G6_AFFORDANCE_FIT_COLLAPSE", **base_result}
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

    engine = G6GatedAffordanceEngine(
        transitions_g4=transitions_g4, topk_masks=topk_masks,
        theta=[theta[a] for a in range(len(routes))],
        tau=[tau[a] for a in range(len(routes))],
        bridge_transitions=bridge_trans,
        bridge_route_flags={a: (routes[a] == "bridge") for a in routes},
        generators=comp["generators"], transitions=comp["transitions"],
        t_pow=comp["t_pow"], recon=comp["recon"],
        action_names=comp.get("action_names"), n_actions=len(comp["generators"]),
        seed=args.seed, horizon=args.horizon, device=device,
        omega_bound=args.omega_bound,
        waypoint_advance_thresh=args.waypoint_advance_thresh,
        langevin_temp=LANGEVIN_TEMP, tau_stall=args.tau_stall,
        ingress=ingress)

    result = engine.run_gauntlet(
        env_names, fast_encoder=FastFullDWaveEncoder(
            d_model=65536, device=device, seed=args.seed),
        steps_per_env=args.steps_per_env, seed=args.seed,
        trajectory_bank=args.trajectory_bank, trajectory_jsonl=args.trajectory_jsonl,
        ingress=ingress, out_dir=args.out_dir, receipt_out=receipt_out,
        pg1_min_auc=pg1_min_auc, env_goals=env_goals)
    result.update(base_result)
    result["pg1_min_auc"] = pg1_min_auc
    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
