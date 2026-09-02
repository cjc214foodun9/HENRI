"""Carrier G5 — Empirical Bayes Shrinkage & Sample-Gated Dual-Subspace Affordance Engine.

Directive: Carrier_G5_Master_Directive___G4_Post-Mortem.md
(HENRI-DIR-2026-09-V3-CARRIER-G5-SHRINKAGE-DIRECTIVE, fc2dd03a..., 17,334 B, 315 lines).
Prereg: docs/spec/g5_shrunk_affordance_preregistration.md (8a6fd670..., sealed
#88e18da9). Branch feat/carrier-g5-shrunk-affordance. Seed 20260928.
Parent: G4_GATES_VERDICT #9cbf1ed1 (24th sealed falsification).

Mechanism (directive-mandated, reconciled with live code per prereg):
  Lever 1 — Empirical Bayes shrinkage of per-block displacement variance:
      s2_m(a) = Var_t(||Psi_{t+1}^(m) - Psi_t^(m)||)  (CENTERED; directive §1.1;
               the directive's own code block used uncentered mean-of-squares —
               prereg locks the centered form per §1.1).
      prior_m = (1/|A|) Sum_a s2_m(a)   (cross-action pooled spatial prior)
      alpha_a = nu0 / (nu0 + N_a * d),  nu0 = 64, d = 8  (monotone in N_a — C2)
      shrunken = (1 - alpha_a) * s2_m(a) + alpha_a * prior_m
      top-k = argmax_k shrunken over m in {1..8192}, k = 64 (PER-ACTION
              invariant support — directive §3.2 text; per-sample top-k rejected).
  Lever 2 — Sample-gated dual-subspace routing (PG3):
      N_moving(a) < 40  -> D=64 bridge arm (_bridge_to_d64_batch, VERIFIED
                          origin arc_f21_edmd_engine.py:79; G4's import from
                          arc_f21_1_vectorized_engine is a latent dead import).
      N_moving(a) >= 40 -> shrunken top-k arm (D=65,536).
      Fit per-block 8x8 ridge transitions on MOVING rows; score = mean quadratic
      residual over the SAME arm support (C1 homology within each arm).
  Pi_a = sigmoid((theta_a - r_a)/tau_a); theta/tau calibrated per arm from the
      action's moving/blocked residual distributions.
  PG1 (binding): min_action_auc >= 0.8800 on the N=128 action-stratified subset;
      PG1a per-action targets a0-a4 >= 0.9500, a5/a6 >= 0.8800. Full-bank AUC
      diagnostic. PG2: flat norm drift <= 1e-6. PG3: routing + top-k CPU == CUDA
      tensor identity.
Verdicts G5_*. Flag HENRI_G5_SHRUNK_AFFORDANCE=1 (default-OFF fail-closed).
W0 (WavePacketPathSearch wiring) is GATED on a PG1 pass — NOT wired here (C3).
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
    from arc_g4_aligned_engine import (
        B_FULL,
        BLK,
        MIN_AUC_SAMPLES,
        FastFullDWaveEncoder,
        G4AlignedEngine,
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

# VERIFIED bridge origin: arc_f21_edmd_engine.py:79 (G4's import from
# arc_f21_1_vectorized_engine is a latent dead import — f21_1 re-exports only
# PatchIngress from arc_f10_live_engine; G4 died at PG1 before exercising it).
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

FLAG = "HENRI_G5_SHRUNK_AFFORDANCE"
SEED = 20260928
TOP_K = 64
RIDGE_G5 = 1e-2
PG1_MIN_AUC = 0.8800
PG2_NORM_TOL = 1e-6
TAU_STALL_G5 = 0.90
N_SUBSET = 128
SUBSET_PER_ACTION = 18
SAMPLE_THRESHOLD = 40
NU0 = 64.0
BRIDGE_BLOCKS = 8

# PG1a per-action subset-AUC targets (directive §3.1).
PG1A_TARGETS = {0: 0.9500, 1: 0.9500, 2: 0.9500, 3: 0.9500,
                4: 0.9500, 5: 0.8800, 6: 0.8800}


def require_flag(flag_name: str = FLAG) -> None:
    if os.environ.get(flag_name) != "1":
        print(f"BLOCKED: {flag_name} not set (default-OFF)", file=os.sys.stderr)
        raise SystemExit(1)


def shrinkage_alpha(n_moving: float, nu0: float = NU0, d: int = BLK) -> float:
    """alpha_a = nu0 / (nu0 + N_a * d); monotone decreasing in N_a (C2)."""
    return float(nu0 / (nu0 + float(n_moving) * d))


def shrunken_variance(sample_var: torch.Tensor, prior: torch.Tensor,
                      alpha: float) -> torch.Tensor:
    """(1 - alpha) * s2 + alpha * prior — convex combination."""
    return (1.0 - alpha) * sample_var + alpha * prior


def compute_shrunken_topk_masks(psi_full: torch.Tensor, nxt_full: torch.Tensor,
                                onehot: torch.Tensor, y: torch.Tensor,
                                nu0: float = NU0, k: int = TOP_K,
                                min_mov: int = 5) -> dict:
    """Per-action shrunken top-k masks {a: [k] tensor}.

    s2_m(a) = centered Var_t over the action's MOVING rows; prior = pooled
    mean of s2 across actions with >= min_mov moving rows; alpha per action.
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
        alpha = shrinkage_alpha(n_mov, nu0)
        shrunk = shrunken_variance(var, prior, alpha)
        masks[a] = select_topk_blocks(shrunk, k)
    return masks


def bridge_fit_transitions(psi64: torch.Tensor, nxt64: torch.Tensor,
                           onehot: torch.Tensor, y: torch.Tensor,
                           ridge: float = RIDGE_G5,
                           bridge_blocks: int = BRIDGE_BLOCKS) -> dict:
    """Fit per-block 8x8 ridge transitions on the D=64 bridge subspace.

    psi64/nxt64: [N, 64] unit rows -> view [N, 8, 8]. Fits on MOVING rows per
    action; returns {a: {m: T_m}} over m in {0..7}.
    """
    n_actions = onehot.shape[1]
    topk = torch.arange(bridge_blocks)
    out = {}
    for a in range(n_actions):
        mask = onehot[:, a].bool()
        mov = mask & (y == 1.0)
        if int(mov.sum().item()) < 5:
            out[a] = {}
            continue
        out[a] = fit_aligned_transitions(
            psi64[mov].view(-1, bridge_blocks, BLK),
            nxt64[mov].view(-1, bridge_blocks, BLK), topk, ridge)
    return out


def bridge_mean_quadratic(psi64: torch.Tensor, nxt64: torch.Tensor,
                          transitions: dict,
                          bridge_blocks: int = BRIDGE_BLOCKS) -> torch.Tensor:
    """[N, 64] -> [N] mean quadratic residual over the 8 bridge blocks (C1)."""
    if not transitions:
        return torch.full((psi64.shape[0],), float("nan"), device=psi64.device)
    return aligned_mean_quadratic(
        psi64.view(-1, bridge_blocks, BLK),
        nxt64.view(-1, bridge_blocks, BLK), transitions)


def route_decision(n_moving: int, threshold: int = SAMPLE_THRESHOLD,
                   min_fit: int = 5) -> str:
    """'bridge' if 5 <= n_moving < 40; 'topk' if n_moving >= 40; else 'skip'."""
    if n_moving < min_fit:
        return "skip"
    return "bridge" if n_moving < threshold else "topk"


def pg3_routing_determinism(psi_full: torch.Tensor, nxt_full: torch.Tensor,
                            onehot: torch.Tensor, y: torch.Tensor,
                            nu0: float = NU0, k: int = TOP_K) -> bool:
    """PG3: shrunken top-k indices identical on CPU and CUDA (when present)."""
    cpu = compute_shrunken_topk_masks(psi_full.cpu(), nxt_full.cpu(),
                                      onehot.cpu(), y.cpu(), nu0, k)
    ref = compute_shrunken_topk_masks(psi_full, nxt_full, onehot, y, nu0, k)
    for a in ref:
        if not torch.equal(cpu[a], ref[a].cpu()):
            return False
    return True


class G5ShrunkAffordanceEngine(G4AlignedEngine):
    """Dual-subspace affordance engine: shrunken top-k + D=64 bridge routing."""

    def __init__(self, transitions_g4, topk_masks, theta, tau,
                 bridge_transitions, bridge_route_flags,
                 generators, transitions, t_pow, recon,
                 action_names=None, n_actions=7, seed=SEED,
                 horizon=DEFAULT_HORIZON, device="cuda",
                 omega_bound=OMEGA_BOUND, waypoints=None,
                 waypoint_advance_thresh=WAYPOINT_ADVANCE_THRESH,
                 langevin_temp=LANGEVIN_TEMP, tau_stall=TAU_STALL_G5,
                 ingress=None):
        super().__init__(
            transitions_g4=transitions_g4, topk_masks=topk_masks,
            theta=theta, tau=tau, generators=generators,
            transitions=transitions, t_pow=t_pow, recon=recon,
            action_names=action_names, n_actions=n_actions, seed=seed,
            horizon=horizon, device=device, omega_bound=omega_bound,
            waypoints=waypoints,
            waypoint_advance_thresh=waypoint_advance_thresh,
            langevin_temp=langevin_temp, tau_stall=tau_stall)
        self.bridge_transitions = bridge_transitions  # {a: {m: T_m}}
        self.bridge_route_flags = bridge_route_flags  # {a: bool}
        self.ingress = ingress

    def affordance_residuals(self, psi_full, action_idx, psi_full_next=None):
        """Route by sample support: bridge arm or shrunken top-k arm (C1)."""
        psi_full = psi_full.float().to(self.device)
        if psi_full.dim() == 2:
            psi_full = psi_full.unsqueeze(0)
        if self.bridge_route_flags.get(int(action_idx), False):
            if self.ingress is None:
                return None  # fail closed: bridge arm needs the D=64 projection
            if psi_full_next is None:
                pair = self.last_pair.get(int(action_idx))
                if pair is None:
                    return None
                _, psi_full_next = pair
            if not self.bridge_transitions.get(int(action_idx)):
                return None
            psi_full_next = psi_full_next.float().to(self.device)
            if psi_full_next.dim() == 2:
                psi_full_next = psi_full_next.unsqueeze(0)
            psi64 = _bridge_to_d64_batch(
                psi_full.reshape(psi_full.shape[0], -1),
                ingress=self.ingress, seed=self.seed)
            nxt64 = _bridge_to_d64_batch(
                psi_full_next.reshape(psi_full_next.shape[0], -1),
                ingress=self.ingress, seed=self.seed)
            return bridge_mean_quadratic(
                psi64, nxt64, self.bridge_transitions[int(action_idx)])
        return super().affordance_residuals(psi_full, action_idx, psi_full_next)

    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        if steps_done > 0 and updates == 0:
            return "G5_NO_AFFORDANCE_ENGAGEMENT"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "G5_GATE_G1_FAILED"
        if solved < G2_MIN_SOLVED:
            return "G5_GATE_G2_FAILED"
        if mean_delta_nu is not None and mean_delta_nu < G3_MIN_DELTA_NU:
            return "G5_GATE_G3_FAILED"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "G5_GATE_G4_FAILED"
        return "G5_ALIGNED_AFFORDANCE_VERIFIED"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Carrier G5 dual-subspace shrunk affordance gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--omega-bound", type=float, default=OMEGA_BOUND)
    ap.add_argument("--waypoint-advance-thresh", type=float,
                    default=WAYPOINT_ADVANCE_THRESH)
    ap.add_argument("--tau-stall", type=float, default=TAU_STALL_G5)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--ridge", type=float, default=RIDGE_G5)
    ap.add_argument("--nu0", type=float, default=NU0)
    ap.add_argument("--sample-threshold", type=int, default=SAMPLE_THRESHOLD)
    ap.add_argument("--trajectory-bank", required=True)
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--out-dir", default="/tmp/henri_g5_shrunk/")
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

    # Per-action moving counts and route decisions (PG3).
    moving_counts = {str(a): int((onehot[:, a].bool() & (y == 1.0)).sum().item())
                     for a in range(onehot.shape[1])}
    routes = {a: route_decision(moving_counts[str(a)], args.sample_threshold)
              for a in range(onehot.shape[1])}
    alpha = {str(a): shrinkage_alpha(moving_counts[str(a)], args.nu0)
             for a in range(onehot.shape[1])}

    # Lever 1: shrunken top-k masks (per-action invariant support).
    topk_masks = compute_shrunken_topk_masks(
        psi_full, nxt_full, onehot, y, args.nu0, args.top_k)

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
        if routes[a] == "skip":
            transitions_g4[a] = {}
            bridge_trans[a] = {}
            theta[a], tau[a] = 0.0, 1.0
            continue
        if routes[a] == "bridge":
            bridge_trans[a] = fit_aligned_transitions(
                psi64[mov].view(-1, BRIDGE_BLOCKS, BLK),
                nxt64[mov].view(-1, BRIDGE_BLOCKS, BLK),
                torch.arange(BRIDGE_BLOCKS), args.ridge)
            transitions_g4[a] = {}
            r_mov = bridge_mean_quadratic(psi64[mov], nxt64[mov], bridge_trans[a])
            r_blk = bridge_mean_quadratic(psi64[mask & (y == 0.0)],
                                          nxt64[mask & (y == 0.0)],
                                          bridge_trans[a])
        else:
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

    # PG3: routing + top-k CPU == CUDA identity.
    pg3_passed = pg3_routing_determinism(psi_full, nxt_full, onehot, y,
                                         args.nu0, args.top_k)

    receipt_out = args.receipt_out or str(pathlib.Path(args.out_dir) / "g5_gates_receipt.json")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    label_counts = {str(a): float(y[onehot[:, a].bool()].mean().item())
                    for a in range(onehot.shape[1])}

    base_result = {
        "steps_done": 0, "min_auc_subset": pg1_min_auc,
        "per_action_auc_subset": per_action_auc_sub,
        "per_action_auc_full": per_action_auc_full,
        "per_action_route": {str(a): routes[a] for a in routes},
        "moving_counts": moving_counts, "alpha": alpha,
        "label_counts": label_counts, "norm_drift": norm_drift,
        "pg2_passed": pg2_passed, "pg3_passed": pg3_passed,
        "n_actions": len(routes), "seed": args.seed, "top_k": args.top_k,
        "ridge": args.ridge, "nu0": args.nu0,
        "sample_threshold": args.sample_threshold,
        "tau_stall": args.tau_stall, "subset_size": int(subset.sum().item()),
        "pg1a_failures": pg1a_failures}

    if pg1_min_auc < PG1_MIN_AUC or pg1a_failures or not pg2_passed or not pg3_passed:
        result = {"verdict": "G5_AFFORDANCE_FIT_COLLAPSE", **base_result}
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

    engine = G5ShrunkAffordanceEngine(
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
