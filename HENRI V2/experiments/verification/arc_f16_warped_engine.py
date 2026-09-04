"""Carrier F16 — Active Lie Generator Warping & Adaptive Affordance Re-Ranking Engine.

Directive: HENRI-DIR-2026-08-F15-POSTMORTEM-ADAPTIVE-RERANKING-ORDER
(c5d14170b2e33d342409b5b55e873a2692afb800840d77ce935abbf3bb7d703c, 21,716 B, 249 lines)

F15 post-mortem RATIFIED (F15_GATE_G2_FAILED #0fb9c7be @ ledger 1,098): static Lie
generators scored against a Slerp waypoint are rank-inert (alignment 0.9909 ->
0.9918, G3 -0.0022, G4 0.9607).

Carrier F16 replaces static scoring with goal-conditioned Hamiltonian warping:
  Tier 1  Omega_goal(t) = Psi_goal Psi_t^T - Psi_t Psi_goal^T   in so(D) (skew)
  Tier 2  D~_a = D_a + alpha_steer * Omega_goal(t);  Psi_hat = exp(D~_a) Psi_t
  Tier 3  vectorized K=8 beam, goal-direct objective:
          J(a_1:8) = |<Psi_hat_{t+8}, Psi_goal>| - beta_Sagnac * sum_k Delta_damped(k)
          Delta_damped(k) = Delta_Sagnac(k) + gamma_damp * ||Psi_{t+k} - Psi_t||^2
  Tier 4  Hebbian goal-valence creep retained from F15 (proven rank-inert there;
          any steering effect is attributable to generator warping, D4).

Fail-closed: no bank -> F16_BLOCKED_NO_TRAJECTORY_BANK (zero steps).
Flag: HENRI_F16_WARPED=1 (or --force-enabled for tests).
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_f15_trajectory_engine import (
    DEFAULT_ENVS,
    LATENCY_BUDGET_MS,
    MIN_TRAJECTORY_DEPTH,
    _bridge_to_d64,
    _safe_levels,
    abs_cos,
    load_environment_indices,
    pg1_pass,
    resolve_trajectory_goal,
)
from arc_f10_live_engine import PatchIngress, SinglePassHorizon, sagnac_delta, _to_flat
from arc_f11_plasticity_engine import ActionPrototypeMemory, compute_valence

SAGNAC_TAU_F16 = 0.050       # G4
G3_MIN_DNU = 0.0200          # G3
MAX_INITIAL_OVERLAP = 0.90   # PG1
DEFAULT_ALPHA_STEER = 0.35
DEFAULT_GAMMA_DAMP = 0.10
DEFAULT_BETA_SAGNAC = 0.05
DEFAULT_HORIZON = 8
DEFAULT_BEAM = 8
DEFAULT_SEED = 20260915
DEFAULT_ETA_FAST = 0.05


def require_f16_enabled(_force_enabled=False):
    if not (_force_enabled or os.environ.get("HENRI_F16_WARPED") == "1"):
        raise RuntimeError("F16 warped engine disabled: set HENRI_F16_WARPED=1")


def omega_goal(psi_t, psi_goal):
    """Tier 1: anti-symmetric goal projection tensor, so(D)."""
    x = psi_t.reshape(-1).float()
    g = psi_goal.reshape(-1).float()
    return torch.outer(g, x) - torch.outer(x, g)


class WarpedLieEngine(nn.Module):
    """Goal-conditioned Hamiltonian Lie steering engine (Tiers 1-4)."""

    def __init__(self, D=64, n_actions=8, seed=0, horizon=DEFAULT_HORIZON,
                 beam=DEFAULT_BEAM, alpha_steer=DEFAULT_ALPHA_STEER,
                 gamma_damp=DEFAULT_GAMMA_DAMP, beta_sagnac=DEFAULT_BETA_SAGNAC,
                 eta_fast=DEFAULT_ETA_FAST):
        super().__init__()
        self.D = D
        self.n_actions = n_actions
        self.horizon = int(horizon)
        self.beam = int(beam)
        self.alpha_steer = float(alpha_steer)
        self.gamma_damp = float(gamma_damp)
        self.beta_sagnac = float(beta_sagnac)
        self.eta_fast = eta_fast
        g = torch.Generator().manual_seed(seed + 20)
        skew = torch.randn(n_actions, D, D, generator=g) * 0.1
        skew = skew - skew.transpose(-1, -2)
        self.register_buffer("D_a", skew)  # static base generators (skew, so(D))
        self.memory = ActionPrototypeMemory(
            n_actions=n_actions, D=D, eta_fast=eta_fast, seed=seed)

    def omega(self, psi_t, psi_goal):
        return omega_goal(psi_t, psi_goal)

    def warped_ops(self, candidates, omega, alpha_steer=None):
        """Tier 2: exp(D_a + alpha_steer * Omega) per candidate action."""
        cand = torch.as_tensor(list(candidates), dtype=torch.long, device=self.D_a.device)
        w = omega.to(self.D_a.device)
        a_steer = self.alpha_steer if alpha_steer is None else float(alpha_steer)
        return torch.linalg.matrix_exp(self.D_a[cand] + a_steer * w)

    def score_action(self, psi, goal, omega, a, horizon=1, beta=None, gamma=None,
                     alpha_steer=None):
        """Single-action objective J(a) (diagnostic / rank-break contract)."""
        beta = self.beta_sagnac if beta is None else float(beta)
        gamma = self.gamma_damp if gamma is None else float(gamma)
        goal_v = F.normalize(goal.reshape(-1).float().to(self.D_a.device), p=2, dim=-1)
        psi0 = F.normalize(psi.reshape(-1).float().to(self.D_a.device), p=2, dim=-1)
        ops = self.warped_ops([a], omega, alpha_steer=alpha_steer)
        state = psi0
        ssum = 0.0
        for _ in range(int(horizon)):
            nxt = F.normalize(ops[0] @ state, p=2, dim=-1)
            raw = float((nxt @ goal_v).abs().clamp(0.0, 1.0).item())
            sag = max(0.0, min(2.0, 1.0 - raw))
            disp = float((nxt - psi0).pow(2).sum().clamp_min(0.0).item())
            ssum += sag + gamma * disp
            state = nxt
        return raw - beta * ssum

    def beam_search(self, psi, goal, omega, candidates=None, horizon=None, beam=None,
                    beta=None, gamma=None):
        """Tier 3: vectorized K-step beam, goal-direct objective, damped Sagnac."""
        horizon = self.horizon if horizon is None else int(horizon)
        beam = self.beam if beam is None else int(beam)
        beta = self.beta_sagnac if beta is None else float(beta)
        gamma = self.gamma_damp if gamma is None else float(gamma)
        if not candidates:
            candidates = list(range(self.n_actions))
        cand = torch.as_tensor(list(candidates), dtype=torch.long, device=self.D_a.device)
        goal_v = F.normalize(goal.reshape(-1).float().to(self.D_a.device), p=2, dim=-1)
        psi0 = F.normalize(psi.reshape(-1).float().to(self.D_a.device), p=2, dim=-1)
        states = psi0.unsqueeze(0)
        ops = self.warped_ops(candidates, omega)  # [A, D, D]
        acts = torch.full((1, 0), -1, dtype=torch.long, device=self.D_a.device)
        ssum = torch.zeros(1, device=self.D_a.device)
        for _ in range(horizon):
            nxt = torch.einsum("bd,axd->bax", states, ops)
            nxt = F.normalize(nxt, p=2, dim=-1)
            raw = nxt @ goal_v
            align = raw.abs().clamp(0.0, 1.0)
            sag = (1.0 - raw).clamp(0.0, 2.0)
            disp = (nxt - psi0).pow(2).sum(-1).clamp_min(0.0)
            damp = sag + gamma * disp
            jp = align - beta * (ssum[:, None] + damp)
            flat = jp.reshape(-1)
            k = min(beam, flat.numel())
            top = torch.topk(flat, k)
            idx = top.indices
            states = nxt.reshape(-1, nxt.shape[-1])[idx]
            b_idx = idx // nxt.shape[1]
            a_idx = idx % nxt.shape[1]
            acts = torch.cat([acts[b_idx], cand[a_idx].unsqueeze(1)], dim=1)
            ssum = ssum[b_idx] + damp.reshape(-1)[idx]
        raw_f = states @ goal_v
        j = raw_f.abs().clamp(0.0, 1.0) - beta * ssum
        best = int(torch.argmax(j))
        action = int(acts[best, 0])
        return action, float(j[best].detach())

    def valence_delta_goal(self, psi_next, psi_t, goal):
        c_next = float(abs_cos(psi_next, goal).item())
        c_t = float(abs_cos(psi_t, goal).item())
        return c_next - c_t

    def creep(self, action, delta_nu, psi):
        if delta_nu == 0.0:
            return
        self.memory.creep(action, delta_nu, psi)


def _verdict(gates, reason=None):
    if reason and reason.startswith("BLOCKED_NO_TRAJECTORY_BANK"):
        return "F16_BLOCKED_NO_TRAJECTORY_BANK"
    if reason and reason.startswith("DEGENERATE_GOAL"):
        return "F16_PREFLIGHT_DEGENERATE_GOAL"
    if reason is not None and reason.startswith("live_error"):
        return "F16_LIVE_ENGINE_BLOCKED"
    if not gates.get("G1"):
        return "F16_LIVE_ENGINE_BLOCKED"
    if all(gates.get(k) for k in ("G1", "G2", "G3", "G4")):
        return "F16_LIVE_LOOP_VERIFIED"
    for name in ("G2", "G3", "G4"):
        if not gates.get(name):
            return "F16_GATE_{}_FAILED".format(name)
    return "F16_INDETERMINATE"


def write_receipt(path, gates, telemetry, meta):
    data = {
        "schema": "f16-warped-engine.v1",
        "gates": gates,
        "telemetry": telemetry,
        "verdict": _verdict(gates, telemetry.get("reason")),
        "meta": meta,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def run_gauntlet(env_names=None, steps_per_env=150, seed=DEFAULT_SEED,
                 horizon=DEFAULT_HORIZON, beam=DEFAULT_BEAM,
                 alpha_steer=DEFAULT_ALPHA_STEER, gamma_damp=DEFAULT_GAMMA_DAMP,
                 beta_sagnac=DEFAULT_BETA_SAGNAC, eta_fast=DEFAULT_ETA_FAST,
                 max_initial_overlap=MAX_INITIAL_OVERLAP,
                 trajectory_bank=None, trajectory_jsonl=None,
                 out_dir=None, receipt_out=None, _force_enabled=False):
    """F16 live gauntlet (directive command)."""
    require_f16_enabled(_force_enabled=_force_enabled)
    env_names = list(env_names) if env_names else list(DEFAULT_ENVS)
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(receipt_out) if receipt_out else out_dir / "f16_gates_receipt.json"

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    telemetry = {
        "envs": list(env_names),
        "steps": 0,
        "resets": 0,
        "mean_latency_ms": None,
        "sagnac_raw_mean": None,
        "sagnac_damped_mean": None,
        "omega_norm_mean": None,
        "progress": 0.0,
        "solved": 0,
        "sum_delta_nu": 0.0,
        "mean_delta_nu_goal": None,
        "creeps": 0,
        "goal_align_first": None,
        "goal_align_last": None,
        "pg1_overlaps": None,
        "goal_source": "trajectory-bank-v2" if trajectory_bank else "NONE",
        "reason": None,
    }

    def fail_closed(reason):
        telemetry["reason"] = reason
        gates = {"PG1": False, "G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": horizon, "beam": beam,
                  "alpha_steer": alpha_steer, "gamma_damp": gamma_damp,
                  "beta_sagnac": beta_sagnac, "p": 32, "device": device,
                  "eta_fast": eta_fast, "max_initial_overlap": max_initial_overlap,
                  "trajectory_bank": str(trajectory_bank) if trajectory_bank else None,
                  "trajectory_jsonl": str(trajectory_jsonl) if trajectory_jsonl else None},
        )

    # --- Pre-flight: no bank -> no goal source -> fail closed -------------
    if not trajectory_bank or not trajectory_jsonl:
        return fail_closed("BLOCKED_NO_TRAJECTORY_BANK: no --trajectory-bank/--trajectory-jsonl; "
                           "goal source unavailable")

    # --- Construct substrate (canonical imports; C14 reachability guard) ---
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=seed).to(device)
    horizon_inst = SinglePassHorizon(d=64, rank=32, K=8, num_blocks=8, seed=seed).to(device)
    engine = WarpedLieEngine(
        D=64, n_actions=8, seed=seed, horizon=horizon, beam=beam,
        alpha_steer=alpha_steer, gamma_damp=gamma_damp, beta_sagnac=beta_sagnac,
        eta_fast=eta_fast,
    ).to(device)

    try:
        from arc_agi import Arcade
        arcade = Arcade()
    except Exception as exc:
        return fail_closed("arcade_unavailable: {!r}".format(exc))

    # --- Resolve trajectory goals per env (PG1 pre-flight) ------------------
    goals = {}
    pg1_overlaps = {}
    for name in env_names:
        try:
            goal, meta = resolve_trajectory_goal(
                trajectory_bank, trajectory_jsonl, name, device=device, ingress=ingress)
        except Exception as exc:
            return fail_closed("trajectory_goal_error: {}: {!r}".format(name, exc))
        goals[name] = goal
        pg1_overlaps[name] = None  # filled after psi0 observed

    latencies, sagnacs_raw, sagnacs_damped, dnus = [], [], [], []
    omega_norms = []
    steps_done = 0
    sum_delta_nu = 0.0
    progress = 0.0
    solved = 0
    creeps = 0
    align_first = None
    align_last = None
    try:
        for name in env_names:
            game = arcade.make(name)
            if game is None:
                return fail_closed("arcade_make_returned_none: {!r}".format(name))
            obs = game.reset()
            if obs is None or not getattr(obs, "frame", None):
                return fail_closed("null_initial_frame: {!r}".format(name))
            prev_levels = _safe_levels(obs)

            frame0 = obs.frame[0]
            raw0 = torch.as_tensor(_to_flat(frame0), dtype=torch.float32, device=device)
            if raw0.numel() < 4096:
                raw0 = F.pad(raw0, (0, 4096 - raw0.numel()))
            else:
                raw0 = raw0[:4096]
            psi0_b = ingress(raw0.unsqueeze(0))
            psi0 = psi0_b[0].detach()

            goal = goals[name]
            ok, overlap = pg1_pass(psi0, goal, max_overlap=max_initial_overlap)
            pg1_overlaps[name] = float(overlap)
            if not ok:
                telemetry["steps"] = steps_done
                telemetry["reason"] = "DEGENERATE_GOAL: env {} overlap {:.4f} > {}".format(
                    name, overlap, max_initial_overlap)
                telemetry["pg1_overlaps"] = pg1_overlaps
                gates = {"PG1": False, "G1": False, "G2": False, "G3": False, "G4": False}
                return write_receipt(
                    receipt_path, gates, telemetry,
                    meta={"seed": seed, "K": horizon, "beam": beam,
                          "alpha_steer": alpha_steer, "gamma_damp": gamma_damp,
                          "beta_sagnac": beta_sagnac, "p": 32, "device": device,
                          "eta_fast": eta_fast, "max_initial_overlap": max_initial_overlap,
                          "trajectory_bank": str(trajectory_bank),
                          "trajectory_jsonl": str(trajectory_jsonl)},
                )

            for _ in range(steps_per_env):
                frame = obs.frame[0]
                raw = torch.as_tensor(_to_flat(frame), dtype=torch.float32, device=device)
                if raw.numel() < 4096:
                    raw = F.pad(raw, (0, 4096 - raw.numel()))
                else:
                    raw = raw[:4096]
                t_start = time.perf_counter()
                psi_b = ingress(raw.unsqueeze(0))
                psi = psi_b[0]
                psi_s = psi.detach()

                # Tier 1 + Tier 2 + Tier 3 (goal-direct beam with warping)
                om = engine.omega(psi_s, goal)
                omega_norms.append(float(om.norm().item()))
                avail = list(getattr(obs, "available_actions", None) or [])
                candidates = [int(a) for a in avail] if avail else list(range(8))
                sel, _info = engine.beam_search(
                    psi_s, goal, om, candidates,
                    horizon=horizon, beam=beam, beta=beta_sagnac, gamma=gamma_damp)
                if align_first is None:
                    align_first = float(abs_cos(psi_s, goal).item())

                # G4 instrument: SinglePassHorizon roll[0,0] vs goal (D3:
                # damped Sagnac per directive §3.3; raw also reported)
                roll = horizon_inst(psi_b)
                raw_sag = float(sagnac_delta(roll[0, 0], goal).item())
                disp = float((roll[0, 0] - psi_s).pow(2).sum().clamp_min(0.0).item())
                damped_sag = raw_sag + gamma_damp * disp
                sagnacs_raw.append(raw_sag)
                sagnacs_damped.append(damped_sag)

                actions = list(game.action_space)
                action = actions[sel % max(1, len(actions))]
                obs = game.step(action)
                latencies.append((time.perf_counter() - t_start) * 1000.0)
                steps_done += 1

                terminal = obs is None or (
                    getattr(obs, "state", None) and obs.state.name == "GAME_OVER")
                was_reset = terminal
                cur_levels = _safe_levels(obs) if not terminal else prev_levels
                r_ext = compute_valence(prev_levels, cur_levels, was_reset)
                if terminal:
                    telemetry["resets"] += 1
                    obs = game.reset()
                    if obs is None or not getattr(obs, "frame", None):
                        break
                    prev_levels = _safe_levels(obs)
                    continue

                frame_next = obs.frame[0]
                raw_next = torch.as_tensor(_to_flat(frame_next), dtype=torch.float32, device=device)
                if raw_next.numel() < 4096:
                    raw_next = F.pad(raw_next, (0, 4096 - raw_next.numel()))
                else:
                    raw_next = raw_next[:4096]
                psi_next = ingress(raw_next.unsqueeze(0))[0].detach()
                dnu = engine.valence_delta_goal(psi_next, psi_s, goal)
                dnus.append(dnu)
                sum_delta_nu += dnu
                progress += r_ext
                if dnu > 0.0:
                    engine.creep(sel, dnu, psi_s)
                    creeps += 1
                if cur_levels > prev_levels:
                    solved += 1
                prev_levels = cur_levels
                align_last = float(abs_cos(psi_next, goal).item())
    except Exception as exc:
        telemetry["steps"] = steps_done
        telemetry["reason"] = "live_error: {!r}".format(exc)
        telemetry["pg1_overlaps"] = pg1_overlaps
        gates = {"PG1": True, "G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": horizon, "beam": beam,
                  "alpha_steer": alpha_steer, "gamma_damp": gamma_damp,
                  "beta_sagnac": beta_sagnac, "p": 32, "device": device,
                  "eta_fast": eta_fast, "max_initial_overlap": max_initial_overlap,
                  "trajectory_bank": str(trajectory_bank),
                  "trajectory_jsonl": str(trajectory_jsonl)},
        )

    mean_latency = float(np.mean(latencies)) if latencies else None
    sagnac_raw_mean = float(np.mean(sagnacs_raw)) if sagnacs_raw else None
    sagnac_damped_mean = float(np.mean(sagnacs_damped)) if sagnacs_damped else None
    omega_norm_mean = float(np.mean(omega_norms)) if omega_norms else None
    mean_dnu = float(np.mean(dnus)) if dnus else None
    telemetry.update({
        "steps": steps_done,
        "mean_latency_ms": mean_latency,
        "sagnac_raw_mean": sagnac_raw_mean,
        "sagnac_damped_mean": sagnac_damped_mean,
        "omega_norm_mean": omega_norm_mean,
        "progress": float(progress),
        "solved": int(solved),
        "sum_delta_nu": float(sum_delta_nu),
        "mean_delta_nu_goal": mean_dnu,
        "creeps": int(creeps),
        "goal_align_first": align_first,
        "goal_align_last": align_last,
        "pg1_overlaps": pg1_overlaps,
    })

    g1 = steps_done >= steps_per_env * len(env_names) and mean_latency is not None and mean_latency <= LATENCY_BUDGET_MS
    g2 = solved > 0
    g3 = mean_dnu is not None and mean_dnu >= G3_MIN_DNU
    g4 = sagnac_damped_mean is not None and sagnac_damped_mean <= SAGNAC_TAU_F16
    gates = {"PG1": True, "G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    return write_receipt(
        receipt_path, gates, telemetry,
        meta={"seed": seed, "K": horizon, "beam": beam,
              "alpha_steer": alpha_steer, "gamma_damp": gamma_damp,
              "beta_sagnac": beta_sagnac, "p": 32, "device": device,
              "eta_fast": eta_fast, "max_initial_overlap": max_initial_overlap,
              "trajectory_bank": str(trajectory_bank),
              "trajectory_jsonl": str(trajectory_jsonl)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=None, help="comma-separated env names (default: F10 cohort)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    ap.add_argument("--alpha-steer", type=float, default=DEFAULT_ALPHA_STEER)
    ap.add_argument("--gamma-damp", type=float, default=DEFAULT_GAMMA_DAMP)
    ap.add_argument("--beta-sagnac", type=float, default=DEFAULT_BETA_SAGNAC)
    ap.add_argument("--eta-fast", type=float, default=DEFAULT_ETA_FAST)
    ap.add_argument("--max-initial-overlap", type=float, default=MAX_INITIAL_OVERLAP)
    ap.add_argument("--trajectory-bank", default=None,
                    help="verified trajectory bank npz path (required)")
    ap.add_argument("--trajectory-jsonl", default=None,
                    help="trajectory metadata jsonl path (required)")
    ap.add_argument("--out-dir", default="/tmp/henri_f16_warped")
    ap.add_argument("--receipt-out", default=None)
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()] if args.envs else None
    receipt = run_gauntlet(
        env_names=envs,
        steps_per_env=args.steps_per_env,
        seed=args.seed,
        horizon=args.horizon,
        beam=args.beam,
        alpha_steer=args.alpha_steer,
        gamma_damp=args.gamma_damp,
        beta_sagnac=args.beta_sagnac,
        eta_fast=args.eta_fast,
        max_initial_overlap=args.max_initial_overlap,
        trajectory_bank=args.trajectory_bank,
        trajectory_jsonl=args.trajectory_jsonl,
        out_dir=args.out_dir,
        receipt_out=args.receipt_out,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
