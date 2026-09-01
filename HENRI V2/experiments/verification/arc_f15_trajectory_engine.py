"""Carrier F15 — Interactive Trajectory Goal Steering Engine.

Directive: HENRI-DIR-2026-08-F15-TRAJECTORY-GOAL-STEERING-ORDER
(e2e3d1efa22e26909b504aee173bf1d6b3891311bf4e2a219e8ec9ec02146808)

Trajectory-grounded goal steering on the F3 v2 verified trajectory bank:
  Tier 1  terminal-state goal extraction (per env, JSONL env rows)
  PG1     |<psi0, goal>| <= 0.90 (all 12 envs; PRE-FLIGHT KILL on violation)
  Tier 2  Slerp waypoint (tau = 0.25)
  Tier 3  vectorized K=8 einsum beam search, commit first action
  Tier 4  Hebbian valence creep (eta_fast = 0.05, zero-valence guard)

Fail-closed: no trajectory bank -> F15_BLOCKED_NO_TRAJECTORY_BANK (zero steps).
Flag: HENRI_F15_TRAJECTORY=1 (or --force-enabled for tests).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_f10_live_engine import PatchIngress, SinglePassHorizon, sagnac_delta, _to_flat
from arc_f11_plasticity_engine import ActionPrototypeMemory, compute_valence
LATENCY_BUDGET_MS = 5.0      # G1
SAGNAC_TAU_F15 = 0.050       # G4
G3_MIN_DNU = 0.0200          # G3
MAX_INITIAL_OVERLAP = 0.90   # PG1
DEFAULT_TAU = 0.25
DEFAULT_HORIZON = 8
DEFAULT_ALPHA = 0.05
DEFAULT_BEAM = 8
DEFAULT_ENVS = [
    "ar25-0c556536", "sc25-635fd71a", "tr87-cd924810", "cd82-fb555c5d",
    "lp85-305b61c3", "wa30-ee6fef47", "ft09-0d8bbf25", "g50t-5849a774",
    "sk48-d8078629", "bp35-0a0ad940", "ka59-38d34dbb", "sb26-7fbdac44",
]  # F10 receipt cohort (12 envs)

MIN_TRAJECTORY_DEPTH = 30  # directive: assert len(env_indices) >= 30


def require_f15_enabled(_force_enabled=False):
    if not (_force_enabled or os.environ.get("HENRI_F15_TRAJECTORY") == "1"):
        raise RuntimeError("F15 trajectory engine disabled: set HENRI_F15_TRAJECTORY=1")


def abs_cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).clamp(0.0, 1.0).squeeze(0)


def slerp(a, b, tau):
    """Spherical linear interpolation on S^{D-1} (signed geodesic, F14 D2)."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    cos_theta = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).clamp(-1.0, 1.0).squeeze(0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    if float(sin_theta.abs()) < 1e-8:
        w = (1.0 - tau) * a + tau * b
        n = torch.linalg.vector_norm(w).clamp_min(1e-8)
        return (w / n).reshape(-1)
    w = (torch.sin((1.0 - tau) * theta) / sin_theta) * a + (
        torch.sin(tau * theta) / sin_theta) * b
    return F.normalize(w, p=2, dim=-1)


def pg1_pass(psi0, goal, max_overlap=MAX_INITIAL_OVERLAP):
    overlap = float(abs_cos(psi0, goal).item())
    return overlap <= max_overlap, overlap


def _bridge_to_d64(wave_65536, device):
    """Deterministic D=65,536 -> D=64 bridge (deviation D1, amended).

    Block-mean pool [65536] -> [4096], unit-normalize, then scale by K=64.
    K matches the raw-frame norm band observed by the F10 PatchIngress
    substrate (sparse grids, norms ~50-100): a unit-norm 4096-d vector has
    per-component values ~0.016, below the MLP's bias floor, so distinct
    inputs collapse to a near-constant output (OBSERVED: overlap 0.9924
    with K=1). K=64 restores per-component ~1.0, the raw-frame operating
    point. Pure averaging + normalize + scale: no learnable params,
    deterministic.
    """
    w = wave_65536.reshape(-1).float().to(device)
    if w.numel() < 65536:
        w = F.pad(w, (0, 65536 - w.numel()))
    else:
        w = w[:65536]
    pooled = w.view(16, 4096).mean(dim=0)  # [4096]
    pooled = F.normalize(pooled, p=2, dim=-1) * 64.0  # scale-matched (K=64)
    return pooled


def load_environment_indices(jsonl_path, env_id):
    """Rows in JSONL whose `env` field == env_id (directive D2: field is `env`)."""
    indices = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("env") == env_id:
                indices.append(i)
    return indices


def resolve_trajectory_goal(bank_npz_path, bank_jsonl_path, env_id,
                            device="cpu", ingress=None):
    """Terminal-state goal extraction (directive §2.1).

    Returns (goal [64] unit tensor, meta dict).
    """
    data = np.load(bank_npz_path)
    if "psi" not in data.files:
        raise KeyError("bank npz missing key 'psi' (schema henri.arc-trajectory-bank.v1)")
    waves = torch.from_numpy(np.asarray(data["psi"])).float()  # [N, 65536]

    env_indices = load_environment_indices(bank_jsonl_path, env_id)
    if len(env_indices) < MIN_TRAJECTORY_DEPTH:
        raise ValueError(
            f"Insufficient trajectory depth for {env_id}: {len(env_indices)} < {MIN_TRAJECTORY_DEPTH}")
    terminal_idx = env_indices[-1]
    psi_goal_full = F.normalize(waves[terminal_idx], p=2, dim=-1)

    pooled = _bridge_to_d64(psi_goal_full, device)
    if ingress is not None:
        with torch.no_grad():
            psi_b = ingress(pooled.unsqueeze(0))
        goal = psi_b[0].detach().reshape(-1)
    else:
        goal = pooled
    goal = F.normalize(goal, p=2, dim=-1)

    meta = {
        "env_id": env_id,
        "rows": len(env_indices),
        "terminal_idx": int(terminal_idx),
        "goal_source": "trajectory-bank-v2",
        "bridge": "block-mean-16x4096" if ingress is not None else "none",
    }
    return goal, meta


class TrajectorySteeringEngine(nn.Module):
    """Trajectory-goal steering engine (Tiers 1-4)."""

    def __init__(self, D=64, n_actions=8, seed=0, horizon=DEFAULT_HORIZON,
                 beam=DEFAULT_BEAM, tau=DEFAULT_TAU, alpha=DEFAULT_ALPHA,
                 eta_fast=0.05):
        super().__init__()
        self.D = D
        self.n_actions = n_actions
        self.horizon = int(horizon)
        self.beam = int(beam)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.eta_fast = eta_fast
        g = torch.Generator().manual_seed(seed + 10)
        skew = torch.randn(n_actions, D, D, generator=g) * 0.1
        skew = skew - skew.transpose(-1, -2)
        self.register_buffer("expD", torch.linalg.matrix_exp(skew))  # [A, D, D]
        self.memory = ActionPrototypeMemory(
            n_actions=n_actions, D=D, eta_fast=eta_fast, seed=seed)

    def waypoint(self, psi_t, goal, tau=None):
        return slerp(psi_t, goal, self.tau if tau is None else tau)

    def rollout(self, psi, a):
        flat = psi.reshape(-1).float()
        return self.expD[a] @ flat

    def beam_search(self, psi, waypoint, candidates, horizon=None, beam=None, alpha=None):
        """Vectorized K-step beam search — F13 C11-verified semantics."""
        horizon = self.horizon if horizon is None else int(horizon)
        beam = self.beam if beam is None else int(beam)
        alpha = self.alpha if alpha is None else float(alpha)
        if not candidates:
            candidates = list(range(self.n_actions))
        cand = torch.as_tensor(list(candidates), dtype=torch.long, device=self.expD.device)
        wp = F.normalize(waypoint.reshape(-1).float().to(self.expD.device), p=2, dim=-1)
        states = F.normalize(psi.reshape(-1).float().to(self.expD.device), p=2, dim=-1).unsqueeze(0)
        acts = torch.full((1, 0), -1, dtype=torch.long, device=self.expD.device)
        ssum = torch.zeros(1, device=self.expD.device)
        for _ in range(horizon):
            ops = self.expD[cand]  # [A, D, D]
            nxt = torch.einsum("bd,axd->bax", states, ops)
            nxt = F.normalize(nxt, p=2, dim=-1)
            raw = nxt @ wp
            align = raw.abs().clamp(0.0, 1.0)
            sag = (1.0 - raw).clamp(0.0, 2.0)
            jp = align - alpha * (ssum[:, None] + sag)
            flat = jp.reshape(-1)
            k = min(beam, flat.numel())
            top = torch.topk(flat, k)
            idx = top.indices
            states = nxt.reshape(-1, nxt.shape[-1])[idx]
            b_idx = idx // nxt.shape[1]
            a_idx = idx % nxt.shape[1]
            acts = torch.cat([acts[b_idx], cand[a_idx].unsqueeze(1)], dim=1)
            ssum = ssum[b_idx] + sag.reshape(-1)[idx]
        raw_f = states @ wp
        j = raw_f.abs().clamp(0.0, 1.0) - alpha * ssum
        best = int(torch.argmax(j))
        action = int(acts[best, 0])
        return action, float(j[best].detach())

    def valence_delta(self, psi_next, psi_t, waypoint):
        c_next = float(abs_cos(psi_next, waypoint).item())
        c_t = float(abs_cos(psi_t, waypoint).item())
        return c_next - c_t

    def creep(self, action, delta_nu, psi):
        if delta_nu == 0.0:
            return
        self.memory.creep(action, delta_nu, psi)


def _verdict(gates, reason=None):
    if reason and reason.startswith("BLOCKED_NO_TRAJECTORY_BANK"):
        return "F15_BLOCKED_NO_TRAJECTORY_BANK"
    if reason and reason.startswith("DEGENERATE_GOAL"):
        return "F15_PREFLIGHT_DEGENERATE_GOAL"
    if reason is not None and reason.startswith("live_error"):
        return "F15_LIVE_ENGINE_BLOCKED"
    if not gates.get("G1"):
        return "F15_LIVE_ENGINE_BLOCKED"
    if all(gates.get(k) for k in ("G1", "G2", "G3", "G4")):
        return "F15_LIVE_LOOP_VERIFIED"
    for name in ("G2", "G3", "G4"):
        if not gates.get(name):
            return "F15_GATE_{}_FAILED".format(name)
    return "F15_INDETERMINATE"


def write_receipt(path, gates, telemetry, meta):
    data = {
        "schema": "f15-trajectory-engine.v1",
        "gates": gates,
        "telemetry": telemetry,
        "verdict": _verdict(gates, telemetry.get("reason")),
        "meta": meta,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def _safe_levels(obs):
    try:
        return int(getattr(obs, "levels_completed", 0) or 0)
    except Exception:
        return 0


def run_gauntlet(env_names=None, steps_per_env=150, seed=20260914,
                 horizon=DEFAULT_HORIZON, tau=DEFAULT_TAU, alpha=DEFAULT_ALPHA,
                 beam=DEFAULT_BEAM, eta_fast=0.05,
                 max_initial_overlap=MAX_INITIAL_OVERLAP,
                 trajectory_bank=None, trajectory_jsonl=None,
                 out_dir=None, receipt_out=None, _force_enabled=False):
    """F15 live gauntlet (directive command)."""
    require_f15_enabled(_force_enabled=_force_enabled)
    env_names = list(env_names) if env_names else list(DEFAULT_ENVS)
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(receipt_out) if receipt_out else out_dir / "f15_gates_receipt.json"

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    telemetry = {
        "envs": list(env_names),
        "steps": 0,
        "resets": 0,
        "mean_latency_ms": None,
        "sagnac_mean": None,
        "progress": 0.0,
        "solved": 0,
        "sum_delta_nu": 0.0,
        "mean_delta_nu_goal": None,
        "creeps": 0,
        "waypoint_align_first": None,
        "waypoint_align_last": None,
        "pg1_overlaps": None,
        "goal_source": "trajectory-bank-v2" if trajectory_bank else "NONE",
        "reason": None,
    }

    def fail_closed(reason):
        telemetry["reason"] = reason
        gates = {"PG1": False, "G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": horizon, "beam": beam, "tau": tau,
                  "alpha": alpha, "p": 32, "device": device, "eta_fast": eta_fast,
                  "max_initial_overlap": max_initial_overlap,
                  "trajectory_bank": str(trajectory_bank) if trajectory_bank else None,
                  "trajectory_jsonl": str(trajectory_jsonl) if trajectory_jsonl else None},
        )

    # --- Pre-flight: no bank -> no goal source -> fail closed -------------
    if not trajectory_bank or not trajectory_jsonl:
        return fail_closed("BLOCKED_NO_TRAJECTORY_BANK: no --trajectory-bank/--trajectory-jsonl; "
                           "goal source unavailable")

    # --- Construct substrate -----------------------------------------------
    # PatchIngress/SinglePassHorizon are module-level imports from
    # arc_f10_live_engine (canonical). No local re-import: a stale local
    # `from henri_vision_encoder import PatchIngress` shadowed the module
    # name with None and fail-closed the live run (OBSERVED 2026-09-01,
    # run-1 receipt F15_LIVE_ENGINE_BLOCKED / live_error: PatchIngress
    # unavailable — harness defect, preserved as evidence).
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=seed).to(device)
    horizon_inst = SinglePassHorizon(d=64, rank=32, K=8, num_blocks=8, seed=seed).to(device)
    engine = TrajectorySteeringEngine(
        D=64, n_actions=8, seed=seed,
        horizon=horizon, beam=beam, tau=tau, alpha=alpha, eta_fast=eta_fast,
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

    latencies, sagnacs, dnus = [], [], []
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
                    meta={"seed": seed, "K": horizon, "beam": beam, "tau": tau,
                          "alpha": alpha, "p": 32, "device": device, "eta_fast": eta_fast,
                          "max_initial_overlap": max_initial_overlap,
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
                wp = engine.waypoint(psi_s, goal, tau)

                avail = list(getattr(obs, "available_actions", None) or [])
                candidates = [int(a) for a in avail] if avail else list(range(8))
                sel, _info = engine.beam_search(
                    psi_s, wp, candidates, horizon=horizon, beam=beam, alpha=alpha)
                if align_first is None:
                    align_first = float(abs_cos(psi_s, wp).item())

                # G4 instrument: SinglePassHorizon roll[0,0] vs goal (D5,
                # byte-identical to F14 for cross-carrier comparability)
                roll = horizon_inst(psi_b)
                sagnacs.append(float(sagnac_delta(roll[0, 0], goal).item()))

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
                dnu = engine.valence_delta(psi_next, psi_s, wp)
                dnus.append(dnu)
                sum_delta_nu += dnu
                progress += r_ext
                if dnu > 0.0:
                    engine.creep(sel, dnu, psi_s)
                    creeps += 1
                if cur_levels > prev_levels:
                    solved += 1
                prev_levels = cur_levels
                align_last = float(abs_cos(psi_next, wp).item())
    except Exception as exc:
        telemetry["steps"] = steps_done
        telemetry["reason"] = "live_error: {!r}".format(exc)
        telemetry["pg1_overlaps"] = pg1_overlaps
        gates = {"PG1": True, "G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": horizon, "beam": beam, "tau": tau,
                  "alpha": alpha, "p": 32, "device": device, "eta_fast": eta_fast,
                  "max_initial_overlap": max_initial_overlap,
                  "trajectory_bank": str(trajectory_bank),
                  "trajectory_jsonl": str(trajectory_jsonl)},
        )

    mean_latency = float(np.mean(latencies)) if latencies else None
    sagnac_mean = float(np.mean(sagnacs)) if sagnacs else None
    mean_dnu = float(np.mean(dnus)) if dnus else None
    telemetry.update({
        "steps": steps_done,
        "mean_latency_ms": mean_latency,
        "sagnac_mean": sagnac_mean,
        "progress": float(progress),
        "solved": int(solved),
        "sum_delta_nu": float(sum_delta_nu),
        "mean_delta_nu_goal": mean_dnu,
        "creeps": int(creeps),
        "waypoint_align_first": align_first,
        "waypoint_align_last": align_last,
        "pg1_overlaps": pg1_overlaps,
    })

    g1 = steps_done >= steps_per_env * len(env_names) and mean_latency is not None and mean_latency <= LATENCY_BUDGET_MS
    g2 = solved > 0
    g3 = mean_dnu is not None and mean_dnu >= G3_MIN_DNU
    g4 = sagnac_mean is not None and sagnac_mean <= SAGNAC_TAU_F15
    gates = {"PG1": True, "G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    return write_receipt(
        receipt_path, gates, telemetry,
        meta={"seed": seed, "K": horizon, "beam": beam, "tau": tau,
              "alpha": alpha, "p": 32, "device": device, "eta_fast": eta_fast,
              "max_initial_overlap": max_initial_overlap,
              "trajectory_bank": str(trajectory_bank),
              "trajectory_jsonl": str(trajectory_jsonl)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=None, help="comma-separated env names (default: F10 cohort)")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--tau-waypoint", type=float, default=DEFAULT_TAU)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    ap.add_argument("--max-initial-overlap", type=float, default=MAX_INITIAL_OVERLAP)
    ap.add_argument("--eta-fast", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trajectory-bank", default=None,
                    help="verified trajectory bank npz path (required)")
    ap.add_argument("--trajectory-jsonl", default=None,
                    help="trajectory metadata jsonl path (required)")
    ap.add_argument("--out-dir", default="/tmp/henri_f15_trajectory")
    ap.add_argument("--receipt-out", default=None)
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()] if args.envs else None
    receipt = run_gauntlet(
        env_names=envs,
        steps_per_env=args.steps_per_env,
        seed=args.seed,
        horizon=args.horizon,
        tau=args.tau_waypoint,
        alpha=args.alpha,
        beam=args.beam,
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
