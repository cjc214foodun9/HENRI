"""Carrier F17 — Candidate-Differential Killing-Form Steering Engine.

Directive: HENRI-DIR-2026-08-F16-POSTMORTEM-DIFFERENTIAL-STEERING
(9962037e2555d77e15b3f78dc7edbad0c01345cd7859f43ee8bdfc8f9b797edf, 22,080 B, 261 lines)

F16 post-mortem RATIFIED (F16_GATE_G2_FAILED #bb6797be @ ledger 1,103): additive
common-mode warp D~_a = D_a + alpha*Omega is rank-inert — the BCH first-order
term exp(alpha*Omega) is identical for every candidate, so relative ranking is
unchanged (goal alignment 0.0 -> 0.0 over 1,800 steps; G3 +0.000214 residue).

Carrier F17 replaces the common-mode shift with a candidate-DIFFERENTIAL
Killing-form modulation:
  Tier 1  Omega_goal(t) = Psi_goal Psi_t^T - Psi_t Psi_goal^T   in so(D) (skew)
  Tier 2  gamma_a = -1/2 Tr(D_a Omega_goal(t))  (Killing projection; by skew-D
          algebra gamma_a = <Psi_goal, D_a Psi_t> — the goal-alignment of the
          displacement each generator would apply)
          D~_a = (1 + kappa_diff * tanh(gamma_a)) * D_a,  kappa_diff = 0.75
  Tier 3  Psi_hat_{t+1}(a) = exp(D~_a) Psi_t;  Lyapunov-damped K=8 beam:
          J(a_1:8) = |<Psi_hat_{t+8}, Psi_goal>| - beta_Sagnac * sum_k Delta_Lyap(k)
          Delta_Lyap(k) = Delta_Sagnac(k) + mu_damp * ||D~_a||_F^2,  mu_damp = 0.15
  Tier 4  Hebbian goal-valence creep retained from F15/F16 (eta=0.05).

Engagement gate (pre-registered): mean over steps of std_a(gamma_a) must exceed
1e-6, else the warp never differentiated candidates -> F17_FALSIFIED_NO_ENGAGEMENT.

Fail-closed: no bank -> F17_BLOCKED_NO_TRAJECTORY_BANK (zero steps).
Flag: HENRI_F17_DIFFERENTIAL=1 (or --force-enabled for tests).
"""
import argparse
import json
import math
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

SAGNAC_TAU_F17 = 0.050       # G4 (Lyapunov-damped)
G3_MIN_DNU = 0.0200          # G3
MAX_INITIAL_OVERLAP = 0.90   # PG1
ENGAGEMENT_MIN_GAMMA_STD = 1e-6   # E1
DEFAULT_KAPPA_DIFF = 0.75
DEFAULT_MU_DAMP = 0.15
DEFAULT_BETA_SAGNAC = 0.05
DEFAULT_HORIZON = 8
DEFAULT_BEAM = 8
DEFAULT_SEED = 20260916
DEFAULT_ETA_FAST = 0.05


def require_f17_enabled(_force_enabled=False):
    if not (_force_enabled or os.environ.get("HENRI_F17_DIFFERENTIAL") == "1"):
        raise RuntimeError("F17 differential engine disabled: set HENRI_F17_DIFFERENTIAL=1")


def omega_goal(psi_t, psi_goal):
    """Tier 1: anti-symmetric goal projection tensor, so(D)."""
    x = psi_t.reshape(-1).float()
    g = psi_goal.reshape(-1).float()
    return torch.outer(g, x) - torch.outer(x, g)


def killing_coeffs(gen, omega):
    """Tier 2: Cartan-Killing projection gamma_a = -1/2 Tr(D_a Omega).

    gen: [A, D, D] skew generators; omega: [D, D] skew goal tensor.
    By skew-D algebra gamma_a = <Psi_goal, D_a Psi_t> (positive when the
    generator displaces the state toward the goal).
    """
    tr = torch.einsum("aij,ji->a", gen, omega)
    return -0.5 * tr


class DifferentialLieEngine(nn.Module):
    """Candidate-differential Killing-steered Lie engine (Tiers 1-4)."""

    def __init__(self, D=64, n_actions=8, seed=0, horizon=DEFAULT_HORIZON,
                 beam=DEFAULT_BEAM, kappa_diff=DEFAULT_KAPPA_DIFF,
                 mu_damp=DEFAULT_MU_DAMP, beta_sagnac=DEFAULT_BETA_SAGNAC,
                 eta_fast=DEFAULT_ETA_FAST):
        super().__init__()
        self.D = D
        self.n_actions = n_actions
        self.horizon = int(horizon)
        self.beam = int(beam)
        self.kappa_diff = float(kappa_diff)
        self.mu_damp = float(mu_damp)
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

    def gamma_all(self, candidates, omega):
        """Killing coefficients for the candidate subset."""
        cand = torch.as_tensor(list(candidates), dtype=torch.long, device=self.D_a.device)
        w = omega.to(self.D_a.device)
        return killing_coeffs(self.D_a[cand], w)

    def warped_ops(self, candidates, omega, kappa_diff=None):
        """Tier 2: exp((1 + kappa*tanh(gamma_a)) * D_a) per candidate action."""
        cand = torch.as_tensor(list(candidates), dtype=torch.long, device=self.D_a.device)
        w = omega.to(self.D_a.device)
        kappa = self.kappa_diff if kappa_diff is None else float(kappa_diff)
        gams = killing_coeffs(self.D_a[cand], w)
        scale = 1.0 + kappa * torch.tanh(gams)
        warped = scale[:, None, None] * self.D_a[cand]
        return torch.linalg.matrix_exp(warped)

    def score_action(self, psi, goal, omega, a, horizon=1, beta=None, mu_damp=None,
                     kappa_diff=None):
        """Single-action objective J(a) (diagnostic / rank-break contract)."""
        beta = self.beta_sagnac if beta is None else float(beta)
        mu = self.mu_damp if mu_damp is None else float(mu_damp)
        goal_v = F.normalize(goal.reshape(-1).float().to(self.D_a.device), p=2, dim=-1)
        psi0 = F.normalize(psi.reshape(-1).float().to(self.D_a.device), p=2, dim=-1)
        ops = self.warped_ops([a], omega, kappa_diff=kappa_diff)
        norm2 = float(ops[0].pow(2).sum().item())
        state = psi0
        ssum = 0.0
        for _ in range(int(horizon)):
            nxt = F.normalize(ops[0] @ state, p=2, dim=-1)
            raw = float((nxt @ goal_v).abs().clamp(0.0, 1.0).item())
            sag = max(0.0, min(2.0, 1.0 - raw))
            ssum += sag + mu * norm2
            state = nxt
        return raw - beta * ssum

    def beam_search(self, psi, goal, omega, candidates=None, horizon=None, beam=None,
                    beta=None, mu_damp=None):
        """Tier 3: vectorized K-step beam, goal-direct objective, Lyapunov damping."""
        horizon = self.horizon if horizon is None else int(horizon)
        beam = self.beam if beam is None else int(beam)
        beta = self.beta_sagnac if beta is None else float(beta)
        mu = self.mu_damp if mu_damp is None else float(mu_damp)
        if not candidates:
            candidates = list(range(self.n_actions))
        cand = torch.as_tensor(list(candidates), dtype=torch.long, device=self.D_a.device)
        goal_v = F.normalize(goal.reshape(-1).float().to(self.D_a.device), p=2, dim=-1)
        psi0 = F.normalize(psi.reshape(-1).float().to(self.D_a.device), p=2, dim=-1)
        states = psi0.unsqueeze(0)
        ops = self.warped_ops(candidates, omega)  # [A, D, D]
        norms2 = ops.pow(2).sum(dim=(-1, -2))     # [A]
        damp_const = mu * norms2                  # per-action constant penalty
        acts = torch.full((1, 0), -1, dtype=torch.long, device=self.D_a.device)
        ssum = torch.zeros(1, device=self.D_a.device)
        for _ in range(horizon):
            nxt = torch.einsum("bd,axd->bax", states, ops)
            nxt = F.normalize(nxt, p=2, dim=-1)
            raw = nxt @ goal_v
            align = raw.abs().clamp(0.0, 1.0)
            sag = (1.0 - raw).clamp(0.0, 2.0)
            damp = sag + damp_const[None, :]
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


def _verdict(gates, telemetry=None):
    telemetry = telemetry or {}
    reason = telemetry.get("reason")
    if reason and reason.startswith("BLOCKED_NO_TRAJECTORY_BANK"):
        return "F17_BLOCKED_NO_TRAJECTORY_BANK"
    if reason and reason.startswith("DEGENERATE_GOAL"):
        return "F17_PREFLIGHT_DEGENERATE_GOAL"
    if reason is not None and reason.startswith("live_error"):
        return "F17_LIVE_ENGINE_BLOCKED"
    if not gates.get("G1"):
        return "F17_LIVE_ENGINE_BLOCKED"
    if all(gates.get(k) for k in ("G1", "G2", "G3", "G4")):
        return "F17_LIVE_LOOP_VERIFIED"
    # E1 engagement gate: without candidate-differential gamma, no steering
    # mechanism was active — subsumes K2/K3 attribution.
    gstd = telemetry.get("killing_gamma_std_mean")
    # fail-closed: non-finite engagement telemetry cannot prove engagement
    # (live run-1 defect: NaN <= threshold is False -> gate bypassed)
    if gstd is None or not math.isfinite(float(gstd)) or gstd <= ENGAGEMENT_MIN_GAMMA_STD:
        return "F17_FALSIFIED_NO_ENGAGEMENT"
    for name in ("G2", "G3", "G4"):
        if not gates.get(name):
            return "F17_GATE_{}_FAILED".format(name)
    return "F17_INDETERMINATE"


def write_receipt(path, gates, telemetry, meta):
    data = {
        "schema": "f17-differential-engine.v1",
        "gates": gates,
        "telemetry": telemetry,
        "verdict": _verdict(gates, telemetry),
        "meta": meta,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def run_gauntlet(env_names=None, steps_per_env=150, seed=DEFAULT_SEED,
                 horizon=DEFAULT_HORIZON, beam=DEFAULT_BEAM,
                 kappa_diff=DEFAULT_KAPPA_DIFF, mu_damp=DEFAULT_MU_DAMP,
                 beta_sagnac=DEFAULT_BETA_SAGNAC, eta_fast=DEFAULT_ETA_FAST,
                 max_initial_overlap=MAX_INITIAL_OVERLAP,
                 trajectory_bank=None, trajectory_jsonl=None,
                 out_dir=None, receipt_out=None, _force_enabled=False):
    """F17 live gauntlet (directive command)."""
    require_f17_enabled(_force_enabled=_force_enabled)
    env_names = list(env_names) if env_names else list(DEFAULT_ENVS)
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(receipt_out) if receipt_out else out_dir / "f17_gates_receipt.json"

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    telemetry = {
        "envs": list(env_names),
        "steps": 0,
        "resets": 0,
        "mean_latency_ms": None,
        "sagnac_raw_mean": None,
        "sagnac_lyapunov_mean": None,
        "omega_norm_mean": None,
        "killing_gamma_std_mean": None,
        "killing_gamma_min_mean": None,
        "killing_gamma_max_mean": None,
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
                  "kappa_diff": kappa_diff, "mu_damp": mu_damp,
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
    engine = DifferentialLieEngine(
        D=64, n_actions=8, seed=seed, horizon=horizon, beam=beam,
        kappa_diff=kappa_diff, mu_damp=mu_damp, beta_sagnac=beta_sagnac,
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

    latencies, sagnacs_raw, sagnacs_lyap, dnus = [], [], [], []
    omega_norms = []
    gamma_stds, gamma_mins, gamma_maxs = [], [], []
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
                          "kappa_diff": kappa_diff, "mu_damp": mu_damp,
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

                # Tier 1 + Tier 2 + Tier 3 (differential Killing beam)
                om = engine.omega(psi_s, goal)
                omega_norms.append(float(om.norm().item()))
                avail = list(getattr(obs, "available_actions", None) or [])
                candidates = [int(a) for a in avail] if avail else list(range(8))
                gams = engine.gamma_all(candidates, om)
                # population std (correction=0): a single-candidate pool has
                # zero variation, not NaN (torch sample-std on 1 element ->
                # NaN, observed live in F17 run-1)
                gamma_stds.append(float(gams.std(correction=0).item()))
                gamma_mins.append(float(gams.min().item()))
                gamma_maxs.append(float(gams.max().item()))
                sel, _info = engine.beam_search(
                    psi_s, goal, om, candidates,
                    horizon=horizon, beam=beam, beta=beta_sagnac, mu_damp=mu_damp)
                if align_first is None:
                    align_first = float(abs_cos(psi_s, goal).item())

                # G4 instrument: SinglePassHorizon roll[0,0] vs goal + Lyapunov
                # norm penalty for the SELECTED action's warped generator (D3)
                roll = horizon_inst(psi_b)
                raw_sag = float(sagnac_delta(roll[0, 0], goal).item())
                sel_ops = engine.warped_ops([sel], om)
                sel_norm2 = float(sel_ops[0].pow(2).sum().item())
                lyap_sag = raw_sag + mu_damp * sel_norm2
                sagnacs_raw.append(raw_sag)
                sagnacs_lyap.append(lyap_sag)

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
                  "kappa_diff": kappa_diff, "mu_damp": mu_damp,
                  "beta_sagnac": beta_sagnac, "p": 32, "device": device,
                  "eta_fast": eta_fast, "max_initial_overlap": max_initial_overlap,
                  "trajectory_bank": str(trajectory_bank),
                  "trajectory_jsonl": str(trajectory_jsonl)},
        )

    mean_latency = float(np.mean(latencies)) if latencies else None
    sagnac_raw_mean = float(np.mean(sagnacs_raw)) if sagnacs_raw else None
    sagnac_lyap_mean = float(np.mean(sagnacs_lyap)) if sagnacs_lyap else None
    omega_norm_mean = float(np.mean(omega_norms)) if omega_norms else None
    gstd_mean = float(np.mean(gamma_stds)) if gamma_stds else None
    gmin_mean = float(np.mean(gamma_mins)) if gamma_mins else None
    gmax_mean = float(np.mean(gamma_maxs)) if gamma_maxs else None
    mean_dnu = float(np.mean(dnus)) if dnus else None
    telemetry.update({
        "steps": steps_done,
        "mean_latency_ms": mean_latency,
        "sagnac_raw_mean": sagnac_raw_mean,
        "sagnac_lyapunov_mean": sagnac_lyap_mean,
        "omega_norm_mean": omega_norm_mean,
        "killing_gamma_std_mean": gstd_mean,
        "killing_gamma_min_mean": gmin_mean,
        "killing_gamma_max_mean": gmax_mean,
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
    g4 = sagnac_lyap_mean is not None and sagnac_lyap_mean <= SAGNAC_TAU_F17
    gates = {"PG1": True, "G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    return write_receipt(
        receipt_path, gates, telemetry,
        meta={"seed": seed, "K": horizon, "beam": beam,
              "kappa_diff": kappa_diff, "mu_damp": mu_damp,
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
    ap.add_argument("--kappa-diff", type=float, default=DEFAULT_KAPPA_DIFF)
    ap.add_argument("--mu-damp", type=float, default=DEFAULT_MU_DAMP)
    ap.add_argument("--beta-sagnac", type=float, default=DEFAULT_BETA_SAGNAC)
    ap.add_argument("--eta-fast", type=float, default=DEFAULT_ETA_FAST)
    ap.add_argument("--max-initial-overlap", type=float, default=MAX_INITIAL_OVERLAP)
    ap.add_argument("--trajectory-bank", default=None,
                    help="verified trajectory bank npz path (required)")
    ap.add_argument("--trajectory-jsonl", default=None,
                    help="trajectory metadata jsonl path (required)")
    ap.add_argument("--out-dir", default="/tmp/henri_f17_differential")
    ap.add_argument("--receipt-out", default=None)
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()] if args.envs else None
    receipt = run_gauntlet(
        env_names=envs,
        steps_per_env=args.steps_per_env,
        seed=args.seed,
        horizon=args.horizon,
        beam=args.beam,
        kappa_diff=args.kappa_diff,
        mu_damp=args.mu_damp,
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
