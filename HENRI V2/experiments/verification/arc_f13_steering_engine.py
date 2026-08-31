"""Carrier F13 — Hierarchical Wave-Optic Goal Steering & Macro-Action Synthesis.

Directive HENRI-DIR-2026-08-F12-POSTMORTEM-HIERARCHICAL-STEERING
(18,857 B / d02eca2cd414bdce52ec5fedad3275214b21414ef273cf97c01e5de9db6b3a2f).
F12 RATIFIED (F12_GATE_G2_FAILED, seal #5145429145): intrinsic curiosity
provably engages plasticity (G3 creeps 1,642, sum dnu +1174.6) but collapses
into the "Noisy TV" epistemic trap (91 hashes / 158 resets / 0 solved). This
engine replaces undirected surprise-seeking with directional sub-goal phase
steering:

  Tier 1  Psi_waypoint = normalize((1-tau) Psi_t + tau Psi_goal), tau = 0.25
  Tier 2  J(a_{1:K}) = |<Psi_hat_{t+K}(a_{1:K}), Psi_waypoint>|
                      - alpha * sum_k Sagnac(Psi_hat_{t+k}, Psi_waypoint)
          a*_t = a_1* from K=8 beam search (zero-trainable exp(D_a) rollouts)
  Tier 3  commit first macro-action through the live Arcade adapter
  Tier 4  dnu_t = |<Psi_{t+1}, Psi_waypoint>| - |<Psi_t, Psi_waypoint>|
          M_{a_t} <- Normalize_L2(M_{a_t} + eta_fast * dnu_t * Psi_t)

Default-OFF: HENRI_F13_STEERING=1 required (fail-closed).
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

from arc_f10_live_engine import PatchIngress, SinglePassHorizon, sagnac_delta, _to_flat
from arc_f11_plasticity_engine import ActionPrototypeMemory, compute_valence

SAGNAC_TAU_F13 = 0.050  # Gate G4 (directive: <= 0.050)
LATENCY_BUDGET_MS = 5.0  # Gate G1 (directive: <= 5.0 ms)
DEFAULT_TAU = 0.25  # Gate/Tier 1 (directive: tau-waypoint 0.25)
DEFAULT_HORIZON = 8  # Tier 2 (directive: horizon 8)
DEFAULT_ALPHA = 0.05  # Tier 2 Sagnac penalty scale (deviation D4)
DEFAULT_BEAM = 8  # Beam width (deviation D3)
RESET_PENALTY = 0.5  # D2 mapping (carried from F11)
DEFAULT_ENVS = [
    "ar25-0c556536", "sc25-635fd71a", "tr87-cd924810", "cd82-fb555c5d",
    "lp85-305b61c3", "wa30-ee6fef47", "ft09-0d8bbf25", "g50t-5849a774",
    "sk48-d8078629", "bp35-0a0ad940", "ka59-38d34dbb", "sb26-7fbdac44",
]  # F10 receipt cohort (12 envs)


def require_f13_enabled(_force_enabled=False):
    if not (_force_enabled or os.environ.get("HENRI_F13_STEERING") == "1"):
        raise RuntimeError(
            "F13 steering engine disabled: set HENRI_F13_STEERING=1"
        )


def abs_cos(a, b):
    """|cosine| between two flat vectors, clamped to [0, 1]."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).clamp(0.0, 1.0).squeeze(0)


class SteeringEngine(nn.Module):
    """Hierarchical goal-steering engine (Tiers 1-4).

    D_a skew-symmetric (seeded, zero-trainable) => exp(D_a) orthogonal on
    S^{D-1}. M in R^{n_actions x D} Hebbian prototypes (Tier 4, signed
    valence). All steering math is pure torch (grad-tensor safe).
    """

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
        skew = skew - skew.transpose(-1, -2)  # skew-symmetric
        self.register_buffer("expD", torch.linalg.matrix_exp(skew))  # [A, D, D]
        self.memory = ActionPrototypeMemory(
            n_actions=n_actions, D=D, eta_fast=eta_fast, seed=seed
        )

    def rollout(self, psi, a):
        """Psi_hat_{t+1}(a) = exp(D_a) Psi_t (norm-preserving)."""
        flat = psi.reshape(-1).float()
        out = self.expD[a] @ flat
        return F.normalize(out, p=2, dim=-1)

    def waypoint(self, psi_t, psi_goal, tau=None):
        """Tier 1: Psi_waypoint = normalize((1-tau) Psi_t + tau Psi_goal)."""
        tau = self.tau if tau is None else float(tau)
        mix = (1.0 - tau) * psi_t.reshape(-1).float() + tau * psi_goal.reshape(-1).float()
        return F.normalize(mix, p=2, dim=-1)

    def sagnac_to(self, state, waypoint):
        """Delta_Sagnac(state, waypoint) in [0, 2] (reuse F10 instrument)."""
        return sagnac_delta(state, waypoint)

    def score_path(self, rollouts, waypoint, alpha=None):
        """J = |<final, waypoint>| - alpha * sum_k Sagnac(rollouts[k], waypoint)."""
        alpha = self.alpha if alpha is None else float(alpha)
        final = rollouts[-1]
        j = abs_cos(final, waypoint).item()
        if alpha > 0.0:
            penalty = sum(
                float(self.sagnac_to(s.reshape(-1), waypoint.reshape(-1)).item())
                for s in rollouts
            )
            j -= alpha * penalty
        return j

    def beam_search(self, psi, waypoint, candidates, horizon=None, beam=None, alpha=None):
        """Tier 2: K-step beam search; commit a_1* (Tier 3).

        Pruning score at depth k: |cos(psi_k, waypoint)| - alpha * sum Sagnac
        (consistent with the directive's terminal J). Returns
        (action, info{horizon, beam, actions, j}).
        """
        horizon = self.horizon if horizon is None else int(horizon)
        beam = self.beam if beam is None else int(beam)
        alpha = self.alpha if alpha is None else float(alpha)

        paths = [([], psi.reshape(-1).float(), 0.0, 0.0)]  # (actions, state, sagnac_sum, j_partial)
        for _ in range(horizon):
            expanded = []
            for acts, state, ssum, _j in paths:
                for a in candidates:
                    nxt = self.rollout(state, a)
                    sk = float(self.sagnac_to(nxt, waypoint.reshape(-1)).item())
                    jp = abs_cos(nxt, waypoint).item() - alpha * (ssum + sk)
                    expanded.append((acts + [a], nxt, ssum + sk, jp))
            expanded.sort(key=lambda x: x[3], reverse=True)
            paths = expanded[:beam]

        best_acts, best_state, best_ssum, _ = max(
            paths, key=lambda x: x[3]
        )
        j = abs_cos(best_state, waypoint).item() - alpha * best_ssum
        info = {
            "horizon": horizon,
            "beam": beam,
            "actions": list(best_acts),
            "j": float(j),
        }
        return best_acts[0], info

    def valence_delta(self, psi_next, psi_t, waypoint):
        """Tier 4: dnu = |<psi_next, waypoint>| - |<psi_t, waypoint>| (signed)."""
        return float(
            abs_cos(psi_next, waypoint).item() - abs_cos(psi_t, waypoint).item()
        )

    def creep(self, action, delta_nu, psi):
        """Tier 4: M_a <- Normalize_L2(M_a + eta_fast * delta_nu * Psi_t).

        Zero valence -> no update (literal formula: M + 0*Psi == M; avoids
        float normalization drift on non-events).
        """
        if float(delta_nu) == 0.0:
            return
        self.memory.creep(action, delta_nu, psi.reshape(1, -1))


def _safe_levels(outcome):
    try:
        if isinstance(outcome, dict):
            return int(outcome.get("levels_completed", 0) or 0)
        if hasattr(outcome, "levels_completed"):
            return int(outcome.levels_completed or 0)
    except Exception:
        pass
    return 0


def _verdict(gates):
    if not gates.get("G1"):
        return "F13_LIVE_ENGINE_BLOCKED"
    if all(gates.get(k) for k in ("G1", "G2", "G3", "G4")):
        return "F13_GOAL_STEERING_VERIFIED"
    for name in ("G2", "G3", "G4"):
        if not gates.get(name):
            return "F13_GATE_{}_FAILED".format(name)
    return "F13_INDETERMINATE"


def write_receipt(path, gates, telemetry, meta):
    data = {
        "schema": "f13-steering.v1",
        "gates": gates,
        "telemetry": telemetry,
        "verdict": _verdict(gates),
        "meta": meta,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def run_gauntlet(env_names=None, steps_per_env=150, seed=20260911,
                 horizon=DEFAULT_HORIZON, tau=DEFAULT_TAU, alpha=DEFAULT_ALPHA,
                 beam=DEFAULT_BEAM, eta_fast=0.05,
                 out_dir=None, receipt_out=None, _force_enabled=False):
    require_f13_enabled(_force_enabled=_force_enabled)
    env_names = list(env_names) if env_names else list(DEFAULT_ENVS)
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(receipt_out) if receipt_out else out_dir / "f13_gates_receipt.json"

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=seed).to(device)
    horizon_inst = SinglePassHorizon(d=64, rank=32, K=8, num_blocks=8, seed=seed).to(device)
    engine = SteeringEngine(
        D=64, n_actions=8, seed=seed,
        horizon=horizon, beam=beam, tau=tau, alpha=alpha, eta_fast=eta_fast,
    ).to(device)

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
        "reason": None,
    }

    def fail_closed(reason):
        telemetry["reason"] = reason
        gates = {"G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": horizon, "beam": beam, "tau": tau,
                  "alpha": alpha, "p": 32, "device": device, "eta_fast": eta_fast},
        )

    try:
        from arc_agi import Arcade
        arcade = Arcade()
    except Exception as exc:
        return fail_closed("arcade_unavailable: {!r}".format(exc))

    latencies, sagnacs, dnus = [], [], []
    steps_done = 0
    sum_delta_nu = 0.0
    progress = 0.0
    solved = 0
    creeps = 0
    align_first = None
    try:
        for name in env_names:
            game = arcade.make(name)
            if game is None:
                return fail_closed("arcade_make_returned_none: {!r}".format(name))
            obs = game.reset()
            if obs is None or not getattr(obs, "frame", None):
                return fail_closed("null_initial_frame: {!r}".format(name))
            goal = None
            prev_levels = _safe_levels(obs)
            for _ in range(steps_per_env):
                frame = obs.frame[0]
                raw = torch.as_tensor(_to_flat(frame), dtype=torch.float32, device=device)
                if raw.numel() < 4096:
                    raw = F.pad(raw, (0, 4096 - raw.numel()))
                else:
                    raw = raw[:4096]
                t_start = time.perf_counter()
                psi = ingress(raw.unsqueeze(0))[0]
                if goal is None:
                    goal = psi.detach()  # D1: episodic first-frame axiom
                psi_s = psi.detach()
                wp = engine.waypoint(psi_s, goal, tau)
                roll = horizon_inst(psi)  # G4 measurement instrument (D9)

                # Tier 2: Sagnac-guided macro-action beam search
                avail = list(getattr(obs, "available_actions", None) or [])
                candidates = [int(a) for a in avail] if avail else list(range(8))
                sel, _info = engine.beam_search(
                    psi_s, wp, candidates, horizon=horizon, beam=beam, alpha=alpha
                )
                if align_first is None:
                    align_first = float(abs_cos(psi_s, wp).item())

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
                r_ext = compute_valence(prev_levels, cur_levels, was_reset)  # D2
                if terminal:
                    telemetry["resets"] += 1
                    obs = game.reset()
                    if obs is None or not getattr(obs, "frame", None):
                        break
                    goal = None
                    prev_levels = _safe_levels(obs)
                    continue

                # Tier 4: signed goal-convergence valence on OBSERVED next state
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
                if dnu > 0.0:  # goal-convergent moves drive active Hebbian updates
                    engine.creep(sel, dnu, psi_s)
                    creeps += 1
                if cur_levels > prev_levels:
                    solved += 1
                prev_levels = cur_levels
                align_last = float(abs_cos(psi_next, wp).item())
    except Exception as exc:
        telemetry["steps"] = steps_done
        telemetry["reason"] = "live_error: {!r}".format(exc)
        gates = {"G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": horizon, "beam": beam, "tau": tau,
                  "alpha": alpha, "p": 32, "device": device, "eta_fast": eta_fast},
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
    })

    g1 = steps_done >= steps_per_env * len(env_names) and mean_latency is not None and mean_latency <= LATENCY_BUDGET_MS
    g2 = solved > 0
    g3 = mean_dnu is not None and mean_dnu > 0.0
    g4 = sagnac_mean is not None and sagnac_mean <= SAGNAC_TAU_F13
    gates = {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    return write_receipt(
        receipt_path, gates, telemetry,
        meta={"seed": seed, "K": horizon, "beam": beam, "tau": tau,
              "alpha": alpha, "p": 32, "device": device, "eta_fast": eta_fast},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=None, help="comma-separated env names (default: F10 cohort)")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--tau-waypoint", type=float, default=DEFAULT_TAU)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    ap.add_argument("--eta-fast", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="/tmp/henri_f13_steering")
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
        out_dir=args.out_dir,
        receipt_out=args.receipt_out,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
