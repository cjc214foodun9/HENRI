"""Carrier F11 — Closed-Loop In-Situ Reward Plasticity & Sub-Goal Ingress
(Δν → Viscoelastic Creep).

Directive HENRI-DIR-2026-08-F10-POSTMORTEM-REWARD-PLASTICITY (17,788 B /
73c4bc56c0af79f757e0a4f383d43aa026511942424c5709f005a43c51296d38).

Couples the live exteroceptive reward channel Δν into the active inference
loop: Tier 1 valence-weighted selection, Tier 2 fast Hebbian action-prototype
creep, Tier 3 anisotropic Langevin escape on negative valence.

Default-OFF: HENRI_F11_PLASTICITY=1 required (fail-closed).
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

SAGNAC_TAU_F11 = 0.050  # Gate G4 (directive: <= 0.050)
LATENCY_BUDGET_MS = 5.0  # Gate G1 (directive: <= 5.0 ms)
RESET_PENALTY = 0.5  # D2 mapping: reset penalizes valence
DEFAULT_ENVS = [
    "ar25-0c556536", "sc25-635fd71a", "tr87-cd924810", "cd82-fb555c5d",
    "lp85-305b61c3", "wa30-ee6fef47", "ft09-0d8bbf25", "g50t-5849a774",
    "sk48-d8078629", "bp35-0a0ad940", "ka59-38d34dbb", "sb26-7fbdac44",
]  # F10 receipt cohort (12 envs)


def require_f11_enabled(_force_enabled=False):
    if not (_force_enabled or os.environ.get("HENRI_F11_PLASTICITY") == "1"):
        raise RuntimeError(
            "F11 plasticity engine disabled: set HENRI_F11_PLASTICITY=1"
        )


def compute_valence(prev_levels, cur_levels, was_reset):
    """Exteroceptive valence (D2 mapping).

    delta_nu := (cur_levels - prev_levels) - 0.5 * reset_penalty.
    """
    delta = float(cur_levels) - float(prev_levels)
    if was_reset:
        delta -= RESET_PENALTY
    return delta


class ActionPrototypeMemory(nn.Module):
    """Tier-2 fast-plasticity Hebbian action prototype matrix.

    M in R^{n_actions x D} (D4: ratified F10 scale, D = num_blocks * 8 = 64).
    Zero-initialized: R-hat = 0 until first valence event (zero pretraining).
    """

    def __init__(self, n_actions=8, D=64, eta_fast=0.05, rhat_decay=0.9, seed=0):
        super().__init__()
        self.n_actions = n_actions
        self.D = D
        self.eta_fast = eta_fast
        self.rhat_decay = rhat_decay
        self.register_buffer("M", torch.zeros(n_actions, D))
        self.register_buffer("rhat", torch.zeros(n_actions))
        self.register_buffer("_rhat_initialized", torch.zeros(n_actions, dtype=torch.bool))

    def update_rhat(self, action, delta_nu):
        """Exponential moving average of valence per action."""
        if not bool(self._rhat_initialized[action]):
            self.rhat[action] = (1.0 - self.rhat_decay) * delta_nu
            self._rhat_initialized[action] = True
        else:
            self.rhat[action] = (
                self.rhat_decay * self.rhat[action]
                + (1.0 - self.rhat_decay) * delta_nu
            )

    def creep(self, action, delta_nu, psi):
        """M_{a} <- Normalize_L2( M_{a} + eta_fast * delta_nu * psi ).

        Positive valence pulls the prototype toward the current wave state;
        negative valence pushes it away.
        """
        self.update_rhat(action, delta_nu)
        psi_flat = psi.reshape(-1).to(self.M.device)
        update = self.eta_fast * delta_nu * psi_flat
        row = self.M[action] + update
        norm = torch.linalg.vector_norm(row).clamp_min(1e-8)
        self.M[action] = row / norm


class F11PlasticityEngine(nn.Module):
    """Valence-coupled active inference engine (Tiers 1-3)."""

    def __init__(self, D=64, n_actions=8, seed=0, lambda_reward=2.0,
                 eta_fast=0.05, t_base=0.15, kappa=0.5, t_escape=0.50,
                 escape_steps=3, streak_window=10, rhat_decay=0.9):
        super().__init__()
        self.D = D
        self.n_actions = n_actions
        self.lambda_reward = lambda_reward
        self.eta_fast = eta_fast
        self.t_base = t_base
        self.kappa = kappa
        self.t_escape = t_escape
        self.escape_steps = escape_steps
        self.streak_window = streak_window
        self.memory = ActionPrototypeMemory(
            n_actions=n_actions, D=D, eta_fast=eta_fast, rhat_decay=rhat_decay, seed=seed
        )
        self.neg_streak = 0

    def active_temperature(self, streak=None, delta_nu=0.0):
        """T_active = T_base + kappa * max(0, -delta_nu), with escape floor."""
        streak = self.neg_streak if streak is None else streak
        t = self.t_base + self.kappa * max(0.0, -float(delta_nu))
        if streak >= self.escape_steps:
            t = max(t, self.t_escape)
        return t

    def select_valence_weighted(self, sagnac, rhat, candidates):
        """a* = argmin_a [ Delta_Sagnac(a) - lambda_rew * R_hat(a) ]."""
        best = None
        best_val = float("inf")
        for a in candidates:
            r = float(rhat.get(a, 0.0))
            val = float(sagnac.get(a, float("inf"))) - self.lambda_reward * r
            if val < best_val:
                best_val = val
                best = a
        return best

    def observe_outcome(self, action, delta_nu, psi):
        """Tier-2 creep + Tier-3 streak tracking after each live step."""
        if delta_nu != 0.0:
            self.memory.creep(action, delta_nu, psi)
        if delta_nu < 0.0:
            self.neg_streak += 1
        else:
            self.neg_streak = 0
        return self.active_temperature(delta_nu=delta_nu)


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
        return "F11_LIVE_ENGINE_BLOCKED"
    if all(gates.get(k) for k in ("G1", "G2", "G3", "G4")):
        return "F11_REWARD_LOOP_VERIFIED"
    for name in ("G2", "G3", "G4"):
        if not gates.get(name):
            return "F11_GATE_{}_FAILED".format(name)
    return "F11_INDETERMINATE"


def write_receipt(path, gates, telemetry, meta):
    data = {
        "schema": "f11-plasticity.v1",
        "gates": gates,
        "telemetry": telemetry,
        "verdict": _verdict(gates),
        "meta": meta,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def run_gauntlet(env_names=None, steps_per_env=60, seed=20260909,
                 lambda_reward=2.0, eta_fast=0.05, out_dir=None,
                 receipt_out=None, _force_enabled=False):
    require_f11_enabled(_force_enabled=_force_enabled)
    env_names = list(env_names) if env_names else list(DEFAULT_ENVS)
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(receipt_out) if receipt_out else out_dir / "f11_gates_receipt.json"

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=seed).to(device)
    horizon = SinglePassHorizon(d=64, rank=32, K=8, num_blocks=8, seed=seed).to(device)
    engine = F11PlasticityEngine(
        D=64, n_actions=8, seed=seed,
        lambda_reward=lambda_reward, eta_fast=eta_fast,
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
        "creeps": 0,
        "escapes": 0,
        "reason": None,
    }

    def fail_closed(reason):
        telemetry["reason"] = reason
        gates = {"G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": 8, "p": 32, "device": device,
                  "lambda_reward": lambda_reward, "eta_fast": eta_fast},
        )

    try:
        from arc_agi import Arcade
        arcade = Arcade()
    except Exception as exc:
        return fail_closed("arcade_unavailable: {!r}".format(exc))

    latencies, sagnacs = [], []
    steps_done = 0
    sum_delta_nu = 0.0
    progress = 0.0
    solved = 0
    creeps = 0
    escapes = 0
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
                psi = ingress(raw.unsqueeze(0))
                if goal is None:
                    goal = psi[0].detach()
                roll = horizon(psi)
                # Tier 1: valence-weighted selection over available actions
                avail = list(getattr(obs, "available_actions", None) or [])
                candidates = [int(a) for a in avail] if avail else list(range(8))
                sagnac_by_a = {}
                rhat_by_a = {}
                for a in candidates:
                    sim = F.cosine_similarity(
                        roll[0, 0].reshape(-1).unsqueeze(0),
                        engine.memory.M[a].reshape(-1).unsqueeze(0),
                        dim=-1,
                    ).clamp(-1.0, 1.0)
                    sagnac_by_a[a] = 1.0 - sim.item()
                    rhat_by_a[a] = float(engine.memory.rhat[a])
                sel = engine.select_valence_weighted(sagnac_by_a, rhat_by_a, candidates)
                sagnacs.append(float(sagnac_delta(roll[0, 0], goal).item()))
                # Tier 3: anisotropic Langevin escape on negative streak
                t_active = engine.active_temperature(delta_nu=0.0)
                if engine.neg_streak >= engine.escape_steps:
                    escapes += 1
                    noise = torch.randn_like(psi) * (2.0 * t_active) ** 0.5
                    psi = F.normalize(psi + noise, dim=-1)
                actions = list(game.action_space)
                action = actions[sel % max(1, len(actions))]
                obs = game.step(action)
                latencies.append((time.perf_counter() - t_start) * 1000.0)
                steps_done += 1
                terminal = obs is None or (
                    getattr(obs, "state", None) and obs.state.name == "GAME_OVER")
                was_reset = terminal
                cur_levels = _safe_levels(obs) if not terminal else prev_levels
                delta_nu = compute_valence(prev_levels, cur_levels, was_reset)
                sum_delta_nu += delta_nu
                progress += delta_nu
                if not terminal and delta_nu != 0.0:
                    engine.memory.creep(sel, delta_nu, psi[0].detach())
                    creeps += 1
                if terminal:
                    telemetry["resets"] += 1
                    obs = game.reset()
                    if obs is None or not getattr(obs, "frame", None):
                        break
                    goal = None
                    prev_levels = _safe_levels(obs)
                    continue
                solved += 1 if cur_levels > prev_levels else 0
                prev_levels = cur_levels
    except Exception as exc:
        telemetry["steps"] = steps_done
        telemetry["reason"] = "live_error: {!r}".format(exc)
        gates = {"G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": 8, "p": 32, "device": device,
                  "lambda_reward": lambda_reward, "eta_fast": eta_fast},
        )

    mean_latency = float(np.mean(latencies)) if latencies else None
    sagnac_mean = float(np.mean(sagnacs)) if sagnacs else None
    telemetry.update({
        "steps": steps_done,
        "mean_latency_ms": mean_latency,
        "sagnac_mean": sagnac_mean,
        "progress": float(progress),
        "solved": int(solved),
        "sum_delta_nu": float(sum_delta_nu),
        "creeps": int(creeps),
        "escapes": int(escapes),
    })

    g1 = steps_done >= steps_per_env * len(env_names) and mean_latency is not None and mean_latency <= LATENCY_BUDGET_MS
    g2 = solved > 0
    g3 = sum_delta_nu > 0.0
    g4 = sagnac_mean is not None and sagnac_mean <= SAGNAC_TAU_F11
    gates = {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    return write_receipt(
        receipt_path, gates, telemetry,
        meta={"seed": seed, "K": 8, "p": 32, "device": device,
              "lambda_reward": lambda_reward, "eta_fast": eta_fast},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=None, help="comma-separated env names (default: F10 cohort)")
    ap.add_argument("--steps-per-env", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--lambda-reward", type=float, default=2.0)
    ap.add_argument("--eta-fast", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="/tmp/henri_f11_plasticity")
    ap.add_argument("--receipt-out", default=None)
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()] if args.envs else None
    receipt = run_gauntlet(
        env_names=envs,
        steps_per_env=args.steps_per_env,
        seed=args.seed,
        lambda_reward=args.lambda_reward,
        eta_fast=args.eta_fast,
        out_dir=args.out_dir,
        receipt_out=args.receipt_out,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
