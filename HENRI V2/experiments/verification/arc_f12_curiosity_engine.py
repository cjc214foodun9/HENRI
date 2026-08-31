"""Carrier F12 — Sub-Goal Phase Hashing & Intrinsic Curiosity Ingress
(Count-Based Phase Epistemic Drive).

Directive HENRI-DIR-2026-08-F11-POSTMORTEM-SUBGOAL-INGRESS (18,149 B /
f19107fdcabc90a1b109d879328c4a77a65ed432599c2ed91135331818888a4d).
F11 RATIFIED (F11_GATE_G2_FAILED, seal #95c17ee1): reward-plasticity machinery
valid; sparse exteroceptive feedback is the bottleneck. This engine replaces
dependence on external scalar rewards with a dense intrinsic information-gain
drive:

  r_intrinsic = r_surprise + r_novelty
  r_surprise(t) = 1 - |<exp(D_{a_t}) Psi_t, Psi_{t+1}>|   (Wave-JEPA surprise)
  r_novelty(t)  = 1 / sqrt(N(Hash(Psi_{t+1})) + 1)        (count-based novelty)

Tiers: 1 single-pass wave-JEPA rollout per candidate action (exp(D_a) orthogonal),
2 intrinsic free-energy action selection, 3 online Hebbian trace update with
dense intrinsic valence, 4 fast state-hash frontier cache.

Default-OFF: HENRI_F12_CURIOSITY=1 required (fail-closed).
"""

import argparse
import json
import os
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_f10_live_engine import PatchIngress, SinglePassHorizon, sagnac_delta, _to_flat
from arc_f11_plasticity_engine import ActionPrototypeMemory, compute_valence

SAGNAC_TAU_F12 = 0.050  # Gate G4 (directive: <= 0.050)
LATENCY_BUDGET_MS = 5.0  # Gate G1 (directive: <= 5.0 ms)
MIN_CREEPS = 100  # Gate G3 (directive: creeps >= 100)
RESET_PENALTY = 0.5  # D2 mapping (carried from F11)
DEFAULT_ENVS = [
    "ar25-0c556536", "sc25-635fd71a", "tr87-cd924810", "cd82-fb555c5d",
    "lp85-305b61c3", "wa30-ee6fef47", "ft09-0d8bbf25", "g50t-5849a774",
    "sk48-d8078629", "bp35-0a0ad940", "ka59-38d34dbb", "sb26-7fbdac44",
]  # F10 receipt cohort (12 envs)


def require_f12_enabled(_force_enabled=False):
    if not (_force_enabled or os.environ.get("HENRI_F12_CURIOSITY") == "1"):
        raise RuntimeError(
            "F12 curiosity engine disabled: set HENRI_F12_CURIOSITY=1"
        )


def abs_cos(a, b):
    """|cosine| between two flat vectors, clamped to [0, 1]."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).clamp(0.0, 1.0).squeeze(0)


class CuriosityEngine(nn.Module):
    """Dual-stream intrinsic curiosity engine (Tiers 1-4).

    D_a skew-symmetric (seeded, zero-trainable) => exp(D_a) orthogonal on
    S^{D-1}. M in R^{n_actions x D} Hebbian prototypes (Tier 3), in-memory
    frontier hash table N(h) (Tier 4).
    """

    def __init__(self, D=64, n_actions=8, seed=0,
                 lambda_cur=1.0, lambda_nov=0.5, eta_fast=0.05):
        super().__init__()
        self.D = D
        self.n_actions = n_actions
        self.lambda_cur = lambda_cur
        self.lambda_nov = lambda_nov
        self.eta_fast = eta_fast
        g = torch.Generator().manual_seed(seed + 10)
        skew = torch.randn(n_actions, D, D, generator=g) * 0.1
        skew = skew - skew.transpose(-1, -2)  # skew-symmetric
        self.register_buffer("expD", torch.linalg.matrix_exp(skew))  # [A, D, D]
        self.memory = ActionPrototypeMemory(
            n_actions=n_actions, D=D, eta_fast=eta_fast, seed=seed
        )
        self.frontier = {}  # h -> N(h)

    def rollout(self, psi, a):
        """Psi_hat_{t+1}(a) = exp(D_a) Psi_t (norm-preserving)."""
        flat = psi.reshape(-1).float()
        out = self.expD[a] @ flat
        return F.normalize(out, p=2, dim=-1)

    def surprise(self, pred, actual):
        """r_surprise = 1 - |<pred, actual>| in [0, 1] (D5)."""
        return (1.0 - abs_cos(pred, actual)).item()

    def hash_wave(self, psi):
        """h = crc32(packed sign bits) in Z_{2^32} (D6).

        Detaches first: the live ingress path produces grad-requiring waves;
        numpy() on a grad tensor is a fail-closed RuntimeError (F12 run 1,
        live_error, 0 steps — preserved as evidence).
        """
        signs = torch.sign(psi.reshape(-1).detach()).cpu().numpy().astype(np.int8)
        bits = np.packbits((signs > 0).astype(np.uint8))
        return int(zlib.crc32(bits.tobytes()) & 0xFFFFFFFF)

    def novelty(self, h):
        """r_novelty = 1 / sqrt(N(h) + 1); N BEFORE increment (D7)."""
        n = int(self.frontier.get(h, 0))
        return 1.0 / float(np.sqrt(n + 1))

    def visit(self, h):
        """Tier 4: increment AFTER reward computation (D7)."""
        self.frontier[h] = self.frontier.get(h, 0) + 1

    def compute_valence_intrinsic(self, r_ext, pred, actual, h):
        """Delta_nu = r_ext + lambda_cur*r_surprise + lambda_nov*r_novelty."""
        return (
            float(r_ext)
            + self.lambda_cur * self.surprise(pred, actual)
            + self.lambda_nov * self.novelty(h)
        )

    def creep(self, action, delta_nu, psi):
        """Tier 3: M_a <- Normalize_L2(M_a + eta_fast * delta_nu * Psi_t)."""
        self.memory.creep(action, delta_nu, psi.reshape(1, -1))

    def select_intrinsic(self, psi, goal, candidates, lambda_cur=None, lambda_nov=None):
        """a* = argmax_a [ lambda_cur*(1-|cos(rollout(a), goal)|)
                          + lambda_nov * novelty(hash(rollout(a))) ]."""
        lambda_cur = self.lambda_cur if lambda_cur is None else lambda_cur
        lambda_nov = self.lambda_nov if lambda_nov is None else lambda_nov
        best = None
        best_val = float("-inf")
        for a in candidates:
            pred = self.rollout(psi, a)
            val = lambda_cur * (1.0 - abs_cos(pred, goal).item())
            if lambda_nov > 0.0:
                val += lambda_nov * self.novelty(self.hash_wave(pred))
            if val > best_val:
                best_val = val
                best = a
        return best


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
        return "F12_LIVE_ENGINE_BLOCKED"
    if all(gates.get(k) for k in ("G1", "G2", "G3", "G4")):
        return "F12_CURIOSITY_LOOP_VERIFIED"
    for name in ("G2", "G3", "G4"):
        if not gates.get(name):
            return "F12_GATE_{}_FAILED".format(name)
    return "F12_INDETERMINATE"


def write_receipt(path, gates, telemetry, meta):
    data = {
        "schema": "f12-curiosity.v1",
        "gates": gates,
        "telemetry": telemetry,
        "verdict": _verdict(gates),
        "meta": meta,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def run_gauntlet(env_names=None, steps_per_env=150, seed=20260910,
                 lambda_curiosity=1.0, lambda_novelty=0.5, eta_fast=0.05,
                 out_dir=None, receipt_out=None, _force_enabled=False):
    require_f12_enabled(_force_enabled=_force_enabled)
    env_names = list(env_names) if env_names else list(DEFAULT_ENVS)
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(receipt_out) if receipt_out else out_dir / "f12_gates_receipt.json"

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=seed).to(device)
    horizon = SinglePassHorizon(d=64, rank=32, K=8, num_blocks=8, seed=seed).to(device)
    engine = CuriosityEngine(
        D=64, n_actions=8, seed=seed,
        lambda_cur=lambda_curiosity, lambda_nov=lambda_novelty, eta_fast=eta_fast,
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
        "surprise_mean": None,
        "novelty_mean": None,
        "frontier_size": 0,
        "reason": None,
    }

    def fail_closed(reason):
        telemetry["reason"] = reason
        gates = {"G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": 8, "p": 32, "device": device,
                  "lambda_curiosity": lambda_curiosity,
                  "lambda_novelty": lambda_novelty, "eta_fast": eta_fast},
        )

    try:
        from arc_agi import Arcade
        arcade = Arcade()
    except Exception as exc:
        return fail_closed("arcade_unavailable: {!r}".format(exc))

    latencies, sagnacs, surprises, novelties = [], [], [], []
    steps_done = 0
    sum_delta_nu = 0.0
    progress = 0.0
    solved = 0
    creeps = 0
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
                    goal = psi[0].detach()  # D1: episodic first-frame axiom
                roll = horizon(psi)  # G4 measurement instrument (D9)

                # Tier 2: intrinsic free-energy selection over available actions
                avail = list(getattr(obs, "available_actions", None) or [])
                candidates = [int(a) for a in avail] if avail else list(range(8))
                sel = engine.select_intrinsic(psi[0], goal, candidates)
                pred = engine.rollout(psi[0], sel)

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

                # Dense intrinsic valence on the OBSERVED next state
                frame_next = obs.frame[0]
                raw_next = torch.as_tensor(_to_flat(frame_next), dtype=torch.float32, device=device)
                if raw_next.numel() < 4096:
                    raw_next = F.pad(raw_next, (0, 4096 - raw_next.numel()))
                else:
                    raw_next = raw_next[:4096]
                psi_next = ingress(raw_next.unsqueeze(0))[0].detach()
                h_next = engine.hash_wave(psi_next)
                nu = engine.compute_valence_intrinsic(r_ext, pred, psi_next, h_next)
                surprises.append(engine.surprise(pred, psi_next))
                novelties.append(engine.novelty(h_next))
                sum_delta_nu += nu
                progress += r_ext
                if nu > 0.0:  # dense intrinsic valence drives active Hebbian updates
                    engine.creep(sel, nu, psi[0].detach())
                    creeps += 1
                engine.visit(h_next)  # Tier 4 (D7: AFTER reward)
                if cur_levels > prev_levels:
                    solved += 1
                prev_levels = cur_levels
    except Exception as exc:
        telemetry["steps"] = steps_done
        telemetry["reason"] = "live_error: {!r}".format(exc)
        gates = {"G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": 8, "p": 32, "device": device,
                  "lambda_curiosity": lambda_curiosity,
                  "lambda_novelty": lambda_novelty, "eta_fast": eta_fast},
        )

    mean_latency = float(np.mean(latencies)) if latencies else None
    sagnac_mean = float(np.mean(sagnacs)) if sagnacs else None
    surprise_mean = float(np.mean(surprises)) if surprises else None
    novelty_mean = float(np.mean(novelties)) if novelties else None
    telemetry.update({
        "steps": steps_done,
        "mean_latency_ms": mean_latency,
        "sagnac_mean": sagnac_mean,
        "progress": float(progress),
        "solved": int(solved),
        "sum_delta_nu": float(sum_delta_nu),
        "creeps": int(creeps),
        "surprise_mean": surprise_mean,
        "novelty_mean": novelty_mean,
        "frontier_size": int(len(engine.frontier)),
    })

    g1 = steps_done >= steps_per_env * len(env_names) and mean_latency is not None and mean_latency <= LATENCY_BUDGET_MS
    g2 = solved > 0
    g3 = creeps >= MIN_CREEPS
    g4 = sagnac_mean is not None and sagnac_mean <= SAGNAC_TAU_F12
    gates = {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    return write_receipt(
        receipt_path, gates, telemetry,
        meta={"seed": seed, "K": 8, "p": 32, "device": device,
              "lambda_curiosity": lambda_curiosity,
              "lambda_novelty": lambda_novelty, "eta_fast": eta_fast},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=None, help="comma-separated env names (default: F10 cohort)")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--lambda-curiosity", type=float, default=1.0)
    ap.add_argument("--lambda-novelty", type=float, default=0.5)
    ap.add_argument("--eta-fast", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="/tmp/henri_f12_curiosity")
    ap.add_argument("--receipt-out", default=None)
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()] if args.envs else None
    receipt = run_gauntlet(
        env_names=envs,
        steps_per_env=args.steps_per_env,
        seed=args.seed,
        lambda_curiosity=args.lambda_curiosity,
        lambda_novelty=args.lambda_novelty,
        eta_fast=args.eta_fast,
        out_dir=args.out_dir,
        receipt_out=args.receipt_out,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
