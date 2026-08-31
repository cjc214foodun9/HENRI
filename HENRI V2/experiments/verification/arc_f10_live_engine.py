"""Carrier F10 — Live Interactive Time-Series World Model
(TimesFM-3 Continuous Ingress & Closed-Loop Active Inference).

Directive HENRI-DIR-2026-08-F9-1-POSTMORTEM-TIMESFM3-SYNTHESIS (18,535 B / 149bd93b02...).
F9/F9.1 bank-optimization line permanently CLOSED (sealed F9_OPTIMIZATION_FAILED /
F9_1_OPTIMIZATION_FAILED at ln 7). This engine runs LIVE: raw grid observations ->
TimesFM-3-style patch ingress (p=32) -> single-pass K=8 Sagnac horizon -> EFE
selection -> arcade step. Default-OFF: HENRI_F10_LIVE=1.
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

FLAG = "HENRI_F10_LIVE"
SAGNAC_TAU = 0.35


def require_f10_enabled(_force_enabled=False):
    if not _force_enabled and os.environ.get(FLAG) != "1":
        raise RuntimeError("HENRI_F10_LIVE != 1: Carrier F10 engine is default-OFF")


class PatchIngress(nn.Module):
    """TimesFM-3-style continuous patch ingress on RAW observation vectors.

    x [B, in_dim] -> contiguous patches (p) -> residual MLP + position embedding
    -> projection -> [B, num_blocks, 8] -> per-block L2 sphere.
    """

    def __init__(self, in_dim=4096, d=64, num_blocks=8, p=32, seed=0):
        super().__init__()
        self.in_dim = in_dim
        self.d = d
        self.num_blocks = num_blocks
        self.p = p
        self.n_patches = max(1, in_dim // p)
        g = torch.Generator().manual_seed(seed)
        self.ln = nn.LayerNorm(p)
        self.proj_in = nn.Linear(p, d)
        self.mlp = nn.Sequential(
            nn.Linear(p, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.pos_emb = nn.Parameter(torch.randn(self.n_patches, d, generator=g) * 0.1)
        self.proj = nn.Linear(self.n_patches * d, num_blocks * 8)
        self._seed_linear_init(seed)

    def _seed_linear_init(self, seed):
        """Deterministic parameter init for all Linear layers.

        Default nn.Linear init consumes the GLOBAL RNG, making module output
        nondeterministic across processes (C7 flake: P@1 0.8281 vs 0.8906 on
        the same recipe). A sealed carrier must be reproducible.
        """
        g = torch.Generator().manual_seed(seed + 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02, generator=g)
                if m.bias is not None:
                    m.bias.data.uniform_(-0.02, 0.02, generator=g)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.n_patches, self.p).float()
        # Residual: raw-amplitude branch preserves patch occupancy and mean
        # offsets (LayerNorm alone collapses constant patches / uniform-offset
        # class signals — C3/C7 defect).
        raw = self.proj_in(x)                     # [B, n_patches, d]
        normed = self.mlp(self.ln(x))             # normalized feature branch
        t = raw + normed + self.pos_emb.unsqueeze(0)
        out = self.proj(t.reshape(B, -1)).view(B, self.num_blocks, 8)
        return F.normalize(out, p=2, dim=-1)


class SinglePassHorizon(nn.Module):
    """Single-pass non-autoregressive K-step horizon.

    psi [B, nb, 8] -> [B, K, nb, 8] via one vectorized low-rank coupling
    (V, W in [D, rank]; no dense [D, D]) with per-step gains, then per-block
    L2 sphere.
    """

    def __init__(self, d=64, rank=8, K=8, num_blocks=8, seed=0):
        super().__init__()
        self.D = num_blocks * 8
        self.K = K
        self.num_blocks = num_blocks
        g = torch.Generator().manual_seed(seed)
        self.V = nn.Parameter(torch.randn(self.D, rank, generator=g) * 0.1)
        self.W = nn.Parameter(torch.randn(self.D, rank, generator=g) * 0.1)
        self.gains = nn.Parameter(torch.randn(K, generator=g) * 0.1)

    def forward(self, psi):
        B = psi.shape[0]
        flat = psi.reshape(B, self.D)
        proj = (flat @ self.W) @ self.V.T  # [B, D]
        out = flat.unsqueeze(1) + self.gains.view(1, self.K, 1) * proj.unsqueeze(1)
        out = out.view(B, self.K, self.num_blocks, 8)
        return F.normalize(out, p=2, dim=-1)


def sagnac_delta(a, b):
    """Normalized Sagnac delta = 1 - cos(a, b), bounded [0, 2]."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).clamp(-1.0, 1.0)
    return 1.0 - sim.squeeze(0)


def veto(delta):
    return float(delta) > SAGNAC_TAU


def efe_select(roll, goal):
    """EFE per step = mean-over-blocks Sagnac distance to goal; argmin over K."""
    B, K, nb, _ = roll.shape
    gn = torch.linalg.vector_norm(goal, dim=-1, keepdim=True).clamp_min(1e-8)  # [nb, 1]
    rn = torch.linalg.vector_norm(roll, dim=-1, keepdim=True).clamp_min(1e-8)  # [B,K,nb,1]
    sim = (roll * goal[None, None]).sum(-1) / (rn.squeeze(-1) * gn[None, None, :, 0])
    delta = 1.0 - sim.mean(-1)  # [B, K]
    return delta.argmin(dim=-1)


def _verdict(gates):
    if not gates.get("G1"):
        return "F10_LIVE_ENGINE_BLOCKED"
    if all(gates.get(k) for k in ("G1", "G2", "G3", "G4")):
        return "F10_LIVE_LOOP_VERIFIED"
    for name in ("G2", "G3", "G4"):
        if not gates.get(name):
            return "F10_GATE_{}_FAILED".format(name)
    return "F10_INDETERMINATE"


def write_receipt(path, gates, telemetry, meta):
    data = {
        "schema": "f10-live-engine.v1",
        "gates": gates,
        "telemetry": telemetry,
        "verdict": _verdict(gates),
        "meta": meta,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(data, indent=2))
    return data


def _to_flat(frame):
    arr = np.asarray(frame)
    return arr.reshape(-1).astype(np.float32)


def _safe_score(outcome):
    try:
        if isinstance(outcome, dict):
            return float(outcome.get("score", 0.0))
        if hasattr(outcome, "score"):
            return float(outcome.score)
        if isinstance(outcome, (tuple, list)) and len(outcome) >= 2 and isinstance(outcome[1], (int, float)):
            return float(outcome[1])
    except Exception:
        pass
    return 0.0


def _safe_solved(outcome):
    try:
        if isinstance(outcome, dict):
            return int(outcome.get("levels_completed", 0) or 0)
        if hasattr(outcome, "levels_completed"):
            return int(outcome.levels_completed or 0)
    except Exception:
        pass
    return 0


def run_gauntlet(env_names, steps=60, seed=20260908, out_dir=None,
                 receipt_out=None, _force_enabled=False):
    require_f10_enabled(_force_enabled=_force_enabled)
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(receipt_out) if receipt_out else out_dir / "f10_gates_receipt.json"

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=seed).to(device)
    horizon = SinglePassHorizon(d=64, rank=32, K=8, num_blocks=8, seed=seed).to(device)

    telemetry = {
        "envs": list(env_names),
        "steps": 0,
        "resets": 0,
        "mean_latency_ms": None,
        "sagnac_mean": None,
        "progress": 0.0,
        "solved": 0,
        "reason": None,
    }

    def fail_closed(reason):
        telemetry["reason"] = reason
        gates = {"G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": 8, "p": 32, "device": device},
        )

    try:
        from arc_agi import Arcade
        arcade = Arcade()
    except Exception as exc:  # live environment unreachable -> BLOCKED_INFRA
        return fail_closed("arcade_unavailable: {!r}".format(exc))

    latencies, sagnacs, progress, solved = [], [], 0.0, 0
    steps_done = 0
    try:
        for name in env_names:
            game = arcade.make(name)
            if game is None:  # production fail-closed: make returns None on download/API failure
                return fail_closed("arcade_make_returned_none: {!r}".format(name))
            obs = game.reset()
            if obs is None or not getattr(obs, "frame", None):
                return fail_closed("null_initial_frame: {!r}".format(name))
            goal = None
            prev_score = 0.0
            for _ in range(steps):
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
                sel = efe_select(roll, goal).item()
                sagnacs.append(float(sagnac_delta(roll[0, sel], goal).item()))
                actions = list(game.action_space)
                action = actions[sel % max(1, len(actions))]
                obs = game.step(action)  # single-return step: next obs/outcome
                latencies.append((time.perf_counter() - t_start) * 1000.0)
                steps_done += 1
                # Terminal boundary: step returned None or GAME_OVER (production
                # pattern production_arc_run.py:2318-2321). Reset and continue
                # interactively; count resets for telemetry.
                terminal = obs is None or (
                    getattr(obs, "state", None) and obs.state.name == "GAME_OVER")
                if terminal:
                    telemetry["resets"] += 1
                    obs = game.reset()
                    if obs is None or not getattr(obs, "frame", None):
                        break  # cannot continue this env
                    goal = None
                    prev_score = 0.0
                    continue
                score = _safe_score(obs)
                progress += score - prev_score
                prev_score = score
                solved += _safe_solved(obs)
    except Exception as exc:  # live pipeline defect -> K1 class, BLOCKED_INFRA
        telemetry["steps"] = steps_done
        telemetry["reason"] = "live_error: {!r}".format(exc)
        gates = {"G1": False, "G2": False, "G3": False, "G4": False}
        return write_receipt(
            receipt_path, gates, telemetry,
            meta={"seed": seed, "K": 8, "p": 32, "device": device},
        )

    mean_latency = float(np.mean(latencies)) if latencies else None
    sagnac_mean = float(np.mean(sagnacs)) if sagnacs else None
    telemetry.update({
        "steps": steps_done,
        "mean_latency_ms": mean_latency,
        "sagnac_mean": sagnac_mean,
        "progress": float(progress),
        "solved": int(solved),
    })

    g1 = steps_done >= steps * len(env_names) and mean_latency is not None and mean_latency <= 50.0
    g2 = solved > 0
    g3 = sagnac_mean is not None and sagnac_mean <= SAGNAC_TAU
    g4 = float(progress) > 0.0
    gates = {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    return write_receipt(
        receipt_path, gates, telemetry,
        meta={"seed": seed, "K": 8, "p": 32, "device": device},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True, help="comma-separated env names")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out-dir", default="/tmp/henri_f10_live")
    ap.add_argument("--receipt-out", default=None)
    args = ap.parse_args()

    receipt = run_gauntlet(
        env_names=[e.strip() for e in args.envs.split(",") if e.strip()],
        steps=args.steps,
        seed=args.seed,
        out_dir=args.out_dir,
        receipt_out=args.receipt_out,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
