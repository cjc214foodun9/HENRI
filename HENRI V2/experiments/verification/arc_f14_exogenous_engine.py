"""Carrier F14 — Exogenous Goal Ingress & Non-Degenerate Waypoint Steering.

Replaces F13's self-referential first-frame goal with an exogenous goal
synthesized from demonstration pairs:

    W_task = StiefelRetract( (1/M) sum_i Psi_Y,i otimes Psi_X,i^T )
    Psi_goal = Normalize( W_task @ Psi_0 )
    PG1: |<Psi_0, Psi_goal>| <= 0.90  (fail-closed pre-flight)

Tiers (directive HENRI-DIR-2026-08-F13-POSTMORTEM-EXOGENOUS-GOAL):
  Tier 1  Slerp waypoint: Psi_wp(tau) on the geodesic (signed theta).
  Tier 2  Vectorized K=8 macro beam search: J = |<hat Psi_{t+8}, Psi_wp>|
          - alpha * sum_k Sagnac(k).
  Tier 3  Commit first action; signed goal-convergence valence
          dnu_t = |<Psi_{t+1}, Psi_wp>| - |<Psi_t, Psi_wp>|.
  Tier 4  Hebbian creep M_a <- Normalize(M_a + eta * dnu_t * Psi_t);
          zero-valence guard (literal M + 0*Psi = M).

Fail-closed gates: PG1 pre-flight, G1 latency <= 5.0 ms, G2 >= 1 solved,
G3 mean dnu_goal >= +0.02, G4 Sagnac <= 0.05.

Pre-registered deviation D1: the live Arcade substrate exposes NO public
demonstration pairs (OBSERVED 2026-08-31: wrapper/inner game/arc_agi package/
env files all zero demo surfaces) and no provenance-pinned ingress manifest
exists in repo or Drive inbox. Without --ingress-manifest the gauntlet
fail-closes with F14_BLOCKED_NO_PUBLIC_DEMOS (zero steps) — honest negative,
never fabricated demos. With a manifest, demos resolve via
arc_public_ingress.resolve_demos (exact SHA-256 provenance).

Flag: HENRI_F14_EXOGENOUS=1 (fail-closed otherwise).
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

# ---------------------------------------------------------------------------
# Gates (directive section 4)
# ---------------------------------------------------------------------------
LATENCY_BUDGET_MS = 5.0   # G1
SAGNAC_TAU_F14 = 0.050    # G4
G3_MIN_DNU = 0.0200       # G3 (directive: >= +0.0200)
MAX_INITIAL_OVERLAP = 0.90  # PG1 (directive: <= 0.9000)
DEFAULT_TAU = 0.25
DEFAULT_HORIZON = 8
DEFAULT_ALPHA = 0.05
DEFAULT_BEAM = 8
DEFAULT_ENVS = [
    "ar25-0c556536", "sc25-635fd71a", "tr87-cd924810", "cd82-fb555c5d",
    "lp85-305b61c3", "wa30-ee6fef47", "ft09-0d8bbf25", "g50t-5849a774",
    "sk48-d8078629", "bp35-0a0ad940", "ka59-38d34dbb", "sb26-7fbdac44",
]  # F10 receipt cohort (12 envs)


def require_f14_enabled(_force_enabled=False):
    if not (_force_enabled or os.environ.get("HENRI_F14_EXOGENOUS") == "1"):
        raise RuntimeError("F14 exogenous engine disabled: set HENRI_F14_EXOGENOUS=1")


def abs_cos(a, b):
    """|cosine| between two flat vectors, clamped to [0, 1]."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).clamp(0.0, 1.0).squeeze(0)


def slerp(a, b, tau):
    """Spherical linear interpolation on S^{D-1} (signed geodesic).

    theta = arccos(<a, b>) (signed; anti-aligned pairs travel through the
    far side of the sphere). When sin(theta) ~ 0, falls back to normalized
    linear interpolation (norm-preserving, no NaN) — deviation D2.

    Returns unit vector for tau in [0, 1]; a and b must be unit vectors.
    """
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    cos_theta = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).clamp(-1.0, 1.0).squeeze(0)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    if float(sin_theta.abs()) < 1e-8:
        # Nearly parallel or anti-parallel: linear interpolation is the
        # limiting geodesic (anti-parallel handled deterministically).
        w = (1.0 - tau) * a + tau * b
        n = torch.linalg.vector_norm(w).clamp_min(1e-8)
        return (w / n).reshape(-1)
    w = (torch.sin((1.0 - tau) * theta) / sin_theta) * a + (
        torch.sin(tau * theta) / sin_theta
    ) * b
    return F.normalize(w, p=2, dim=-1)


def stiefel_retract(M):
    """Polar decomposition U V^T (orthogonal projection onto O(D)).

    For a square M, SVD M = U S V^T; polar = U V^T. Deterministic.
    """
    u, _, vt = torch.linalg.svd(M)
    return u @ vt


def synthesize_goal(demo_pairs, psi0, ingress=None):
    """W_task = StiefelRetract((1/M) sum_i Psi_Y,i otimes Psi_X,i^T).

    demo_pairs: list of (x, y) 2D grid numpy arrays (ARC train pairs).
    psi0: Tensor [D] initial observation wave. When ingress is provided,
    each grid is encoded to a wave via PatchIngress (flat -> [8, 8] -> [64])
    so the functor operates in WAVE space (exogenous terminal-target
    semantics). Returns (W, goal) with goal = Normalize(W @ psi0).
    """
    if ingress is None:
        # Direct tensor/array fixtures (unit-wave test path): pairs are
        # already [D] vectors.
        mats = []
        for x, y in demo_pairs:
            xv = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)
            yv = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1)
            mats.append(torch.outer(yv, xv))
        M = torch.stack(mats).mean(dim=0)
        W = stiefel_retract(M)
        goal = F.normalize((W @ psi0.reshape(-1).float()), p=2, dim=-1)
        return W, goal

    mats = []
    for x, y in demo_pairs:
        xw = _grid_wave(ingress, x)
        yw = _grid_wave(ingress, y)
        mats.append(torch.outer(yw, xw))
    M = torch.stack(mats).mean(dim=0)
    W = stiefel_retract(M)
    goal = F.normalize((W @ psi0.reshape(-1).float()), p=2, dim=-1)
    return W, goal


def _grid_wave(ingress, grid):
    """Encode a 2D grid to a flat [D] wave via PatchIngress (flat -> [1,8,8])."""
    arr = np.asarray(grid, dtype=np.float32)
    flat = torch.as_tensor(arr.reshape(-1), dtype=torch.float32, device=next(ingress.parameters()).device)
    if flat.numel() < 4096:
        flat = F.pad(flat, (0, 4096 - flat.numel()))
    else:
        flat = flat[:4096]
    with torch.no_grad():
        psi_b = ingress(flat.unsqueeze(0))
    return psi_b[0].detach().reshape(-1)


def pg1_pass(psi0, goal, max_overlap=MAX_INITIAL_OVERLAP):
    """PG1: |<Psi_0, Psi_goal>| <= max_overlap. Fail-closed if False."""
    overlap = float(abs_cos(psi0, goal).item())
    return overlap <= max_overlap, overlap


class ExogenousSteeringEngine(nn.Module):
    """Exogenous goal steering engine (Tiers 1-4).

    D_a skew-symmetric (seeded, zero-trainable) => exp(D_a) orthogonal.
    M in R^{n_actions x D} Hebbian prototypes (Tier 4, signed valence).
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
        skew = skew - skew.transpose(-1, -2)
        self.register_buffer("expD", torch.linalg.matrix_exp(skew))  # [A, D, D]
        self.memory = ActionPrototypeMemory(
            n_actions=n_actions, D=D, eta_fast=eta_fast, seed=seed
        )

    def waypoint(self, psi_t, goal, tau=None):
        return slerp(psi_t, goal, self.tau if tau is None else tau)

    def rollout(self, psi, a):
        """Psi_hat_{t+1}(a) = exp(D_a) Psi_t (norm-preserving)."""
        flat = psi.reshape(-1).float()
        return self.expD[a] @ flat

    def beam_search(self, psi, waypoint, candidates, horizon=None, beam=None, alpha=None):
        """Vectorized K-step beam search — F13 C11-verified semantics.

        J = |cos(final, waypoint)| - alpha * sum_k Sagnac(psi_k, waypoint).
        Tracks the full action sequence per beam row; commits a_1* (Tier 3).
        Identical argmax to the naive loop (contract C8).
        Returns (selected_action, best_score).
        """
        horizon = self.horizon if horizon is None else int(horizon)
        beam = self.beam if beam is None else int(beam)
        alpha = self.alpha if alpha is None else float(alpha)
        if not candidates:
            candidates = list(range(self.n_actions))
        cand = torch.as_tensor(list(candidates), dtype=torch.long, device=self.expD.device)
        wp = F.normalize(waypoint.reshape(-1).float().to(self.expD.device), p=2, dim=-1)
        states = F.normalize(psi.reshape(-1).float().to(self.expD.device), p=2, dim=-1).unsqueeze(0)  # [1, D]
        acts = torch.full((1, 0), -1, dtype=torch.long, device=self.expD.device)
        ssum = torch.zeros(1, device=self.expD.device)
        for _ in range(horizon):
            ops = self.expD[cand]  # [A, D, D]
            nxt = torch.einsum("bd,axd->bax", states, ops)  # [B, A, D]
            nxt = F.normalize(nxt, p=2, dim=-1)
            raw = nxt @ wp  # [B, A] signed cosine
            align = raw.abs().clamp(0.0, 1.0)
            sag = (1.0 - raw).clamp(0.0, 2.0)
            jp = align - alpha * (ssum[:, None] + sag)  # [B, A]
            flat = jp.reshape(-1)
            k = min(beam, flat.numel())
            top = torch.topk(flat, k)
            idx = top.indices
            states = nxt.reshape(-1, nxt.shape[-1])[idx]  # [k, D]
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
        """dnu = |<psi_next, wp>| - |<psi_t, wp>| (Tier 3, signed)."""
        c_next = float(abs_cos(psi_next, waypoint).item())
        c_t = float(abs_cos(psi_t, waypoint).item())
        return c_next - c_t

    def creep(self, action, delta_nu, psi):
        """Tier 4 Hebbian update with zero-valence guard (D4)."""
        if delta_nu == 0.0:
            return
        self.memory.creep(action, delta_nu, psi)


def _verdict(gates, reason=None):
    if reason and reason.startswith("BLOCKED_NO_PUBLIC_DEMOS"):
        return "F14_BLOCKED_NO_PUBLIC_DEMOS"
    if reason and reason.startswith("DEGENERATE_GOAL"):
        return "F14_PREFLIGHT_DEGENERATE_GOAL"
    if reason is not None and reason.startswith("live_error"):
        return "F14_LIVE_ENGINE_BLOCKED"
    if not gates.get("G1"):
        return "F14_LIVE_ENGINE_BLOCKED"
    if all(gates.get(k) for k in ("G1", "G2", "G3", "G4")):
        return "F14_LIVE_LOOP_VERIFIED"
    for name in ("G2", "G3", "G4"):
        if not gates.get(name):
            return "F14_GATE_{}_FAILED".format(name)
    return "F14_INDETERMINATE"


def write_receipt(path, gates, telemetry, meta):
    data = {
        "schema": "f14-exogenous-engine.v1",
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


def run_gauntlet(env_names=None, steps_per_env=150, seed=20260912,
                 horizon=DEFAULT_HORIZON, tau=DEFAULT_TAU, alpha=DEFAULT_ALPHA,
                 beam=DEFAULT_BEAM, eta_fast=0.05, max_initial_overlap=MAX_INITIAL_OVERLAP,
                 ingress_manifest=None, out_dir=None, receipt_out=None,
                 _force_enabled=False):
    """F14 live gauntlet (directive command).

    Fail-closed D1: without a provenance-pinned ingress manifest, the
    goal source has no data path -> F14_BLOCKED_NO_PUBLIC_DEMOS, zero steps.
    With a manifest, demos resolve per env (exact SHA), goal synthesized,
    PG1 asserted per env (any violation -> PRE-FLIGHT KILL).
    """
    require_f14_enabled(_force_enabled=_force_enabled)
    env_names = list(env_names) if env_names else list(DEFAULT_ENVS)
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(receipt_out) if receipt_out else out_dir / "f14_gates_receipt.json"

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
        "goal_source": "functor" if ingress_manifest else "NONE",
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
                  "ingress_manifest": str(ingress_manifest) if ingress_manifest else None},
        )

    # --- D1 pre-flight: no manifest -> no goal source -> fail closed --------
    if not ingress_manifest:
        return fail_closed("BLOCKED_NO_PUBLIC_DEMOS: no --ingress-manifest; "
                           "live Arcade exposes no public demo pairs (OBSERVED 2026-08-31)")

    # --- Resolve manifest + demos -------------------------------------------
    from arc_public_ingress import resolve_demos

    manifest_path = str(ingress_manifest)
    demo_map = {}
    for name in env_names:
        res = resolve_demos(manifest_path, name)
        if not res.ok:
            return fail_closed(
                "BLOCKED_NO_PUBLIC_DEMOS: manifest resolution {}: {} for {}".format(
                    res.status, res.reason, name))
        demo_map[name] = res.demo_pairs

    # --- Construct substrate -------------------------------------------------
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32, seed=seed).to(device)
    horizon_inst = SinglePassHorizon(d=64, rank=32, K=8, num_blocks=8, seed=seed).to(device)
    engine = ExogenousSteeringEngine(
        D=64, n_actions=8, seed=seed,
        horizon=horizon, beam=beam, tau=tau, alpha=alpha, eta_fast=eta_fast,
    ).to(device)

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
    pg1_overlaps = {}
    try:
        for name in env_names:
            game = arcade.make(name)
            if game is None:
                return fail_closed("arcade_make_returned_none: {!r}".format(name))
            obs = game.reset()
            if obs is None or not getattr(obs, "frame", None):
                return fail_closed("null_initial_frame: {!r}".format(name))
            prev_levels = _safe_levels(obs)

            # --- Exogenous goal synthesis + PG1 ------------------------------
            frame0 = obs.frame[0]
            raw0 = torch.as_tensor(_to_flat(frame0), dtype=torch.float32, device=device)
            if raw0.numel() < 4096:
                raw0 = F.pad(raw0, (0, 4096 - raw0.numel()))
            else:
                raw0 = raw0[:4096]
            psi0_b = ingress(raw0.unsqueeze(0))
            psi0 = psi0_b[0].detach()
            W_task, goal = synthesize_goal(demo_map[name], psi0, ingress=ingress)
            ok, overlap = pg1_pass(psi0, goal, max_overlap=max_initial_overlap)
            pg1_overlaps[name] = float(overlap)
            if not ok:
                telemetry["steps"] = steps_done
                telemetry["reason"] = "DEGENERATE_GOAL: env {} overlap {:.4f} > {}".format(
                    name, overlap, max_initial_overlap)
                gates = {"PG1": False, "G1": False, "G2": False, "G3": False, "G4": False}
                telemetry["pg1_overlaps"] = pg1_overlaps
                return write_receipt(
                    receipt_path, gates, telemetry,
                    meta={"seed": seed, "K": horizon, "beam": beam, "tau": tau,
                          "alpha": alpha, "p": 32, "device": device, "eta_fast": eta_fast,
                          "max_initial_overlap": max_initial_overlap,
                          "ingress_manifest": str(ingress_manifest)},
                )

            for _ in range(steps_per_env):
                frame = obs.frame[0]
                raw = torch.as_tensor(_to_flat(frame), dtype=torch.float32, device=device)
                if raw.numel() < 4096:
                    raw = F.pad(raw, (0, 4096 - raw.numel()))
                else:
                    raw = raw[:4096]
                t_start = time.perf_counter()
                psi_b = ingress(raw.unsqueeze(0))  # [1, 8, 8] batched
                psi = psi_b[0]  # [8, 8] flat 64
                psi_s = psi.detach()
                wp = engine.waypoint(psi_s, goal, tau)
                roll = horizon_inst(psi_b)  # G4 instrument

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
                  "ingress_manifest": str(ingress_manifest)},
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
    g4 = sagnac_mean is not None and sagnac_mean <= SAGNAC_TAU_F14
    gates = {"PG1": True, "G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    return write_receipt(
        receipt_path, gates, telemetry,
        meta={"seed": seed, "K": horizon, "beam": beam, "tau": tau,
              "alpha": alpha, "p": 32, "device": device, "eta_fast": eta_fast,
              "max_initial_overlap": max_initial_overlap,
              "ingress_manifest": str(ingress_manifest)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=None, help="comma-separated env names (default: F10 cohort)")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--tau-waypoint", type=float, default=DEFAULT_TAU)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    ap.add_argument("--max-initial-overlap", type=float, default=MAX_INITIAL_OVERLAP)
    ap.add_argument("--eta-fast", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ingress-manifest", default=None,
                    help="provenance-pinned public-ingress manifest path (required for live run)")
    ap.add_argument("--out-dir", default="/tmp/henri_f14_exogenous")
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
        ingress_manifest=args.ingress_manifest,
        out_dir=args.out_dir,
        receipt_out=args.receipt_out,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
