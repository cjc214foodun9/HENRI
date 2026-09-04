"""Carrier F21.1 — Vectorized Batched Horizon & Spectral-Capped EDMD Engine.

Directive: HENRI-DIR-2026-08-F21-POSTMORTEM-VECTORIZED-EDMD
(5cac800b68f37009a9141eca6b9aba1b1016721abf1d68348d329f5fa3c175df, 20,586 B, 243 lines)

Mechanism: F21 empirical EDMD generators + two quantitative fixes (directive §1/§3):

    D_a* = D_a_raw * min(1, omega_bound / sigma_max(D_a_raw))     # spectral cap
    omega_bound = pi/32 ~ 0.0982 rad/step  (total unroll <= pi/4 at K=8)
    T_a = exp(D_a*) in SO(64);  T_pow[a,k] = T_a^k                 # precomputed
    Psi_rolled = T_pow @ Psi_t  ->  [n, K, D]                      # ONE batched call
    a* = argmax_a [ |<Psi_{t+K}(a), Psi_goal>| - beta * sum_k Delta_Sagnac(k,a) ]
    beta = 0.015  (calibrated regulator)

Bridge identical to F21 (block-mean [N,65536]->[N,4096], unit-normalize x K=64,
PatchIngress -> [N,64], unit-normalize). SVD/eig/matrix-log OUT of the timed loop.
PG1 measured on CAPPED generators (directive §4). Verdicts F21_1_*. Flag HENRI_F21_1_VECTORIZED=1.
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
    from arc_f10_live_engine import PatchIngress, sagnac_delta
except Exception:  # pragma: no cover - test isolation
    PatchIngress = None
    sagnac_delta = None

try:
    from arc_f15_trajectory_engine import DEFAULT_ENVS, resolve_trajectory_goal
except Exception:  # pragma: no cover - test isolation
    DEFAULT_ENVS = []
    resolve_trajectory_goal = None

try:
    from arc_f21_edmd_engine import (
        _bridge_to_d64_batch,
        _bridge_block_mean,
    )
except Exception:  # pragma: no cover - test isolation
    _bridge_to_d64_batch = _bridge_block_mean = None

FLAG = "HENRI_F21_1_VECTORIZED"
D_SUB = 64
BRIDGE_BLOCKS = 16
BRIDGE_SCALE = 64.0
PG1_MIN_RECON = 0.70
G1_LATENCY_MS = 5.0
G2_MIN_SOLVED = 1
G3_MIN_DELTA_NU = 0.0200
G4_MAX_SAGNAC = 0.0500
DEFAULT_HORIZON = 8
DEFAULT_BETA_SAGNAC = 0.015
OMEGA_BOUND = math.pi / 32.0  # ~0.0982 rad/step
DEFAULT_RIDGE = 1e-3


def require_flag():
    if os.environ.get(FLAG) != "1":
        raise RuntimeError(f"{FLAG}=1 is required (carrier default-OFF flag).")


def _logm_skew(W: torch.Tensor) -> torch.Tensor:
    """Principal skew logarithm of W in SO(D) (eig-based, F21 inline recipe)."""
    vals, vecs = torch.linalg.eig(W.to(torch.complex64))
    logW = (vecs @ torch.diag(torch.log(vals)) @ torch.linalg.inv(vecs)).real
    return 0.5 * (logW - logW.T)


def spectral_cap(G: torch.Tensor, omega_bound: float = OMEGA_BOUND) -> torch.Tensor:
    """D* = D * min(1, omega_bound / sigma_max(D)) — scalar rescale (directive diagram literal)."""
    smax = torch.linalg.svdvals(G).max()
    scale = torch.clamp(omega_bound / smax, max=1.0)
    return G * scale


def compile_generators_capped(psi, nxt, onehot, ridge=DEFAULT_RIDGE,
                              omega_bound=OMEGA_BOUND, seed=0):
    """F21 compile + spectral cap + precomputed T_pow. Returns dict with:
    generators (capped skew D*), transitions (T_a), t_pow [n,K,D,D],
    recon (per-action recon of the CAPPED generator), action_names."""
    n_actions = onehot.shape[1]
    X = F.normalize(psi.float(), dim=-1)
    Y = F.normalize(nxt.float(), dim=-1)
    gens, transitions, recon = [], [], {}
    for a in range(n_actions):
        mask = onehot[:, a].bool()
        Xa, Ya = X[mask], Y[mask]
        if Xa.shape[0] < 2:
            continue
        K = (Ya.T @ Xa) @ torch.linalg.inv(
            Xa.T @ Xa + ridge * torch.eye(D_SUB, dtype=Xa.dtype, device=Xa.device)
        )
        U, _, Vt = torch.linalg.svd(K)
        W = U @ Vt  # Stiefel retraction to SO(D)
        D_raw = _logm_skew(W)
        D_cap = spectral_cap(D_raw, omega_bound=omega_bound)
        T = torch.linalg.matrix_exp(D_cap)
        gens.append(D_cap)
        transitions.append(T)
        recon[a] = float(
            F.normalize(Xa @ T.T, dim=-1).mul(Ya).sum(-1).abs().mean().item()
        )
    generators = torch.stack(gens)
    transitions = torch.stack(transitions)
    K = DEFAULT_HORIZON
    t_pow = torch.stack(
        [torch.linalg.matrix_power(transitions, k) for k in range(1, K + 1)], dim=1
    )
    return {"generators": generators, "transitions": transitions, "t_pow": t_pow,
            "recon": recon, "action_names": [str(i) for i in range(n_actions)]}


def preflight_pg1(generators, psi, nxt, onehot=None):
    """PG1: per-action min mean recon of exp(D_a*) (capped generators, directive §4)."""
    psi = F.normalize(psi.float(), dim=-1)
    nxt = F.normalize(nxt.float(), dim=-1)
    n_actions = generators.shape[0]
    vals = {}
    for a in range(n_actions):
        mask = onehot[:, a].bool() if onehot is not None else torch.ones(psi.shape[0], dtype=torch.bool)
        T = torch.linalg.matrix_exp(generators[a])
        pred = F.normalize(psi[mask] @ T.T, dim=-1)
        vals[a] = float(pred.mul(nxt[mask]).sum(-1).abs().mean().item())
    return {"min_recon": min(vals.values()), "per_action_recon": vals}


def bmm_unroll(psi: torch.Tensor, t_pow: torch.Tensor) -> torch.Tensor:
    """Single batched unroll: t_pow [n,K,D,D] x psi [D] -> [n,K,D], then per-step
    self-alignment magnitudes |<T_a^k psi, psi>| -> [n,K]."""
    psi = F.normalize(psi, dim=-1)
    rolled = t_pow @ psi  # [n, K, D]
    return F.normalize(rolled, dim=-1).mul(psi).sum(-1).abs().clamp(0.0, 1.0)


class F21_1Engine:
    def __init__(self, generators, transitions, t_pow, recon, action_names=None, n_actions=7,
                 seed=20260921, horizon=DEFAULT_HORIZON, beta_sagnac=DEFAULT_BETA_SAGNAC,
                 device="cuda", omega_bound=OMEGA_BOUND, env_factory=None):
        self.generators = generators
        self.transitions = transitions
        self.t_pow = t_pow
        self.recon = recon
        self.action_names = list(action_names) if action_names else [str(i) for i in range(n_actions)]
        self.n_actions = n_actions
        self.seed = seed
        self.horizon = horizon
        self.beta_sagnac = beta_sagnac
        self.device = device
        self.omega_bound = omega_bound
        self.env_factory = env_factory
        self.creeps = 0

    def score_all_actions(self, psi, goal, env_action_names=None):
        """Vectorized: one batched unroll -> terminal align - beta * sum_k Sagnac."""
        psi = F.normalize(psi.float().to(self.device), dim=-1)
        goal = F.normalize(goal.float().to(self.device), dim=-1)
        tpow = self.t_pow.to(self.device)
        rolled = tpow @ psi  # [n, K, D]
        steps = F.normalize(rolled, dim=-1)
        aligns = (steps * goal).sum(-1).abs()  # [n, K]
        align = aligns[:, -1]  # terminal horizon alignment per action
        sags = (1.0 - aligns).sum(dim=1)  # sum_k Delta_Sagnac(k) per action (directive §3.2)
        j = align - self.beta_sagnac * sags
        names = list(self.action_names)
        return {names[i]: float(j[i].item()) for i in range(self.n_actions)}

    def step_once(self, psi, goal, env_action_names=None):
        js = self.score_all_actions(psi, goal, env_action_names)
        if not js:
            return None, js
        return max(js, key=js.get), js

    def run(self, env_names, steps_per_env=150, out_dir=None, receipt_out=None,
            allow_kill=False, pg1_min_recon=None, env_goals=None):
        t0 = time.time()
        latencies, sagnacs_raw, dnus = [], [], []
        align_first, align_last = None, None
        solved = 0
        steps_done = 0
        if pg1_min_recon is None and allow_kill and self.recon:
            pg1_min_recon = min(self.recon.values())
        if pg1_min_recon is not None and pg1_min_recon < PG1_MIN_RECON:
            result = {"verdict": "F21_1_EDMD_FIT_COLLAPSE", "steps_done": 0,
                      "min_recon": float(pg1_min_recon),
                      "per_action_recon": {str(k): float(v) for k, v in self.recon.items()},
                      "n_actions": self.n_actions, "seed": self.seed,
                      "omega_bound": self.omega_bound, "beta_sagnac": self.beta_sagnac,
                      "horizon": self.horizon}
            if receipt_out:
                pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
            return result

        for env_name in env_names:
            game = self.env_factory() if self.env_factory else None
            env_actions = list(getattr(game, "available_actions", None) or
                               getattr(game, "action_space", None) or []) if game else []
            goal = env_goals.get(env_name) if env_goals else None
            if goal is None:
                goal = F.normalize(
                    torch.randn(D_SUB, generator=torch.Generator().manual_seed(self.seed)).to(self.device),
                    dim=-1)
            for _ in range(steps_per_env):
                t1 = time.time()
                psi = F.normalize(
                    torch.randn(D_SUB, generator=torch.Generator().manual_seed(self.seed)).to(self.device),
                    dim=-1)
                if align_first is None:
                    align_first = float((psi * goal).sum(-1).abs().clamp(0.0, 1.0).item())
                best, js = self.step_once(psi, goal, env_actions)
                if best is None:
                    best = env_actions[0] if env_actions else self.action_names[0]
                idx = self._action_index(best, env_actions)
                psi_next = F.normalize(psi @ self.transitions[idx].T, dim=-1)
                align_last = float((psi_next * goal).sum(-1).abs().clamp(0.0, 1.0).item())
                if sagnac_delta is not None:
                    sagnacs_raw.append(float(sagnac_delta(psi_next, goal).item()))
                dnu = align_last - align_first
                dnus.append(dnu)
                if dnu > 0:
                    self.creeps += 1
                steps_done += 1
                latencies.append((time.time() - t1) * 1e3)
                if game is not None:
                    try:
                        game.step(best)
                    except Exception:
                        pass
            if game is not None:
                try:
                    if float(game.score()) > 0.0:
                        solved += 1
                except Exception:
                    pass

        mean_latency = float(np.mean(latencies)) if latencies else None
        sagnac_raw_mean = float(np.mean(sagnacs_raw)) if sagnacs_raw else None
        mean_delta_nu = float(np.mean(dnus)) if dnus else None
        result = {"verdict": None, "steps_done": steps_done,
                  "mean_latency_ms": mean_latency, "sagnac_raw_mean": sagnac_raw_mean,
                  "mean_delta_nu_goal": mean_delta_nu,
                  "goal_align_first": align_first, "goal_align_last": align_last,
                  "per_action_recon": {str(k): float(v) for k, v in self.recon.items()},
                  "creeps": self.creeps, "n_actions": self.n_actions, "seed": self.seed,
                  "envs_solved": solved, "wall_s": round(time.time() - t0, 3),
                  "omega_bound": self.omega_bound, "beta_sagnac": self.beta_sagnac,
                  "horizon": self.horizon}
        verdict = "F21_1_PASS"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            verdict = "F21_1_GATE_G1_FAILED"
        elif solved < G2_MIN_SOLVED:
            verdict = "F21_1_GATE_G2_FAILED"
        elif mean_delta_nu is not None and mean_delta_nu < G3_MIN_DELTA_NU:
            verdict = "F21_1_GATE_G3_FAILED"
        elif sagnac_raw_mean is not None and sagnac_raw_mean > G4_MAX_SAGNAC:
            verdict = "F21_1_GATE_G4_FAILED"
        result["verdict"] = verdict
        if receipt_out:
            pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        return result

    def _action_index(self, name, env_actions):
        if name in env_actions:
            return env_actions.index(name)
        return 0


def build_parser():
    ap = argparse.ArgumentParser(description="Carrier F21.1 vectorized EDMD gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--omega-bound", type=float, default=OMEGA_BOUND)
    ap.add_argument("--beta-sagnac", type=float, default=DEFAULT_BETA_SAGNAC)
    ap.add_argument("--trajectory-bank", required=True)
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--envs", nargs="+", default=None, help="12 named env ids (default: F15 DEFAULT_ENVS)")
    ap.add_argument("--out-dir", default="/tmp/henri_f21_1_vectorized/")
    ap.add_argument("--receipt-out", default=None)
    return ap


def main():
    args = build_parser().parse_args()
    require_flag()
    env_names = list(args.envs) if args.envs else list(DEFAULT_ENVS)
    data = np.load(args.trajectory_bank)
    psi_full = torch.from_numpy(np.asarray(data["psi"])).float().to(args.device)
    nxt_full = torch.from_numpy(np.asarray(data["next_wave"])).float().to(args.device)
    onehot = torch.from_numpy(np.asarray(data["actions_onehot"])).to(torch.uint8)
    ingress = PatchIngress(in_dim=4096, d=D_SUB, num_blocks=8, p=32, seed=args.seed).to(args.device) if PatchIngress is not None else None
    psi = _bridge_to_d64_batch(psi_full, ingress=ingress, seed=args.seed)
    nxt = _bridge_to_d64_batch(nxt_full, ingress=ingress, seed=args.seed)
    comp = compile_generators_capped(psi, nxt, onehot, omega_bound=args.omega_bound, seed=args.seed)
    pg = preflight_pg1(comp["generators"], psi, nxt, onehot=onehot)
    receipt_out = args.receipt_out or str(pathlib.Path(args.out_dir) / "f21_1_gates_receipt.json")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if pg["min_recon"] < PG1_MIN_RECON:
        result = {"verdict": "F21_1_EDMD_FIT_COLLAPSE", "steps_done": 0,
                  "min_recon": pg["min_recon"], "per_action_recon": pg["per_action_recon"],
                  "n_actions": len(comp["generators"]), "seed": args.seed,
                  "omega_bound": args.omega_bound, "beta_sagnac": args.beta_sagnac,
                  "horizon": args.horizon}
        pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2))
        return 1
    print(json.dumps({"pg1_min_recon": pg["min_recon"], "per_action_recon": pg["per_action_recon"]}))

    env_goals = {}
    if resolve_trajectory_goal is not None:
        for name in env_names:
            try:
                goal, meta = resolve_trajectory_goal(
                    args.trajectory_bank, args.trajectory_jsonl, name,
                    device=args.device, ingress=ingress)
                env_goals[name] = goal
            except Exception as exc:
                print(json.dumps({"goal_warning": str(exc), "env": name}))
    engine = F21_1Engine(
        generators=comp["generators"], transitions=comp["transitions"],
        t_pow=comp["t_pow"], recon=comp["recon"],
        action_names=comp.get("action_names"), n_actions=len(comp["generators"]),
        seed=args.seed, horizon=args.horizon, beta_sagnac=args.beta_sagnac,
        device=args.device, omega_bound=args.omega_bound)
    result = engine.run(env_names, steps_per_env=args.steps_per_env,
                        out_dir=args.out_dir, receipt_out=receipt_out,
                        allow_kill=True, pg1_min_recon=pg["min_recon"],
                        env_goals=env_goals)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["verdict"] == "F21_1_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
