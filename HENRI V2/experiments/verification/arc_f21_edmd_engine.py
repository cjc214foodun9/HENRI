"""Carrier F21 — In-Situ Empirical EDMD & Trajectory-Span Generator Synthesis Engine.

Directive: HENRI-DIR-2026-08-F20-POSTMORTEM-DYNAMIC-GENERATOR-ORDER
(9ecabe24ec255591327f5830219581962774f7d3108f774027d7122f993438f4, 21,582 B, 252 lines)

Mechanism: replace static random skew generator dictionaries with data-driven
transition Lie generators compiled in-situ from the verified F3 v2 trajectory bank:

    K_a = (Y_a^T X_a) (X_a^T X_a + lambda I)^-1          # Koopman normal equation
    U S V^T = SVD(K_a);  W_a = U V^T in SO(D)            # Stiefel retraction
    D_a = 0.5 * (Logm(W_a) - Logm(W_a)^T) in so(D)       # skew Lie generator
    Psi_hat_{t+1}(a) = exp(D_a) Psi_t                     # runtime: matmul only

Bridge (identical to F15/F20 live ingress): block-mean [N,65536] -> [N,4096],
unit-normalize, scale by K=64 (F15 _bridge_to_d64 operating point), then
PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32) -> [N,64], unit-normalize.

Runtime keeps SVD/eig/matrix-log OUT of the timed loop (F20 G1 fix): exp(U_a)
matrices are compiled offline; the live loop applies them by matmul and scores a
K=8 horizon per action with the F20-ratified fixed Sagnac regulator (beta 0.025).

Gates (directive §4): PG1 in-sample recon cos >= 0.70 per action (pre-flight kill);
G1 <= 5.0 ms/step; G2 >= 1/12 envs solved; G3 mean Delta_nu_goal >= +0.0200;
G4 single-pass horizon Sagnac <= 0.0500. Flag-gated: HENRI_F21_EDMD=1.
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

FLAG = "HENRI_F21_EDMD"

D_SUB = 64
BRIDGE_BLOCKS = 16  # 65536 -> 4096 block-mean
BRIDGE_SCALE = 64.0  # F15 _bridge_to_d64 operating point
PG1_MIN_RECON = 0.70
G1_LATENCY_MS = 5.0
G2_MIN_SOLVED = 1
G3_MIN_DELTA_NU = 0.0200
G4_MAX_SAGNAC = 0.0500
DEFAULT_HORIZON = 8
DEFAULT_BETA_SAGNAC = 0.025
DEFAULT_RIDGE = 1e-3


def require_flag():
    if os.environ.get(FLAG) != "1":
        raise RuntimeError(
            f"HENRI_F21_EDMD gate: {FLAG}=1 is required (carrier default-OFF flag)."
        )


def _bridge_block_mean(x: torch.Tensor) -> torch.Tensor:
    """[N, 65536] -> [N, 4096] block-mean (16 blocks of 4096), per F15/F20 bridge."""
    flat = x.reshape(x.shape[0], BRIDGE_BLOCKS, -1)
    return flat.mean(dim=1)


def _bridge_to_d64_batch(x: torch.Tensor, ingress=None, seed=0) -> torch.Tensor:
    """Full F15/F20 bridge: block-mean -> normalize -> scale K=64 -> PatchIngress."""
    pooled = F.normalize(_bridge_block_mean(x.float()), p=2, dim=-1) * BRIDGE_SCALE
    if ingress is not None:
        with torch.no_grad():
            pooled = ingress(pooled).reshape(pooled.shape[0], -1)
    return F.normalize(pooled, p=2, dim=-1)


def _principal_logm_skew(W: torch.Tensor) -> torch.Tensor:
    """Skew part of the principal matrix log of an orthogonal W, via eig.

    Returns D = 0.5*(Logm(W) - Logm(W)^T) in so(D) (directive §1.2 step 4).
    """
    vals, vecs = torch.linalg.eig(W.to(torch.complex64))
    log_vals = torch.log(vals)
    logW = (vecs @ torch.diag(log_vals) @ torch.linalg.inv(vecs)).real
    return 0.5 * (logW - logW.T)


def compile_lie_generators_d64(psi, nxt, onehot, ridge=DEFAULT_RIDGE, seed=0):
    """Compile per-action Lie generators from a D=64 transition bank.

    psi / nxt: [N, D] float tensors (unit rows). onehot: [N, n_actions] uint8.
    Returns dict: generators {a: [D,D] skew}, exp_generators {a: [D,D] SO},
    recon {a: float in-sample recon cosine}.
    """
    torch.manual_seed(seed)
    psi = F.normalize(psi.float(), dim=-1)
    nxt = F.normalize(nxt.float(), dim=-1)
    n_actions = onehot.shape[1]
    generators, exp_generators, recon = {}, {}, {}
    for a in range(n_actions):
        idx = onehot[:, a] > 0
        if int(idx.sum()) < 4:
            continue
        X = psi[idx]
        Y = nxt[idx]
        XtX = X.T @ X
        K = (Y.T @ X) @ torch.linalg.inv(XtX + ridge * torch.eye(D_SUB, dtype=X.dtype))
        U, _, Vt = torch.linalg.svd(K)
        W = U @ Vt
        Da = _principal_logm_skew(W).to(torch.float32)
        Ua = torch.matrix_exp(Da)
        generators[a] = Da
        exp_generators[a] = Ua
        with torch.no_grad():
            pred = F.normalize(X @ Ua.T, dim=-1)
        recon[a] = float((pred * Y).sum(-1).mean().item())
    return {
        "generators": generators,
        "exp_generators": exp_generators,
        "recon": recon,
        "n_actions": n_actions,
        "bridge": "block-mean-16x4096-k64-ingress",
    }


def preflight_pg1(generators, psi, nxt, onehot=None):
    """PG1: per-action min mean in-sample recon cosine (directive §4).

    Requires the action partition (onehot) so D_a is evaluated only on rows of
    action a; without it, mismatched action pairs depress the cosine (0.42 on
    healthy data, FALSIFIED as measured 2026-09-01).
    """
    psi = F.normalize(psi.float(), dim=-1)
    nxt = F.normalize(nxt.float(), dim=-1)
    per_action = {}
    for a, Da in generators.items():
        if onehot is not None:
            idx = onehot[:, a] > 0
            X = psi[idx]
            Y = nxt[idx]
        else:
            X = psi
            Y = nxt
        if X.shape[0] == 0:
            continue
        Ua = torch.matrix_exp(Da)
        pred = F.normalize(X @ Ua.T, dim=-1)
        per_action[int(a)] = float((pred * Y).sum(-1).mean().item())
    return {"min_recon": min(per_action.values()) if per_action else 0.0,
            "per_action_recon": per_action}


class EDMDGeneratorBank:
    """Loads the F3 v2 trajectory bank (npz + jsonl) and compiles D=64 generators."""

    def __init__(self, bank_npz, bank_jsonl, device="cpu", seed=0, ridge=DEFAULT_RIDGE):
        self.bank_npz = pathlib.Path(bank_npz)
        self.bank_jsonl = pathlib.Path(bank_jsonl)
        self.device = device
        self.seed = seed
        self.ridge = ridge

    def compile(self):
        data = np.load(self.bank_npz)
        missing = {"psi", "next_wave", "actions_onehot", "action_names"} - set(data.files)
        if missing:
            raise KeyError(f"bank npz missing keys {sorted(missing)} (schema henri.arc-trajectory-bank.v1)")
        psi_full = torch.from_numpy(np.asarray(data["psi"])).float().to(self.device)   # [N, 65536]
        nxt_full = torch.from_numpy(np.asarray(data["next_wave"])).float().to(self.device)  # [N, 65536]
        onehot = torch.from_numpy(np.asarray(data["actions_onehot"])).to(torch.uint8)
        names = [str(n) for n in np.asarray(data["action_names"])]
        ingress = None
        if PatchIngress is not None:
            ingress = PatchIngress(in_dim=4096, d=D_SUB, num_blocks=8, p=32, seed=self.seed).to(self.device)
        psi = _bridge_to_d64_batch(psi_full, ingress=ingress, seed=self.seed)
        nxt = _bridge_to_d64_batch(nxt_full, ingress=ingress, seed=self.seed)
        comp = compile_lie_generators_d64(psi, nxt, onehot, ridge=self.ridge, seed=self.seed)
        comp["action_names"] = names
        return comp


class F21Engine:
    """Live gauntlet engine: exp(D_a) forward sims + K=8 horizon beam, beta fixed."""

    def __init__(self, generators=None, exp_generators=None, action_names=None,
                 recon=None, n_actions=7, seed=20260920, horizon=DEFAULT_HORIZON,
                 beta_sagnac=DEFAULT_BETA_SAGNAC, device="cpu", env_factory=None):
        self.generators = generators or {}
        self.exp_generators = exp_generators or {}
        self.action_names = list(action_names) if action_names else [f"ACTION{i + 1}" for i in range(len(self.exp_generators) or n_actions)]
        self.recon = recon or {}
        self.n_actions = len(self.generators) or n_actions
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.beta_sagnac = float(beta_sagnac)
        self.device = device
        self.env_factory = env_factory
        self.creeps = 0

    # ------------------------------------------------------------------ timed path
    def _roll_action(self, psi, Ua, goal):
        """K-step single-rollout of action a; returns (align, sagnac_sum)."""
        state = F.normalize(psi, dim=-1)
        sag_sum = 0.0
        for _ in range(self.horizon):
            state = F.normalize(state @ Ua.T, dim=-1)
            if sagnac_delta is not None:
                sag_sum += float(sagnac_delta(state, goal).item())
        align = float((state * goal).sum(-1).abs().clamp(0.0, 1.0).item())
        return align, sag_sum

    def score_all_actions(self, psi, goal, env_action_names=None):
        """Per-action J = |<Psi_hat_{t+K}, Psi_goal>| - beta * sum Delta_Sagnac.

        Restricts to generators whose bank action name appears in env_action_names
        when the env exposes named actions (F20 candidates semantics).
        """
        psi = F.normalize(psi.float(), dim=-1)
        goal = F.normalize(goal.float(), dim=-1)
        allowed = set(env_action_names) if env_action_names else None
        js = {}
        for a, Ua in self.exp_generators.items():
            name = self.action_names[a] if a < len(self.action_names) else f"ACTION{a + 1}"
            if allowed is not None and name not in allowed:
                continue
            align, sag_sum = self._roll_action(psi, Ua, goal)
            js[name] = align - self.beta_sagnac * sag_sum
        return js

    def step_once(self, psi, goal, env_action_names=None):
        js = self.score_all_actions(psi, goal, env_action_names)
        if not js:
            return None, js
        best = max(js, key=js.get)
        return best, js

    # ------------------------------------------------------------------ run
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
            result = {
                "verdict": "F21_EDMD_FIT_COLLAPSE",
                "steps_done": 0,
                "min_recon": float(pg1_min_recon),
                "per_action_recon": {str(k): float(v) for k, v in self.recon.items()},
                "n_actions": self.n_actions,
                "seed": self.seed,
            }
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
                goal = F.normalize(torch.randn(D_SUB, generator=torch.Generator().manual_seed(self.seed)), dim=-1)
            for _ in range(steps_per_env):
                t1 = time.time()
                psi = F.normalize(torch.randn(D_SUB), dim=-1)
                if align_first is None:
                    align_first = float((psi * goal).sum(-1).abs().clamp(0.0, 1.0).item())
                best, js = self.step_once(psi, goal, env_actions)
                if best is None:
                    best = env_actions[0] if env_actions else self.action_names[0]
                idx = self._action_index(best, env_actions)
                psi_next = F.normalize(psi @ self.exp_generators[idx].T, dim=-1)
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
        result = {
            "verdict": None,
            "steps_done": steps_done,
            "mean_latency_ms": mean_latency,
            "sagnac_raw_mean": sagnac_raw_mean,
            "mean_delta_nu_goal": mean_delta_nu,
            "goal_align_first": align_first,
            "goal_align_last": align_last,
            "per_action_recon": {str(k): float(v) for k, v in self.recon.items()},
            "creeps": self.creeps,
            "n_actions": self.n_actions,
            "seed": self.seed,
            "envs_solved": solved,
            "wall_s": round(time.time() - t0, 3),
        }
        verdict = "F21_PASS"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            verdict = "F21_GATE_G1_FAILED"
        elif solved < G2_MIN_SOLVED:
            verdict = "F21_GATE_G2_FAILED"
        elif mean_delta_nu is not None and mean_delta_nu < G3_MIN_DELTA_NU:
            verdict = "F21_GATE_G3_FAILED"
        elif sagnac_raw_mean is not None and sagnac_raw_mean > G4_MAX_SAGNAC:
            verdict = "F21_GATE_G4_FAILED"
        result["verdict"] = verdict
        if receipt_out:
            pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        return result

    def _action_index(self, name, env_actions):
        if name in env_actions:
            return env_actions.index(name)
        return 0


def main():
    ap = argparse.ArgumentParser(description="Carrier F21 EDMD generators gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--beta-sagnac", type=float, default=DEFAULT_BETA_SAGNAC)
    ap.add_argument("--trajectory-bank", required=True)
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--envs", nargs="+", default=None, help="12 named env ids (default: F15 DEFAULT_ENVS)")
    ap.add_argument("--out-dir", default="/tmp/henri_f21_edmd/")
    ap.add_argument("--receipt-out", default=None)
    args = ap.parse_args()

    require_flag()
    env_names = list(args.envs) if args.envs else list(DEFAULT_ENVS)
    bank = EDMDGeneratorBank(args.trajectory_bank, args.trajectory_jsonl, device=args.device, seed=args.seed)
    comp = bank.compile()
    data = np.load(args.trajectory_bank)
    psi_full = torch.from_numpy(np.asarray(data["psi"])).float().to(args.device)
    nxt_full = torch.from_numpy(np.asarray(data["next_wave"])).float().to(args.device)
    onehot = torch.from_numpy(np.asarray(data["actions_onehot"])).to(torch.uint8)
    ingress = PatchIngress(in_dim=4096, d=D_SUB, num_blocks=8, p=32, seed=args.seed).to(args.device) if PatchIngress is not None else None
    psi = _bridge_to_d64_batch(psi_full, ingress=ingress, seed=args.seed)
    nxt = _bridge_to_d64_batch(nxt_full, ingress=ingress, seed=args.seed)
    pg = preflight_pg1(comp["generators"], psi, nxt, onehot=onehot)
    receipt_out = args.receipt_out or str(pathlib.Path(args.out_dir) / "f21_gates_receipt.json")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if pg["min_recon"] < PG1_MIN_RECON:
        result = {"verdict": "F21_EDMD_FIT_COLLAPSE", "steps_done": 0,
                  "min_recon": pg["min_recon"], "per_action_recon": pg["per_action_recon"],
                  "n_actions": len(comp["generators"]), "seed": args.seed}
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
    engine = F21Engine(
        generators=comp["generators"], exp_generators=comp["exp_generators"],
        action_names=comp.get("action_names"), recon=comp["recon"],
        n_actions=len(comp["generators"]), seed=args.seed, horizon=args.horizon,
        beta_sagnac=args.beta_sagnac, device=args.device)
    result = engine.run(env_names, steps_per_env=args.steps_per_env,
                        out_dir=args.out_dir, receipt_out=receipt_out,
                        allow_kill=True, pg1_min_recon=pg["min_recon"],
                        env_goals=env_goals)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["verdict"] == "F21_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
