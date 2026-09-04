"""Carrier G4 — Functionally Aligned Sparse Affordance Engine (HENRI V3).

Directive: Sprint_Closeout_Synthesis___Carrier_G4_Master_Directive.md
(HENRI-DIR-2026-09-V3-SPRINT-CLOSEOUT-G2-G3-G4, bb25dfe247..., 18,288 B) +
Project_HENRI_V3_Carrier_G4_Master_Directive___Functional_Consistency_Synthesis.md
(HENRI-DIR-2026-09-V3-CARRIER-G4-FUNCTIONAL-CONSISTENCY, 1203d7d8..., 4,845 B).
Prereg: docs/spec/g4_aligned_affordance_preregistration.md (dcc09dcc..., sealed
#fd47cb46). Branch feat/carrier-g4-aligned-affordance. Seed 20260927.

Mechanism (directive-mandated, sign-corrected per prereg):
  PG3: per-action top-k=64 block masks by per-block displacement variance
       argmax_k Var_t(||Psi_t^(m) - Psi_(t-1)^(m)||).
  Fit: per-(action, topk-block) ridge 8x8 transitions T_a^(m) on MOVING rows
       (stall-cosine label, tau_stall 0.90, flat norm-divided — G2-verified).
  Score (C1 homology): mean quadratic residual over the SAME top-k support,
       r_a = (1/k) Sum_{m in TopK(a)} ||psi_next^(m) - T_a^(m) psi_t^(m)||^2
       — the exact functional minimized during fitting (shared code path).
  Pi_a = sigmoid((theta_a - r_a)/tau_a); theta/tau calibrated from the
       action's moving/blocked residual distributions. SIGN CORRECTION: the
       directive exec-flow sigmoid (mean - theta)/tau would INVERT affordance
       (blocked => Pi->1); implemented as (theta - mean)/tau matching the
       directive's engine code semantics (exp(-mean_loss), higher = more
       affordable). Guarded by contract C13.
  PG1 (binding): min_action_auc >= 0.8800 on the N=128 action-stratified
       subset (18-19 rows/action, seeded draw); full-bank AUC diagnostic.
  PG2: flat norm drift |norm(Psi) - 1| <= 1e-6 on canonicalized rows.
  Live: one-step lagged observed pair per action (causal, zero leakage).
Verdicts G4_*. Flag HENRI_G4_ALIGNED_AFFORDANCE=1 (default-OFF fail-closed).
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
import torch.nn as nn
import torch.nn.functional as F

try:
    from arc_g2_local_gauge_engine import (
        B_FULL,
        BLK,
        FastFullDWaveEncoder,
        stall_cosine_labels,
    )
except Exception:  # pragma: no cover - test isolation
    B_FULL = 8192
    BLK = 8
    FastFullDWaveEncoder = None
    stall_cosine_labels = None

try:
    from arc_g1_topological_engine import (
        compute_auc,
        compile_free_generators_capped,
        _safe_levels,
        DEFAULT_HORIZON,
        G1_LATENCY_MS,
        G2_MIN_SOLVED,
        G3_MIN_DELTA_NU,
        G4_MAX_AFFORDANCE,
        LANGEVIN_TEMP,
        MIN_AUC_SAMPLES,
        MOVING_THRESH,
        OMEGA_BOUND,
        WAYPOINT_ADVANCE_THRESH,
        _bridge_to_d64_single,
        advance_waypoint_index,
        extract_waypoints,
        langevin_escape_tick,
    )
except Exception:  # pragma: no cover - test isolation
    compute_auc = compile_free_generators_capped = None
    _safe_levels = None
    DEFAULT_HORIZON = 8
    G1_LATENCY_MS = 2.0
    G2_MIN_SOLVED = 1
    G3_MIN_DELTA_NU = 0.0150
    G4_MAX_AFFORDANCE = 0.0500
    LANGEVIN_TEMP = 0.50
    MIN_AUC_SAMPLES = 10
    MOVING_THRESH = 0.05
    OMEGA_BOUND = math.pi / 32.0
    WAYPOINT_ADVANCE_THRESH = 0.60
    _bridge_to_d64_single = advance_waypoint_index = None
    extract_waypoints = langevin_escape_tick = None

try:
    from arc_f21_1_vectorized_engine import _bridge_to_d64_batch, PatchIngress
except Exception:  # pragma: no cover - test isolation
    _bridge_to_d64_batch = None
    PatchIngress = None

try:
    from arc_f15_trajectory_engine import DEFAULT_ENVS, resolve_trajectory_goal
except Exception:  # pragma: no cover - test isolation
    DEFAULT_ENVS = []
    resolve_trajectory_goal = None

FLAG = "HENRI_G4_ALIGNED_AFFORDANCE"
SEED = 20260927
TOP_K = 64
RIDGE_G4 = 1e-2
PG1_MIN_AUC = 0.8800
PG2_NORM_TOL = 1e-6
TAU_STALL_G4 = 0.90
N_SUBSET = 128
SUBSET_PER_ACTION = 18
TOTAL_BLOCKS = 8192


def require_flag(flag_name: str) -> None:
    if os.environ.get(flag_name) != "1":
        print(f"BLOCKED: {flag_name} not set (default-OFF)", file=os.sys.stderr)
        raise SystemExit(1)


def per_block_displacement_variance(psi_t: torch.Tensor,
                                    psi_next: torch.Tensor) -> torch.Tensor:
    """[N, M, 8] pair -> [M] Var_t(||psi_next^(m) - psi_t^(m)||)."""
    d = torch.norm(psi_next - psi_t, p=2, dim=-1)          # [N, M]
    return d.var(dim=0)                                    # [M]


def select_topk_blocks(variance: torch.Tensor, k: int = TOP_K) -> torch.Tensor:
    """[M] -> [k] argmax_k indices (PG3 selection rule)."""
    return torch.topk(variance, k=min(k, variance.numel()), dim=-1).indices


def fit_aligned_transitions(psi_t: torch.Tensor, psi_next: torch.Tensor,
                            topk: torch.Tensor,
                            ridge: float = RIDGE_G4) -> dict:
    """Ridge per-block 8x8 transitions on the action's rows (moving subset).

    psi_t/psi_next [N, M, 8]; topk [k]. Returns {m: T_m [8, 8]}.
    T_m minimizes ||Y - X T^T|| (forward operator, pred = T psi):
      T = [solve(X^T X + ridge I, X^T Y)]^T = Y^T X (X^T X)^-1.
    C1 HOMOLOGY: this is the exact functional the score evaluates
    (aligned_mean_quadratic), so fit functional == score functional.
    """
    out = {}
    for m in topk.tolist():
        X = psi_t[:, m, :].double()
        Y = psi_next[:, m, :].double()
        A = X.T @ X + ridge * torch.eye(8, dtype=X.dtype, device=X.device)
        B = X.T @ Y
        T = torch.linalg.solve(A, B).T.float()  # forward operator
        out[int(m)] = T
    return out


def aligned_mean_quadratic(psi_t: torch.Tensor, psi_next: torch.Tensor,
                           transitions: dict) -> torch.Tensor:
    """[N, M, 8] -> [N] mean quadratic residual over the top-k support.

    THE shared functional: identical for fitting and scoring (C1).
    """
    if not transitions:
        return torch.full((psi_t.shape[0],), float("nan"), device=psi_t.device)
    rs = []
    for m, T in transitions.items():
        pred = psi_t[:, m, :] @ T.T
        rs.append(torch.sum((psi_next[:, m, :] - pred) ** 2, dim=-1))
    return torch.stack(rs, dim=-1).mean(dim=-1)


def calibrate_theta_tau(res_moving: torch.Tensor,
                        res_blocked: torch.Tensor) -> tuple:
    """theta = midpoint of the two residual means; tau = half sum of stds."""
    mu_m, mu_b = res_moving.mean(), res_blocked.mean()
    sd = 0.5 * (res_moving.std() + res_blocked.std()).clamp(min=1e-3)
    return float(0.5 * (mu_m + mu_b)), float(sd)


def aligned_affordance_pi(residuals: torch.Tensor, theta: float,
                          tau: float) -> torch.Tensor:
    """Pi = sigmoid((theta - r)/tau): higher affordability as residual falls."""
    return torch.sigmoid((theta - residuals) / max(tau, 1e-6))


def stratified_subset(onehot: torch.Tensor, n_sub: int = SUBSET_PER_ACTION,
                      seed: int = SEED) -> torch.Tensor:
    """[N, A] -> boolean mask of an action-stratified N=128 draw.

    19 rows for the two largest actions, 18 for the rest (126 + 2 = 128).
    Device-safe: indices are moved to the onehot device before gathering.
    """
    n = onehot.shape[0]
    sizes = onehot.sum(0)
    order = torch.argsort(sizes, descending=True)
    counts = {int(a): n_sub for a in range(onehot.shape[1])}
    for a in order[:2].tolist():
        counts[a] += 1
    mask = torch.zeros(n, dtype=torch.bool, device=onehot.device)
    for a in range(onehot.shape[1]):
        idx = torch.nonzero(onehot[:, a].bool()).squeeze(-1)
        g = torch.Generator().manual_seed(seed + a)
        perm = torch.randperm(idx.numel(), generator=g)
        pick = idx[perm.to(idx.device)][: counts[a]]
        mask[pick] = True
    return mask


class G4AlignedEngine:
    """Functionally aligned top-k affordance; kinematics D=64 carried."""

    def __init__(self, transitions_g4, topk_masks, theta, tau,
                 generators, transitions, t_pow, recon,
                 action_names=None, n_actions=7, seed=SEED,
                 horizon=DEFAULT_HORIZON, device="cuda",
                 omega_bound=OMEGA_BOUND,
                 waypoints=None, waypoint_advance_thresh=WAYPOINT_ADVANCE_THRESH,
                 langevin_temp=LANGEVIN_TEMP, tau_stall=TAU_STALL_G4):
        self.transitions_g4 = transitions_g4       # {a: {m: T_m}}
        self.topk_masks = topk_masks               # {a: [k] tensor}
        self.theta = theta                         # [A]
        self.tau = tau                             # [A]
        self.generators = generators
        self.transitions = list(transitions)
        self.t_pow = t_pow
        self.recon = recon
        self.action_names = list(action_names) if action_names else [str(i) for i in range(n_actions)]
        self.n_actions = n_actions
        self.seed = seed
        self.horizon = horizon
        self.device = device
        self.omega_bound = omega_bound
        self.waypoints = list(waypoints) if waypoints else None
        self.waypoint_advance_thresh = waypoint_advance_thresh
        self.langevin_temp = langevin_temp
        self.tau_stall = tau_stall
        self.creeps = 0
        self.waypoint_advances = 0
        self.langevin_escapes = 0
        self.affordance_updates = 0
        self.escape_state = {"steps": 0, "active": False}
        self._wp_idx = 0
        self.last_pair = {}                        # {a: (psi_t, psi_next)} lagged

    def _active_waypoint(self):
        if self.waypoints is None or len(self.waypoints) == 0:
            return F.normalize(
                torch.randn(64, generator=torch.Generator().manual_seed(self.seed)),
                dim=-1)
        return self.waypoints[min(self._wp_idx, len(self.waypoints) - 1)]

    def affordance_residuals(self, psi_full, action_idx, psi_full_next=None):
        """Mean quadratic residual on the action's top-k support (C1)."""
        psi_full = psi_full.float().to(self.device)
        if psi_full.dim() == 2:
            psi_full = psi_full.unsqueeze(0)
        if psi_full_next is None:
            pair = self.last_pair.get(int(action_idx))
            if pair is None:
                return None
            _, psi_full_next = pair
        if not self.transitions_g4.get(int(action_idx)):
            return None
        psi_full_next = psi_full_next.float().to(self.device)
        if psi_full_next.dim() == 2:
            psi_full_next = psi_full_next.unsqueeze(0)
        topk = self.topk_masks[int(action_idx)]
        trans = {int(m): self.transitions_g4[int(action_idx)][int(m)]
                 for m in topk.tolist()}
        return aligned_mean_quadratic(psi_full, psi_full_next, trans)

    def predict_affordance(self, psi_full, psi_full_next=None):
        """Pi [B, A]: sigmoid((theta_a - r_a)/tau_a); lagged pair when no next."""
        psi_full = psi_full.float().to(self.device)
        if psi_full.dim() == 2:
            psi_full = psi_full.unsqueeze(0)
        pis = []
        for a in range(self.n_actions):
            r = self.affordance_residuals(psi_full, a, psi_full_next)
            if r is None:
                pis.append(torch.full((psi_full.shape[0],), 0.5,
                                      device=self.device))
            else:
                pis.append(aligned_affordance_pi(
                    r, float(self.theta[a]), float(self.tau[a])))
        return torch.stack(pis, dim=-1)

    def score_all_actions(self, psi64, psi_full, waypoint=None):
        psi64 = F.normalize(psi64.float().to(self.device), dim=-1)
        wp = F.normalize((waypoint if waypoint is not None else self._active_waypoint())
                         .float().to(self.device), dim=-1)
        rolled = self.t_pow.to(self.device) @ psi64
        steps = F.normalize(rolled, dim=-1)
        aligns = (steps * wp).sum(-1).abs()
        align = aligns[:, -1]
        pi = self.predict_affordance(psi_full)[0]
        j = align * (pi ** self.horizon)
        if self.escape_state.get("active"):
            g = torch.Generator().manual_seed(self.seed + self.escape_state.get("steps", 0))
            noise = torch.sqrt(torch.tensor(2.0 * self.langevin_temp, device=self.device)) * \
                torch.randn(self.n_actions, generator=g).to(self.device)
            j = j + noise
        return {self.action_names[i]: float(j[i].item()) for i in range(self.n_actions)}

    def step_once(self, psi64, psi_full, waypoint=None):
        js = self.score_all_actions(psi64, psi_full, waypoint)
        if not js:
            return None, js
        return max(js, key=js.get), js

    def update_online_affordance(self, psi_full, action_idx, psi_full_next,
                                 eta=None):
        """Store the OBSERVED pair (lagged, causal); no weight mutation."""
        psi_full = psi_full.float().to(self.device)
        psi_full_next = psi_full_next.float().to(self.device)
        self.last_pair[int(action_idx)] = (psi_full, psi_full_next)
        self.affordance_updates += 1
        return 0.0

    def g4_single_pass(self, psi64, psi_full, action_idx, psi64_actual):
        """Aligned residual (per-dimension normalized) vs actual transition."""
        pair = self.last_pair.get(int(action_idx))
        if pair is None:
            return 1.0
        _, psi_full_next = pair
        r = self.affordance_residuals(psi_full, action_idx, psi_full_next)
        if r is None or not torch.isfinite(r).all():
            return 1.0
        return float((r / BLK).clamp(0.0, 2.0).mean().item())

    def advance_waypoint_index(self, psi, waypoint, k):
        if advance_waypoint_index is None:
            return k
        return advance_waypoint_index(psi, waypoint, k, thresh=self.waypoint_advance_thresh)

    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        if steps_done > 0 and updates == 0:
            return "G4_NO_AFFORDANCE_ENGAGEMENT"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "G4_GATE_G1_FAILED"
        if solved < G2_MIN_SOLVED:
            return "G4_GATE_G2_FAILED"
        if mean_delta_nu is not None and mean_delta_nu < G3_MIN_DELTA_NU:
            return "G4_GATE_G3_FAILED"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "G4_GATE_G4_FAILED"
        return "G4_ALIGNED_AFFORDANCE_VERIFIED"

    def run_gauntlet(self, env_names, fast_encoder, steps_per_env=150,
                     seed=SEED, trajectory_bank=None, trajectory_jsonl=None,
                     ingress=None, out_dir=None, receipt_out=None,
                     allow_kill=True, pg1_min_auc=None, env_goals=None):
        t0 = time.time()
        latencies, g4s, dnus = [], [], []
        solved, steps_done, resets = 0, 0, 0
        env_levels = {}
        if pg1_min_auc is not None and allow_kill and pg1_min_auc < PG1_MIN_AUC:
            result = {"verdict": "G4_AFFORDANCE_FIT_COLLAPSE", "steps_done": 0,
                      "min_auc": float(pg1_min_auc), "n_actions": self.n_actions,
                      "seed": self.seed, "horizon": self.horizon}
            if receipt_out:
                pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
            return result
        try:
            from arc_agi import Arcade
            arcade = Arcade()
        except Exception as exc:
            result = {"verdict": "G4_ARCADE_UNAVAILABLE", "steps_done": 0,
                      "reason": f"arcade_unavailable: {exc!r}"}
            if receipt_out:
                pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
            return result
        for env_name in env_names:
            self._wp_idx = 0
            goal = env_goals.get(env_name) if env_goals else None
            if goal is None and resolve_trajectory_goal is not None:
                try:
                    goal, _m = resolve_trajectory_goal(
                        trajectory_bank, trajectory_jsonl, env_name,
                        device=self.device, ingress=ingress)
                except Exception:
                    goal = None
            if goal is None:
                goal = F.normalize(torch.randn(
                    64, generator=torch.Generator().manual_seed(seed)).to(self.device), dim=-1)
            wps = [goal]
            self.waypoints = wps
            # Carrier P1 guarded hook: subclasses may bind a full-wave goal per
            # env (no-op when the method is absent; default path byte-identical).
            _p1_hook = getattr(self, "p1_bind_env_goal", None)
            if _p1_hook is not None:
                _p1_hook(env_name)
            game = None
            try:
                game = arcade.make(env_name)
            except Exception as exc:
                result = {"verdict": "G4_ARCADE_MAKE_FAILED", "steps_done": steps_done,
                          "reason": f"arcade_make: {exc!r}"}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            if game is None:
                result = {"verdict": "G4_ARCADE_MAKE_NONE", "steps_done": steps_done}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            try:
                obs = game.reset()
            except Exception as exc:
                result = {"verdict": "G4_ARCADE_RESET_FAILED", "steps_done": steps_done,
                          "reason": f"arcade_reset: {exc!r}"}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            if obs is None or not getattr(obs, "frame", None):
                result = {"verdict": "G4_NULL_INITIAL_FRAME", "steps_done": steps_done,
                          "env": env_name}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            prev_levels = _safe_levels(obs) if _safe_levels else 0
            for _ in range(steps_per_env):
                t1 = time.time()
                frame = obs.frame[0]
                raw = torch.as_tensor(np.asarray(frame).reshape(-1).astype(np.float32),
                                      dtype=torch.float32, device=self.device)
                if raw.numel() < 4096:
                    raw = F.pad(raw, (0, 4096 - raw.numel()))
                else:
                    raw = raw[:4096]
                psi64 = ingress(raw.unsqueeze(0)).reshape(1, -1)[0].detach()
                psi_full = fast_encoder.encode_grid(
                    torch.as_tensor(np.asarray(frame), dtype=torch.long,
                                    device=self.device)).reshape(B_FULL, BLK).detach()
                wp = self._active_waypoint()
                best, js = self.step_once(psi64, psi_full, wp)
                if best is None:
                    best = self.action_names[0]
                env_actions = list(getattr(obs, "available_actions", None) or
                                   getattr(game, "action_space", None) or [])
                idx = self._action_index(best, env_actions)
                # Carrier G8 guarded meter hook: when a G8 chain is bound, the
                # M1 delta-nu meter measures FULL-domain alignment to the ACTIVE
                # waypoint (reference captured at decision time; rebind on
                # promotion = packet's "reset baseline to wp[k*+1]"). For all
                # non-G8 engines `_g8_meter_ref` is absent -> legacy D=64 meter
                # (default path byte-identical).
                _g8_ref = getattr(self, "_g8_meter_ref", None)
                if _g8_ref is not None:
                    _g8_ref = _g8_ref.to(self.device).detach()
                    c_t = float((F.normalize(
                        psi_full.reshape(-1), p=2, dim=-1) * _g8_ref
                    ).sum(-1).abs().clamp(0.0, 1.0).item())
                else:
                    c_t = float((psi64 * wp).sum(-1).abs().clamp(0.0, 1.0).item())
                try:
                    if env_actions:
                        action = env_actions[idx % max(1, len(env_actions))]
                    else:
                        action = best
                    obs = game.step(action)
                except Exception:
                    obs = None
                latencies.append((time.time() - t1) * 1e3)
                steps_done += 1
                terminal = obs is None or (
                    getattr(obs, "state", None) and obs.state.name == "GAME_OVER")
                if terminal:
                    cur_levels = _safe_levels(obs) if obs is not None else prev_levels
                    if cur_levels > prev_levels:
                        solved += 1
                    prev_levels = cur_levels
                    env_levels[env_name] = prev_levels
                    self.escape_state = {"steps": 0, "active": True}
                    self.langevin_escapes += 1
                    resets += 1
                    try:
                        obs = game.reset()
                    except Exception:
                        obs = None
                    if obs is None or not getattr(obs, "frame", None):
                        break
                    prev_levels = _safe_levels(obs) if _safe_levels else 0
                    continue
                frame_next = obs.frame[0]
                psi_full_next = fast_encoder.encode_grid(
                    torch.as_tensor(np.asarray(frame_next), dtype=torch.long,
                                    device=self.device)).reshape(B_FULL, BLK).detach()
                # Carrier M1 (measurement repair): derive the post-step D=64
                # bridge state from the ACTUAL next frame. Previously c_next
                # reused the stale pre-step psi64, making mean_delta_nu_wp
                # structurally 0.0 and creeps unreachable (G4..P1 defect).
                raw_next = torch.as_tensor(
                    np.asarray(frame_next).reshape(-1).astype(np.float32),
                    dtype=torch.float32, device=self.device)
                if raw_next.numel() < 4096:
                    raw_next = F.pad(raw_next, (0, 4096 - raw_next.numel()))
                else:
                    raw_next = raw_next[:4096]
                psi64_next = ingress(raw_next.unsqueeze(0)).reshape(1, -1)[0].detach()
                g4s.append(float(self.g4_single_pass(psi64, psi_full, idx, None)))
                self.update_online_affordance(psi_full, idx, psi_full_next)
                # Carrier G8 meter hook (post-step; same reference as decision)
                _g8_ref_n = getattr(self, "_g8_meter_ref", None)
                if _g8_ref_n is not None:
                    _g8_ref_n = _g8_ref_n.to(self.device).detach()
                    c_next = float((F.normalize(
                        psi_full_next.reshape(-1), p=2, dim=-1) * _g8_ref_n
                    ).sum(-1).abs().clamp(0.0, 1.0).item())
                else:
                    c_next = float((psi64_next * wp).sum(-1).abs().clamp(0.0, 1.0).item())
                dnus.append(c_next - c_t)
                # Carrier K3 guarded meter hook: per-env dnu observer for the
                # goal-available seal basis (absent on non-K3 engines -> no-op,
                # default path byte-identical).
                _k3_obs = getattr(self, "k3_observe_dnu", None)
                if _k3_obs is not None:
                    _k3_obs(c_next - c_t)
                if c_next > c_t:
                    self.creeps += 1
                cur_levels = _safe_levels(obs) if _safe_levels else 0
                if cur_levels > prev_levels:
                    solved += 1
                prev_levels = cur_levels
            env_levels[env_name] = prev_levels
        mean_latency = float(np.mean(latencies)) if latencies else None
        g4_mean = float(np.mean(g4s)) if g4s else None
        mean_dnu = float(np.mean(dnus)) if dnus else None
        result = {"verdict": None, "steps_done": steps_done, "resets": resets,
                  "mean_latency_ms": mean_latency, "g4_affordance_mean": g4_mean,
                  "mean_delta_nu_wp": mean_dnu, "waypoint_advances": self.waypoint_advances,
                  "langevin_escapes": self.langevin_escapes,
                  "affordance_updates": self.affordance_updates, "creeps": self.creeps,
                  "n_actions": self.n_actions, "seed": self.seed,
                  "envs_solved": solved, "env_levels": env_levels,
                  "wall_s": round(time.time() - t0, 3),
                  "horizon": self.horizon, "tau_stall": self.tau_stall}
        result["verdict"] = self._decide_verdict(
            mean_latency, solved, mean_dnu, g4_mean, steps_done, self.affordance_updates)
        if receipt_out:
            pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        return result

    def _action_index(self, name, env_actions):
        if name in env_actions:
            return env_actions.index(name)
        try:
            return int(name) % max(1, len(env_actions))
        except (TypeError, ValueError):
            return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Carrier G4 aligned sparse affordance gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--omega-bound", type=float, default=OMEGA_BOUND)
    ap.add_argument("--waypoint-advance-thresh", type=float, default=WAYPOINT_ADVANCE_THRESH)
    ap.add_argument("--tau-stall", type=float, default=TAU_STALL_G4)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--ridge", type=float, default=RIDGE_G4)
    ap.add_argument("--trajectory-bank", required=True)
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--out-dir", default="/tmp/henri_g4_aligned/")
    ap.add_argument("--receipt-out", default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    require_flag(FLAG)
    device = args.device
    env_names = list(args.envs) if args.envs else list(DEFAULT_ENVS)

    data = np.load(args.trajectory_bank)
    psi_flat = torch.from_numpy(np.asarray(data["psi"])).float().to(device)
    nxt_flat = torch.from_numpy(np.asarray(data["next_wave"])).float().to(device)
    # onehot MUST live on `device`: every downstream mask operation ANDs it
    # with y (CUDA, from stall_cosine_labels on CUDA waves). A CPU onehot
    # raised "Expected all tensors to be on the same device" at main()'s
    # `mask & (y == 1.0)` on the remote CUDA run (harness defect, 0 live
    # steps, relaunched with identical bounds — G4 launch #1).
    onehot = torch.from_numpy(np.asarray(data["actions_onehot"])).to(torch.uint8).to(device)

    # Canonical flat-unit geometry (matches bank next_wave; G2-verified label).
    psi_full = F.normalize(psi_flat.float().reshape(psi_flat.shape[0], -1), p=2, dim=-1) \
        .view(-1, B_FULL, BLK)
    nxt_full = F.normalize(nxt_flat.float().reshape(nxt_flat.shape[0], -1), p=2, dim=-1) \
        .view(-1, B_FULL, BLK)

    # PG2: flat norm drift on the working geometry.
    norm_drift = float((psi_full.reshape(psi_full.shape[0], -1).norm(dim=-1) - 1.0).abs().max().item())

    # Stall-cosine labels (G2-verified flat norm-divided; tau_stall 0.90).
    y, _cos = stall_cosine_labels(psi_flat, nxt_flat, args.tau_stall)

    # Per-action top-k masks (PG3: displacement variance over all transitions).
    topk_masks = {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        var = per_block_displacement_variance(psi_full[mask], nxt_full[mask])
        topk_masks[a] = select_topk_blocks(var, args.top_k)

    # Fit per-(action, topk-block) ridge transitions on MOVING rows only.
    transitions_g4 = {}
    theta, tau = {}, {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        mov = mask & (y == 1.0)
        if mov.sum() < 5:
            transitions_g4[a] = {}
            theta[a], tau[a] = 0.0, 1.0
            continue
        transitions_g4[a] = fit_aligned_transitions(
            psi_full[mov], nxt_full[mov], topk_masks[a], args.ridge)
        r_mov = aligned_mean_quadratic(psi_full[mov], nxt_full[mov], transitions_g4[a])
        r_blk = aligned_mean_quadratic(psi_full[mask & (y == 0.0)],
                                       nxt_full[mask & (y == 0.0)], transitions_g4[a])
        theta[a], tau[a] = calibrate_theta_tau(r_mov, r_blk)

    # Per-action affordance pi + AUC (full bank diagnostic + PG1 subset).
    pi_all = torch.zeros(onehot.shape[0], onehot.shape[1], device=device)
    for a in range(onehot.shape[1]):
        if transitions_g4[a]:
            r = aligned_mean_quadratic(psi_full, nxt_full, transitions_g4[a])
            pi_all[:, a] = aligned_affordance_pi(r, theta[a], tau[a])
        else:
            pi_all[:, a] = 0.5

    subset = stratified_subset(onehot, seed=args.seed).to(device)
    per_action_auc_full, per_action_auc_sub = {}, {}
    for a in range(onehot.shape[1]):
        mask = onehot[:, a].bool()
        if mask.sum() >= MIN_AUC_SAMPLES:
            per_action_auc_full[str(a)] = compute_auc(pi_all[mask, a], y[mask])
        else:
            per_action_auc_full[str(a)] = None
        mask_sub = mask & subset
        if mask_sub.sum() >= 10:
            per_action_auc_sub[str(a)] = compute_auc(pi_all[mask_sub, a], y[mask_sub])
        else:
            per_action_auc_sub[str(a)] = None
    sub_vals = [v for v in per_action_auc_sub.values() if v is not None]
    pg1_min_auc = min(sub_vals) if sub_vals else 0.0

    receipt_out = args.receipt_out or str(pathlib.Path(args.out_dir) / "g4_gates_receipt.json")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    label_counts = {str(a): float(y[onehot[:, a].bool()].mean().item())
                    for a in range(onehot.shape[1])}
    moving_counts = {str(a): int((onehot[:, a].bool() & (y == 1.0)).sum().item())
                     for a in range(onehot.shape[1])}

    if pg1_min_auc < PG1_MIN_AUC:
        result = {"verdict": "G4_AFFORDANCE_FIT_COLLAPSE", "steps_done": 0,
                  "min_auc_subset": pg1_min_auc,
                  "per_action_auc_subset": per_action_auc_sub,
                  "per_action_auc_full": per_action_auc_full,
                  "label_counts": label_counts, "moving_counts": moving_counts,
                  "norm_drift": norm_drift, "n_actions": len(topk_masks),
                  "seed": args.seed, "top_k": args.top_k, "ridge": args.ridge,
                  "tau_stall": args.tau_stall, "subset_size": int(subset.sum().item())}
        pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str))
        return 1

    # Kinematics (D=64, carried) for the live loop.
    ingress = PatchIngress(in_dim=4096, d=64, num_blocks=8, p=32,
                           seed=args.seed).to(device)
    psi64 = _bridge_to_d64_batch(psi_full.reshape(psi_full.shape[0], -1),
                                 ingress=ingress, seed=args.seed)
    nxt64 = _bridge_to_d64_batch(nxt_full.reshape(nxt_full.shape[0], -1),
                                 ingress=ingress, seed=args.seed)
    comp = compile_free_generators_capped(
        psi64, nxt64, onehot.float(), omega_bound=args.omega_bound,
        moving_thresh=MOVING_THRESH, seed=args.seed)

    env_goals = {}
    if resolve_trajectory_goal is not None:
        for name in env_names:
            try:
                goal, meta = resolve_trajectory_goal(
                    args.trajectory_bank, args.trajectory_jsonl, name,
                    device=device, ingress=ingress)
                env_goals[name] = goal
            except Exception:
                pass

    engine = G4AlignedEngine(
        transitions_g4=transitions_g4, topk_masks=topk_masks,
        theta=[theta[a] for a in range(len(topk_masks))],
        tau=[tau[a] for a in range(len(topk_masks))],
        generators=comp["generators"], transitions=comp["transitions"],
        t_pow=comp["t_pow"], recon=comp["recon"],
        action_names=comp.get("action_names"), n_actions=len(comp["generators"]),
        seed=args.seed, horizon=args.horizon, device=device,
        omega_bound=args.omega_bound,
        waypoint_advance_thresh=args.waypoint_advance_thresh,
        langevin_temp=LANGEVIN_TEMP, tau_stall=args.tau_stall)

    result = engine.run_gauntlet(
        env_names, fast_encoder=FastFullDWaveEncoder(
            d_model=65536, device=device, seed=args.seed),
        steps_per_env=args.steps_per_env, seed=args.seed,
        trajectory_bank=args.trajectory_bank, trajectory_jsonl=args.trajectory_jsonl,
        ingress=ingress, out_dir=args.out_dir, receipt_out=receipt_out,
        pg1_min_auc=pg1_min_auc, env_goals=env_goals)
    result["pg1_min_auc"] = pg1_min_auc
    result["per_action_auc_subset"] = per_action_auc_sub
    result["per_action_auc_full"] = per_action_auc_full
    result["norm_drift"] = norm_drift
    result["label_counts"] = label_counts
    result["moving_counts"] = moving_counts
    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    os.sys.exit(main())
