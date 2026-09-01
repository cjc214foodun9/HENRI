"""Carrier G1 — Topological Gauge-Wave Scattering Engine (HENRI V3).

Directive: HENRI-DIR-2026-09-V3-TOPOLOGICAL-GAUGE-WAVE-ORDER
(818968573ada74e28353af6d2779390ac09a203cecfacd495183ac2f06c6e0b0, 23,536 B, 352 lines)

Mechanism (directive §1.2, §1.3, §4) — the Homogeneous Manifold Fallacy fix:
  Tier 1: T_free,a = exp(D_free,a) in SO(64) compiled STRICTLY from non-zero
          displacement transitions (||Psi_{t+1} - Psi_t||_2 > 0.05), via
          regularized normal equations -> SVD Stiefel retraction -> skew log
          -> spectral cap (F21.1 _logm_skew / spectral_cap reused verbatim).
          Actions with < 5 moving transitions get T_free,a = I.
  Tier 2: Pi_pass,a(Psi_t) = sigmoid(Psi_t^T W_contact,a Psi_t + b_contact,a),
          a state-dependent bilinear affordance classifier fit as a centered
          quadratic correlation against moving-vs-blocked labels.
  Tier 3: Psi_hat_{t+1}(a) = Pi_pass,a * T_free,a Psi_t + (1-Pi_pass,a) * Psi_t
          (Symplectic Scattering Operator). Blocked action predicts ZERO
          movement -> no hallucinated progress (F21/F22/F23 defect).
  Tier 4: Vectorized Homotopy Beam (K=8): J_a = |<Psi_hat_{t+K}(a), Psi_wp>| *
          Pi_pass,a^K  (directive §1.3 formula; no per-step SVD, no beta term).
  Tier 5: Dual-Speed online affordance plasticity on the EXECUTED action:
          b_a += eta*error; W_a += eta*error*outer(Psi_t, Psi_t),
          eta = 0.10, error = was_moving - Pi_pred. Unexpected collision
          lowers Pi; unexpected motion raises Pi.
  Tier 6: ground-truth waypoint advancement on the ACTUAL post-action frame
          (|cos(Psi_{t+1}, Psi_wp,k)| >= 0.60 -> k+1). Reset -> 3-step Langevin.

G4 (directive §3): single-pass horizon physical consistency
  Delta_Affordance = 1 - |<Psi_hat_{t+1}(a_exec), Psi_{t+1}>| — the scattered
  prediction vs the ACTUAL post-action observation. External physics, not an
  internal axiom (F23 defect 0.9230) and not a random reference (F22 defect).

PG1 (directive §3): in-sample moving-vs-blocked AUC >= 0.8500 per action with
>= 10 samples. PRE-FLIGHT KILL -> no live run.

Verdicts G1_*. Flag HENRI_G1_TOPOLOGICAL=1 (default-OFF fail-closed).
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
    from arc_f21_1_vectorized_engine import (
        _bridge_to_d64_batch,
        _logm_skew,
        spectral_cap,
        sagnac_delta,
        PatchIngress,
    )
except Exception:  # pragma: no cover - test isolation
    _bridge_to_d64_batch = None
    _logm_skew = None
    spectral_cap = None
    sagnac_delta = None
    PatchIngress = None

try:
    from arc_f15_trajectory_engine import DEFAULT_ENVS, resolve_trajectory_goal
except Exception:  # pragma: no cover - test isolation
    DEFAULT_ENVS = []
    resolve_trajectory_goal = None

FLAG = "HENRI_G1_TOPOLOGICAL"
D_SUB = 64
PG1_MIN_AUC = 0.8500
G1_LATENCY_MS = 2.0
G2_MIN_SOLVED = 1
G3_MIN_DELTA_NU = 0.0150
G4_MAX_AFFORDANCE = 0.0500
DEFAULT_HORIZON = 8
OMEGA_BOUND = math.pi / 32.0  # ~0.0982 rad/step
MOVING_THRESH = 0.05
WAYPOINT_ADVANCE_THRESH = 0.60
WAYPOINT_STRIDE = 15
WAYPOINT_DELTA_THETA = 0.35  # rad
WAYPOINT_MIN = 2
WAYPOINT_MAX = 6
LANGEVIN_TEMP = 0.50
LANGEVIN_STEPS = 3
ETA_AFFORDANCE = 0.10
TAU_SHARP = 0.05
RIDGE = 1e-4
MIN_MOVING = 5
MIN_AUC_SAMPLES = 10


def require_flag():
    if os.environ.get(FLAG) != "1":
        raise RuntimeError(f"{FLAG}=1 is required (carrier default-OFF flag).")


def _bridge_to_d64_single(wave_65536, ingress=None, seed=0, device="cpu"):
    w = torch.as_tensor(wave_65536, dtype=torch.float32, device="cpu").reshape(-1)
    if w.numel() < 65536:
        w = F.pad(w, (0, 65536 - w.numel()))
    else:
        w = w[:65536]
    pooled = w.view(16, 4096).mean(dim=0)
    pooled = F.normalize(pooled, p=2, dim=-1) * 64.0
    if ingress is not None:
        with torch.no_grad():
            pooled = ingress(pooled.unsqueeze(0).to(device))[0].detach().reshape(-1)
    return F.normalize(pooled, p=2, dim=-1)


def _geodesic_angle(a, b):
    c = F.cosine_similarity(a.reshape(-1).unsqueeze(0), b.reshape(-1).unsqueeze(0), dim=-1).clamp(-1.0, 1.0)
    return torch.acos(c.abs()).squeeze(0)


def extract_waypoints(curve, goal, delta_theta=WAYPOINT_DELTA_THETA,
                      max_waypoints=WAYPOINT_MAX, min_waypoints=WAYPOINT_MIN):
    """Greedy geodesic waypoint sampling over an observed trajectory curve."""
    curve = F.normalize(curve.float().reshape(curve.shape[0], -1), p=2, dim=-1)
    goal = F.normalize(goal.float().reshape(-1), p=2, dim=-1)
    wps = []
    last = None
    for i in range(0, curve.shape[0], WAYPOINT_STRIDE):
        w = curve[i]
        if last is not None and _geodesic_angle(last, w).item() < delta_theta:
            continue
        if len(wps) >= max_waypoints - 1:
            break
        wps.append(w.clone())
        last = w
    if len(wps) == 0:
        wps.append(goal.clone())
    elif float((wps[-1] * goal).sum().abs().clamp(0.0, 1.0).item()) >= 0.9999:
        wps[-1] = goal.clone()
    else:
        wps.append(goal.clone())
    if len(wps) < min_waypoints and len(wps) < max_waypoints:
        wps = [curve[0].clone()] + wps
    return wps[:max_waypoints]


def advance_waypoint_index(psi, waypoint, k, thresh=WAYPOINT_ADVANCE_THRESH):
    """k(t+1) = k+1 if |<psi, wp_k>| >= thresh and k < WAYPOINT_MAX-1, else k."""
    c = float((psi.reshape(-1) * waypoint.reshape(-1)).sum().abs().clamp(0.0, 1.0).item())
    if c >= thresh and k < WAYPOINT_MAX - 1:
        return k + 1
    return k


def langevin_escape_tick(state, langevin_steps=LANGEVIN_STEPS):
    steps = state.get("steps", 0)
    active = state.get("active", False)
    if active:
        steps += 1
        if steps >= langevin_steps:
            return {"steps": 0, "active": False}
        return {"steps": steps, "active": True}
    return {"steps": 0, "active": False}


def _safe_levels(obs):
    try:
        return int(getattr(obs, "levels_completed", 0) or 0)
    except Exception:
        return 0


def compute_auc(scores, labels):
    """Rank-based in-sample AUC (Mann-Whitney U). scores/labels: 1-D tensors."""
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1)
    pos = scores[labels == 1.0]
    neg = scores[labels == 0.0]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    n_pos = pos.numel()
    n_neg = neg.numel()
    # AUC = (1/(n_pos*n_neg)) * sum_i sum_j 1[s_pos_i > s_neg_j] + 0.5*1[==]
    cmp = (pos.unsqueeze(1) > neg.unsqueeze(0)).float()
    tie = (pos.unsqueeze(1) == neg.unsqueeze(0)).float()
    return float((cmp.sum() + 0.5 * tie.sum()).item() / (n_pos * n_neg))


def compile_free_generators_capped(psi, nxt, onehot, omega_bound=OMEGA_BOUND,
                                   moving_thresh=MOVING_THRESH, ridge=RIDGE,
                                   min_moving=MIN_MOVING, seed=0):
    """Fit T_free,a = exp(D_free,a) STRICTLY on moving transitions (directive §1.2).

    psi/nxt: [N, D] unit states; onehot: [N, n_actions] float.
    Returns dict: generators [n,D,D] (capped skew), transitions [n,D,D] (SO(D)),
    t_pow [n,K,D,D], recon (moving-only per action), is_moving [N], counts.
    """
    psi = F.normalize(psi.float(), dim=-1)
    nxt = F.normalize(nxt.float(), dim=-1)
    n_actions = onehot.shape[1]
    displacement = torch.norm(nxt - psi, p=2, dim=-1)
    is_moving = (displacement > moving_thresh).float()
    gens, transitions, recon = [], [], {}
    counts = {}
    for a in range(n_actions):
        mask_a = onehot[:, a].bool()
        mask_moving_a = mask_a & (is_moving == 1.0)
        counts[a] = {"samples": int(mask_a.sum().item()),
                     "moving": int(mask_moving_a.sum().item())}
        if mask_moving_a.sum() > min_moving:
            X_a = psi[mask_moving_a]
            Y_a = nxt[mask_moving_a]
            reg = ridge * torch.eye(D_SUB, dtype=X_a.dtype, device=X_a.device)
            K_a = (Y_a.T @ X_a) @ torch.linalg.inv(X_a.T @ X_a + reg)
            U, _, Vt = torch.linalg.svd(K_a)
            W_a = U @ Vt  # Stiefel retraction to SO(D)
            D_raw = _logm_skew(W_a) if _logm_skew is not None else 0.5 * (W_a - W_a.T)
            D_cap = spectral_cap(D_raw, omega_bound=omega_bound) if spectral_cap is not None else D_raw
            T = torch.linalg.matrix_exp(D_cap)
            gens.append(D_cap)
            transitions.append(T)
            recon[a] = float(
                F.normalize(X_a @ T.T, dim=-1).mul(Y_a).sum(-1).abs().mean().item()
            )
        else:
            gens.append(torch.zeros(D_SUB, D_SUB, dtype=psi.dtype))
            transitions.append(torch.eye(D_SUB, dtype=psi.dtype))
            recon[a] = float(
                F.normalize(psi[mask_a] @ torch.eye(D_SUB).T, dim=-1).mul(nxt[mask_a]).sum(-1).abs().mean().item()
            ) if mask_a.sum() > 0 else 0.0
    generators = torch.stack(gens)
    transitions = torch.stack(transitions)
    K = DEFAULT_HORIZON
    t_pow = torch.stack(
        [torch.linalg.matrix_power(transitions, k) for k in range(1, K + 1)], dim=1
    )
    return {"generators": generators, "transitions": transitions, "t_pow": t_pow,
            "recon": recon, "is_moving": is_moving, "counts": counts,
            "action_names": [str(i) for i in range(n_actions)]}


def fit_affordance_classifiers(psi, onehot, is_moving, ridge=RIDGE):
    """State-dependent bilinear affordance classifiers (directive §1.2).

    W_contact,a = (1/N_a) sum_i (y_i - mean_a) outer(psi_i, psi_i)  (centered
    quadratic correlation); b_contact,a = logit(clamp(mean_a, 0.05, 0.95)).
    Actions with < 10 samples get W = 0, b = logit(prior) (neutral).
    Returns (W [n,D,D], b [n]).
    """
    psi = F.normalize(psi.float(), dim=-1)
    y = is_moving.float()
    n_actions = onehot.shape[1]
    W = torch.zeros(n_actions, D_SUB, D_SUB, dtype=psi.dtype)
    b = torch.zeros(n_actions, dtype=psi.dtype)
    for a in range(n_actions):
        mask_a = onehot[:, a].bool()
        if mask_a.sum() >= MIN_AUC_SAMPLES:
            y_a = y[mask_a]
            mean_a = float(y_a.mean().item())
            yc = y_a - mean_a
            W[a] = torch.einsum("i,id,ie->de", yc, psi[mask_a], psi[mask_a]) / y_a.numel()
            b[a] = torch.logit(torch.clamp(torch.tensor(mean_a), 0.05, 0.95))
        else:
            prior = float(y[mask_a].mean().item()) if mask_a.sum() > 0 else 0.5
            b[a] = torch.logit(torch.clamp(torch.tensor(prior), 0.05, 0.95))
    return W, b


def scatter_prediction(psi, T, pi):
    """Symplectic scattering: pi*T@psi + (1-pi)*psi. psi [D] or [B,D], pi scalar/[B]."""
    psi = F.normalize(psi.float(), dim=-1)
    if psi.dim() == 1:
        return pi * (psi @ T.T) + (1.0 - pi) * psi
    pi = pi.unsqueeze(-1)
    return pi * (psi @ T.T) + (1.0 - pi) * psi


class G1Engine:
    def __init__(self, generators, transitions, t_pow, recon, W_contact, b_contact,
                 action_names=None, n_actions=7, seed=20260924,
                 horizon=DEFAULT_HORIZON, device="cuda", omega_bound=OMEGA_BOUND,
                 waypoints=None, waypoint_advance_thresh=WAYPOINT_ADVANCE_THRESH,
                 langevin_temp=LANGEVIN_TEMP, eta_affordance=ETA_AFFORDANCE,
                 moving_thresh=MOVING_THRESH, tau_sharp=TAU_SHARP):
        self.generators = generators
        self.transitions = list(transitions)
        self.t_pow = t_pow
        self.recon = recon
        self.W_contact = W_contact
        self.b_contact = b_contact
        self.action_names = list(action_names) if action_names else [str(i) for i in range(n_actions)]
        self.n_actions = n_actions
        self.seed = seed
        self.horizon = horizon
        self.device = device
        self.omega_bound = omega_bound
        self.waypoints = list(waypoints) if waypoints else None
        self.waypoint_advance_thresh = waypoint_advance_thresh
        self.langevin_temp = langevin_temp
        self.eta_affordance = eta_affordance
        self.moving_thresh = moving_thresh
        self.tau_sharp = tau_sharp
        self.creeps = 0
        self.waypoint_advances = 0
        self.langevin_escapes = 0
        self.affordance_updates = 0
        self.escape_state = {"steps": 0, "active": False}
        self._wp_idx = 0

    def _active_waypoint(self):
        if self.waypoints is None or len(self.waypoints) == 0:
            return F.normalize(torch.randn(D_SUB, generator=torch.Generator().manual_seed(self.seed)), dim=-1)
        k = min(self._wp_idx, len(self.waypoints) - 1)
        return self.waypoints[k]

    def predict_affordance(self, psi):
        """Pi_pass [B, n_actions] = sigmoid(psi^T W_a psi + b_a).

        Accepts a single [D] state or a [B, D] batch; always returns [B, n].
        """
        psi = F.normalize(psi.float().to(self.device), dim=-1)
        if psi.dim() == 1:
            psi = psi.unsqueeze(0)
        W = self.W_contact.to(self.device)
        b = self.b_contact.to(self.device)
        # Directive §1.2: Pi = sigmoid((<Psi, W_a Psi> - theta)/tau_sharp).
        quad = torch.einsum("bd,ade,be->ba", psi, W, psi) / self.tau_sharp
        return torch.sigmoid(quad + b.unsqueeze(0))

    def score_all_actions(self, psi, waypoint=None):
        """J_a = |<Psi_hat_{t+K}(a), Psi_wp>| * Pi_pass,a^K (directive §1.3)."""
        psi = F.normalize(psi.float().to(self.device), dim=-1)
        wp = F.normalize((waypoint if waypoint is not None else self._active_waypoint()).float().to(self.device), dim=-1)
        tpow = self.t_pow.to(self.device)
        rolled = tpow @ psi  # [n, K, D]
        steps = F.normalize(rolled, dim=-1)
        aligns = (steps * wp).sum(-1).abs()  # [n, K]
        align = aligns[:, -1]  # terminal horizon alignment vs waypoint
        pi = self.predict_affordance(psi)[0]  # [n]
        j = align * (pi ** self.horizon)
        if self.escape_state.get("active"):
            g = torch.Generator().manual_seed(self.seed + self.escape_state.get("steps", 0))
            noise = torch.sqrt(torch.tensor(2.0 * self.langevin_temp, device=self.device)) * torch.randn(self.n_actions, generator=g).to(self.device)
            j = j + noise
        names = list(self.action_names)
        return {names[i]: float(j[i].item()) for i in range(self.n_actions)}

    def step_once(self, psi, waypoint=None):
        js = self.score_all_actions(psi, waypoint)
        if not js:
            return None, js
        return max(js, key=js.get), js

    def g4_single_pass(self, psi, action_idx, psi_actual):
        """Scattered prediction vs ACTUAL post-action observation (directive §3)."""
        psi = F.normalize(psi.float().to(self.device), dim=-1)
        psi_actual = F.normalize(psi_actual.float().to(self.device), dim=-1)
        pi = self.predict_affordance(psi)[0, action_idx].item()
        pred = scatter_prediction(psi, self.transitions[action_idx], pi)
        sim = (pred * psi_actual).sum(-1).abs().clamp(0.0, 1.0)
        return 1.0 - sim

    def update_online_affordance(self, psi, action_idx, psi_next, eta=None):
        """Dual-speed in-situ affordance plasticity on the EXECUTED action (§4)."""
        eta = self.eta_affordance if eta is None else eta
        psi = F.normalize(psi.float().to(self.device), dim=-1)
        psi_next = F.normalize(psi_next.float().to(self.device), dim=-1)
        displacement = float(torch.norm(psi_next - psi, p=2).item())
        was_moving = 1.0 if displacement > self.moving_thresh else 0.0
        pi_pred = self.predict_affordance(psi)[0, action_idx].item()
        error = was_moving - pi_pred
        self.b_contact[action_idx] = self.b_contact[action_idx] + eta * error
        self.W_contact[action_idx] = self.W_contact[action_idx] + eta * error * torch.outer(psi, psi)
        self.affordance_updates += 1
        return error

    def advance_waypoint_index(self, psi, waypoint, k):
        return advance_waypoint_index(psi, waypoint, k, thresh=self.waypoint_advance_thresh)

    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        if steps_done > 0 and updates == 0:
            return "G1_NO_AFFORDANCE_ENGAGEMENT"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "G1_GATE_G1_FAILED"
        if solved < G2_MIN_SOLVED:
            return "G1_GATE_G2_FAILED"
        if mean_delta_nu is not None and mean_delta_nu < G3_MIN_DELTA_NU:
            return "G1_GATE_G3_FAILED"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "G1_GATE_G4_FAILED"
        return "G1_PASS"

    def run_gauntlet(self, env_names, steps_per_env=150, seed=20260924,
                     trajectory_bank=None, trajectory_jsonl=None, ingress=None,
                     out_dir=None, receipt_out=None, allow_kill=True,
                     pg1_min_auc=None, env_goals=None):
        t0 = time.time()
        latencies, g4s, dnus = [], [], []
        waypoint_align_first, waypoint_align_last = None, None
        solved, steps_done, resets = 0, 0, 0
        env_levels = {}
        if pg1_min_auc is None and allow_kill:
            pg1_min_auc = self.pg1_min_auc if hasattr(self, "pg1_min_auc") else None
        if pg1_min_auc is not None and pg1_min_auc < PG1_MIN_AUC:
            result = {"verdict": "G1_AFFORDANCE_FIT_COLLAPSE", "steps_done": 0,
                      "min_auc": float(pg1_min_auc), "n_actions": self.n_actions,
                      "seed": self.seed, "omega_bound": self.omega_bound,
                      "horizon": self.horizon}
            if receipt_out:
                pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
            return result

        try:
            from arc_agi import Arcade
            arcade = Arcade()
        except Exception as exc:
            result = {"verdict": "G1_ARCADE_UNAVAILABLE", "steps_done": 0,
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
                    goal, _meta = resolve_trajectory_goal(
                        trajectory_bank, trajectory_jsonl, env_name,
                        device=self.device, ingress=ingress)
                except Exception:
                    goal = None
            if goal is None:
                goal = F.normalize(
                    torch.randn(D_SUB, generator=torch.Generator().manual_seed(seed)).to(self.device),
                    dim=-1)
            wps = [goal]
            if trajectory_bank:
                try:
                    data = np.load(trajectory_bank)
                    if "psi" in data.files:
                        waves = torch.from_numpy(np.asarray(data["psi"])).float()
                        env_indices = []
                        if trajectory_jsonl:
                            with open(trajectory_jsonl, "r", encoding="utf-8") as f:
                                for i, line in enumerate(f):
                                    try:
                                        rec = json.loads(line)
                                    except json.JSONDecodeError:
                                        continue
                                    if rec.get("env") == env_name:
                                        env_indices.append(i)
                        if len(env_indices) >= WAYPOINT_MIN:
                            curve_full = waves[env_indices]
                            curve = torch.stack([
                                _bridge_to_d64_single(w, ingress=ingress, seed=seed, device=self.device)
                                for w in curve_full
                            ])
                            extracted = extract_waypoints(curve, goal)
                            if len(extracted) >= WAYPOINT_MIN:
                                wps = extracted
                except Exception:
                    pass
            self.waypoints = wps

            game = None
            try:
                game = arcade.make(env_name)
            except Exception as exc:
                result = {"verdict": "G1_ARCADE_MAKE_FAILED", "steps_done": steps_done,
                          "reason": f"arcade_make: {exc!r}"}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            if game is None:
                result = {"verdict": "G1_ARCADE_MAKE_NONE", "steps_done": steps_done}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            try:
                obs = game.reset()
            except Exception as exc:
                result = {"verdict": "G1_ARCADE_RESET_FAILED", "steps_done": steps_done,
                          "reason": f"arcade_reset: {exc!r}"}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            if obs is None or not getattr(obs, "frame", None):
                result = {"verdict": "G1_NULL_INITIAL_FRAME", "steps_done": steps_done,
                          "env": env_name}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            prev_levels = _safe_levels(obs)

            for _ in range(steps_per_env):
                t1 = time.time()
                frame = obs.frame[0]
                raw = torch.as_tensor(np.asarray(frame).reshape(-1).astype(np.float32), dtype=torch.float32, device=self.device)
                if raw.numel() < 4096:
                    raw = F.pad(raw, (0, 4096 - raw.numel()))
                else:
                    raw = raw[:4096]
                psi_b = ingress(raw.unsqueeze(0)).reshape(1, -1)
                if psi_b.shape[-1] != D_SUB:
                    raise RuntimeError(f"PatchIngress boundary must flatten to [{D_SUB}], got {tuple(psi_b.shape)}")
                psi = psi_b[0].detach()
                wp = self._active_waypoint()
                if waypoint_align_first is None:
                    waypoint_align_first = float((psi * wp).sum(-1).abs().clamp(0.0, 1.0).item())
                best, js = self.step_once(psi, wp)
                if self.escape_state.get("active"):
                    self.escape_state = langevin_escape_tick(self.escape_state)
                if best is None:
                    best = self.action_names[0]
                env_actions = list(getattr(obs, "available_actions", None) or
                                   getattr(game, "action_space", None) or [])
                idx = self._action_index(best, env_actions)
                # G4 uses the scattered prediction vs the ACTUAL post-action
                # observation. G3 and waypoint advancement also use the actual
                # frame, so they cannot reward an action that did not change
                # the environment.
                c_t = float((psi * wp).sum(-1).abs().clamp(0.0, 1.0).item())
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
                    prev_levels = _safe_levels(obs)
                    continue
                frame_next = obs.frame[0]
                raw_next = torch.as_tensor(np.asarray(frame_next).reshape(-1).astype(np.float32), dtype=torch.float32, device=self.device)
                if raw_next.numel() < 4096:
                    raw_next = F.pad(raw_next, (0, 4096 - raw_next.numel()))
                else:
                    raw_next = raw_next[:4096]
                psi_next = ingress(raw_next.unsqueeze(0)).reshape(1, -1)[0].detach()
                g4s.append(float(self.g4_single_pass(psi, idx, psi_next).item()))
                # Dual-speed online affordance plasticity on the executed action.
                self.update_online_affordance(psi, idx, psi_next)
                c_next = float((psi_next * wp).sum(-1).abs().clamp(0.0, 1.0).item())
                dnu = c_next - c_t
                dnus.append(dnu)
                if dnu > 0:
                    self.creeps += 1
                new_k = self.advance_waypoint_index(
                    psi_next, self.waypoints[min(self._wp_idx, len(self.waypoints) - 1)],
                    self._wp_idx)
                if new_k > self._wp_idx:
                    self._wp_idx = new_k
                    self.waypoint_advances += 1
                waypoint_align_last = c_next
                cur_levels = _safe_levels(obs)
                if cur_levels > prev_levels:
                    solved += 1
                prev_levels = cur_levels
            env_levels[env_name] = prev_levels

        mean_latency = float(np.mean(latencies)) if latencies else None
        g4_affordance_mean = float(np.mean(g4s)) if g4s else None
        mean_delta_nu = float(np.mean(dnus)) if dnus else None
        result = {"verdict": None, "steps_done": steps_done, "resets": resets,
                  "mean_latency_ms": mean_latency, "g4_affordance_mean": g4_affordance_mean,
                  "mean_delta_nu_wp": mean_delta_nu,
                  "waypoint_align_first": waypoint_align_first,
                  "waypoint_align_last": waypoint_align_last,
                  "waypoint_advances": self.waypoint_advances,
                  "langevin_escapes": self.langevin_escapes,
                  "per_action_recon": {str(k): float(v) for k, v in self.recon.items()},
                  "affordance_updates": self.affordance_updates,
                  "creeps": self.creeps, "n_actions": self.n_actions, "seed": self.seed,
                  "envs_solved": solved, "env_levels": env_levels,
                  "wall_s": round(time.time() - t0, 3),
                  "omega_bound": self.omega_bound, "horizon": self.horizon,
                  "waypoint_advance_thresh": self.waypoint_advance_thresh,
                  "langevin_temp": self.langevin_temp,
                  "eta_affordance": self.eta_affordance, "moving_thresh": self.moving_thresh}
        verdict = self._decide_verdict(
            mean_latency=mean_latency, solved=solved, mean_delta_nu=mean_delta_nu,
            g4_mean=g4_affordance_mean, steps_done=steps_done,
            updates=self.affordance_updates)
        result["verdict"] = verdict
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


def build_parser():
    ap = argparse.ArgumentParser(description="Carrier G1 topological gauge-wave scattering gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--omega-bound", type=float, default=OMEGA_BOUND)
    ap.add_argument("--waypoint-advance-thresh", type=float, default=WAYPOINT_ADVANCE_THRESH)
    ap.add_argument("--langevin-temp", type=float, default=LANGEVIN_TEMP)
    ap.add_argument("--eta-affordance", type=float, default=ETA_AFFORDANCE)
    ap.add_argument("--moving-thresh", type=float, default=MOVING_THRESH)
    ap.add_argument("--trajectory-bank", required=True)
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--envs", nargs="+", default=None, help="12 named env ids (default: F15 DEFAULT_ENVS)")
    ap.add_argument("--out-dir", default="/tmp/henri_g1_topological/")
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
    comp = compile_free_generators_capped(
        psi, nxt, onehot.float(), omega_bound=args.omega_bound,
        moving_thresh=args.moving_thresh, seed=args.seed)
    W, b = fit_affordance_classifiers(psi, onehot.float(), comp["is_moving"])
    # PG1: per-action in-sample moving-vs-blocked AUC (>= 10 samples).
    pi_all = torch.sigmoid(torch.einsum("bd,ade,be->ba", psi, W, psi) / TAU_SHARP + b.unsqueeze(0))
    per_action_auc = {}
    for a in range(pi_all.shape[1]):
        mask_a = onehot[:, a].bool()
        if mask_a.sum() >= MIN_AUC_SAMPLES:
            per_action_auc[str(a)] = compute_auc(pi_all[mask_a, a], comp["is_moving"][mask_a])
        else:
            per_action_auc[str(a)] = None
    auc_vals = [v for v in per_action_auc.values() if v is not None]
    pg1_min_auc = min(auc_vals) if auc_vals else 0.0
    receipt_out = args.receipt_out or str(pathlib.Path(args.out_dir) / "g1_gates_receipt.json")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if pg1_min_auc < PG1_MIN_AUC:
        result = {"verdict": "G1_AFFORDANCE_FIT_COLLAPSE", "steps_done": 0,
                  "min_auc": pg1_min_auc, "per_action_auc": per_action_auc,
                  "n_actions": len(comp["generators"]), "seed": args.seed,
                  "omega_bound": args.omega_bound, "horizon": args.horizon}
        pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2))
        return 1
    print(json.dumps({"pg1_min_auc": pg1_min_auc, "per_action_auc": per_action_auc,
                      "per_action_recon": {str(k): float(v) for k, v in comp["recon"].items()},
                      "counts": comp["counts"]}))

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
    engine = G1Engine(
        generators=comp["generators"], transitions=comp["transitions"],
        t_pow=comp["t_pow"], recon=comp["recon"], W_contact=W, b_contact=b,
        action_names=comp.get("action_names"), n_actions=len(comp["generators"]),
        seed=args.seed, horizon=args.horizon, device=args.device,
        omega_bound=args.omega_bound,
        waypoint_advance_thresh=args.waypoint_advance_thresh,
        langevin_temp=args.langevin_temp, eta_affordance=args.eta_affordance,
        moving_thresh=args.moving_thresh)
    engine.pg1_min_auc = pg1_min_auc
    result = engine.run_gauntlet(
        env_names, steps_per_env=args.steps_per_env, seed=args.seed,
        trajectory_bank=args.trajectory_bank, trajectory_jsonl=args.trajectory_jsonl,
        ingress=ingress, out_dir=args.out_dir, receipt_out=receipt_out,
        pg1_min_auc=pg1_min_auc, env_goals=env_goals)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
