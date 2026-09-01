"""Carrier F22 — Dynamic Affordance Sub-Goal Stepping & Metric-Realigned Task Resolution Engine.

Directive: HENRI-DIR-2026-08-F21-1-POSTMORTEM-TASK-RESOLUTION
(841c73a5b43058261ba31b0a17760f0674f8a3126e5a111ecb631f35666eaa49, 20,690 B, 235 lines)

Mechanism (directive §3):
  Tier 1: W^(e) = ExtractGeodesicWaypoints(D_bank^(e), stride=15) — per-env waypoint
          chain (greedy geodesic, dtheta = 0.35 rad, 4-6 waypoints + terminal).
          T_a = exp(SpectralCap(Logm(W_a^EDMD), pi/32)) in SO(64) — F21.1 capped
          generators reused verbatim.
  Tier 2: a* = argmax_a [ |<Psi_hat_{t+K}(a), Psi_wp,k(t)>| - beta * sum_k Delta_Sagnac(k) ]
          — F21.1 batched t_pow unroll, scored against the ACTIVE waypoint.
  Tier 3: |<Psi_{t+1}, Psi_wp,k>| >= 0.60 -> k <- k+1 (waypoint advance).
          Reset event -> inject T_active = 0.50 into action scores for 3 steps.

G4 realigned (directive §2.1): Delta_Sagnac = 1 - |<T_a Psi_t, Psi_axiom>| — physical
operator coherence, NOT goal distance.

Live interaction (G2 prerequisite): real Arcade loop per F15 pattern — arcade.make ->
reset() -> frame[0] -> flat -> pad 4096 -> PatchIngress -> Psi_t [64] -> select ->
game.step(action) -> terminal (GAME_OVER) -> reset; solved = levels_completed increase.

Verdicts F22_*. Flag HENRI_F22_RESOLUTION=1 (default-OFF fail-closed).
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
        compile_generators_capped,
        preflight_pg1,
        _bridge_to_d64_batch,
        sagnac_delta,
        PatchIngress,
    )
except Exception:  # pragma: no cover - test isolation
    compile_generators_capped = None
    preflight_pg1 = None
    _bridge_to_d64_batch = None
    sagnac_delta = None
    PatchIngress = None

try:
    from arc_f15_trajectory_engine import DEFAULT_ENVS, resolve_trajectory_goal
except Exception:  # pragma: no cover - test isolation
    DEFAULT_ENVS = []
    resolve_trajectory_goal = None

FLAG = "HENRI_F22_RESOLUTION"
D_SUB = 64
PG1_MIN_RECON = 0.85
G1_LATENCY_MS = 2.0
G2_MIN_SOLVED = 1
G3_MIN_DELTA_NU = 0.0200
G4_MAX_SAGNAC = 0.0500
DEFAULT_HORIZON = 8
DEFAULT_BETA_SAGNAC = 0.015
OMEGA_BOUND = math.pi / 32.0  # ~0.0982 rad/step
WAYPOINT_ADVANCE_THRESH = 0.60
WAYPOINT_STRIDE = 15
WAYPOINT_DELTA_THETA = 0.35  # rad
WAYPOINT_MIN = 2
WAYPOINT_MAX = 6
LANGEVIN_TEMP = 0.50
LANGEVIN_STEPS = 3


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
    """Greedy geodesic waypoint sampling over an observed trajectory curve.

    curve: [N, D] unit states along the trajectory; goal: [D] unit terminal.
    Returns a list of [D] unit waypoints, last EXACTLY == goal. Monotone
    geodesic progress toward the goal. Pads to min_waypoints with curve[0] if
    needed (only when room remains under max_waypoints).
    """
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
        wps[-1] = goal.clone()  # last sample ~= goal: replace with exact terminal
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
    """Return an updated escape state dict: {'steps': int, 'active': bool}."""
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


class F22Engine:
    def __init__(self, generators, transitions, t_pow, recon, action_names=None,
                 n_actions=7, seed=20260922, horizon=DEFAULT_HORIZON,
                 beta_sagnac=DEFAULT_BETA_SAGNAC, device="cuda",
                 omega_bound=OMEGA_BOUND, waypoints=None, axiom=None,
                 waypoint_advance_thresh=WAYPOINT_ADVANCE_THRESH,
                 langevin_temp=LANGEVIN_TEMP):
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
        self.waypoints = list(waypoints) if waypoints else None
        self.axiom = axiom
        self.waypoint_advance_thresh = waypoint_advance_thresh
        self.langevin_temp = langevin_temp
        self.creeps = 0
        self.waypoint_advances = 0
        self.langevin_escapes = 0
        self.escape_state = {"steps": 0, "active": False}
        self._wp_idx = 0
        g = torch.Generator().manual_seed(seed)
        if self.axiom is None:
            self.axiom = F.normalize(torch.randn(D_SUB, generator=g), dim=-1)

    def _active_waypoint(self):
        if self.waypoints is None or len(self.waypoints) == 0:
            return self.axiom
        k = min(self._wp_idx, len(self.waypoints) - 1)
        return self.waypoints[k]

    def score_all_actions(self, psi, waypoint=None):
        psi = F.normalize(psi.float().to(self.device), dim=-1)
        wp = F.normalize((waypoint if waypoint is not None else self._active_waypoint()).float().to(self.device), dim=-1)
        tpow = self.t_pow.to(self.device)
        rolled = tpow @ psi  # [n, K, D]
        steps = F.normalize(rolled, dim=-1)
        aligns = (steps * wp).sum(-1).abs()  # [n, K]
        align = aligns[:, -1]  # terminal horizon alignment per action vs waypoint
        sags = (1.0 - aligns).sum(dim=1)  # sum_k Delta_Sagnac(k) per action
        j = align - self.beta_sagnac * sags
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

    def g4_single_pass(self, psi, action_idx):
        """Physical Sagnac coherence: 1 - |<T_a Psi_t, Psi_axiom>| (directive §2.1)."""
        psi = F.normalize(psi.float().to(self.device), dim=-1)
        pred = F.normalize(psi @ self.transitions[action_idx].T, dim=-1)
        ax = F.normalize(self.axiom.float().to(self.device), dim=-1)
        sim = (pred * ax).sum(-1).abs().clamp(0.0, 1.0)
        return 1.0 - sim

    def run_gauntlet(self, env_names, steps_per_env=150, seed=20260922,
                     trajectory_bank=None, trajectory_jsonl=None, ingress=None,
                     out_dir=None, receipt_out=None, allow_kill=True,
                     pg1_min_recon=None, env_goals=None):
        t0 = time.time()
        latencies, sagnacs, dnus = [], [], []
        waypoint_align_first, waypoint_align_last = None, None
        solved, steps_done, resets, waypoint_advances = 0, 0, 0, 0
        env_levels = {}
        if pg1_min_recon is None and allow_kill and self.recon:
            pg1_min_recon = min(self.recon.values())
        if pg1_min_recon is not None and pg1_min_recon < PG1_MIN_RECON:
            result = {"verdict": "F22_EDMD_FIT_COLLAPSE", "steps_done": 0,
                      "min_recon": float(pg1_min_recon),
                      "per_action_recon": {str(k): float(v) for k, v in self.recon.items()},
                      "n_actions": self.n_actions, "seed": self.seed,
                      "omega_bound": self.omega_bound, "beta_sagnac": self.beta_sagnac,
                      "horizon": self.horizon}
            if receipt_out:
                pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
            return result

        try:
            from arc_agi import Arcade
            arcade = Arcade()
        except Exception as exc:
            result = {"verdict": "F22_ARCADE_UNAVAILABLE", "steps_done": 0,
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
                result = {"verdict": "F22_ARCADE_MAKE_FAILED", "steps_done": steps_done,
                          "reason": f"arcade_make: {exc!r}"}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            if game is None:
                result = {"verdict": "F22_ARCADE_MAKE_NONE", "steps_done": steps_done}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            try:
                obs = game.reset()
            except Exception as exc:
                result = {"verdict": "F22_ARCADE_RESET_FAILED", "steps_done": steps_done,
                          "reason": f"arcade_reset: {exc!r}"}
                if receipt_out:
                    pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
                return result
            if obs is None or not getattr(obs, "frame", None):
                result = {"verdict": "F22_NULL_INITIAL_FRAME", "steps_done": steps_done,
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
                psi_b = ingress(raw.unsqueeze(0))
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
                # G4 uses the predicted operator transition. G3 and waypoint
                # advancement use the actual post-action observation, so they
                # cannot reward an action that did not change the environment.
                if sagnac_delta is not None:
                    sagnacs.append(float(self.g4_single_pass(psi, idx).item()))
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
                    # A terminal response may still carry the final frame and
                    # level count. Do not invent a post-state when it is absent.
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
                psi_next = ingress(raw_next.unsqueeze(0))[0].detach()
                c_next = float((psi_next * wp).sum(-1).abs().clamp(0.0, 1.0).item())
                dnu = c_next - c_t
                dnus.append(dnu)
                if dnu > 0:
                    self.creeps += 1
                new_k = advance_waypoint_index(
                    psi_next, self.waypoints[min(self._wp_idx, len(self.waypoints) - 1)],
                    self._wp_idx, thresh=self.waypoint_advance_thresh)
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
        sagnac_axiom_mean = float(np.mean(sagnacs)) if sagnacs else None
        mean_delta_nu = float(np.mean(dnus)) if dnus else None
        result = {"verdict": None, "steps_done": steps_done, "resets": resets,
                  "mean_latency_ms": mean_latency, "sagnac_axiom_mean": sagnac_axiom_mean,
                  "mean_delta_nu_wp": mean_delta_nu,
                  "waypoint_align_first": waypoint_align_first,
                  "waypoint_align_last": waypoint_align_last,
                  "waypoint_advances": self.waypoint_advances,
                  "langevin_escapes": self.langevin_escapes,
                  "per_action_recon": {str(k): float(v) for k, v in self.recon.items()},
                  "creeps": self.creeps, "n_actions": self.n_actions, "seed": self.seed,
                  "envs_solved": solved, "env_levels": env_levels,
                  "wall_s": round(time.time() - t0, 3),
                  "omega_bound": self.omega_bound, "beta_sagnac": self.beta_sagnac,
                  "horizon": self.horizon, "waypoint_advance_thresh": self.waypoint_advance_thresh,
                  "langevin_temp": self.langevin_temp}
        verdict = "F22_PASS"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            verdict = "F22_GATE_G1_FAILED"
        elif solved < G2_MIN_SOLVED:
            verdict = "F22_GATE_G2_FAILED"
        elif mean_delta_nu is not None and mean_delta_nu < G3_MIN_DELTA_NU:
            verdict = "F22_GATE_G3_FAILED"
        elif sagnac_axiom_mean is not None and sagnac_axiom_mean > G4_MAX_SAGNAC:
            verdict = "F22_GATE_G4_FAILED"
        result["verdict"] = verdict
        if receipt_out:
            pathlib.Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(receipt_out).write_text(json.dumps(result, indent=2, default=str))
        return result

    def _action_index(self, name, env_actions):
        if name in env_actions:
            return env_actions.index(name)
        try:
            # Candidate names are stable numeric indices; Arcade action
            # objects are mapped through the current legal list as in F15.
            return int(name) % max(1, len(env_actions))
        except (TypeError, ValueError):
            return 0


def build_parser():
    ap = argparse.ArgumentParser(description="Carrier F22 dynamic task resolution gauntlet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-per-env", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--omega-bound", type=float, default=OMEGA_BOUND)
    ap.add_argument("--beta-sagnac", type=float, default=DEFAULT_BETA_SAGNAC)
    ap.add_argument("--waypoint-advance-thresh", type=float, default=WAYPOINT_ADVANCE_THRESH)
    ap.add_argument("--langevin-temp", type=float, default=LANGEVIN_TEMP)
    ap.add_argument("--trajectory-bank", required=True)
    ap.add_argument("--trajectory-jsonl", required=True)
    ap.add_argument("--envs", nargs="+", default=None, help="12 named env ids (default: F15 DEFAULT_ENVS)")
    ap.add_argument("--out-dir", default="/tmp/henri_f22_resolution/")
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
    receipt_out = args.receipt_out or str(pathlib.Path(args.out_dir) / "f22_gates_receipt.json")
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if pg["min_recon"] < PG1_MIN_RECON:
        result = {"verdict": "F22_EDMD_FIT_COLLAPSE", "steps_done": 0,
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
    engine = F22Engine(
        generators=comp["generators"], transitions=comp["transitions"],
        t_pow=comp["t_pow"], recon=comp["recon"],
        action_names=comp.get("action_names"), n_actions=len(comp["generators"]),
        seed=args.seed, horizon=args.horizon, beta_sagnac=args.beta_sagnac,
        device=args.device, omega_bound=args.omega_bound,
        waypoint_advance_thresh=args.waypoint_advance_thresh,
        langevin_temp=args.langevin_temp)
    result = engine.run_gauntlet(
        env_names, steps_per_env=args.steps_per_env, seed=args.seed,
        trajectory_bank=args.trajectory_bank, trajectory_jsonl=args.trajectory_jsonl,
        ingress=ingress, out_dir=args.out_dir, receipt_out=receipt_out,
        allow_kill=True, pg1_min_recon=pg["min_recon"], env_goals=env_goals)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["verdict"] == "F22_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
