"""Carrier G8 Phase B — Sub-Goal Steering Engine (default-OFF).

Packet: Carrier_G8_PhaseB_Engine_Wiring_and_Promotion_Protocol.md
(SHA-256 2c5f70b51062b4c63015995f6114353b26e45d5ebf5df088550a19f148e520d5,
159 lines / 8,959 B, HENRI-SPEC-2026-09-V3-CARRIER-G8-PHASE-B-WIRING).
Base: arc_p1_goal_steering_engine.P1GoalSteeringEngine @ 71276baf
(G8 Phase A tip; local == origin).

Mechanism (packet eq. 1-3): the agent locks onto the immediate NEXT waypoint
k* instead of the terminal attractor. Promotion occurs when the CURRENT full
state aligns to chain[k*] >= 0.60 (packet threshold); scoring and the M1 meter
then reference chain[k*+1].

Representation: waypoint chain = selected bank rows
(arc_g8_waypoint_extractor.extract_waypoints) normalized to S^{D-1} flat
[K, D]. NO dense operators (packet memory invariant ~6.9 MB for 12x9x64KB).

Meter: G8 re-targets the repaired M1 delta-nu measurement to the ACTIVE
waypoint in the FULL domain (|align(psi_full_t, ref)| ->
|align(psi_full_{t+1}, ref)| with the post-step ACTUAL frame), replacing the
legacy D=64 static-terminal meter ONLY while a G8 chain is bound (guarded hook
in arc_g4_aligned_engine.run_gauntlet; default path byte-identical). Meter
values are NOT comparable to the P1/P2 [64]-domain meter.

Flag: HENRI_G8_SUBGOAL=1 (fail-closed via require_g8_flag). Launcher wiring in
arc_g7_calibrated_engine.main(): lazy import under the flag; the default G7/P1
paths are untouched when the flag is absent (differential default-OFF proof).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from arc_g8_waypoint_extractor import extract_waypoints
from arc_p1_goal_steering_engine import P1GoalSteeringEngine

try:  # pragma: no cover - test isolation
    from arc_g1_topological_engine import (
        G1_LATENCY_MS,
        G2_MIN_SOLVED,
        G4_MAX_AFFORDANCE,
    )
except Exception:  # pragma: no cover - test isolation
    G1_LATENCY_MS = 2.0
    G2_MIN_SOLVED = 1
    G4_MAX_AFFORDANCE = 0.0500

FLAG_G8 = "HENRI_G8_SUBGOAL"
G8_PROMOTE_THRESHOLD = 0.60
G8_MIN_PROMOTED_ENVS = 6
G8_LG1_MIN_DELTA_NU = 0.0100
G8_MIN_CHAIN_LEN = 2


def require_g8_flag() -> None:
    if os.environ.get(FLAG_G8) != "1":
        raise SystemExit(f"BLOCKED: {FLAG_G8} not set (default-OFF)")


def g8_step_promotion(align: float, k: int, K: int,
                      thresh: float = G8_PROMOTE_THRESHOLD):
    """Pure promotion state transition (packet eq. 2).

    Returns (promoted: bool, new_k: int). k is monotone increasing and capped
    at K-1. Alignment is the |inner product| of two unit vectors in [0,1].
    """
    if k < 0 or K < 1 or k >= K:
        raise ValueError(f"invalid waypoint index k={k} K={K}")
    if align >= thresh and k < (K - 1):
        return True, k + 1
    return False, k


def build_g8_waypoint_chains(bank_npz_path: str, bank_jsonl_path: str,
                             env_names, device="cpu", min_sep: int = 16):
    """Per-env waypoint chains from the trajectory bank.

    Rows are selected by extract_waypoints (curvature peaks + terminal) and
    the chain is the normalized bank wave at each selected global row index,
    flat [K, D] on S^{D-1}, ordered by row, terminal last. Returns
    {env_name: Tensor[K, D]} for envs present in the bank with K >= 2.
    """
    data = np.load(bank_npz_path)
    if "psi" not in data.files:
        raise KeyError("bank npz missing key 'psi' (henri.arc-trajectory-bank.v1)")
    psi_np = np.asarray(data["psi"])

    rows_per_env: dict = {}
    with open(bank_jsonl_path, "r", encoding="utf-8") as fh:
        import json as _json
        for i, line in enumerate(fh):
            rec = _json.loads(line)
            rows_per_env.setdefault(rec["env"], []).append(i)

    waypoints = extract_waypoints(rows_per_env, psi_np, min_sep=min_sep)
    chains: dict = {}
    for name in env_names:
        entries = waypoints.get(name) or []
        if len(entries) < G8_MIN_CHAIN_LEN:
            continue
        entries = sorted(entries, key=lambda t: t[0])
        idxs = [r for r, _ in entries]
        rows = psi_np[idxs].astype(np.float32)
        chain = torch.from_numpy(rows).float()
        chain = F.normalize(chain, p=2, dim=-1)
        chains[name] = chain.to(device).detach()
    return chains


class G8SubgoalSteeringEngine(P1GoalSteeringEngine):
    """P1 goal steering with a promoted sub-goal waypoint chain.

    When a G8 chain is bound for the current env (via p1_bind_env_goal
    override), the P1 full-wave goal reference is rebound to the ACTIVE
    waypoint each step and the M1 meter targets that same active reference.
    When no chain exists the engine is byte-identical to P1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._g8_chains = None      # {env_name: Tensor[K, D]} set by launcher
        self._g8_chain = None       # current env chain Tensor[K, D]
        self._g8_k = 0              # active waypoint index k*
        self._g8_meter_ref = None   # flat unit [D] reference for the M1 meter
        self._g8_promotions = 0
        self._g8_promos_by_env = {}
        self._g8_env = None

    # -- env-bound chain injection (replaces the P1 single-goal hook) --------
    def p1_bind_env_goal(self, env_name: str) -> None:
        chain = None
        if self._g8_chains is not None:
            chain = self._g8_chains.get(env_name)
        if chain is not None and chain.shape[0] >= G8_MIN_CHAIN_LEN:
            self._g8_chain = chain.float().to(self.device)
            self._g8_k = 0
            self._g8_meter_ref = self._g8_chain[0].detach()
            self._p1_goal_full = self._g8_chain[0].detach()
            self._g8_env = env_name
            self._g8_promos_by_env.setdefault(env_name, 0)
        else:
            self._g8_chain = None
            self._g8_meter_ref = None
            super().p1_bind_env_goal(env_name)

    # -- per-step promotion + active-goal rebind -----------------------------
    def _g8_promote_current(self, psi_full) -> bool:
        """Check promotion against the ACTIVE waypoint and rebind the goal.

        psi_full: [8192, 8] (or flat [D]) current full wave. Returns True when
        the pointer advanced. Called before every scoring pass while a G8
        chain is bound.
        """
        chain = self._g8_chain
        if chain is None:
            return False
        flat = F.normalize(psi_full.float().reshape(-1), p=2, dim=-1).to(self.device)
        k = self._g8_k
        K = chain.shape[0]
        if k >= K - 1:
            ref = chain[k].detach()
            self._g8_meter_ref = ref
            self._p1_goal_full = ref
            return False
        align = float((flat * chain[k]).sum(-1).abs().clamp(0.0, 1.0).item())
        promoted, new_k = g8_step_promotion(align, k, K, G8_PROMOTE_THRESHOLD)
        if promoted:
            self._g8_k = new_k
            self._g8_promotions += 1
            self._g8_promos_by_env[self._g8_env or ""] = \
                self._g8_promos_by_env.get(self._g8_env or "", 0) + 1
        ref = chain[self._g8_k].detach()
        self._g8_meter_ref = ref
        self._p1_goal_full = ref
        return promoted

    # -- scoring override: promote first, then delegate to P1 scoring --------
    def score_all_actions(self, psi64, psi_full, waypoint=None):
        if self._g8_chain is not None:
            self._g8_promote_current(psi_full)
        return super().score_all_actions(psi64, psi_full, waypoint)

    # -- verdict: G8 gates (reachability before outcome; flags after) --------
    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        if self._p1_latencies_ms:
            mean_latency = sum(self._p1_latencies_ms) / len(self._p1_latencies_ms)
        if steps_done > 0 and updates == 0:
            return "G8_NO_AFFORDANCE_ENGAGEMENT"
        promoted_envs = sum(1 for v in self._g8_promos_by_env.values() if v > 0)
        if promoted_envs < G8_MIN_PROMOTED_ENVS:
            return "G8_FALSIFY_SUBGOAL_REACHABILITY"
        if solved < G2_MIN_SOLVED:
            return "G8_FALSIFY_TASK_SOLVE_LG2"
        if mean_delta_nu is not None and mean_delta_nu < G8_LG1_MIN_DELTA_NU:
            return "G8_FLAG_LOW_VELOCITY_PROGRESSION"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "G8_FLAG_KERNEL_PERF_REGRESSION"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "G8_GATE_G4_AFFORDANCE_FAILED"
        return "G8_SUBGOAL_STEERING_VERIFIED"

    # -- receipt fields --------------------------------------------------------
    def g8_receipt_fields(self) -> dict:
        return {
            "g8_promote_threshold": G8_PROMOTE_THRESHOLD,
            "g8_promotions_total": self._g8_promotions,
            "g8_promotions_by_env": dict(self._g8_promos_by_env),
            "g8_envs_with_promotion": sum(
                1 for v in self._g8_promos_by_env.values() if v > 0),
            "g8_meter_active": self._g8_meter_ref is not None,
        }
