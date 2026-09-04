"""Carrier P1 — Goal-Grounded Policy Steering Engine (default-OFF).

Packet: Carrier_P1_SpecContract___Alignment_Probe.md
(SHA-256 06e667c33134f82924c0d9500dfa8c8ee8ab5c1e2dd6484478296b4161fd3989,
388 lines / 18,819 B, HENRI-SPEC-2026-09-V3-CARRIER-P1-POLICY-GROUNDING).
Carrier: P1_GOAL_GROUNDED_POLICY_STEERING.
Base: arc_g7_calibrated_engine.G7CalibratedAffordanceEngine @ 4c71d4d.

Pathology (G7 launch #2 receipt a95e4ee6, OBSERVED): the live policy scored
  j(a) = align * (pi_a)^H with align = |<T_a^K psi64, wp>| computed in the
  D=64 kinematics domain. 1,800 live steps, 0 solved, 0 waypoint advances,
  mean_delta_nu_wp = 0.0. The goal term was unable to discriminate actions.

P1 mechanism (packet eq. 2.2-3.2): action-conditioned potential drop
  V(psi) = 1 - |<psi_norm, g_norm>|^2
  DeltaV(a) = V(psi_t) - V(T_a psi_t) = |<T_a psi_t, g>|^2 - |<psi_t, g>|^2
  j(a) = (clamp(DeltaV(a), -1, 1) + 1) * (pi_a)^H
Candidates are 1-step route-aware predictions with the SAME fitted operators
that define the affordance residual (C1 homology preserved):
  - topk/dense arm: per-block T_m on the action's top-k full-wave blocks;
    DeltaV measured against the FULL-wave terminal goal g_full [8192,8].
  - bridge/sparse arm: per-block T_m on the D=64 bridge state; DeltaV
    measured against the D=64 goal (waypoint) in the bridge domain.
When no full-wave goal is supplied the engine is byte-identical to G7
(default-OFF). Flag HENRI_P1_GOAL_STEERING=1 (fail-closed).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from arc_g7_calibrated_engine import G7CalibratedAffordanceEngine

try:
    from arc_g1_topological_engine import (
        DEFAULT_HORIZON,
        G1_LATENCY_MS,
        G2_MIN_SOLVED,
        G4_MAX_AFFORDANCE,
    )
except Exception:  # pragma: no cover - test isolation
    DEFAULT_HORIZON = 8
    G1_LATENCY_MS = 2.0
    G2_MIN_SOLVED = 1
    G4_MAX_AFFORDANCE = 0.0500

FLAG_P1 = "HENRI_P1_GOAL_STEERING"
DV_CLAMP = 1.0
P1_LG1_MIN_DELTA_NU = 0.0500


def build_p1_full_goals(bank_npz_path, bank_jsonl_path, env_names, device="cpu"):
    """Per-env FULL-wave terminal goals from the bank (row == jsonl line).

    Uses the alignment proven by probe_trajectory_bank_alignment (rows map
    1:1 to jsonl lines; per-env rows contiguous; the engine's goal row is the
    env's last jsonl line). Returns {env_name: goal [65536] unit flat} using
    only envs with rows in the bank.
    """
    import numpy as np

    from arc_f15_trajectory_engine import load_environment_indices

    data = np.load(bank_npz_path)
    if "psi" not in data.files:
        raise KeyError("bank npz missing key 'psi' (henri.arc-trajectory-bank.v1)")
    waves = torch.from_numpy(np.asarray(data["psi"])).float()
    goals = {}
    for name in env_names:
        idxs = load_environment_indices(bank_jsonl_path, name)
        if not idxs:
            continue
        g = F.normalize(waves[int(idxs[-1])].to(device), p=2, dim=-1)
        goals[name] = g.detach()
    return goals


def require_p1_flag() -> None:
    if os.environ.get(FLAG_P1) != "1":
        raise SystemExit(f"BLOCKED: {FLAG_P1} not set (default-OFF)")


class P1GoalSteeringEngine(G7CalibratedAffordanceEngine):
    """G7 + action-conditioned potential-drop steering toward a full-wave goal.

    The full-wave goal is injected per environment through `_p1_full_goals`
    (dict env_name -> goal wave [8192, 8] or [65536] flat, unit-normalized)
    by the P1 launcher before `run_gauntlet`. The G4 run loop assigns it to
    `self._p1_goal_full` at each environment boundary via a guarded hook.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p1_full_goals = None      # {env_name: full goal wave}
        self._p1_goal_full = None       # current env full goal [8192,8]
        self.p1_last = {}               # telemetry: potential drops per action
        self._p1_latencies_ms = []      # LG3: local score-path kernel latency
        self._p1_drop_accum = None      # per-action potential-drop accumulator
        self._p1_score_calls = 0

    # -- full-goal injection (guarded hook consumed by the run loop) --------
    def p1_bind_env_goal(self, env_name: str) -> None:
        if self._p1_full_goals and self._p1_full_goals.get(env_name) is not None:
            g = self._p1_full_goals[env_name].float().to(self.device)
            self._p1_goal_full = F.normalize(g.reshape(-1), p=2, dim=-1)
        else:
            self._p1_goal_full = None

    # -- conditioned scoring -------------------------------------------------
    def score_all_actions(self, psi64, psi_full, waypoint=None):
        """Action-conditioned potential-drop policy (packet eq. 3.2).

        Falls back to the inherited G7/G4 scorer (byte-identical) when no
        full-wave goal is bound. Returns the same dict-of-floats contract.
        """
        if self._p1_goal_full is None:
            return super().score_all_actions(psi64, psi_full, waypoint)

        psi_full = psi_full.float().to(self.device)
        batched = psi_full.dim() == 3
        if not batched:
            psi_full = psi_full.unsqueeze(0)          # [1, 8192, 8]
        B = psi_full.shape[0]
        flat = F.normalize(psi_full.reshape(B, -1), p=2, dim=-1)   # [1, D]
        g = self._p1_goal_full.to(self.device)                     # [D]
        g = F.normalize(g, p=2, dim=-1)
        base_align = (flat * g.unsqueeze(0)).sum(-1).abs().clamp(0.0, 1.0) ** 2  # [B]

        _use_events = (self.device.startswith("cuda")
                       and torch.cuda.is_available())
        _ev_start = _ev_end = None
        if _use_events:
            _ev_start = torch.cuda.Event(enable_timing=True)
            _ev_end = torch.cuda.Event(enable_timing=True)
            _ev_start.record()

        pi = self.predict_affordance(psi_full[0])[0]               # [A] calibrated
        wp64 = F.normalize(
            (waypoint if waypoint is not None else self._active_waypoint())
            .float().to(self.device), p=2, dim=-1)                 # [64]

        drops = torch.zeros(self.n_actions, device=self.device)
        for a in range(self.n_actions):
            if self.bridge_route_flags.get(int(a), False):
                # Bridge arm: candidate in the D=64 domain.
                if self.ingress is None or not self.bridge_transitions.get(int(a)):
                    drops[a] = 0.0
                    continue
                psi64b = F.normalize(
                    psi64.float().to(self.device).reshape(-1), p=2, dim=-1)  # [64]
                cand64 = psi64b.clone().view(8, 8)
                for m, tm in self.bridge_transitions[int(a)].items():
                    cand64[m] = F.normalize(tm.to(self.device) @ cand64[m], p=2, dim=-1)
                cand64 = F.normalize(cand64.reshape(-1), p=2, dim=-1)
                align_t = (cand64 * wp64).sum(-1).abs().clamp(0.0, 1.0) ** 2
                align_0 = (psi64b * wp64).sum(-1).abs().clamp(0.0, 1.0) ** 2
                drops[a] = align_t - align_0
            else:
                # Topk arm: candidate in the full-wave domain.
                topk = self.topk_masks.get(int(a))
                trans = self.transitions_g4.get(int(a))
                if topk is None or not trans:
                    drops[a] = 0.0
                    continue
                idx = topk.to(self.device)
                cand = psi_full.clone()
                for m in idx.tolist():
                    tm = trans[int(m)].to(self.device)             # [8, 8]
                    cand[:, m, :] = torch.einsum(
                        "ij,bj->bi", tm, cand[:, m, :])
                cand_flat = F.normalize(cand.reshape(B, -1), p=2, dim=-1)
                align_t = (cand_flat * g.unsqueeze(0)).sum(-1).abs().clamp(0.0, 1.0) ** 2
                align_0 = base_align
                drops[a] = (align_t - align_0)[0]

        eff = torch.clamp(drops, min=-DV_CLAMP, max=DV_CLAMP) + 1.0
        j = eff * torch.pow(pi, self.horizon)
        if self.escape_state.get("active"):
            ggen = torch.Generator().manual_seed(self.seed + self.escape_state.get("steps", 0))
            noise = torch.sqrt(torch.tensor(2.0 * self.langevin_temp, device=self.device)) * \
                torch.randn(self.n_actions, generator=ggen).to(self.device)
            j = j + noise

        if _use_events and _ev_end is not None:
            _ev_end.record()
            torch.cuda.synchronize()
            self._p1_latencies_ms.append(float(_ev_start.elapsed_time(_ev_end)))

        if self._p1_drop_accum is None:
            self._p1_drop_accum = drops.detach().cpu().clone()
        else:
            self._p1_drop_accum += drops.detach().cpu()
        self._p1_score_calls += 1

        self.p1_last = {
            "selected_action": int(torch.argmax(j).item()),
            "potential_drops": [float(v) for v in drops.detach().cpu().tolist()],
            "base_align": float(base_align[0].item()),
        }
        return {self.action_names[i]: float(j[i].item()) for i in range(self.n_actions)}

    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        # LG3 scope (packet YAML): the local score path, timed with CUDA
        # events, NOT the remote-arcade round trip. When local timings exist
        # they replace the inherited wall-clock mean for the latency gate.
        if self._p1_latencies_ms:
            mean_latency = sum(self._p1_latencies_ms) / len(self._p1_latencies_ms)
        if steps_done > 0 and updates == 0:
            return "P1_NO_AFFORDANCE_ENGAGEMENT"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "P1_GATE_LG3_LATENCY_FAILED"
        if solved < G2_MIN_SOLVED:
            return "P1_GATE_LG2_SOLVED_FAILED"
        if mean_delta_nu is not None and mean_delta_nu < P1_LG1_MIN_DELTA_NU:
            return "P1_GATE_LG1_STAGNATION"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "P1_GATE_G4_AFFORDANCE_FAILED"
        return "P1_POLICY_GROUNDING_VERIFIED"
