"""Carrier C1 — SO(8) rotor steering engine (default-OFF, G-series gauntlet).

Directive: Carrier_C1_Master_Directive_SO8_Rotor_Action_Generators.md
(SHA-256 2554c3fc4f2169bc5324219f91b839653134ce99a35972518e7fcc70ee728814).
Prereg: docs/spec/c1_rotor_generators_preregistration.md (sealed before code).
Base: P1GoalSteeringEngine (arc_p1_goal_steering_engine.py) @ b0f76ab.

Pathology (P2-0, sealed P2_NO_PROGRESS #81cf42c1, receipt 4c14d189):
actions entered the score path as learned per-block transitions T_a whose
effective displacement collapsed (~1e-4 micro-creeps; 395 creeps / 1,800
steps; 0 waypoint advances; 0 solves; mean_delta_nu_wp 2.07e-4).

C1 mechanism: replace the learned T_a candidate generation with EXACT
per-action SO(8) rotors R_a (Cayley, block-wise on [8192, 8]):
    cand_a = R_a psi_full        (all blocks, exact orthogonal rotation)
    drops[a] = |<cand_a, g>|^2 - |<psi, g>|^2      (P1 potential-drop form)
    j(a) = (clamp(drops[a], -1, 1) + 1) * pi_a^H   (calibrated affordance)
The M1 meter in the run loop is UNCHANGED (post-step actual frame vs active
waypoint), so mean_delta_nu_wp stays directly comparable to P2-0.

Flag: HENRI_C1_SO8_ROTORS=1 (fail-closed via require_c1_flag; launcher
routing in arc_g7_calibrated_engine.main mirrors use_g8/use_p1).

Verdicts (prereg C1 taxonomy): C1_ROTOR_VERIFIED (coupling >= 0.02 AND
solved >= 1) | C1_FALSIFIED_ACTION_COUPLING (mean_delta_nu < 0.02) |
C1_FALSIFIED_TASK_SOLVE_LG2 (coupling passed, solved == 0) |
C1_NO_AFFORDANCE_ENGAGEMENT | C1_GATE_LG3_LATENCY_FAILED | BLOCKED_INFRA.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from arc_g7_calibrated_engine import (
    G1_LATENCY_MS,
    G2_MIN_SOLVED,
    G4_MAX_AFFORDANCE,
)
from arc_p1_goal_steering_engine import (
    DV_CLAMP,
    P1GoalSteeringEngine,
)
from arc_c1_rotor_engine import (
    C1_FLAG,
    C1_SEED,
    FactorizedSO8ActionGenerators,
    require_c1_flag,
)

C1_LG1_MIN_DELTA_NU = 0.0200  # prereg C1_GATE_ACTION_COUPLING (P2-0: 2.07e-4)
# require_c1_flag is re-exported from arc_c1_rotor_engine for launcher parity
# with G8/P1 (which expose require_g8_flag/require_p1_flag from the engine
# module). Importing it here makes it available as arc_c1_steering_engine's
# attribute without a local def that would shadow the import.


class C1RotorSteeringEngine(P1GoalSteeringEngine):
    """G7/P1 + exact SO(8) rotor action candidates toward the full goal.

    Inherits P1's full-goal binding (_p1_full_goals, p1_bind_env_goal hook,
    guarded in the G4 run loop). score_all_actions falls back to the
    inherited P1/G7 scorer (byte-identical) when no full goal is bound.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._c1_rotors = FactorizedSO8ActionGenerators(
            num_actions=max(1, int(self.n_actions)), seed=int(self.seed),
        ).to(self.device)
        self.c1_last = {}
        self.c1_displacements = []
        self.c1_score_calls = 0
        self.c1_orth_err = None
        self.c1_latencies_ms = []

    # -- rotor-conditioned scoring (replaces P1's learned-T_a candidates) ----

    def score_all_actions(self, psi64, psi_full, waypoint=None):
        if self._p1_goal_full is None:
            return super().score_all_actions(psi64, psi_full, waypoint)

        psi_full = psi_full.float().to(self.device)
        batched = psi_full.dim() == 3
        if not batched:
            psi_full = psi_full.unsqueeze(0)          # [1, 8192, 8]
        B = psi_full.shape[0]
        flat = F.normalize(psi_full.reshape(B, -1), p=2, dim=-1)   # [1, D]
        g = F.normalize(self._p1_goal_full.to(self.device), p=2, dim=-1)  # [D]
        base_align = (flat * g.unsqueeze(0)).sum(-1).abs().clamp(0.0, 1.0) ** 2

        _use_events = (self.device.startswith("cuda")
                       and torch.cuda.is_available())
        _ev_start = _ev_end = None
        if _use_events:
            _ev_start = torch.cuda.Event(enable_timing=True)
            _ev_end = torch.cuda.Event(enable_timing=True)
            _ev_start.record()

        pi = self.predict_affordance(psi_full[0])[0]               # [A]
        drops = torch.zeros(self.n_actions, device=self.device)
        disps = torch.zeros(self.n_actions, device=self.device)
        n = psi_full.shape[1]                                       # blocks
        for a in range(self.n_actions):
            cand = self._c1_rotors.rotate(psi_full[0], a)           # [8192, 8]
            cand_flat = F.normalize(cand.reshape(1, -1), p=2, dim=-1)
            align_t = (cand_flat * g.unsqueeze(0)).sum(-1).abs().clamp(0.0, 1.0) ** 2
            drops[a] = (align_t - base_align)[0]
            disps[a] = (cand - psi_full[0]).norm().item() / (n ** 0.5)

        eff = torch.clamp(drops, min=-DV_CLAMP, max=DV_CLAMP) + 1.0
        j = eff * torch.pow(pi, self.horizon)
        if self.escape_state.get("active"):
            ggen = torch.Generator().manual_seed(
                self.seed + self.escape_state.get("steps", 0))
            noise = torch.sqrt(torch.tensor(
                2.0 * self.langevin_temp, device=self.device)) * \
                torch.randn(self.n_actions, generator=ggen).to(self.device)
            j = j + noise

        if _use_events and _ev_end is not None:
            _ev_end.record()
            torch.cuda.synchronize()
            self.c1_latencies_ms.append(float(_ev_start.elapsed_time(_ev_end)))

        sel = int(torch.argmax(j).item())
        self.c1_last = {
            "selected_action": sel,
            "potential_drops": [float(v) for v in drops.detach().cpu().tolist()],
            "base_align": float(base_align[0].item()),
            "selected_displacement": float(disps[sel].item()),
        }
        if self.c1_orth_err is None:
            self.c1_orth_err = self._c1_rotors.max_orth_error()
        self.c1_displacements.append(float(disps[sel].item()))
        self.c1_score_calls += 1
        return {self.action_names[i]: float(j[i].item())
                for i in range(self.n_actions)}

    # -- verdicts (prereg C1 taxonomy) --------------------------------------

    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        if self.c1_latencies_ms:
            mean_latency = sum(self.c1_latencies_ms) / len(self.c1_latencies_ms)
        if steps_done > 0 and updates == 0:
            return "C1_NO_AFFORDANCE_ENGAGEMENT"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "C1_GATE_LG3_LATENCY_FAILED"
        if mean_delta_nu is not None and mean_delta_nu < C1_LG1_MIN_DELTA_NU:
            return "C1_FALSIFIED_ACTION_COUPLING"
        if solved < G2_MIN_SOLVED:
            return "C1_FALSIFIED_TASK_SOLVE_LG2"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "C1_GATE_G4_AFFORDANCE_FAILED"
        return "C1_ROTOR_VERIFIED"

    def c1_receipt_fields(self) -> dict:
        return {
            "c1_score_calls": self.c1_score_calls,
            "c1_orth_err": self.c1_orth_err,
            "c1_mean_selected_displacement": (
                sum(self.c1_displacements) / len(self.c1_displacements)
                if self.c1_displacements else None),
            "c1_min_selected_displacement": (
                min(self.c1_displacements) if self.c1_displacements else None),
        }
