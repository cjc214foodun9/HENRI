"""henri_causal_planner.py — bounded exteroceptive EFE planner (HENRI-DIR-2026-09-BUNDLE-VLA-GROUNDING, Component 2).

Functional form (directive §3, Component 2):

    G(a) = beta_prag * d_goal(Psi_hat_{t+1}^a, Psi_goal)
         + lambda_epis * S_RMS(Psi_hat_{t+1}^a, Psi_baseplate)
         - gamma_val * E[Delta_nu_a]

Normalization constraints (all in [0, 1]):
    d_goal   = (1 - cos(Psi_hat^a, Psi_goal)) / 2   (hyperspherical cosine distance)
    S_RMS    = 0.5 * ||Psi_hat^a - Psi_base||_F / sqrt(D)   (unit waves: antipodal = 1.0)
    Delta_nu = clamp(Score_t - Score_{t-1}, -1.0, 1.0)      (observable external delta)

Failure-trace coupling (directive §3, Component 2): if Delta_nu <= 0 over a
rolling window of k=5 steps for the selected action, retroactively assign
nu = -1.0 and inject anisotropic thermal noise into the failing action's
operator coordinates.

Default-OFF: gated by HENRI_CAUSAL_PLANNER=1; imported lazily by the
approved orchestrator consumer only when enabled.  This module does NOT
reimplement transition dynamics.  It consumes a caller-supplied prediction map
{action_index: predicted_wave} (the live LowRankCoupledTransition supplies that
map in the approved integration).

All methods are CPU-testable at reduced scale; no checkpoint requirement.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


class BoundedExteroceptiveEFEPlanner:
    """Exteroceptive expected-free-energy action scorer with bounded channels.

    Selection rule: argmin_a G(a).  Empty predictions fail closed (ValueError).
    """

    def __init__(
        self,
        d_model: int,
        num_actions: int = 8,
        beta_prag: float = 1.0,
        lambda_epis: float = 1.0,
        gamma_val: float = 1.0,
        k_failure: int = 5,
        noise_amp: float = 1e-2,
    ) -> None:
        if d_model <= 0 or d_model % 8 != 0:
            raise ValueError(f"d_model must be a positive multiple of 8; got {d_model}")
        self.d_model = d_model
        self.num_actions = num_actions
        self.beta_prag = beta_prag
        self.lambda_epis = lambda_epis
        self.gamma_val = gamma_val
        self.k_failure = k_failure
        self.noise_amp = noise_amp

        self._goal: Optional[torch.Tensor] = None
        self._baseplate: Optional[torch.Tensor] = None  # [B, D] unit rows

        # per-action observable outcome history (bounded rolling)
        self._delta_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max(1, k_failure))
        )
        self._retroactive_fail: Dict[int, bool] = defaultdict(bool)
        # per-action operator coordinate tensors (noise injection targets)
        self.operators: Dict[int, torch.Tensor] = {
            a: torch.zeros(d_model) for a in range(num_actions)
        }

    # ------------------------------------------------------------------ state
    def register_goal(self, wave: torch.Tensor) -> None:
        w = F.normalize(wave.reshape(-1).to(torch.float32), p=2, dim=-1)
        if w.shape[-1] != self.d_model:
            raise ValueError(f"goal dim {w.shape[-1]} != d_model {self.d_model}")
        self._goal = w

    def register_baseplate(self, waves: torch.Tensor) -> None:
        """waves: [B, D] unit-norm constraint (frozen boundary reference) rows."""
        w = F.normalize(waves.to(torch.float32), p=2, dim=-1)
        if w.shape[-1] != self.d_model:
            raise ValueError(f"baseplate dim {w.shape[-1]} != d_model {self.d_model}")
        self._baseplate = w

    def reset(self) -> None:
        self._delta_history.clear()
        self._retroactive_fail.clear()
        self.operators = {a: torch.zeros(self.d_model) for a in range(self.num_actions)}

    # ------------------------------------------------------------- distances
    @staticmethod
    def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Normalized hyperspherical cosine distance in [0, 1]."""
        a = F.normalize(a.reshape(-1).to(torch.float32), p=2, dim=-1)
        b = F.normalize(b.reshape(-1).to(torch.float32), p=2, dim=-1)
        return (1.0 - (a @ b).clamp(-1.0, 1.0)) / 2.0

    @classmethod
    def rms_drift(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """RMS Frobenius phase drift scaled to [0, 1] for unit waves."""
        D = a.reshape(-1).shape[-1]
        diff = a.reshape(-1).to(torch.float32) - b.reshape(-1).to(torch.float32)
        return (0.5 * diff.norm(p=2) / math.sqrt(D)).clamp(0.0, 1.0)

    # ------------------------------------------------------------- outcomes
    def observe_outcome(self, action_idx: int, delta_nu: float) -> None:
        """Record an observable external score delta, clamped to [-1, 1]."""
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"action {action_idx} out of range")
        d = max(-1.0, min(1.0, float(delta_nu)))
        hist = self._delta_history[action_idx]
        hist.append(d)
        # rolling failure trace: last k outcomes all <= 0 -> retroactive nu = -1
        if len(hist) >= self.k_failure and all(x <= 0.0 for x in hist):
            self._retroactive_fail[action_idx] = True
        else:
            self._retroactive_fail[action_idx] = False

    def expected_delta(self, action_idx: int) -> float:
        hist = self._delta_history[action_idx]
        if not hist:
            return 0.0
        if self._retroactive_fail[action_idx]:
            return -1.0
        return sum(hist) / len(hist)

    def apply_failure_trace(self, action_idx: int, pred_wave: torch.Tensor) -> float:
        """Anisotropic noise injection on a failing action's operator coordinates.

        Direction = goal - prediction (normalized); amplitude = noise_amp.
        Returns the injected noise norm (0.0 when no failure flag is set).
        """
        if not self._retroactive_fail[action_idx]:
            return 0.0
        if self._goal is None:
            direction = torch.randn(self.d_model)
        else:
            direction = self._goal - F.normalize(
                pred_wave.reshape(-1).to(torch.float32), p=2, dim=-1
            )
        direction = F.normalize(direction, p=2, dim=-1)
        op = self.operators[action_idx]
        noise = self.noise_amp * direction
        self.operators[action_idx] = (op + noise).detach().clone()
        return float(noise.norm(p=2))

    # ------------------------------------------------------------------ G(a)
    def score_action(
        self,
        pred_wave: torch.Tensor,
        action_idx: int,
        goal_wave: Optional[torch.Tensor] = None,
        baseplate: Optional[torch.Tensor] = None,
    ) -> float:
        """G(a) = beta_prag*d_goal + lambda_epis*S_RMS - gamma_val*E[Delta_nu]."""
        goal = self._goal if goal_wave is None else goal_wave
        d_goal = 0.5  # no goal -> neutral (midpoint) so it never dominates alone
        if goal is not None:
            d_goal = float(self.cosine_distance(pred_wave, goal))
        plate = self._baseplate if baseplate is None else baseplate
        s_rms = 0.0
        if plate is not None:
            s_rms = float(
                torch.stack([self.rms_drift(pred_wave, b) for b in plate]).mean()
                if plate.ndim == 2
                else self.rms_drift(pred_wave, plate)
            )
        e_dnu = self.expected_delta(action_idx)
        return self.beta_prag * d_goal + self.lambda_epis * s_rms - self.gamma_val * e_dnu

    def select_action(
        self,
        predictions: Dict[int, torch.Tensor],
        legal: Optional[Sequence[int]] = None,
    ) -> int:
        """argmin_a G(a) over legal candidates; fail-closed on empty input."""
        if not predictions:
            raise ValueError("empty predictions map (fail-closed: no action selected)")
        cand = list(predictions.keys()) if legal is None else [
            a for a in legal if a in predictions
        ]
        if not cand:
            raise ValueError("no legal action has a prediction (fail-closed)")
        scored = [(self.score_action(predictions[a], a), a) for a in cand]
        scored.sort(key=lambda t: (t[0], t[1]))
        return scored[0][1]
