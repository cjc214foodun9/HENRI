"""henri_curriculum_governor.py -- dynamic entropy escalation (default-OFF, additive).

Defect 6 / arXiv:2609.30063 amendment.

OBSERVED failure it fixes: HENRI's Stage-0 run on a stationary 1D byte-tape VM
converged in ~2,000 rounds and spent 99.66% of a 10-billion-token budget flat.
The 1D tape was the defect: a stationary grammar exhausts its algorithmic
entropy, so

    lim_{t->inf} K(curriculum_t | curriculum_{<t}) = 0

and the learner receives no new learnable structure (epiplexity -> 0).  The
governor monitors empirical loss variance and ESCALATES program complexity when
the curriculum has been exhausted.

Rungs (each adds a structural axis the previous rung cannot express):
    1  byte-tape VM arithmetic           (1D, scalar)
    2  2D torus shift + colour perm      (2D, spatial)
    3  Jordan-curve interior/exterior    (topology)
    4  hierarchical role-filler binding  (composition)
    5  action-conditioned causal graphs  (multi-step dynamics)

Default-OFF: constructors raise unless HENRI_CURRICULUM_GOVERNOR=1.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ENV_ENABLE_FLAG = "HENRI_CURRICULUM_GOVERNOR"

RUNGS: List[str] = [
    "byte_tape_arithmetic",
    "torus_shift_colour",
    "jordan_containment",
    "hierarchical_binding",
    "action_conditioned_causal",
]


def curriculum_governor_enabled() -> bool:
    return os.environ.get(ENV_ENABLE_FLAG, "").strip() in {"1", "true", "True", "yes"}


class CurriculumGovernorError(RuntimeError):
    """Base class for governor failures."""


class CurriculumGovernorDisabledError(CurriculumGovernorError):
    """Raised when the governor is used with HENRI_CURRICULUM_GOVERNOR unset or 0."""


@dataclass
class CurriculumParams:
    """Program-complexity parameters emitted for the current rung."""

    rung: int
    name: str
    token_budget: int
    grid_dim: int
    entities: int
    relations: int
    causal_steps: int


@dataclass
class GovernorState:
    rung: int = 0
    rounds_at_rung: int = 0
    escalations: int = 0
    last_variance: float = float("inf")
    history: List[Dict[str, Any]] = field(default_factory=list)


class CurriculumGovernor:
    """Escalate program complexity when the curriculum stops teaching.

    Escalation rule (pre-registered, fixed before any run):
        escalate iff  sigma^2(loss) < variance_floor  for `patience` consecutive
        observations.  The variance floor default 1e-4 is the observed Stage-0
        plateau level (sigma^2 < 1.92e-4), set below it deliberately.

    The rule never de-escalates and never skips a rung: monotone, deterministic.
    """

    def __init__(
        self,
        *,
        variance_floor: float = 1e-4,
        patience: int = 3,
        window: int = 5,
        max_rung: Optional[int] = None,
    ) -> None:
        if not curriculum_governor_enabled():
            raise CurriculumGovernorDisabledError(
                f"{ENV_ENABLE_FLAG} is not set; curriculum governor is disabled"
            )
        if variance_floor <= 0:
            raise CurriculumGovernorError("variance_floor must be positive")
        if patience < 1 or window < 2:
            raise CurriculumGovernorError("patience >= 1 and window >= 2 required")
        self.variance_floor = float(variance_floor)
        self.patience = int(patience)
        self.window = int(window)
        self.max_rung = len(RUNGS) - 1 if max_rung is None else int(max_rung)
        self._losses: List[float] = []
        self.state = GovernorState()

    def _variance(self) -> float:
        w = self._losses[-self.window:]
        if len(w) < 2:
            return float("inf")
        mu = sum(w) / len(w)
        return sum((x - mu) ** 2 for x in w) / (len(w) - 1)

    def observe(self, loss: float) -> Dict[str, Any]:
        """Feed one loss observation; return the (possibly escalated) state."""
        if not math.isfinite(loss):
            raise CurriculumGovernorError(f"loss must be finite; got {loss!r}")
        self._losses.append(float(loss))
        var = self._variance()
        self.state.last_variance = var
        escalated = False
        if var < self.variance_floor:
            self.state.rounds_at_rung += 1
        else:
            self.state.rounds_at_rung = 0
        if self.state.rounds_at_rung >= self.patience and self.state.rung < self.max_rung:
            self.state.rung += 1
            self.state.rounds_at_rung = 0
            self.state.escalations += 1
            escalated = True
        record = {
            "rung": self.state.rung,
            "variance": var,
            "rounds_at_rung": self.state.rounds_at_rung,
            "escalated": escalated,
            "loss": float(loss),
        }
        self.state.history.append(record)
        return record

    def params(self, z: int = 4, action_space: int = 8) -> CurriculumParams:
        """Program-complexity parameters for the current rung."""
        r = self.state.rung
        return CurriculumParams(
            rung=r,
            name=RUNGS[r],
            token_budget=2 ** (r + 3) * 16,
            grid_dim=0 if r == 0 else min(64, 2 ** (r + 1)),
            entities=1 + r,
            relations=r,
            causal_steps=0 if r < 4 else 1 + (r - 4),
        )

    def exhausted(self) -> bool:
        """True when every rung has been consumed -- stop, do not burn compute."""
        return self.state.rung >= self.max_rung and self.state.rounds_at_rung >= self.patience
