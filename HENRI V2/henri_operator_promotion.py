"""Operator-family registry and the ratified promotion gate.

WHY THIS EXISTS
---------------
`arc_task_functor.py` fits a per-slot DIAGONAL ridge, which is the family
measured best on real ARC.  Nothing in that module PREVENTED a later session
from excising it and swapping in a different family on the strength of a
document.  That is exactly what happened once already: a supplied directive
asked for the tripartite resonator to replace it, and the resonator had
already been measured WORSE.

This module makes the prohibition machine-readable and testable.

MEASURED FACTS ENCODED HERE (do not restate as opinion)
-------------------------------------------------------
* Incumbent `diag_ls` (per-slot diagonal ridge): held-out 0.430607 on 60 real
  ARC-AGI-2 training tasks, controls paired.
* `resonator_tripartite`: held-out 0.413310, delta -0.017296 vs tau 0.01 ->
  FALSIFIED_NO_IMPROVEMENT.  Argmax picked the identity triple (12,3,0) on
  48/60 tasks.  Receipt:
  experiments/verification/action2_resonator_paired_ab_observed.json
* Reflections ARE expressible by the resonator class (cos 1.000000); interior
  fill is NOT (0/8), because the class composes GLOBAL operators. Receipt:
  experiments/verification/action2_acceptance_reflection_containment_observed.json

A family is promoted ONLY by a paired held-out A/B that beats the incumbent by
tau.  Training loss and reward magnitude are NOT gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Pre-registered acceptance margin.  Same constant as the A/B harness.
TAU = 0.01

INCUMBENT_FAMILY = "diag_ls"

# Measured held-out scores on the same 60-task split.
MEASURED_HELDOUT: Dict[str, float] = {
    "diag_ls": 0.430607,
    "identity": 0.409158,
    "resonator_tripartite": 0.413310,
}

# Families that measured NO better than the incumbent.  Deliberately NOT
# promotable, so a future session cannot swap them in by assertion.
FALSIFIED_FAMILIES: Tuple[str, ...] = ("resonator_tripartite",)


class PromotionBlocked(Exception):
    """Raised when a candidate family fails the ratified promotion gate."""


@dataclass(frozen=True)
class Family:
    name: str
    promotable: bool
    heldout: float
    note: str


_REGISTRY: Dict[str, Family] = {}


def register(name: str, heldout: float, note: str = "") -> Family:
    """Register a family.  A falsified family is registered NON-promotable."""
    if name in FALSIFIED_FAMILIES:
        fam = Family(name, False, heldout,
                     note or "measured no better than the incumbent; not promotable")
    else:
        fam = Family(name, True, heldout, note)
    _REGISTRY[name] = fam
    return fam


def registry() -> Dict[str, Family]:
    return dict(_REGISTRY)


def assert_promotion_allowed(
    candidate: str,
    delta: float,
    tau: float = TAU,
) -> None:
    """Raise unless `candidate` is promotable AND strictly beats by `tau`.

    This is the single ratified gate.  A negative delta, a falsified family, or
    the incumbent itself all fail closed.
    """
    if candidate == INCUMBENT_FAMILY:
        raise PromotionBlocked(
            "PROMOTION_BLOCKED: %r is the incumbent; nothing to promote" % candidate
        )
    if candidate in FALSIFIED_FAMILIES:
        raise PromotionBlocked(
            "PROMOTION_BLOCKED: %r measured no better than the incumbent "
            "(held-out %.6f vs %.6f). The incumbent diagonal ridge stays."
            % (candidate, MEASURED_HELDOUT.get(candidate, float("nan")),
               MEASURED_HELDOUT[INCUMBENT_FAMILY])
        )
    if delta < tau:
        raise PromotionBlocked(
            "PROMOTION_BLOCKED: candidate delta %.6f < tau %.6f" % (delta, tau)
        )


def measured_delta(candidate: str) -> float:
    """Held-out delta of `candidate` against the incumbent, from the receipts."""
    return MEASURED_HELDOUT[candidate] - MEASURED_HELDOUT[INCUMBENT_FAMILY]


# Register the measured facts at import.
register(INCUMBENT_FAMILY, MEASURED_HELDOUT[INCUMBENT_FAMILY], "measured-best; live fit path")
register("identity", MEASURED_HELDOUT["identity"], "baseline only; not a candidate")
register("resonator_tripartite", MEASURED_HELDOUT["resonator_tripartite"])

# END
