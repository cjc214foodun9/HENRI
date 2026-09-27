"""Operator-family registry and the ratified promotion gate.

WHY THIS EXISTS
---------------
`arc_task_functor.py` fits a per-slot DIAGONAL ridge, the family measured best
on real ARC.  Nothing in that module PREVENTED a later session from excising it
and swapping in a different family on the strength of a document.  That already
happened once: a supplied directive asked for the tripartite resonator to
replace it, and the resonator had already been measured WORSE.

This module makes the prohibition machine-readable and testable.

PROVENANCE
----------
Every constant below is copied at FULL precision from the measured receipt
    experiments/verification/action2_resonator_paired_ab_observed.json
    sha256 c18a4ad2b5f59df2889d4106cdca617ea52d07a28087c58e09bcb0c17c5d710a
`tests/unit/test_operator_promotion.py` re-reads that receipt and asserts these
constants equal it, so the numbers cannot silently drift.

A family is promoted ONLY by a paired held-out A/B that beats the incumbent by
tau.  Training loss and reward magnitude are NOT gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

RECEIPT_PATH = "experiments/verification/action2_resonator_paired_ab_observed.json"
RECEIPT_SHA256 = "c18a4ad2b5f59df2889d4106cdca617ea52d07a28087c58e09bcb0c17c5d710a"

# Pre-registered acceptance margin, identical to the A/B harness.
TAU = 0.01

INCUMBENT_FAMILY = "diag_ls"

# Measured mean held-out cosine on the same 60-task paired split,
# at FULL precision as recorded in the receipt.
MEASURED_HELDOUT: Dict[str, float] = {
    "diag_ls": 0.4306067231169436,
    "identity": 0.409158394626138,
    "resonator_tripartite": 0.41331040988528306,
}

# The receipt's own delta.  Asserted equal to the difference of the two means.
MEASURED_DELTA = -0.01729631323166053

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


def measured_delta(candidate: str) -> float:
    """Held-out delta of `candidate` against the incumbent."""
    return MEASURED_HELDOUT[candidate] - MEASURED_HELDOUT[INCUMBENT_FAMILY]


def assert_promotion_allowed(candidate: str, delta: float, tau: float = TAU) -> None:
    """Raise unless `candidate` is promotable AND strictly beats by `tau`.

    This is the single ratified gate.  A falsified family, the incumbent, or a
    delta below tau all fail closed.
    """
    if candidate == INCUMBENT_FAMILY:
        raise PromotionBlocked(
            "PROMOTION_BLOCKED: " + repr(candidate) + " is the incumbent; nothing to promote"
        )
    if candidate in FALSIFIED_FAMILIES:
        raise PromotionBlocked(
            "PROMOTION_BLOCKED: " + repr(candidate) + " measured no better than the "
            "incumbent (held-out " + repr(MEASURED_HELDOUT.get(candidate))
            + " vs " + repr(MEASURED_HELDOUT[INCUMBENT_FAMILY]) + "). "
            "The incumbent diagonal ridge stays."
        )
    if delta < tau:
        raise PromotionBlocked(
            "PROMOTION_BLOCKED: candidate delta " + repr(delta) + " < tau " + repr(tau)
        )


# Register the measured facts at import.
register(INCUMBENT_FAMILY, MEASURED_HELDOUT[INCUMBENT_FAMILY],
         "measured-best; the live fit path")
register("identity", MEASURED_HELDOUT["identity"], "baseline only; not a candidate")
register("resonator_tripartite", MEASURED_HELDOUT["resonator_tripartite"])

# END
