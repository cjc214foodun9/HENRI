"""Machine-readable promotion gate for the egress Hopfield temperature (beta).

WHY THIS EXISTS
---------------
`henri_hopfield_egress.CanonicalCodebookEgress` changed its default beta from the
sealed 8.0 to the directive's 26.10. That change was measured on REAL encoded waves
at d=512, M=64 and M=174 -- strictly better on a noise-tolerance instrument, with
four controls passing.

It was NOT measured at the scale the constant's own prior receipt names:

    receipts/hopfield_beta_calibration.json, `next_step`:
        "Changing the sealed constant requires replicating the M=10,000 / D=65,536
         capacity contract and passing the receipt-pinned promotion gate.
         This sweep deliberately does not."

Before this module, that requirement existed ONLY AS PROSE in a receipt and in a
docstring. A prose-only gate is bypassable by the next session that reads the doc
and not the receipt -- which is the SAME failure `henri_operator_promotion.py` was
written to prevent for operator families ("a supplied directive asked for the
tripartite resonator ... and the resonator had already been measured WORSE").

So the requirement is now a callable gate plus tests that re-read the receipts, and
`tests/unit/test_egress_promotion.py` asserts the constants equal them. The default
stays 26.10 and the gate stays CLOSED, so the code is honest about both facts at once.

CRLF NOTE
---------
The pinned digests are computed over bytes with CRLF normalised to LF. The sibling
gate (`henri_operator_promotion.py`) hashes raw bytes, which is checkout-fragile on
a host that rewrites line endings (this repo's worktree warns "LF will be replaced
by CRLF the next time Git touches it"). Normalising is the fix, and the digests here
were derived from the git-STORED blob, not the working tree.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))

# Receipts that carry the evidence for the 8.0 -> 26.10 change. Digests are over
# LF-normalised bytes of the git-stored blobs.
RECEIPTS: Dict[str, Tuple[str, str]] = {
    "m64": (
        "receipts/egress_beta_gate_m64.json",
        "7608f9f53249e5a7c680510b97ede5b6d393678720c585eb1f4793c8c3f32334",
    ),
    "m174": (
        "receipts/egress_beta_gate_m174.json",
        "124d73d5e3bd55d8018cba7bcae52cdb62c2712d8736cf4037240bbd0a671c67",
    ),
}

# The two constants, and the state of the change between them.
SEALED_BETA = 8.0
ADOPTED_BETA = 26.10

# The promotion scale named by the prior receipt. NOT yet replicated.
PROMOTION_M = 10_000
PROMOTION_D = 65_536

# What was actually measured. (M, d_model) pairs.
MEASURED_AT: Tuple[Tuple[int, int], ...] = ((64, 512), (174, 512))

# Noise tolerance (largest ||noise||/||engram|| keeping clean_cos >= 0.90), read at
# FULL precision from the receipts. Up is better.
MEASURED_TOLERANCE: Dict[str, Dict[float, float]] = {
    "m64": {8.0: 2.0, 26.1: 4.0, 64.0: 6.0},
    "m174": {8.0: 1.5, 26.1: 4.0, 64.0: 4.0},
}

# The evidence backed ONLY these. beta >= 64 reached tolerance 6.0 at M/d=0.125,
# i.e. 26.10 is a good value, NOT the argmax.
MEASURED_CLAIM = "strictly better than the sealed 8.0 at both tested densities"
NOT_MEASURED_CLAIM = "that 26.10 is the optimum"


class PromotionGated(Exception):
    """Raised when a beta promotion is attempted below the named promotion scale."""


def _normalised_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        raw = fh.read()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def audit() -> Dict[str, object]:
    """Re-read the receipts and report what they actually support.

    This is the single source of truth for `is_provisional()` and
    `assert_promotion_allowed()`. It fails closed: a missing, unreadable, or
    digest-mismatched receipt reports `receipts_ok = False`.
    """
    digests: Dict[str, str] = {}
    verdicts: Dict[str, str] = {}
    tolerances: Dict[str, Dict[str, float]] = {}
    ok = True
    for tag, (rel, pinned) in RECEIPTS.items():
        path = os.path.join(_HERE, rel)
        if not os.path.exists(path):
            digests[tag] = "MISSING"
            ok = False
            continue
        got = _normalised_sha256(path)
        digests[tag] = got
        if got != pinned:
            ok = False
        with open(path, "rb") as fh:
            d = json.loads(fh.read().decode("utf-8"))
        verdicts[tag] = str(d.get("verdict"))
        tolerances[tag] = {
            str(r["beta"]): r["tolerance"]
            for r in d.get("real", [])
            if r.get("tolerance") is not None
        }
    return {
        "receipts_ok": ok,
        "digests": digests,
        "verdicts": verdicts,
        "tolerances": tolerances,
        "measured_at": list(MEASURED_AT),
        "promotion_scale": (PROMOTION_M, PROMOTION_D),
        "gate_met": max(MEASURED_AT) >= (PROMOTION_M, PROMOTION_D)
        if MEASURED_AT
        else False,
    }


def is_provisional() -> bool:
    """True while the adopted beta has not passed its named promotion scale."""
    return not bool(audit()["gate_met"])


def assert_promotion_allowed(m: int, d: int) -> None:
    """Raise unless a measurement was taken at or above the promotion scale.

    Fail-closed on every path: a sub-scale measurement, or a receipt set that does
    not verify, both raise.
    """
    rep = audit()
    if not rep["receipts_ok"]:
        raise PromotionGated(
            "PROMOTION_GATED: receipt provenance does not verify (see audit()); "
            "cannot certify beta at any scale."
        )
    if m < PROMOTION_M or d < PROMOTION_D:
        raise PromotionGated(
            "PROMOTION_GATED: beta={0} was measured at M={1}, d={2}, below the named "
            "promotion scale M={3}, d={4}. The change stays PROVISIONAL. The current "
            "fixture cannot close it (its unique-grid family caps at 174), so this "
            "needs a real production engram source.".format(
                ADOPTED_BETA, m, d, PROMOTION_M, PROMOTION_D
            )
        )


def status() -> Dict[str, object]:
    """Compact runtime status. Carries PROVISIONAL to any telemetry surface."""
    rep = audit()
    prov = not bool(rep["gate_met"])
    return {
        "sealed_beta": SEALED_BETA,
        "adopted_beta": ADOPTED_BETA,
        "provisional": prov,
        "measured_at": {"M": [m for m, _ in MEASURED_AT], "d": [d for _, d in MEASURED_AT]},
        "promotion_scale": {"M": PROMOTION_M, "d": PROMOTION_D},
        "measured_claim": MEASURED_CLAIM,
        "not_measured_claim": NOT_MEASURED_CLAIM,
        "receipts_ok": rep["receipts_ok"],
        "note": (
            "PROVISIONAL -- promotion gate not met. Do not cite as production-validated."
            if prov
            else "promotion gate met at the named scale."
        ),
    }
