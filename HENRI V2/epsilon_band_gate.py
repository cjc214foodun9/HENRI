"""Symmetrical epsilon-band gate probe (defect D1: the self-destroying gate).

WHY THIS EXISTS
---------------
Defect D1, as measured on 2026-09-12 (receipts in
`carrier/e6-verifier/gate_receipt.json` and
`experiments/verification/e6_gauto1_p1_bias_curve.md`):

    A one-sided rejection rule `G > eps => reject` was applied to streams whose
    statistic approaches zero FROM BELOW. The rule therefore rejected its OWN
    negative controls. The gate destroyed its own acceptance evidence. Two
    independent sessions found this by different criteria (n_min = 16384 on a
    control-band criterion, n_min = 128 on a structured-onset criterion).

That is the general failure this module detects: **a gate whose rejection rule
is incompatible with its own control stream.** The fix is not a threshold
relaxation. The fix is a two-sided band plus an explicit pre-condition.

CONTRACT
--------
A gate is epsilon-band SYMMETRIC when, over a set of labelled streams:

  1. every CONTROL stream (dead / noise / constant) reports |G| <= eps AND is
     ACCEPTed, and
  2. every SIGNAL stream (structured) reports G > eps AND is REJECTed, and
  3. the pre-condition holds: any stream below the minimum length n_min is
     quarantined (never scored), so the finite-length compressor warm-up cost
     cannot bias a verdict.

An ASYMMETRIC gate is one where a control is rejected, or where the acceptance
band is one-sided ([0, eps] but not [-eps, eps]). Such a gate CANNOT produce a
valid verdict, and it is detected here rather than trusted.

Evidence boundary: this module measures a gate's own consistency. It does not
establish that the gate's statistic is the right statistic for the task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Verdict strings (typed; never a bare bool at a boundary).
BAND_SYMMETRIC = "EPSILON_BAND_SYMMETRIC"
BAND_ASYMMETRIC = "EPSILON_BAND_ASYMMETRIC"
BAND_UNDERDETERMINED = "EPSILON_BAND_UNDERDETERMINED"


@dataclass(frozen=True)
class StreamObservation:
    """One scored stream. `n` is required: the pre-condition depends on it."""

    name: str
    role: str          # "control" | "signal"
    n: int
    statistic: float   # G
    verdict: str       # the gate's own verdict for this stream


@dataclass
class BandReport:
    status: str
    eps: float
    n_min: int
    controls_inside: int
    controls_total: int
    signals_outside: int
    signals_total: int
    rejected_controls: Tuple[str, ...]
    admitted_signals: Tuple[str, ...]
    quarantined: Tuple[str, ...]
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == BAND_SYMMETRIC

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "eps": self.eps,
            "n_min": self.n_min,
            "controls_inside": self.controls_inside,
            "controls_total": self.controls_total,
            "signals_outside": self.signals_outside,
            "signals_total": self.signals_total,
            "rejected_controls": list(self.rejected_controls),
            "admitted_signals": list(self.admitted_signals),
            "quarantined": list(self.quarantined),
            "reason": self.reason,
        }


def two_sided_band(
    statistic: float,
    eps: float,
    n: int,
    n_min: int,
) -> str:
    """Two-sided classification with an explicit pre-condition.

    Returns "BLOCKED_SUB_MINIMUM_N" | "ACCEPT_CONTROL_BAND" | "ACCEPT_STRUCTURED"
    | "REJECT_OUT_OF_BAND".

    The pre-condition is checked FIRST. A finite-length compressor has a warm-up
    cost of O(300..500) bits, so |G| for a dead stream is biased at small n. The
    order matters: scoring before the pre-condition is the D1 defect.
    """
    if n < n_min:
        return "BLOCKED_SUB_MINIMUM_N"
    if abs(statistic) <= eps:
        return "ACCEPT_CONTROL_BAND"
    if statistic > eps:
        return "ACCEPT_STRUCTURED"
    return "REJECT_OUT_OF_BAND"


def one_sided_band(
    statistic: float,
    eps: float,
    n: int,
    n_min: int = 0,
) -> str:
    """The FALSIFIED one-sided rule, kept as a negative control.

    This is the rule the audit identified. It has no i/o pre-condition and no
    lower bound, so a dead stream (statistic slightly negative) is REJECTed
    while being a control. Running the probe against this rule must produce
    BAND_ASYMMETRIC. If it does not, the probe is not discriminating.
    """
    if statistic > eps:
        return "ACCEPT_STRUCTURED"
    return "REJECT_OUT_OF_BAND"


def evaluate_band(
    observations: Sequence[StreamObservation],
    eps: float,
    n_min: int,
    control_accept_prefix: str = "ACCEPT",
) -> BandReport:
    """Decide whether a gate's eps-band is symmetric over labelled streams."""
    if eps <= 0.0:
        raise ValueError(f"eps must be positive; got {eps}")
    if n_min < 0:
        raise ValueError("n_min must be >= 0")

    controls = [o for o in observations if o.role == "control"]
    signals = [o for o in observations if o.role == "signal"]
    quarantined = tuple(o.name for o in observations if o.n < n_min)

    rejected_controls = tuple(
        o.name for o in controls
        if not (abs(o.statistic) <= eps and o.verdict.startswith(control_accept_prefix))
    )
    admitted_signals = tuple(
        o.name for o in signals
        if not (o.statistic > eps and o.verdict == "ACCEPT_STRUCTURED")
    )

    inside = sum(
        1 for o in controls
        if abs(o.statistic) <= eps and o.verdict.startswith(control_accept_prefix)
    )
    outside = sum(
        1 for o in signals
        if o.statistic > eps and o.verdict == "ACCEPT_STRUCTURED"
    )

    if not controls or not signals:
        return BandReport(
            status=BAND_UNDERDETERMINED, eps=eps, n_min=n_min,
            controls_inside=inside, controls_total=len(controls),
            signals_outside=outside, signals_total=len(signals),
            rejected_controls=rejected_controls, admitted_signals=admitted_signals,
            quarantined=quarantined,
            reason="need at least one control and one signal stream to decide",
        )

    if rejected_controls or admitted_signals:
        parts = []
        if rejected_controls:
            parts.append(
                "the gate rejected its own control stream(s): "
                + ", ".join(rejected_controls)
            )
        if admitted_signals:
            parts.append(
                "the gate admitted signal stream(s) into the control band: "
                + ", ".join(admitted_signals)
            )
        return BandReport(
            status=BAND_ASYMMETRIC, eps=eps, n_min=n_min,
            controls_inside=inside, controls_total=len(controls),
            signals_outside=outside, signals_total=len(signals),
            rejected_controls=rejected_controls, admitted_signals=admitted_signals,
            quarantined=quarantined,
            reason="; ".join(parts),
        )

    return BandReport(
        status=BAND_SYMMETRIC, eps=eps, n_min=n_min,
        controls_inside=inside, controls_total=len(controls),
        signals_outside=outside, signals_total=len(signals),
        rejected_controls=(), admitted_signals=(), quarantined=quarantined,
        reason="both controls inside the band and signals outside it",
    )


def probe_gate(
    gate: Callable[[float, int], str],
    streams: Sequence[Tuple[str, str, int, float]],
    eps: float,
    n_min: int,
) -> BandReport:
    """Run a gate function over raw streams and evaluate its band.

    `gate(statistic, n) -> verdict`; `streams` is (name, role, n, statistic).
    """
    obs = [
        StreamObservation(name=name, role=role, n=n, statistic=g,
                          verdict=gate(g, n))
        for name, role, n, g in streams
    ]
    return evaluate_band(obs, eps=eps, n_min=n_min)


def zlib_warmup_n_min(eps: float, overhead_bits: float = 400.0) -> int:
    """Derive n_min = ceil(overhead_bits / eps).

    This is the carrier-correct form of the D1 fix: the pre-condition is
    DERIVED from the compressor's warm-up cost and the band width, not chosen.
    At eps=0.02 and a 400-bit warm-up, n_min = 20000; the measured carrier
    boundary was 16384, consistent to the order of the overhead estimate.
    """
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    return int(math.ceil(float(overhead_bits) / float(eps)))
