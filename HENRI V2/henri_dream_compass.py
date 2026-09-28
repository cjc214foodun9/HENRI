"""Dream compass + Hopfield terminator — the two unbuilt synthesis bridges.

WHY THIS MODULE EXISTS
======================
My own grep of the live tree found two gaps the synthesis document names:

  * `henri_latent_dreamer.py` mentions `gradient_alignment`  0 times and
    `alignment_reward` 0 times. The dream loop creeps on a raw SAGNAC-derived loss
    and has no LEARNING-PROGRESS compass. Measured cost of that: the alignment reward
    is the one signal validated in this project to separate true learning frontier
    from noise (raw +6.30 vs noise -3.89 in the earlier sprint), and the dream loop
    uses none of it.
  * neither the dreamer nor the Koopman leaf mentions `hopfield` / `Hopfield` 0 times.
    Candidate actions therefore never terminate in the associative codebook; the
    synthesis directive to "terminate in henri_hopfield_egress.py (beta = 8.0)" is
    unwired.

This module supplies both as FAIL-OPEN, DEFAULT-OFF helpers:

  AlignmentCompass   the preconditioned gradient-alignment reward as a dream-step
                     PROGRESS signal: r = |<grad, P * displacement>|.
  HopfieldTerminator a wrap of CanonicalCodebookEgress (sealed beta=8.0) that snaps a
                     continuous wave to a discrete codebook id.
  DreamEgressRouter  the combined path: the compass decides CONTINUE vs TERMINATE, the
                     terminator snaps the final wave, and a RATIFIED flag gates
                     emission.

GATE-D IS PRESERVED, NOT BYPASSED
=================================
`henri_latent_dreamer.py` carries a two-sided latch: `guarded_egress` raises
`DreamEgressBlocked` until `ratify_for_egress(receipt)`. A wiring that emitted actions
without the ratification would silently defeat that latch. So `DreamEgressRouter.route`
REFUSES to emit unless `ratified=True`, and returns a named reason instead -- the gate
stays load-bearing and the mechanism is available behind it.

WHY THE COMPASS NORMALISES NOTHING
==================================
The earlier sprint MEASURED that directional cosine normalisation destroys the
alignment signal's discrimination (raw separated +6.30 progress from -3.89 noise;
the normalised form did not). `AlignmentCompass` therefore reports the RAW reward and
decides with a threshold on it. No normalisation is applied anywhere in this module.

HONEST LIMITS
=============
* Naming: `grad`, `displacement`, `exp_avg_sq` are FLAT vectors of any width. This
  module does not know which model they came from and does not claim to.
* The compass is a DECISION HELPER. It does not update weights. The caller owns any
  parameter change, exactly as the dreamer's own creep does.
* The terminator snaps to a caller-supplied codebook. It makes no claim that the
  codebook is good; a codebook of one entry snaps everything to that entry, which the
  tests demonstrate explicitly rather than hiding.
* No ARC / SciCode score is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from henri_gradient_alignment_reward import alignment_reward, parameter_displacement


class DreamCompassError(RuntimeError):
    """Fail-closed contract violation (configuration, not measurement)."""


@dataclass
class CompassVerdict:
    """One dream-step judgement. `reward` is None when the measurement was not made."""
    reward: Optional[float]
    continue_dreaming: bool
    valid: bool
    reason: str


@dataclass
class SnapVerdict:
    """One terminal snap. `snapped_id` is None whenever no snap was made."""
    snapped_id: Optional[int]
    similarity: Optional[float]
    valid: bool
    emitted: bool
    reason: str


class AlignmentCompass:
    """Preconditioned gradient-alignment reward as a dream CONTINUE/TERMINATE signal.

    Parameters
    ----------
    lr, eps : passed through to the AdamW preconditioner (the measured kernel)
    terminate_below : terminate when the raw reward falls BELOW this value
    consecutive : how many consecutive low readings terminate (default 1)
    """

    def __init__(self, lr: float = 1e-4, eps: float = 1e-8,
                 terminate_below: float = 0.0, consecutive: int = 1) -> None:
        if lr <= 0:
            raise DreamCompassError("lr must be positive")
        if eps <= 0:
            raise DreamCompassError("eps must be positive")
        if consecutive < 1:
            raise DreamCompassError("consecutive must be >= 1")
        self.lr = float(lr)
        self.eps = float(eps)
        self.terminate_below = float(terminate_below)
        self.consecutive = int(consecutive)
        self._low_run = 0
        self.history: List[float] = []

    # ------------------------------------------------------------------ measure
    @staticmethod
    def _flat(v: Any) -> torch.Tensor:
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
        if t.is_complex():
            t = t.real
        return t.reshape(-1).to(torch.float32)

    def reward(self, grad: Any, displacement: Any, exp_avg_sq: Any) -> Optional[float]:
        """Raw preconditioned alignment reward, or None when unmeasurable."""
        try:
            g = self._flat(grad)
            d = self._flat(displacement)
            e = self._flat(exp_avg_sq)
            if not (g.shape == d.shape == e.shape):
                return None
            if not (torch.isfinite(g).all() and torch.isfinite(d).all()
                    and torch.isfinite(e).all()):
                return None
            r = float(alignment_reward(g, d, e, self.lr, self.eps))
            if r != r:
                return None
            return r
        except Exception:
            return None

    def displacement(self, theta_current: Any, theta_lookback: Any) -> Optional[torch.Tensor]:
        """`parameter_displacement(current, lookback)`, or None on a shape mismatch.

        CONVENTION (measured, not assumed): the repo's function returns
        `lookback - current`, and the Stage-0 driver uses it the same way
        (`d_theta = theta_lb - learner.flat()`). This helper passes that through
        unchanged so there is ONE sign convention in the project. The sign is
        immaterial to the reward, which takes an absolute value.
        """
        try:
            a, b = self._flat(theta_current), self._flat(theta_lookback)
            if a.shape != b.shape:
                return None
            return parameter_displacement(a, b)
        except Exception:
            return None

    # ------------------------------------------------------------------- decide
    def judge(self, grad: Any, displacement: Any, exp_avg_sq: Any) -> CompassVerdict:
        """CONTINUE unless the raw reward stays below `terminate_below` too long.

        A None reward is FAIL-OPEN toward continuing: an unmeasurable step must not
        silently end a dream, because this helper only ever ADDS a stop condition.
        """
        r = self.reward(grad, displacement, exp_avg_sq)
        if r is None:
            return CompassVerdict(None, True, False, "UNMEASURABLE")
        self.history.append(r)
        if r < self.terminate_below:
            self._low_run += 1
        else:
            self._low_run = 0
        if self._low_run >= self.consecutive:
            return CompassVerdict(r, False, True,
                                  "TERMINATE_BELOW_%g_X%d" % (self.terminate_below,
                                                              self.consecutive))
        return CompassVerdict(r, True, True, "OK")

    def reset(self) -> None:
        self._low_run = 0
        self.history.clear()

    def report(self) -> Dict[str, Any]:
        return {
            "schema": "henri.dream.compass.v1",
            "lr": self.lr, "eps": self.eps,
            "terminate_below": self.terminate_below, "consecutive": self.consecutive,
            "readings": len(self.history),
            "last_reward": self.history[-1] if self.history else None,
            "normalisation": "NONE (raw reward; normalisation was measured to destroy "
                             "discrimination in the earlier sprint)",
            "fail_mode": "FAIL_OPEN toward continuing; an unmeasurable step never ends a dream",
        }


class HopfieldTerminator:
    """Snap a continuous wave to a discrete codebook id via the sealed Hopfield egress.

    Uses `CanonicalCodebookEgress` (beta defaults to 8.0, the SEALED value). The
    codebook and its ids are supplied by the caller; a validator may reject an id,
    in which case the snap reports status without emitting.
    """

    # Statuses the live egress uses to mean "no snap". Measured vocabulary: a
    # successful decode returns "SNAPPED"; anything listed here is treated as a
    # refusal even when an index is present.
    REJECT_STATUSES = frozenset({"REJECTED", "INVALID", "REFUSED", "REJECT"})

    def __init__(self, dim: int, beta: float = 8.0,
                 validator: Optional[Callable[[int], bool]] = None) -> None:
        if dim < 2:
            raise DreamCompassError("dim must be >= 2")
        if beta <= 0:
            raise DreamCompassError("beta must be positive")
        from henri_hopfield_egress import CanonicalCodebookEgress
        self.dim = int(dim)
        self.beta = float(beta)
        self._validator = validator
        self._egress = CanonicalCodebookEgress(dim=self.dim, beta=self.beta)
        self._registered = 0
        self._ids: List[int] = []

    # ------------------------------------------------------------------ register
    def register(self, code_waves: torch.Tensor, canonical_ids: Sequence[int]) -> int:
        n = self._egress.register(code_waves, list(canonical_ids), self._validator)
        self._registered = int(n)
        self._ids = [int(c) for c in canonical_ids]
        return self._registered

    @property
    def registered(self) -> int:
        return self._registered

    # ---------------------------------------------------------------------- snap
    def snap(self, wave: Any, *, emit: bool = True) -> SnapVerdict:
        """Snap `wave` to a codebook id. Fail-open: any anomaly -> no snap, no emit."""
        if self._registered <= 0:
            return SnapVerdict(None, None, False, False, "NO_CODEBOOK")
        try:
            w = AlignmentCompass._flat(wave)
            if w.numel() != self.dim:
                return SnapVerdict(None, None, False, False, "DIM_MISMATCH")
            if not torch.isfinite(w).all():
                return SnapVerdict(None, None, False, False, "NON_FINITE")
            res = self._egress.decode(w)
            status = str(getattr(res, "status", ""))
            idx = getattr(res, "snapped_index", None)
            sim = getattr(res, "similarity", None)
            # DEFECT FIXED 2026-09-27: the first form required the status to start with
            # "OK". The live egress returns status == "SNAPPED", so `emitted` was
            # ALWAYS False -- a terminator that could never emit, while every test that
            # only checked `valid` would still have passed. The success condition is now
            # measured from the observable result (an index came back and the status is
            # not a declared rejection) rather than from a hard-coded foreign vocabulary.
            rejected = status.upper() in self.REJECT_STATUSES
            ok = (idx is not None) and (not rejected) and emit
            return SnapVerdict((int(idx) if idx is not None else None),
                               (float(sim) if sim is not None else None),
                               bool(idx is not None and not rejected), bool(ok),
                               status or "UNKNOWN")
        except Exception as exc:
            return SnapVerdict(None, None, False, False, "ERROR:%s" % type(exc).__name__)

    def report(self) -> Dict[str, Any]:
        return {
            "schema": "henri.dream.hopfield-terminator.v1",
            "dim": self.dim, "beta": self.beta, "registered": self.registered,
            "ids": list(self._ids),
            "sealed_beta_note": ("8.0 is the SEALED egress value; the document's 26.10 was "
                                 "measured NOT better than the seal and is not adopted"),
            "fail_mode": "FAIL_OPEN: any anomaly -> snapped None, emitted False",
        }


class DreamEgressRouter:
    """Compass + terminator + the GATE-D ratification check.

    Emission requires `ratified=True`. Without it the router reports the snap but does
    NOT emit, preserving the dreamer's two-sided latch rather than bypassing it.
    """

    def __init__(self, compass: AlignmentCompass, terminator: HopfieldTerminator) -> None:
        self.compass = compass
        self.terminator = terminator

    def route(self, wave: Any, *, grad: Any = None, displacement: Any = None,
              exp_avg_sq: Any = None, ratified: bool = False) -> Dict[str, Any]:
        """Judge (when the three tensors are given) then snap; emit only if ratified."""
        verdict = None
        if grad is not None and displacement is not None and exp_avg_sq is not None:
            verdict = self.compass.judge(grad, displacement, exp_avg_sq)
        snap = self.terminator.snap(wave, emit=bool(ratified))
        if not ratified:
            return {"compass": verdict, "snap": snap, "emitted": False,
                    "reason": "EGRESS_NOT_RATIFIED (GATE-D latch honoured)"}
        if not snap.valid:
            return {"compass": verdict, "snap": snap, "emitted": False,
                    "reason": "SNAP_INVALID:%s" % snap.reason}
        return {"compass": verdict, "snap": snap, "emitted": bool(snap.emitted),
                "reason": snap.reason}


def make_codebook(n: int, dim: int, seed: int) -> Tuple[torch.Tensor, List[int]]:
    """Deterministic unit-modulus codebook + canonical ids (test/utility helper)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=g)
    v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return v, list(range(n))
