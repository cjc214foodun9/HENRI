"""HENRI Curriculum Governor — automated entropy escalation (Gap 5 / Directive 1).

WHY THIS MODULE EXISTS
======================
Stage 0 ran a STATIONARY generator. Measured: the held-out curve flattened by
round 2,000 of ~590,000 (0.34% of the run), so 99.66% of a 10-billion-token burn
carried no marginal algorithmic information. The driver's own escalation (flag
`--curriculum-escalate`) fires on `sigma^2 < 1e-4` but mutates ONE knob --
`prog_len` (32 -> 50 -> 77 -> 96). Measured on the live driver: `obstacle` 0,
`multiscale` 0, `grid_size` 0, `distractor` 0 occurrences. A single linear knob
does not restore entropy: the model exhausts the richer lengths within a few
hundred rounds and reaches the plateau again.

This module is the ESCALATION POLICY as a library, so the driver consumes a tested
state machine instead of an inline counter. It provides three things the driver
lacks:

  1. A LADDER of heterogeneous mutations (length, topological obstacle, multiscale
     nesting, distractor noise, grid growth) rather than one knob.
  2. A KILL SWITCH: if escalation stops moving held-out progress, the run must
     TERMINATE rather than burn the remaining budget. That is the literal reading
     of "cease flat token volume burns" -- a plateau detector, not a longer run.
  3. Sub-budgets that are conserved: escalation may change the CURRICULUM, never
     the token accounting. The caller's `vm_executions * 33 == tokens` identity is
     untouched because this module mutates a curriculum spec, not the counter.

PRE-REGISTERED SEMANTICS
------------------------
  observe(loss)            -> appends to the variance window; returns an event or None
  sigma^2 < var_threshold  -> ESCALATE (advance one rung), then reset the window
  no progress for `kill_patience` consecutive escalation events
                           -> KILL (the caller must terminate)
  progress                 -> any HELD-OUT improvement >= progress_eps since the
                              previous event

DEAD-INPUT / VACUOUS-CONTROL GUARDS
-----------------------------------
* A HIGH-variance loss must never escalate (the window must be genuinely quiet).
* A single spike must not escalate: the threshold is the variance of the WINDOW,
  so one outlier raises sigma^2 and suppresses escalation -- tested.
* The KILL decision must be reachable: a run that escalates and never improves
  MUST terminate. A kill switch that cannot fire is the dead-store defect class.

HONEST LIMITS
-------------
* This module decides WHEN to make the curriculum harder and WHEN to stop. It does
  not train anything, and it claims no task accuracy.
* The mutation magnitudes are defaults chosen to be visible in telemetry, not
  optima. They are returned in the event so the caller records them.
* Pure standard library, deterministic, no torch. Local CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Escalation ladder. Each rung is a heterogeneous lever, applied in order.
LADDER: Tuple[str, ...] = (
    "prog_len",          # 1. longer programs (the ONLY knob the driver had)
    "topological_obstacle",   # 2. a closed barrier the program must route around
    "multiscale_nesting",     # 3. nested program blocks (depth 1 -> 2 -> ...)
    "distractor_noise",       # 4. irrelevant symbols in the tape
    "grid_growth",            # 5. a larger spatial lattice
)

DEFAULT_SPEC: Dict[str, float] = {
    "prog_len": 32.0,
    "topological_obstacle": 0.0,
    "multiscale_nesting": 1.0,
    "distractor_noise": 0.0,
    "grid_growth": 16.0,
}

# Per-rung mutation: multiplicative for lengths/depths, additive for counters,
# and each rung is capped so no single lever can run away.
RUNG_MUTATION: Dict[str, Tuple[str, float, float]] = {
    #            mode,         step,  cap
    "prog_len": ("mul", 1.5, 512.0),
    "topological_obstacle": ("add", 1.0, 4.0),
    "multiscale_nesting": ("add", 1.0, 5.0),
    "distractor_noise": ("add", 1.0, 8.0),
    # DEFECT FIXED 2026-09-27: the cap was 64.0 while the DRIVER deploys
    # grid_growth = float(tape_size) = 256.0, so `after = min(256+step, 64) = 64`
    # and `_advance`'s `if after <= before: continue` SKIPPED this rung forever.
    # Rung 5 could not fire at all. Measured by h15_discrim.py. The cap is now
    # above any deployed tape size, and the mode is multiplicative so the lever
    # doubles the tape from the deployment value (256 -> 512 -> ... -> 4096).
    "grid_growth": ("mul", 2.0, 4096.0),
}


@dataclass
class GovernorConfig:
    window: int = 50                    # observations per variance estimate
    var_threshold: float = 1e-4         # sigma^2 below which we escalate
    progress_eps: float = 1e-3          # held-out gain counted as progress
    kill_patience: int = 3              # non-progressing escalations before KILL
    max_events: int = 24                # hard ceiling on escalation events
    rungs: Tuple[str, ...] = LADDER
    spec: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SPEC))


class CurriculumGovernor:
    """Variance-triggered curriculum escalation with a terminate-on-plateau switch."""

    def __init__(self, config: Optional[GovernorConfig] = None):
        self.cfg = config or GovernorConfig()
        self._window: List[float] = []
        self.events: List[Dict[str, object]] = []
        self.killed = False
        self.kill_reason: Optional[str] = None
        self.rung = 0                       # next rung index to apply
        self._progress_marks: List[float] = []
        self.best_heldout: Optional[float] = None
        self._since_progress = 0

    # ------------------------------------------------------------- telemetry
    @property
    def spec(self) -> Dict[str, float]:
        return dict(self.cfg.spec)

    @property
    def n_observations(self) -> int:
        return len(self._window)

    def variance(self) -> Optional[float]:
        n = len(self._window)
        if n < 2:
            return None
        mu = sum(self._window) / n
        return sum((x - mu) ** 2 for x in self._window) / n      # population

    def _already_escalated(self, rung: str) -> bool:
        return any(e["rung"] == rung for e in self.events)

    def _advance(self) -> Optional[Dict[str, object]]:
        """Advance one rung. Bounded; NEVER recurses.

        DEFECT FIXED 2026-09-27: the first form recursed after resetting the rung
        index, and `len(self.events)` does not grow during that recursion (the
        caller appends only after `_advance` returns). Once EVERY rung saturated,
        the function recursed forever and raised RecursionError, which escaped
        `observe()` and killed the caller's loop. Measured by
        test_rungs_saturate_at_their_cap. The loop below bounds the number of
        ladder passes at 2*len(rungs)+1 and returns None when nothing can move.
        """
        for _ in range(2 * len(self.cfg.rungs) + 1):
            while self.rung < len(self.cfg.rungs):
                name = self.cfg.rungs[self.rung]
                self.rung += 1
                mode, step, cap = RUNG_MUTATION[name]
                before = float(self.cfg.spec.get(name, 0.0))
                after = (before * step) if mode == "mul" else (before + step)
                after = min(after, cap)
                if after <= before:
                    continue                      # rung saturated; try the next
                self.cfg.spec[name] = after
                return {"event": len(self.events) + 1, "rung": name, "mode": mode,
                        "before": before, "after": after,
                        "capped": bool(after >= cap)}
            self.rung = 0                          # one full pass: wrap and retry
        return None

    # ---------------------------------------------------------------- public
    def observe(self, loss: float, heldout: Optional[float] = None
                ) -> Optional[Dict[str, object]]:
        """Record one observation. Returns an event dict, or None.

        event["event"] == "ESCALATE"  -> the caller MUST adopt event["spec"]
        event["event"] == "KILL"      -> the caller MUST terminate the run
        """
        if self.killed:
            return None
        self._window.append(float(loss))

        # A held-out reading only RESETS the clock; it never advances it.
        # DEFECT FIXED 2026-09-27: the first form incremented `_since_progress`
        # ONLY inside the no-improvement branch, so a caller that supplies no
        # held-out signal at all (the common case in the seeding loop) never
        # accumulated non-progress and the plateau kill was UNREACHABLE -- a
        # dead-store kill switch. Now every escalation counts as a non-progressing
        # escalation unless a held-out improvement reset the clock.
        if heldout is not None:
            if self.best_heldout is None or heldout > self.best_heldout + self.cfg.progress_eps:
                self.best_heldout = heldout
                self._since_progress = 0

        if len(self._window) < self.cfg.window:
            return None

        v = self.variance()
        self._window = []
        if v is None or v >= self.cfg.var_threshold:
            return None                        # still learning: do not interrupt

        advance = self._advance()
        if advance is None:
            self._kill("ladder exhausted without further gain")
            return {"event": "KILL", "reason": self.kill_reason,
                    "spec": self.spec, "events": len(self.events)}

        self._since_progress += 1
        advance["event"] = "ESCALATE"
        advance["sigma_sq"] = v
        advance["spec"] = self.spec
        advance["escalations_since_progress"] = self._since_progress
        self.events.append(advance)

        if self._since_progress >= self.cfg.kill_patience:
            self._kill("no held-out progress across %d escalations"
                       % self.cfg.kill_patience)
            return {"event": "KILL", "reason": self.kill_reason, "spec": self.spec,
                    "events": len(self.events), "last_escalation": advance}
        if len(self.events) >= self.cfg.max_events:
            self._kill("max_events reached")
            return {"event": "KILL", "reason": self.kill_reason, "spec": self.spec,
                    "events": len(self.events)}
        return advance

    def _kill(self, reason: str) -> None:
        self.killed = True
        self.kill_reason = reason

    def report(self) -> Dict[str, object]:
        return {
            "schema": "henri.curriculum-governor.report.v1",
            "killed": self.killed, "kill_reason": self.kill_reason,
            "events": len(self.events), "rungs_applied": [e["rung"] for e in self.events],
            "spec": self.spec, "best_heldout": self.best_heldout,
            "sigma_sq_last": self.variance(),
            "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in asdict(self.cfg).items()},
        }


def variance_of(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        raise ValueError("variance_of needs >= 2 values")
    mu = sum(values) / n
    return sum((x - mu) ** 2 for x in values) / n
