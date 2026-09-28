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
    # ---- DIRECTIVE 1: escalation TRIGGER SELECTION.
    # "variance" (default) reproduces the previous behaviour byte-for-byte, so no
    # existing caller changes. "progress" uses the moving-window loss DERIVATIVE.
    # WHY: the variance floor sigma^2 < 1e-4 is a PLATEAU DETECTOR -- it can only fire
    # once there is nothing left to learn. Measured on the committed receipt: the first
    # escalation landed with 99.82% of the total loss drop already spent. A trigger
    # based on the loss DERIVATIVE fires while the learner is still improving but
    # SLOWING, i.e. before the manifold settles into a local attractor.
    trigger: str = "variance"
    progress_rate_threshold: float = 1e-3   # absolute relative-improvement floor
    # ---- DIRECTIVE 1, corrected by measurement. Firing on an ABSOLUTE rate floor is
    # ALSO a post-convergence detector: measured, it fired at the SAME step as the
    # variance floor (both step 239) because both wait for learning to stop. The
    # directive wants escalation BEFORE the weights settle, so the primary condition is
    # a DECAY of the learning rate relative to its own peak: once the rate has halved,
    # the task family is yielding less marginal information than it can.
    progress_decay_fraction: float = 0.5
    validate_trigger: bool = True
    # ---- CADENCE RULE (the mode that actually beats the confound).
    # MEASURED: on an exponential decay, |dL/dt| = a*(L-floor), so the RELATIVE rate
    # a*(L-floor)/L is nearly CONSTANT while L >> floor and collapses only as L -> floor
    # (threshold fires at ~98% of the drop). ANY threshold on that metric is therefore a
    # post-convergence detector -- the directive's literal metric included. The
    # directive's own wording is the escape: "escalate WHILE gradient velocity is
    # non-zero" is a CADENCE rule. Fire every `cadence_windows` windows for as long as
    # the loss is still falling by more than `velocity_noise_frac` of its own scale.
    cadence_windows: int = 1
    velocity_noise_frac: float = 1e-3

    def __post_init__(self) -> None:
        """FAIL CLOSED on an unknown trigger.

        Measured defect: with an unrecognised trigger the dispatch fell through to the
        variance branch, so a typo silently selected a different policy. An unknown
        trigger now raises instead of guessing.
        """
        if self.validate_trigger and self.trigger not in ("variance", "progress",
                                                         "cadence"):
            raise ValueError(
                "unknown governor trigger %r; expected 'variance', 'progress' or "
                "'cadence'" % (self.trigger,))



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
        # ---- DIRECTIVE 1 state. ROOT CAUSE of a 23-test regression: an earlier patch
        # anchored on `self._since_progress = 0`, which occurs TWICE in this file, so the
        # edit was SKIPPED and these four attributes were never initialized while
        # learning_progress()/progress_rate()/observe() already read them ->
        # AttributeError on every observe() call. The anchor is now unique.
        self._last_progress: Optional[float] = None
        self._last_rate: Optional[float] = None
        self._peak_rate: Optional[float] = None
        self._last_velocity: Optional[float] = None
        self._n_windows = 0
        self._windows_since_escalation = 0

    # ------------------------------------------------------------- telemetry
    @property
    def spec(self) -> Dict[str, float]:
        return dict(self.cfg.spec)

    @property
    def n_observations(self) -> int:
        return len(self._window)

    def learning_progress(self) -> Optional[float]:
        """Moving-window loss progress (Directive 1).

            Delta L_progress = (1/W) * sum_{w=0}^{W-1} L_{t-w}  -  L_t

        POSITIVE means the loss is still FALLING (the older window is worse than the
        current value). This is the derivative-based replacement for the variance
        floor. Telemetry only; the escalation decision lives in `progress_rate`.
        """
        n = len(self._window)
        if n < self.cfg.window or n < 2:
            # DEFECT FIXED 2026-09-27: observe() consumes the window after computing
            # this, so the live value is only available DURING observe(). Returning the
            # stored last value keeps the accessor meaningful for report()/telemetry
            # instead of silently reporting None (a dead-store).
            return self._last_progress
        cur = float(self._window[-1])
        older = self._window[:-1]
        return (sum(older) / len(older)) - cur

    def progress_rate(self) -> Optional[float]:
        """Relative per-step improvement = learning_progress / W / max(|L_t|, eps).

        Dimensionless, so ONE threshold transfers across loss scales. Escalation
        fires when this falls BELOW `progress_rate_threshold`: the learner is still
        improving but SLOWING. That is EARLIER than a variance collapse by
        construction, because variance only collapses after the loss has flattened.
        """
        if len(self._window) < self.cfg.window or len(self._window) < 2:
            return self._last_rate
        p = self._last_progress if self._last_progress is not None else self.learning_progress()
        if p is None:
            return None
        cur = abs(float(self._window[-1]))
        return (p / float(self.cfg.window)) / max(cur, 1e-9)

    def velocity(self) -> Optional[float]:
        """||loss change across the window|| normalized by the window's own scale.

        NON-ZERO means the learner is still moving. This is the CADENCE signal: the
        directive says to escalate while velocity is non-zero, which is a rule about
        WHEN to keep stepping, not a threshold on a decaying rate.
        """
        if len(self._window) < self.cfg.window or len(self._window) < 2:
            return self._last_velocity
        first = float(self._window[0])
        last = float(self._window[-1])
        scale = max(abs(first), abs(last), 1e-9)
        return (first - last) / scale

    def peak_rate(self) -> Optional[float]:
        """The best relative progress rate seen since the last escalation."""
        return self._peak_rate

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
        prog = self.learning_progress()
        rate = self.progress_rate()
        self._last_progress = prog
        self._last_rate = rate
        vel = self.velocity()
        self._last_velocity = vel
        self._n_windows += 1
        self._windows_since_escalation += 1
        if rate is not None and (self._peak_rate is None or rate > self._peak_rate):
            self._peak_rate = rate
        self._window = []

        # ---- TRIGGER DISPATCH (Directive 1). Default "variance" is byte-identical to
        # the previous behaviour; "progress" fires on the loss derivative.
        if self.cfg.trigger == "cadence":
            # Escalate on a CADENCE while velocity is still non-zero. This fires at the
            # FIRST cadence boundary at which the loss is still moving -- EARLY by
            # construction, and it does not wait for a rate to decay.
            _moving = (vel is not None and abs(vel) > self.cfg.velocity_noise_frac)
            _due = self._windows_since_escalation >= max(1, self.cfg.cadence_windows)
            fire = bool(_moving and _due)
            _why = ("cadence |velocity| %.3e > %.3e after %d window(s)"
                    % (abs(vel) if vel is not None else float("nan"),
                       self.cfg.velocity_noise_frac, self._windows_since_escalation))
        elif self.cfg.trigger == "progress":
            # PRIMARY: decay of the rate relative to its own peak (fires EARLY).
            # SECONDARY: the absolute floor (fires only after near-convergence).
            _decay_ref = (self.cfg.progress_decay_fraction * self._peak_rate
                          if self._peak_rate else None)
            _by_decay = (rate is not None and _decay_ref is not None and rate < _decay_ref)
            _by_floor = (rate is not None
                         and rate < self.cfg.progress_rate_threshold)
            fire = bool(_by_decay or _by_floor)
            _why = ("decay rate %.3e < %.3e*peak(%.3e)=%.3e"
                    % (rate if rate is not None else float("nan"),
                       self.cfg.progress_decay_fraction,
                       self._peak_rate if self._peak_rate else float("nan"),
                       _decay_ref if _decay_ref is not None else float("nan"))
                    if _by_decay else
                    "floor rate %.3e < %.3e"
                    % (rate if rate is not None else float("nan"),
                       self.cfg.progress_rate_threshold))
        else:
            fire = (v is not None and v < self.cfg.var_threshold)
            _why = "sigma2 %.3e < %.3e" % (v if v is not None else float("nan"),
                                           self.cfg.var_threshold)
        if not fire:
            return None                        # still learning: do not interrupt

        advance = self._advance()
        if advance is None:
            self._kill("ladder exhausted without further gain")
            return {"event": "KILL", "reason": self.kill_reason,
                    "spec": self.spec, "events": len(self.events)}

        self._since_progress += 1
        # a new task family resets the learning-rate reference and the cadence clock
        self._peak_rate = None
        self._windows_since_escalation = 0
        advance["event"] = "ESCALATE"
        advance["sigma_sq"] = v
        # ---- DIRECTIVE 1 telemetry: the derivative channel that DECIDED this event.
        advance["learning_progress"] = prog
        advance["progress_rate"] = rate
        advance["trigger"] = self.cfg.trigger
        advance["trigger_reason"] = _why
        advance["peak_rate"] = self._peak_rate
        advance["n_windows"] = self._n_windows
        advance["velocity"] = vel
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
            "trigger": self.cfg.trigger,
            "progress_rate_last": self.progress_rate(),
            "velocity_last": self.velocity(),
            "learning_progress_last": self.learning_progress(),
            "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in asdict(self.cfg).items()},
        }


def variance_of(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        raise ValueError("variance_of needs >= 2 values")
    mu = sum(values) / n
    return sum((x - mu) ** 2 for x in values) / n
