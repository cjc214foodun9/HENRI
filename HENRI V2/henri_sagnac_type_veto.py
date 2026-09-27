"""HENRI-Code — Sagnac phase veto over a toy typed IR (type / borrow / termination).

HONEST BOUNDARIES (do not restate the addendum's claims as fact)
----------------------------------------------------------------
* This models a TOY IR: 5 types, three ownership states, and a phase-neutral
  loop detector.  It is NOT a Rust / C++ / TypeScript checker.
* NO claim of "zero syntax errors" or "100% type soundness".  The gate is sound
  for the violation classes it enumerates ON THIS IR and returns ADMIT
  (silent) for everything it does not model.  A silent gate is not a proof.
* DEFAULT-OFF SIDECAR: `enabled=False` admits everything and sets
  `bypass=True`, so the untouched default path is preserved.  No live egress
  module imports this file.
* `content_blind=True` is the DEAD-INPUT NEGATIVE CONTROL: it derives the
  residual from instruction index only, never from content, and MUST fail to
  veto.  If the control ever vetoes, the gate is reading something other than
  what it claims to read.

Admission rule (pre-registered, single threshold):
    admitted  <=>  max_k |residual_k| <= tau   AND   no recorded violation

Violation classes
    UNBOUND_SLOT         operate on a slot that was never bound
    USE_AFTER_MOVE       USE/MOVE/LET on a slot already moved out
    MOVE_WHILE_BORROWED  move a slot while a borrow is outstanding
    BORROW_WHILE_BUSY    take a second borrow of the same slot
    RELEASE_NOT_BORROWED release a slot that is not borrowed
    TYPE_MISMATCH        ASSERT_TYPE against a different type
    NON_TERMINATING      backward jump whose body is phase-NEUTRAL
    UNKNOWN_OP           unrecognised instruction
    STEP_BUDGET_EXCEEDED the run hit the step budget (see note below)

Termination note: the detector PROVES non-termination only for phase-neutral
loops (the state digest repeats at the same label).  A loop whose state keeps
advancing is admitted up to the step budget; budget exhaustion is NOT a
termination proof and is reported separately via `budget_exhausted`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

TWO_PI = 2.0 * math.pi
TYPES: Tuple[str, ...] = ("Int", "Float", "Str", "Nullable", "Unit")
_TYPE_PHASE: Dict[str, float] = {t: TWO_PI * i / len(TYPES) for i, t in enumerate(TYPES)}

# Ownership states are angular offsets on the same circle as the type phase.
OWNED = 0.0
BORROWED = math.pi / 2.0
MOVED = math.pi

# Synthetic non-cyclic advance so a "progressing" loop has a distinguishable
# state digest.  Irrational multiple of 2*pi => effectively non-repeating.
ADVANCE_STEP = TWO_PI * 0.6180339887498949

# Pre-registered veto threshold on the phase residual (radians).
DELTA_TAU = 1e-6


def wrap_pi(x: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return (x + math.pi) % TWO_PI - math.pi


@dataclass(frozen=True)
class Ins:
    """One instruction.  `args` is a tuple of operands."""
    op: str
    args: Tuple = ()


@dataclass
class Verdict:
    admitted: bool
    delta_phi: float
    violations: List[Tuple[int, str, float]] = field(default_factory=list)
    bypass: bool = False
    n_steps: int = 0
    state_digest: str = ""
    content_blind: bool = False
    tau: float = DELTA_TAU
    budget_exhausted: bool = False


class SagnacTypeVeto:
    """Deterministic type / borrow / termination gate over a toy typed IR."""

    def __init__(
        self,
        enabled: bool = True,
        tau: float = DELTA_TAU,
        max_steps: int = 4096,
        content_blind: bool = False,
    ) -> None:
        self.enabled = enabled
        self.tau = tau
        self.max_steps = max_steps
        self.content_blind = content_blind

    # ------------------------------------------------------------- internals
    @staticmethod
    def _phase(type_name: Optional[str], offset: float) -> float:
        base = _TYPE_PHASE[type_name] if type_name is not None else 0.0
        return (base + offset) % TWO_PI

    @staticmethod
    def _digest(state: Dict[str, Tuple[Optional[str], float]]) -> str:
        h = hashlib.sha256()
        for slot in sorted(state):
            tname, off = state[slot]
            h.update(("%s|%s|%.12f;" % (slot, tname, off % TWO_PI)).encode())
        return h.hexdigest()[:16]

    # ----------------------------------------------------------------- check
    def check(self, program: Sequence[Ins]) -> Verdict:
        if not self.enabled:
            return Verdict(admitted=True, delta_phi=0.0, bypass=True,
                           content_blind=self.content_blind, tau=self.tau)

        labels = {ins.args[0]: i for i, ins in enumerate(program) if ins.op == "LABEL"}
        state: Dict[str, Tuple[Optional[str], float]] = {}
        violations: List[Tuple[int, str, float]] = []
        seen: Dict[str, str] = {}

        max_res = 0.0
        pc = 0
        steps = 0
        budget_exhausted = False

        while 0 <= pc < len(program):
            if steps >= self.max_steps:
                budget_exhausted = True
                break
            steps += 1
            ins = program[pc]
            op, a = ins.op, ins.args
            residual, code = 0.0, None
            nxt = pc + 1
            terminate = False

            if op == "LABEL":
                pass

            elif op == "LET":
                slot, tname = a
                cur = state.get(slot)
                if cur is not None and cur[1] != OWNED:
                    residual, code = abs(wrap_pi(cur[1] - OWNED)), "USE_AFTER_MOVE"
                else:
                    state[slot] = (tname, OWNED)

            elif op == "MOVE":
                src, dst = a
                cur = state.get(src)
                if cur is None:
                    residual, code = math.pi, "UNBOUND_SLOT"
                elif cur[1] == MOVED:
                    residual, code = abs(wrap_pi(cur[1] - OWNED)), "USE_AFTER_MOVE"
                elif cur[1] == BORROWED:
                    residual, code = abs(wrap_pi(cur[1] - OWNED)), "MOVE_WHILE_BORROWED"
                else:
                    state[dst] = (cur[0], OWNED)
                    state[src] = (cur[0], MOVED)

            elif op == "USE":
                cur = state.get(a[0])
                if cur is None:
                    residual, code = math.pi, "UNBOUND_SLOT"
                elif cur[1] == MOVED:
                    residual, code = abs(wrap_pi(cur[1] - OWNED)), "USE_AFTER_MOVE"
                # BORROWED and OWNED are both legal reads.

            elif op == "BORROW":
                src, dst = a
                cur = state.get(src)
                if cur is None:
                    residual, code = math.pi, "UNBOUND_SLOT"
                elif cur[1] == MOVED:
                    residual, code = abs(wrap_pi(cur[1] - OWNED)), "USE_AFTER_MOVE"
                elif cur[1] == BORROWED:
                    residual, code = abs(wrap_pi(cur[1] - OWNED)), "BORROW_WHILE_BUSY"
                else:
                    state[dst] = (cur[0], BORROWED)
                    state[src] = (cur[0], BORROWED)

            elif op == "RELEASE":
                # RELEASE takes (src, handle).  A handle-only release left the
                # SOURCE marked BORROWED forever, so a later MOVE of that source
                # was reported MOVE_WHILE_BORROWED on a slot that was no longer
                # borrowed -- a FALSE VETO.  Found by this module's own ADMIT
                # fixture (2026-09-27); the fixture is what caught it.
                src, handle = a
                cur = state.get(handle)
                if cur is None:
                    residual, code = math.pi, "UNBOUND_SLOT"
                elif cur[1] != BORROWED:
                    residual, code = abs(wrap_pi(cur[1] - BORROWED)), "RELEASE_NOT_BORROWED"
                else:
                    state[handle] = (cur[0], OWNED)
                    srcc = state.get(src)
                    if srcc is not None:
                        state[src] = (srcc[0], OWNED)

            elif op == "ASSERT_TYPE":
                slot, tname = a
                cur = state.get(slot)
                if cur is None:
                    residual, code = math.pi, "UNBOUND_SLOT"
                elif cur[0] != tname:
                    residual = abs(wrap_pi(_TYPE_PHASE[tname] - _TYPE_PHASE[cur[0]]))
                    code = "TYPE_MISMATCH"

            elif op == "ADVANCE":
                cur = state.get(a[0])
                if cur is None:
                    residual, code = math.pi, "UNBOUND_SLOT"
                else:
                    state[a[0]] = (cur[0], (cur[1] + ADVANCE_STEP) % TWO_PI)

            elif op == "JUMP":
                lbl = a[0]
                dg = self._digest(state)
                if lbl in seen and seen[lbl] == dg:
                    residual, code = math.pi, "NON_TERMINATING"
                    terminate = True
                else:
                    seen[lbl] = dg
                nxt = labels[lbl] if lbl in labels else len(program)

            else:
                residual, code = math.pi, "UNKNOWN_OP"

            # DEAD-INPUT CONTROL: ignore content entirely.
            if self.content_blind:
                residual, code = 0.0, None

            max_res = max(max_res, residual)
            if code is not None:
                violations.append((pc, code, residual))
            if terminate:
                break
            pc = nxt

        admitted = (max_res <= self.tau) and (not violations)
        return Verdict(
            admitted=admitted,
            delta_phi=max_res,
            violations=violations,
            bypass=False,
            n_steps=steps,
            state_digest=self._digest(state),
            content_blind=self.content_blind,
            tau=self.tau,
            budget_exhausted=budget_exhausted,
        )


# END
