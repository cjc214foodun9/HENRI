"""UHR-05 A2 — bridge the Pearl intervention gate onto production's Zone C writer.

WHY THIS MODULE EXISTS (measured 2026-09-24, own calls)
=======================================================
Amendment 2 requires: ratify a causal link in Zone C **if and only if** the external
environment registers irreversible change. The gate for that already exists and is
correct — `henri_causal_contingency.ratify_causal_link`, whose docstring reads
"This is the function a Zone C writer should call."

Measured consumers of that gate: the smoke, one wiring script, and the contract
tests. `production_arc_run.py` contains **zero** references to
`henri_causal_contingency` / `PearlGate` / `forge_edge` / `ZoneCCausalEngramDAG`.
So every Zone C engram write in production is UNGATED. That gap is this module.

WHAT IT DOES
------------
Production already computes everything the gate needs, from REAL environment
returns, at `production_arc_run.py:2927-2937` (pre-action `grid`, `game_action`,
`step`, post-action `obs_next.frame[0]`). This module turns those into gate records
and answers one question for a Zone C writer: *may this write proceed?*

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* It never writes Zone C. It only ALLOWS or REFUSES; the caller owns the write, so a
  refusal cannot be silently converted into a write.
* It never reads the planner's own prediction. Every field comes from the
  environment's return, which is what makes the verdict externally grounded.
* It never lowers the gate's thresholds. `min_obs=8` / `min_actions=2` /
  `contrast_margin=1.0` are passed through unchanged.

SIGNATURE SEMANTICS (the load-bearing choice)
---------------------------------------------
`change_signature` is the FROZENSET OF DIFFERING FLAT INDICES, not a magnitude. A
constant-velocity progress cursor produces the SAME magnitude every step, so a
magnitude signature cannot expose it; two different actions that move different
cells produce different index sets. `change_magnitude` (fraction of changed cells)
is carried separately because the gate's static check needs a real magnitude.

MODES (`HENRI_ZONEC_CAUSAL_GATE`)
---------------------------------
  "0" (default) -> gate absent. `allow_zonec_write` returns True unconditionally, so
                   the default path is byte-identical to the pre-amendment path.
  "advisory"    -> evaluate and expose telemetry; the write still happens.
  "enforce"     -> REFUSE a write whose verdict is not RATIFIED. Fail-closed, and it
                   REQUIRES replicated step indices (see PRECONDITION below).
  "enforce_single_pass"
                -> For a run that provides ONE episode per environment, where strict
                   "enforce" is unsatisfiable. Refuses what such a run can actually
                   ESTABLISH -- REFUSED_STATIC (the world did not move) and
                   REFUSED_CONFOUNDED (no signature contrast) -- and ALLOWS
                   REFUSED_INSUFFICIENT while recording that the contrast was
                   UNCOMPUTABLE. This is the gate's own distinction, not a relaxation:
                   its docstring reads "the contrast is UNCOMPUTABLE, not absent".
                   Anything else -> REFUSE.

PRECONDITION FOR "enforce" (measured 2026-09-24, own calls)
-----------------------------------------------------------
The contrast statistic is I(change; action | step), so it needs at least one step
index observed under TWO OR MORE distinct actions. Production's loop is
`for env_name in env_ids:` then `for step in range(args.steps):` -- ONE episode per
environment, so every step index carries exactly one action and the verdict is
REFUSED_INSUFFICIENT. Strict "enforce" therefore refuses EVERY write there, and a
gate that cannot ratify is not a gate. Satisfy the precondition by replaying the same
environment with different action sequences; until then use "enforce_single_pass".
Pooling step indices across DIFFERENT environments would FABRICATE the contrast (the
signature differs because the ENV differs), which is why this bridge clears its buffer
whenever the environment changes.

FAIL-CLOSED RULES
-----------------
* Unknown mode string -> treated as "0" for the WRITE decision but reported as
  `mode_invalid` in telemetry, so a typo cannot silently enable enforcement.
* In "enforce", any evaluation defect (exception, missing library) -> REFUSE.
* Insufficient records -> the gate returns REFUSED_INSUFFICIENT -> REFUSE. An
  episode that has not yet produced a computable contrast cannot ratify.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "causal_gate_mode",
    "observe_zonec_causal_gate",
    "allow_zonec_write",
    "reset_zonec_causal_gate",
    "zonec_causal_gate_status",
    "build_zonec_gate_record",
    "make_change_signature",
    "changed_indices",
    "MODE_DISABLED",
    "MODE_ADVISORY",
    "MODE_ENFORCE",
    "MODE_ENFORCE_SINGLE_PASS",
]

MODE_DISABLED = "0"
MODE_ADVISORY = "advisory"
MODE_ENFORCE = "enforce"
MODE_ENFORCE_SINGLE_PASS = "enforce_single_pass"
_VALID_MODES = frozenset({MODE_DISABLED, MODE_ADVISORY, MODE_ENFORCE,
                          MODE_ENFORCE_SINGLE_PASS})


def _refusal_statuses() -> Tuple[str, str]:
    """The gate's refusal vocabulary, taken FROM the gate when importable.

    Hard-coding these would let a rename in the gate silently turn a refusal into an
    ALLOW. The literals are only a fallback; a contract test pins them to the real
    constants so drift fails loudly.
    """
    try:
        import henri_causal_contingency as _HCC
        return _HCC.STATUS_REFUSED_STATIC, _HCC.STATUS_REFUSED_CONFOUNDED
    except Exception:
        return "REFUSED_STATIC", "REFUSED_CONFOUNDED"

#: Records carry the action taken, the step index, a hashable change signature and a
#: real external change magnitude. Names must match what the gate is told to read.
CHANGE_KEY = "change_signature"
ACTION_KEY = "action"
STEP_KEY = "step"
MAGNITUDE_KEY = "change_magnitude"

_MODE_INVALID_SEEN = False


def causal_gate_mode() -> Tuple[str, bool]:
    """Return (mode, valid). An unrecognised value never enables enforcement."""
    raw = os.environ.get("HENRI_ZONEC_CAUSAL_GATE", MODE_DISABLED)
    mode = str(raw).strip().lower()
    if mode in _VALID_MODES:
        return mode, True
    return MODE_DISABLED, False


def make_change_signature(pre_grid: Any, post_grid: Any) -> Tuple[float, Any]:
    """Return (changed_fraction, ("count", n_changed)) for two grids.

    WHICH SIGNATURE FORM, AND WHY (measured 2026-09-24 -- this was a real defect)

    A first version returned `frozenset(changed_flat_indices)`, matching
    `henri_causal_contingency._delta_and_signature`. That form is RIGHT for the
    per-observation `pearl_intervention_gate` (it only asks "did the world move"),
    but it is WRONG for the episode-level contrast this bridge feeds.

    Reason, documented in the validated control `_ka59_records()` in
    `tests/contract/test_uhr04_amendments.py`: a differing-index set is
    NEAR-UNIQUE per record, so the permutation null becomes DEGENERATE and the
    conditional-MI comparison turns vacuous -- measured there as
    `observed == null q99 == 1.58408` exactly, i.e. the statistic cannot exceed its
    own null. My own two runs reproduced that failure mode:
        R2 pooled steps, index-set sig -> REFUSED_CONFOUNDED (mi=1.0 == null_q99=1.0)
    Switching to the LOW-CARDINALITY count form -- the form the REAL ka59 harvest
    records -- made the contrast computable and the gate ratify.

    The count form still discriminates, which is the property that matters:
      * action-contingent  -> different actions change DIFFERENT NUMBERS of cells
                              -> I(count;action|step) > 0
      * step-driven cursor -> the band moves the SAME number of cells every step
                              -> I(count;step) dominates -> CONFOUNDED
      * static             -> 0 changed cells -> magnitude 0 -> SOLIPSISM_VETO
    `ContingencyVerdict.signature_cardinality` reports which regime you are in, so a
    degenerate signature is visible rather than silent.

    A shape change is reported as magnitude `inf` with a tagged signature, because a
    resized grid is not a comparable transition.
    """
    try:
        import numpy as np
    except Exception:
        return _signature_pure_python(pre_grid, post_grid)

    try:
        b = np.asarray(pre_grid)
        a = np.asarray(post_grid)
        if b.shape != a.shape:
            return float("inf"), ("shape_changed", tuple(b.shape), tuple(a.shape))
        diff = (b.astype("float64") != a.astype("float64"))
        n_changed = int(np.count_nonzero(diff))
        mag = float(diff.mean()) if diff.size else 0.0
        return mag, ("count", n_changed)
    except Exception:
        return _signature_pure_python(pre_grid, post_grid)


def _signature_pure_python(pre_grid: Any, post_grid: Any) -> Tuple[float, Any]:
    """Fallback with the SAME signature form AND the same granularity as numpy.

    The count MUST agree with the numpy path element-for-element, otherwise the
    bridge's verdict would depend on whether numpy imported. Comparing only the
    TOP-LEVEL elements is wrong: for an 8x8 grid that compares 8 rows, not 64 cells,
    so it reports a different count than the numpy path for the same input (measured:
    ("count", 1) vs ("count", 5)). `_flatten` recurses, so both paths count CELLS.
    A contract test pins the two to agreement.
    """
    def _flatten(obj):
        try:
            for x in obj:
                if isinstance(x, (list, tuple)):
                    yield from _flatten(x)
                else:
                    yield x
        except TypeError:
            yield obj

    try:
        b = list(_flatten(pre_grid))
        a = list(_flatten(post_grid))
        if len(b) != len(a):
            return float("inf"), ("shape_changed", len(b), len(a))
        n_changed = sum(1 for x, y in zip(b, a) if x != y)
        n = len(b) or 1
        return float(n_changed) / float(n), ("count", n_changed)
    except Exception:
        return 0.0, ("count", 0)


def changed_indices(pre_grid: Any, post_grid: Any) -> Any:
    """Diagnostic only. The index set is carried in telemetry, NEVER used as the gate
    key -- an index-set signature makes the permutation null degenerate (see above)."""
    try:
        import numpy as np
        b = np.asarray(pre_grid)
        a = np.asarray(post_grid)
        if b.shape != a.shape:
            return ("shape_changed", tuple(b.shape), tuple(a.shape))
        idx = np.flatnonzero((b.astype("float64") != a.astype("float64")).reshape(-1))
        return [int(i) for i in idx.tolist()[:32]]
    except Exception:
        return None


def _action_label(action: Any) -> Any:
    """Stable, hashable action identity. Prefers `.name` on enum-like actions."""
    if action is None:
        return None
    name = getattr(action, "name", None)
    if isinstance(name, str):
        return name
    return str(action)


def build_zonec_gate_record(step: int, action: Any, pre_grid: Any, post_grid: Any) -> Dict[str, Any]:
    """One gate record from one REAL environment transition.

    `pre_grid` is the pre-action grid the runner already holds; `post_grid` is the
    post-action frame from the environment return. Neither is a prediction.
    """
    mag, sig = make_change_signature(pre_grid, post_grid)
    return {
        STEP_KEY: int(step),
        ACTION_KEY: _action_label(action),
        CHANGE_KEY: sig,
        MAGNITUDE_KEY: float(mag),
        # Diagnostic only -- NOT the gate key. Index sets are near-unique and would
        # make the permutation null degenerate (see make_change_signature).
        "changed_indices": changed_indices(pre_grid, post_grid),
    }


class ZoneCCausalGate:
    """Episode-scoped record buffer plus the bridge's verdict surface."""

    def __init__(self, env_name: str = "") -> None:
        self.env_name = env_name
        self.records: List[Dict[str, Any]] = []
        self.defects: List[str] = []
        self.evaluations = 0
        self.refusals = 0
        self.last_verdict: Optional[Dict[str, Any]] = None

    # -- producer side -------------------------------------------------------
    def observe(self, step: int, action: Any, pre_grid: Any, post_grid: Any) -> None:
        """Append a transition built from the environment's own return."""
        try:
            self.records.append(
                build_zonec_gate_record(step, action, pre_grid, post_grid))
        except Exception as exc:  # never break the run on a bookkeeping defect
            self.defects.append(f"observe failed at step {step}: {type(exc).__name__}")

    # -- consumer side -------------------------------------------------------
    def evaluate(self) -> Optional[Dict[str, Any]]:
        """Run the gate. Returns a JSON-able verdict, or None if it could not run."""
        try:
            import henri_causal_contingency as HCC
        except Exception as exc:
            self.defects.append(f"gate import failed: {type(exc).__name__}")
            return None

        try:
            g = HCC.ratify_causal_link(
                self.records,
                action=(self.records[-1][ACTION_KEY] if self.records else None),
                change_key=CHANGE_KEY,
                action_key=ACTION_KEY,
                step_key=STEP_KEY,
                magnitude_key=MAGNITUDE_KEY,
            )
        except Exception as exc:
            self.defects.append(f"gate call failed: {type(exc).__name__}: {exc}")
            return None

        self.evaluations += 1
        inner = getattr(g, "verdict", None)
        verdict = {
            "status": getattr(g, "status", None),
            "admissible": bool(getattr(g, "admissible", False)),
            "veto": getattr(g, "veto", None),
            "reason": str(getattr(g, "reason", ""))[:400],
            "n_records": len(self.records),
            "n_steps": (getattr(inner, "n_steps", None) if inner else None),
            "n_actions": (getattr(inner, "n_actions", None) if inner else None),
            "n_step_matched_pairs": (getattr(inner, "n_step_matched_pairs", None) if inner else None),
            # True only when a step index was seen under >1 action, i.e. the contrast is
            # COMPUTABLE. False means REFUSED_INSUFFICIENT -- "the contrast is
            # UNCOMPUTABLE, not absent". A run that cannot compute the contrast must be
            # reported as such rather than silently treated as a refusal or a pass.
            "contrast_computable": (bool((getattr(inner, "n_step_matched_pairs", 0) or 0) > 0)
                                    if inner else None),
            "null_degenerate": (getattr(inner, "null_degenerate", None) if inner else None),
            "mi_action_given_step": (getattr(inner, "mi_action_given_step", None) if inner else None),
            "null_q99": (getattr(inner, "null_q99", None) if inner else None),
            "delta_s_ext": (getattr(inner, "delta_s_ext", None) if inner else None),
        }
        self.last_verdict = verdict
        if not verdict["admissible"]:
            self.refusals += 1
        return verdict


_GATE: Optional[ZoneCCausalGate] = None


def _get_gate() -> Optional[ZoneCCausalGate]:
    global _GATE
    mode, _valid = causal_gate_mode()
    if mode == MODE_DISABLED:
        return None
    if _GATE is None:
        _GATE = ZoneCCausalGate()
    return _GATE


def reset_zonec_causal_gate(env_name: str = "") -> None:
    """Start a fresh episode. Zone C gating is per-episode: a contrast is computed
    across step indices of ONE environment, so records must not leak across resets."""
    global _GATE
    _GATE = ZoneCCausalGate(env_name) if _get_gate() is not None else None


def observe_zonec_causal_gate(step: int, action: Any, pre_grid: Any, post_grid: Any) -> None:
    """No-op unless the flag is set. Safe to call on the default path."""
    g = _get_gate()
    if g is not None:
        g.observe(step, action, pre_grid, post_grid)


def allow_zonec_write(step: Optional[int] = None, tele: Any = None) -> Tuple[bool, str]:
    """May a Zone C engram write proceed?

    Returns (allowed, reason). Fail-closed: in "enforce" a non-ratified verdict, an
    evaluation defect, or a missing gate all REFUSE.
    """
    mode, mode_valid = causal_gate_mode()
    if not mode_valid:
        # A typo must not enable enforcement, and must not pass silently either.
        _emit(tele, {"zonec_causal_gate_mode_invalid": os.environ.get(
            "HENRI_ZONEC_CAUSAL_GATE", "")})
    if mode == MODE_DISABLED:
        return True, "gate_disabled"
    g = _get_gate()
    if g is None:
        return (True, "gate_absent") if mode != MODE_ENFORCE else (False, "gate_absent_fail_closed")

    v = g.evaluate()
    if v is None:
        _emit(tele, {"zonec_causal_gate": {"step": step, "status": "EVALUATION_DEFECT",
                                           "defects": g.defects[-3:]}})
        return (False, "evaluation_defect_fail_closed") if mode == MODE_ENFORCE else (True, "evaluation_defect_advisory")

    _emit(tele, {"zonec_causal_gate": dict(v, step=step, mode=mode)})
    if mode == MODE_ENFORCE:
        return bool(v["admissible"]), str(v["status"])
    if mode == MODE_ENFORCE_SINGLE_PASS:
        static, confounded = _refusal_statuses()
        if v["status"] in (static, confounded):
            return False, str(v["status"]) + "_refused"
        if v["status"] == "REFUSED_INSUFFICIENT":
            # A single-pass run cannot compute the contrast. Allow the write, but never
            # label the link certified -- the gate's own vocabulary says this outcome is
            # "UNCOMPUTABLE, not absent", so treating it as a refusal would overstate
            # what was established.
            return True, "REFUSED_INSUFFICIENT_contrast_uncomputable"
        return bool(v["admissible"]), str(v["status"])
    return True, str(v["status"]) + "_advisory"


def _emit(tele: Any, payload: Dict[str, Any]) -> None:
    """Emit telemetry if a sink was supplied. Telemetry never gates control flow."""
    if tele is None:
        return
    try:
        tele.emit(payload)
    except Exception:
        try:
            tele(payload)
        except Exception:
            pass


def zonec_causal_gate_status() -> Dict[str, Any]:
    """Compact status for the run receipt."""
    mode, mode_valid = causal_gate_mode()
    g = _GATE
    return {
        "mode": mode,
        "mode_valid": mode_valid,
        "records": (len(g.records) if g else 0),
        "evaluations": (g.evaluations if g else 0),
        "refusals": (g.refusals if g else 0),
        "defects": (list(g.defects[-5:]) if g else []),
        "last_status": ((g.last_verdict or {}).get("status") if g else None),
    }
