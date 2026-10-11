"""Pin the ratified G-SJ3a / G-SJ3b bars (operator ruling 1).

RULING 1 (ratified): G-SJ3 was formally bifurcated.
  G-SJ3a  deduplicated 206-candidate bank, Top-1 >= 0.0450, a VERIFIED 10x lift over the
          0.0046 matched random control, with |r(slot, pick)| <= 0.005.
  G-SJ3b  full multimodal bank, Top-1 >= 0.1600. BLOCKED ON SUBSYSTEM: it presupposes
          henri_resonator_flow.py, which has ZERO files in the tree, and the dedup
          containment ceiling is 0.1367.

MEASURED REALITY (from the earlier audit, not re-derived here): the Two-Stage Selector
scores 0.0481 on the deduplicated 206-candidate bank against a 0.0046 matched control.

TENSION IN THE RULING, RECORDED RATHER THAN RECONCILED
  The two numeric clauses in G-SJ3a do not agree at the stated bar:
      bar  = 0.0450
      10x  = 10 * 0.0046 = 0.0460
  0.0460 > 0.0450, so a top-1 of exactly 0.0450 satisfies the BAR and fails the LIFT.
  The gate therefore reports the clauses SEPARATELY and only passes when both hold, so
  the disagreement is visible instead of hidden behind whichever clause is enforced
  first. The measured 0.0481 clears both (lift 10.456x), so the practical verdict is
  unaffected -- but the bar itself should be raised to 0.0460 to be self-consistent.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_V2, os.path.join(_V2, "henri_core"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from henri_core.wave_readout import (  # noqa: E402
    G_SJ3A_BAR, G_SJ3A_INDEX_BAR, G_SJ3A_LIFT_REQUIRED, G_SJ3A_RANDOM_CONTROL,
    G_SJ3B_BAR, G_SJ3B_REQUIRES, G_SJ3B_STATUS, gate_sj3a, gate_sj3b)


def test_gsj3_constants_are_the_ratified_values():
    """The bars on disk must equal the ruling's numbers, not a drifted copy."""
    assert G_SJ3A_BAR == 0.0450, G_SJ3A_BAR
    assert G_SJ3A_RANDOM_CONTROL == 0.0046, G_SJ3A_RANDOM_CONTROL
    assert G_SJ3A_LIFT_REQUIRED == 10.0, G_SJ3A_LIFT_REQUIRED
    assert G_SJ3A_INDEX_BAR == 0.005, G_SJ3A_INDEX_BAR
    assert G_SJ3B_BAR == 0.1600, G_SJ3B_BAR


def test_gsj3a_dedup_bank_index_is_UNMEASURED_so_the_gate_is_BLOCKED():
    """BANK CONFLATION DEFECT IN MY OWN FIRST DRAFT, corrected here.

    The first draft asserted `gate_sj3a(0.0481, slot_r=-0.0216)` PASSES. It does not,
    and the gate was right: |−0.0216| = 0.0216 > 0.005. But the deeper error was
    MINE -- r = −0.0216 is the FULL 1092-row bank statistic (the 0.1556 row). G-SJ3a's
    index clause is specified on the DEDUPLICATED 206-candidate bank, where r has
    NEVER been measured. A statistic from one bank does not license a claim about
    another. The honest report is BLOCKED, not PASS and not FAIL.
    """
    r = gate_sj3a(0.0481, slot_r=None)
    assert r["passes_bar"], r
    assert r["passes_lift"], r
    assert not r["passes"], r
    assert r["verdict"] == "G_SJ3A_BLOCKED_INDEX_UNMEASURED", r
    assert "DEDUPLICATED" in r["blocked_reason"], r["blocked_reason"]


def test_gsj3a_passes_only_with_a_MEASURED_index_neutral_slot_r():
    """Conditional pass: IF the dedup-bank r is measured and neutral, the gate passes.

    This pins the gate's positive path without asserting an unmeasured number.
    """
    r = gate_sj3a(0.0481, slot_r=-0.002)
    assert r["passes_bar"] and r["passes_lift"], r
    assert r["passes_index_neutrality"], r
    assert r["passes"], r
    assert r["verdict"] == "G_SJ3A_PASSED", r
    assert r["lift"] >= 10.0, r["lift"]


def test_gsj3a_full_bank_r_does_NOT_license_a_dedup_bank_claim():
    """The specific conflation, named: applying the full-bank r to the dedup bank FAILS.

    So the number cannot be borrowed across banks; it must be measured where claimed.
    """
    r = gate_sj3a(0.0481, slot_r=-0.0216)
    assert not r["passes_index_neutrality"], r
    assert not r["passes"], r
    assert r["verdict"] == "G_SJ3A_FAILED_INDEX_BOUND", r


def test_gsj3a_negative_control_fails_the_bar():
    """A selector at the control rate must FAIL, or the bar is vacuous."""
    r = gate_sj3a(0.0046, slot_r=0.0)
    assert not r["passes"], r
    assert not r["passes_lift"], r


def test_gsj3a_the_bars_own_tension_is_detectable():
    """PIN THE INCONSISTENCY: at exactly the bar, the lift clause fails.

    10 * 0.0046 = 0.0460 > 0.0450. This test asserts the disagreement EXISTS, so a
    future edit that silently changes one number without the other is caught.
    """
    r = gate_sj3a(G_SJ3A_BAR, slot_r=0.0)
    assert r["passes_bar"], r
    assert not r["passes_lift"], r
    assert not r["passes"], "the combined gate must fail while the clauses disagree"
    assert G_SJ3A_LIFT_REQUIRED * G_SJ3A_RANDOM_CONTROL > G_SJ3A_BAR, (
        "if this becomes false the ruling was made self-consistent; update this pin")


def test_gsj3a_slot_bound_selector_is_rejected_even_at_high_accuracy():
    """Index drift must fail the gate no matter how high the accuracy.

    argmax scored top-1 0.0000 with 19.51 sigma permutation drift; an index-bound
    selector is the defect this program keeps catching.
    """
    r = gate_sj3a(0.90, slot_r=0.80)
    assert r["passes_bar"] and r["passes_lift"], r
    assert not r["passes_index_neutrality"], r
    assert not r["passes"], "an index-bound selector must not pass"


def test_gsj3a_missing_slot_r_fails_closed():
    """No index measurement means NO pass. Absence of evidence is not neutrality."""
    r = gate_sj3a(0.0481, slot_r=None)
    assert not r["passes"], r
    assert not r["passes_index_neutrality"], r


def test_gsj3b_is_blocked_and_never_passes():
    """G-SJ3b must never report a pass, and must name its missing subsystem."""
    r = gate_sj3b(top1=0.1556)
    assert not r["passes"], r
    assert r["status"] == G_SJ3B_STATUS == "BLOCKED_ON_SUBSYSTEM", r
    assert r["requires_subsystem"] == G_SJ3B_REQUIRES, r


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except AssertionError as exc:
            bad += 1
            print("FAIL", fn.__name__, exc)
    print(f"{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
