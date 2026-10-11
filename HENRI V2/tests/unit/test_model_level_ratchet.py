"""Discriminating tests for the MODEL-LEVEL ratchet (thesis item 2).

Every test CAN FAIL. The load-bearing ones pin a NEGATIVE result, so a later reader cannot
mistake a weak readout for a working ratchet.

WHAT WAS MEASURED (henri_model_ratchet_observed.json)
    ARM R real      exact 0.1111 0.1296 0.1759 0.1759 0.1667 0.1667   growth +0.0556
    ARM S shuffled  exact 0.0000 0.0926 0.1019 0.0556 0.0648 0.0556
    ARM N synthetic exact 0.0648 0.1204 0.1204 0.1204 0.1204 0.1204
    ceiling (ridge on ALL waking rows) exact 0.2037
    bar = max(unigram trace floor, shuffled final) + 0.05 = 0.3188
    VERDICT: MODEL_RATCHET_NOT_CONFIRMED_READOUT_WEAK__FROZEN_WAVE_INSUFFICIENT

WHY THE NEGATIVE IS INFORMATIVE, AND WHY IT IS PINNED
    A store-level accumulator is monotone BY CONSTRUCTION. An argmax ACCURACY is not. ARM R
    rises 0.1111 -> 0.1759 then DECAYS to 0.1667. That decay is the ratchet failing, and it
    is the single most important number here: more experience did not buy more competence
    past a point. The pre-registered rule then returns its second branch: the frozen wave
    carries insufficient information, so a readout fix cannot pass and learnable ingress is
    a prerequisite. That matches the in-tree M4 result independently (held-out token acc
    0.0 against floor 0.1171; wave weak signal 0.1667 vs shuffled 0.0185).

SCOPE. Frozen encoder, ridge readout, synthetic digit-program substrate, single seed, CPU.
NOT an AAII claim, NOT a capability claim, NOT a CUDA result.
"""
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(V, ".."))
RECEIPT = os.path.join(REPO, "design", "zone_a", "evidence", "henri_model_ratchet_observed.json")


def _load():
    if not os.path.exists(RECEIPT):
        pytest.skip("model-level ratchet receipt not produced yet")
    return json.load(open(RECEIPT, encoding="utf-8"))


# ============================================== THE SETUP IS SOUND
def test_waves_are_finite_and_the_control_is_not_a_noop():
    """A no-op shuffle would make the shuffled arm identical to the real arm and the
    control vacuous. The receipt must report a non-trivial distinguishable fraction."""
    R = _load()
    assert R["setup"]["waves_finite"] is True
    d = R["shuffle_distinguishable_fraction"]
    assert d["min"] is not None
    assert d["min"] >= R["preregistered"]["min_distinguishable_fraction"], \
        f"shuffle moved only {d['min']:.2f} of positions; control is vacuous"
    assert R["checks"]["control_is_not_a_noop"] is True


def test_bar_is_the_preregistered_rule_not_an_invented_number():
    """The bar must come from the in-tree rule, so it cannot be tuned to the result.

    bar = max(unigram_floor_trace, shuffled_final) + 0.05
    """
    R = _load()
    floor = R["unigram_floor"]["trace_floor"]
    shuf = R["decision"]["shuffled_final"]
    expected = max(floor, shuf) + 0.05
    assert R["decision"]["bar"] == pytest.approx(expected, abs=1e-9), \
        f"bar {R['decision']['bar']} != preregistered {expected}"
    assert "exp_m4_readout_capacity.py" in R["reuses_bar_from"]


# ============================================== THE NEGATIVE RESULT IS PINNED
def test_arm_R_is_not_monotone_and_the_decay_is_recorded():
    """THE MOST IMPORTANT TEST HERE. A ratchet is MONOTONE NON-DECREASING by definition.

    Measured: 0.1111 0.1296 0.1759 0.1759 0.1667 0.1667. ARM R DECAYS. If a later change
    makes this monotone, that change must be re-examined -- the decay is the finding.
    """
    R = _load()
    series = R["ratchet"]["exact_match_series_R"]
    assert len(series) >= 3
    assert R["ratchet"]["monotone"] is False, "if this becomes True the finding changed"
    assert R["checks"]["arm_R_is_monotone"] is False
    # the decay is real: a later value is strictly below an earlier maximum
    peak = max(series)
    assert series[-1] < peak, f"expected decay from peak {peak}, series {series}"


def test_arm_R_does_not_clear_the_bar_and_the_verdict_says_so():
    """The ratchet is NOT confirmed. Pin the verdict string and the distance to the bar."""
    R = _load()
    series = R["ratchet"]["exact_match_series_R"]
    assert R["checks"]["arm_R_beats_preregistered_bar"] is False
    assert series[-1] < R["decision"]["bar"]
    assert R["verdict"] == "MODEL_RATCHET_NOT_CONFIRMED_READOUT_WEAK__FROZEN_WAVE_INSUFFICIENT"


def test_the_frozen_wave_is_named_as_the_blocker_not_the_readout():
    """The pre-registered rule's second branch. Naming the blocker keeps the next step
    honest: a bigger readout is not the fix; learnable ingress is."""
    R = _load()
    joined = (R["verdict"] + " " + " ".join(R["non_claims"])).lower()
    assert "frozen_wave_insufficient" in joined or "frozen wave" in joined
    assert "train_body=false" in joined
    # the residual must refuse an omni-modal reading
    assert "does not measure an omni-modal capability" in R["residual"].lower()
    assert R["evidence_class"] == "OBSERVED"


# ============================================== CONTROL SEPARATION
def test_real_beats_shuffled_but_by_a_narrow_margin():
    """The wave carries WEAK signal. It must beat the shuffled control, and the margin must
    be recorded as narrow -- that is the measured condition, not a pass."""
    R = _load()
    exR = R["ratchet"]["exact_match_series_R"]
    assert R["checks"]["arm_R_beats_shuffled_control"] is True
    shuf = R["decision"]["shuffled_final"]
    assert exR[-1] > shuf, f"real {exR[-1]} must exceed shuffled {shuf}"
    # and the gap is small relative to the bar -- the honest characterisation
    assert (exR[-1] - shuf) < R["decision"]["bar"]


def test_nothing_here_is_a_cuda_or_capability_claim():
    """Scope guards. This is a CPU readout measurement on a synthetic substrate."""
    R = _load()
    joined = " ".join(R["non_claims"]).lower()
    assert "not an aaii claim" in joined
    assert "not a capability claim" in joined
    assert "not a cuda result" in joined
    assert "prerequisite" in joined
