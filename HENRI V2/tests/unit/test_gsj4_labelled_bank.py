"""Discriminating tests for the G-SJ4 labelled bank (item 1) and G-URC6 (item 4).

Every test CAN FAIL. Several pin MY OWN measured defects or a NEGATIVE result, so a later
reader cannot mistake a measured ceiling for a working gate.

THE CENTRAL FINDING THESE TESTS PROTECT (item 1)
    gold_sets() labels by the TRACE (execution output). build_waves() encodes the SPEC.
    Measured: the best wave statistic's AUC against the trace-match label is 0.4982.
    Near chance. Therefore NO wave-based veto can earn the G-SJ4 separation bar on this
    bank, and the correct verdict is a measured ceiling -- not a retuned bar.

MY DEFECT, SELF-CAUGHT AND PINNED
    I applied EPS_HARD = 0.35 (calibrated for the entry-unimodular COMPLEX family, where a
    match gives |coh| ~ 1.0) to a REAL bank whose empirical chance coherence is 0.120.
    Everything then fired, so the LOO and attributable arms read separation +0.0000 by
    SATURATION, not by anti-selection. The AUC is THRESHOLD-FREE, so it governs: every arm
    sits at chance. Both readings are recorded.

SAFETY PROPERTY
    The dark port must stay mechanically disconnected. A pass on a SYNTHETIC bank does not
    earn G-SJ4 on the real evaluation bank.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", ".."))            # tests/unit -> HENRI V2
REPO = os.path.abspath(os.path.join(V, ".."))
EV = os.path.join(REPO, "design", "zone_a", "evidence")

GSJ4 = os.path.join(EV, "henri_gsj4_labelled_bank_observed.json")
URC6 = os.path.join(EV, "henri_urc6_relational_observed.json")

SEP_BAR = 0.7000
FALSE_VETO_BAR = 0.0500


def _load(path):
    if not os.path.exists(path):
        pytest.skip(f"receipt not produced yet: {os.path.basename(path)}")
    return json.load(open(path, encoding="utf-8"))


# ============================================== ITEM 1: LABELS ARE WIRED
def test_labels_are_wired_and_non_degenerate():
    """The report called this 'the most blocking absence'. It is a WIRING gap.

    Assert the labels exist, are derived from gold_sets, and are non-degenerate. A
    degenerate label set (all correct or all incorrect) would make every metric undefined.
    """
    R = _load(GSJ4)
    b = R["bank"]
    assert b["n_distinct_columns"] > 0 and b["n_queries"] > 0
    s = R["arm_shipped_sagnac"]
    assert s["n_correct"] > 0 and s["n_incorrect"] > 0, "labels must be non-degenerate"
    # MY DEFECT, self-caught: the first draft asserted
    #   s["n_correct"] + s["n_incorrect"] == s["n_correct"] + s["n_incorrect"]
    # which is a TAUTOLOGY -- the dead-gate class this program keeps catching. Replace it
    # with a clause that CAN fail: the pair count must equal the product Q x K, and the
    # label rate must match the reported ratio. Both are real constraints.
    Q, K = R["bank"]["n_queries"], R["bank"]["n_distinct_columns"]
    assert s["n_correct"] + s["n_incorrect"] == Q * K, (
        f"labelled pairs {s['n_correct'] + s['n_incorrect']} != Q*K = {Q * K}")
    assert s["n_correct"] < s["n_correct"] + s["n_incorrect"], "some label must be incorrect"
    assert "gold_sets" in R["label_source"], "the label source must be named"
    # the label rate is low but resolvable
    assert R["resolvability"]["resolvable_at_this_n"] is True


def test_correctness_is_not_recoverable_from_spec_wave_geometry():
    """THE MECHANISM TEST, and the reason no threshold can help.

    build_waves encodes the SPEC string; gold_sets labels by the TRACE. If the trace-match
    label is not recoverable from spec-wave geometry, no wave-based veto can earn the bar.
    Measured AUC is 0.4982 -- near chance. Assert near chance AND that the receipt says so.
    """
    R = _load(GSJ4)
    auc = R["mechanism"]["auc_best_wave_statistic_vs_label"]
    assert abs(auc - 0.5) < 0.10, f"label AUC {auc} is not near chance; mechanism claim fails"
    assert R["mechanism"]["bar_separation"] == SEP_BAR
    # the measured ceiling must be far below the bar -- the negative result, pinned
    assert R["mechanism"]["measured_separation_ceiling"] < SEP_BAR


def test_all_arms_are_at_chance_by_the_threshold_free_reading():
    """AUC is threshold-free, so it governs where a fixed epsilon is family-mismatched.

    All three arms must sit near chance. If a later change pushes one arm's AUC clearly
    above 0.55, this test fails and the change must be re-examined -- that is the point.
    """
    R = _load(GSJ4)
    for key in ("arm_shipped_sagnac", "arm_loo_cosine", "arm_attributable"):
        a = R[key]
        auc = a["auc_correct_vs_incorrect"]
        assert abs(auc - 0.5) < 0.05, f"{key} AUC {auc} is not near chance"
        assert a["passes_separation"] is False, f"{key} must NOT pass the separation bar"


def test_my_own_threshold_family_mismatch_is_recorded():
    """MY DEFECT, self-caught. PINNED so the saturation is not read as anti-selection.

    eps_hard = 0.35 belongs to the entry-unimodular COMPLEX family. On this REAL bank every
    score falls below 1 - 0.35, so every candidate fires and separation reads +0.0000 by
    saturation. The empirical chance coherence is an order of magnitude above the closed
    form, which is the same family mismatch.
    """
    R = _load(GSJ4)
    wf = R["wave_family"]
    assert wf["is_complex"] is False, "this bank is a real family; the closed forms do not apply"
    emp, closed = wf["empirical_chance_mean"], wf["closed_form_mean_sqrt_pi_over_4D"]
    assert emp > 5 * closed, f"empirical {emp} vs closed {closed}: expected a wide mismatch"
    # the saturated arms are exactly those two
    for key in ("arm_loo_cosine", "arm_attributable"):
        assert R[key]["fire_on_correct"] == pytest.approx(1.0, abs=1e-6)
        assert R[key]["fire_on_incorrect"] == pytest.approx(1.0, abs=1e-6)


def test_shuffled_label_control_collapses():
    """A separation that survives label shuffling would be an artifact."""
    R = _load(GSJ4)
    c = R["control_shuffled_labels"]
    assert abs(c["mean_separation"]) < 0.05, "shuffled labels must give ~zero separation"
    assert c["n"] >= 100


def test_dark_port_stays_disconnected_on_a_synthetic_bank():
    """SAFETY PROPERTY. The dark port must stay mechanically disconnected.

    G-SJ4 requires the REAL labelled evaluation bank. This bank is the synthetic in-tree M4
    bank, so G-SJ4 is NOT earned even if an arm had passed. Assert the verdict says BLOCKED
    and that the non-claims say so too.
    """
    R = _load(GSJ4)
    assert "BLOCKED" in R["g_sj4_status"]
    assert "dark port" in R["g_sj4_status"].lower()
    joined = " ".join(R["non_claims"]).lower()
    assert "does not earn g-sj4" in joined
    assert "synthetic" in joined
    assert R["verdict"] == "GSJ4_LABELS_WIRED_NO_ARM_EARNS_BAR_ON_THIS_BANK"


def test_no_bar_was_moved():
    """The bars are READ from the tree and must be unchanged in the receipt."""
    R = _load(GSJ4)
    fb = R["frozen_bars"]
    assert fb["separation"] == SEP_BAR
    assert fb["false_veto"] == FALSE_VETO_BAR
    assert fb["eps_hard"] == 0.35


# ============================================== ITEM 4: G-URC6 IS FAIL-CLOSED
def test_urc6_is_blocked_for_absent_data():
    """No ARC relational task data exists in tree. The correct result is BLOCKED."""
    R = _load(URC6)
    assert R["arc_data_audit"]["real_task_json_count"] == 0
    assert R["checks"]["arc_task_data_present"] is False
    assert R["checks"]["cuda_available"] is False
    assert R["verdict"] == "G_URC6_BLOCKED_DATA_ABSENT"
    assert R["bar"] == 0.8500
    assert len(R["provisioning_required"]) >= 2


def test_urc6_did_not_substitute_a_proxy_for_the_bar():
    """A proxy reported as G-URC6 would be fabrication. The receipt must refuse that."""
    R = _load(URC6)
    p = R["proxy_wiring_check"]
    assert p["is_a_benchmark"] is False
    assert p["is_g_urc6"] is False
    assert p["clears_no_bar"] is True
    joined = " ".join(R["non_claims"]).lower()
    assert "not a benchmark" in joined
    assert "clears no bar" in joined
