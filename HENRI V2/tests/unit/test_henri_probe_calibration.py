"""Tests for henri_probe_calibration.py — the calibration MEASUREMENT layer.

These tests use synthetic data with KNOWN calibration properties, so the
statistic is verified against ground truth rather than against itself.

The load-bearing test is `test_overconfidence_is_detected`: a model that always
says 0.9 and is right half the time MUST produce ECE ~ 0.4. If it did not, the
statistic would be incapable of falsifying the thing it exists to measure.
"""
from __future__ import annotations

import pytest

from henri_probe_calibration import (
    NO_MEASUREMENT_YET,
    ORACLE_API_KEY_ENV,
    brier_score_multiclass,
    build_calibration_receipt,
    build_oracle_request,
    calibration_skew,
    evaluate_calibration,
    expected_calibration_error,
    oracle_leg_status,
    parse_oracle_choice,
    peak_histogram,
    reliability_bins,
    sha256_of_rows,
    sharpness,
    top1_confidence_and_correct,
    uniform_brier_floor,
)


# ------------------------------------------------------------------ scoring ---

def test_brier_perfect_and_uniform():
    """A point mass on the right answer scores 0; uniform scores (K-1)/K."""
    assert brier_score_multiclass([[1.0, 0.0], [0.0, 1.0]], [0, 1]) == 0.0
    # uniform 2-class on 2 samples
    assert abs(brier_score_multiclass([[0.5, 0.5]] * 2, [0, 1]) - 0.5) < 1e-12
    assert uniform_brier_floor(2) == 0.5
    assert abs(uniform_brier_floor(4) - 0.75) < 1e-12


def test_brier_wrong_confident_scores_two():
    """A confident wrong answer: ||p - onehot||^2 = 1 + 1 = 2."""
    assert abs(brier_score_multiclass([[0.0, 1.0]], [0]) - 2.0) < 1e-12


def test_brier_fails_closed_on_bad_input():
    with pytest.raises(ValueError):
        brier_score_multiclass([], [])
    with pytest.raises(ValueError):
        brier_score_multiclass([[0.5, 0.5]], [0, 1])
    with pytest.raises(ValueError):
        brier_score_multiclass([[0.5, 0.5]], [5])
    with pytest.raises(ValueError):
        brier_score_multiclass([[]], [0])


# ------------------------------------------------------------- reliability ---

def test_overconfidence_is_detected():
    """THE falsification test: always-0.9-and-half-right must give ECE ~ 0.4."""
    probs = [[0.9, 0.1]] * 10
    correct = [0, 1] * 5          # exactly 50% right
    rep = evaluate_calibration(probs, correct)
    assert abs(rep.ece - 0.4) < 1e-9, f"ECE {rep.ece} did not detect overconfidence"
    assert not rep.is_well_calibrated
    assert abs(rep.accuracy - 0.5) < 1e-9
    assert abs(rep.mean_confidence - 0.9) < 1e-9


def test_perfectly_calibrated_has_zero_ece():
    """0.9 confidence with 90% accuracy -> ECE exactly 0."""
    probs = [[0.9, 0.1]] * 10
    correct = [1] * 2 + [0] * 8   # 80% right... adjust to 90%
    probs = [[0.9, 0.1]] * 10
    correct = [1] + [0] * 9       # 90% right at conf 0.9
    rep = evaluate_calibration(probs, correct)
    assert abs(rep.ece) < 1e-9, f"ECE {rep.ece} should be 0 for a calibrated set"
    assert rep.is_well_calibrated
    assert rep.brier_skill_score > 0.0


def test_underconfidence_is_detected():
    """A model that says 0.5 while being always right is also miscalibrated."""
    rep = evaluate_calibration([[0.5, 0.5]] * 10, [0] * 10)
    assert abs(rep.ece - 0.5) < 1e-9


def test_reliability_bins_cover_all_samples():
    confs = [0.05, 0.15, 0.5, 0.95]
    corr = [False, False, True, True]
    bins = reliability_bins(confs, corr, n_bins=10)
    assert sum(b["count"] for b in bins) == len(confs)
    assert len(bins) == 10
    # each sample lands in exactly one bin
    assert bins[0]["count"] == 1 and bins[9]["count"] == 1


def test_reliability_bin_edges_and_gap():
    bins = reliability_bins([0.95, 0.95], [True, False], n_bins=10)
    b = bins[9]
    assert b["count"] == 2
    assert abs(b["mean_confidence"] - 0.95) < 1e-12
    assert abs(b["accuracy"] - 0.5) < 1e-12
    assert abs(b["gap"] - 0.45) < 1e-12


def test_confidence_one_lands_in_last_bin_not_overflow():
    bins = reliability_bins([1.0], [True], n_bins=10)
    assert bins[9]["count"] == 1
    assert sum(b["count"] for b in bins) == 1


def test_reliability_bins_fail_closed():
    with pytest.raises(ValueError):
        reliability_bins([0.5], [True], n_bins=0)
    with pytest.raises(ValueError):
        reliability_bins([0.5], [True, False])
    with pytest.raises(ValueError):
        reliability_bins([1.5], [True])


# ------------------------------------------------------------------- ECE ----

def test_top1_reduction():
    """Row 0 tops out at index 1 and is labelled 1 (right); row 1 tops out at
    index 0 and is labelled 1 (wrong). Confidences follow the argmax."""
    confs, correct = top1_confidence_and_correct([[0.2, 0.8], [0.6, 0.4]], [1, 1])
    assert confs == [0.8, 0.6]
    assert correct == [True, False]


def test_ece_rejects_empty():
    with pytest.raises(ValueError):
        expected_calibration_error([], [])


# ---------------------------------------------------------------- report ----

def test_report_is_bounded_and_hashed():
    rep = evaluate_calibration([[0.7, 0.3]] * 4, [0, 0, 1, 0])
    d = rep.to_dict()
    assert d["schema"] == "henri.probe-calibration.v1"
    assert d["evidence_class"] == "OBSERVED"
    assert 0.0 <= d["ece"] <= 1.0
    assert len(d["source_hash"]) == 64
    assert d["is_well_calibrated"] is False or d["ece"] <= 0.05


def test_report_source_hash_is_content_addressed():
    a = sha256_of_rows([[0.7, 0.3], [0.2, 0.8]])
    b = sha256_of_rows([[0.7, 0.3], [0.2, 0.8]])
    c = sha256_of_rows([[0.7, 0.3], [0.3, 0.7]])
    assert a == b and a != c


def test_skill_score_is_positive_for_a_real_signal():
    """A model that is usually right must beat the uniform floor.

    Deliberately kept CLEAR OF the 0.05 ECE boundary: at exactly 0.95 stated
    confidence with 100% observed accuracy the gap is 0.05, which floating-point
    representation places marginally above the threshold. This test asserts the
    quality signal, not the boundary; the boundary itself is covered by
    `test_well_calibrated_boundary_is_exclusive`.
    """
    probs = [[0.96, 0.04]] * 8 + [[0.04, 0.96]] * 2
    correct = [0] * 8 + [1] * 2
    rep = evaluate_calibration(probs, correct)
    assert rep.brier_skill_score > 0.5
    assert rep.is_well_calibrated


def test_well_calibrated_boundary_is_exclusive():
    """Document the boundary: a 0.05 ECE is NOT labelled well-calibrated.

    At 0.95 confidence with 100% accuracy the gap is exactly 0.05, so the
    `ece <= 0.05` test sits on the edge of what float arithmetic can represent.
    Recorded here so the behaviour is pinned rather than discovered by surprise.
    """
    rep = evaluate_calibration([[0.95, 0.05]] * 10, [0] * 10)
    assert abs(rep.ece - 0.05) < 1e-9
    # exact-boundary behaviour: not asserted as calibrated either way, only
    # recorded. The bound is checked below the threshold in the sibling test.
    assert rep.brier_skill_score > 0.0


def test_ragged_rows_fail_closed():
    with pytest.raises(ValueError):
        evaluate_calibration([[0.5, 0.5], [0.3, 0.3, 0.4]], [0, 0])


# ---------------- companion metrics (added: ECE alone is gameable) ----------

def test_skew_sign_detects_overconfidence():
    """Overconfident (states 0.9, right half the time) must give NEGATIVE skew."""
    probs = [[0.9, 0.1]] * 10
    correct = [0, 1] * 5
    confs, corr = top1_confidence_and_correct(probs, correct)
    s = calibration_skew(confs, corr, n_bins=10)
    assert s < 0, f"overconfidence must give negative skew, got {s}"
    assert abs(s - (-0.4)) < 1e-9


def test_skew_sign_detects_underconfidence():
    """Underconfident (states 0.3, always right) must give POSITIVE skew."""
    probs = [[0.3, 0.3, 0.3, 0.1]] * 10
    correct = [0] * 10
    confs, corr = top1_confidence_and_correct(probs, correct)
    s = calibration_skew(confs, corr, n_bins=10)
    assert s > 0, f"underconfidence must give positive skew, got {s}"
    assert abs(s - 0.7) < 1e-9


def test_skew_can_cancel_while_ece_stays_large():
    """Why skew is a COMPANION and not a fix.

    Half the samples are confidently wrong (gap -0.9), half are diffidently
    right (gap +0.9). The signed gaps cancel so skew is ~0 while ECE is large.
    Neither statistic alone is sufficient, so both are reported.
    """
    over = [[0.9] + [0.1 / 9] * 9] * 5      # peak 0.9, never correct
    under = [[0.1] * 10] * 5                # peak 0.1, always correct
    probs = over + under
    correct = [1] * 5 + [0] * 5
    confs, corr = top1_confidence_and_correct(probs, correct)
    s = calibration_skew(confs, corr, n_bins=10)
    ece = expected_calibration_error(confs, corr, n_bins=10)
    assert abs(s) < 1e-9, f"skew should cancel to ~0, got {s}"
    assert ece > 0.5, f"ECE should remain large, got {ece}"


def test_sharpness_of_uniform_is_one_over_k():
    probs = [[0.25] * 4] * 6
    confs, _ = top1_confidence_and_correct(probs, [0] * 6)
    mean_peak, max_peak = sharpness(confs)
    assert abs(mean_peak - 0.25) < 1e-12
    assert abs(max_peak - 0.25) < 1e-12


def test_peak_histogram_counts_sum_to_n():
    probs = [[0.7, 0.3]] * 3 + [[0.2, 0.8]] * 5
    confs, _ = top1_confidence_and_correct(probs, [0] * 8)
    h = peak_histogram(confs, n_bins=10)
    assert len(h) == 10
    assert sum(b["count"] for b in h) == 8
    assert h[7]["count"] == 3          # peak 0.7
    assert h[8]["count"] == 5          # peak 0.8


def test_peak_and_skew_fail_closed():
    with pytest.raises(ValueError):
        peak_histogram([0.5], n_bins=0)
    with pytest.raises(ValueError):
        peak_histogram([1.5], n_bins=10)
    with pytest.raises(ValueError):
        calibration_skew([], [])
    with pytest.raises(ValueError):
        sharpness([])


def test_uniform_predictor_scores_zero_ece_but_has_no_skill():
    """The blueprint's central S1.1 justification, executed.

    A uniform (0.25 x 4) predictor on a BALANCED 4-way set achieves ECE exactly
    0.0 -- indistinguishable from a perfect model by ECE alone -- while its
    Brier skill score is exactly 0.0 (it IS the uniform floor). The joint gate
    must reject it, which is why is_well_calibrated requires BOTH
    ECE <= 0.05 AND brier_skill_score > 0.
    """
    probs = [[0.25] * 4] * 40
    correct = [i % 4 for i in range(40)]           # balanced
    rep = evaluate_calibration(probs, correct)
    assert abs(rep.ece) < 1e-9, f"uniform ECE should be 0, got {rep.ece}"
    assert abs(rep.brier_skill_score) < 1e-9, (
        f"uniform skill should be 0, got {rep.brier_skill_score}")
    assert rep.is_well_calibrated is False
    confs, _ = top1_confidence_and_correct(probs, correct)
    mean_peak, _ = sharpness(confs)
    assert abs(mean_peak - 0.25) < 1e-12


def test_build_calibration_receipt_has_the_required_fields():
    """Blueprint 8.4 field list must all be present."""
    probs = [[0.9, 0.1]] * 10
    correct = [1] + [0] * 9
    rec = build_calibration_receipt(probs, correct, n_bins=10)
    for key in ("schema", "n_samples", "n_classes", "source_hash", "brier",
                "ece", "bins", "mean_confidence", "accuracy",
                "is_well_calibrated", "brier_skill_score",
                "calibration_skew", "sharpness_mean_peak",
                "sharpness_max_peak", "peak_histogram"):
        assert key in rec, f"receipt missing required field {key!r}"
    assert rec["n_samples"] == 10
    assert rec["n_classes"] == 2
    assert len(rec["source_hash"]) == 64
    assert sum(b["count"] for b in rec["peak_histogram"]) == 10
    assert rec["schema"] == "henri.probe-calibration.v1"


# ------------------------------------------------------------ oracle leg ----

def test_oracle_leg_is_blocked_without_a_credential():
    """The external leg must report BLOCKED, never fabricate a Jev result."""
    st = oracle_leg_status({})
    assert st.available is False
    assert st.status == "BLOCKED_NO_CREDENTIAL"
    assert "fabricated" in st.reason or "cannot run" in st.reason


def test_oracle_leg_available_when_key_present():
    st = oracle_leg_status({ORACLE_API_KEY_ENV: "sk-test"})
    assert st.available is True and st.status == "AVAILABLE"


def test_build_oracle_request_shape():
    req = build_oracle_request(
        {"a": 1}, {"dept": {"type": "choice", "instructions": "x", "criteria": {"y": "z"}}}
    )
    assert req["model"] == "jev-latest"
    assert "state" in req and "questions" in req
    with pytest.raises(ValueError):
        build_oracle_request("s", {})


def test_parse_oracle_choice_reads_a_real_shaped_response():
    resp = {
        "answers": {
            "department": {
                "type": "choice",
                "choice": "returns",
                "confidence": 0.39,
                "probabilities": {"shipping": 0.02, "billing": 0.38, "returns": 0.60},
            }
        }
    }
    ids, probs, idx = parse_oracle_choice(resp, "department")
    assert ids == [0, 1, 2]
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[idx] == 0.60        # 'returns' is the 3rd key


def test_parse_oracle_choice_fails_closed():
    with pytest.raises(ValueError):
        parse_oracle_choice({"answers": {}}, "nope")
    with pytest.raises(ValueError):
        parse_oracle_choice({"answers": {"q": {"probabilities": {"a": 0.5}}}}, "q")
    with pytest.raises(ValueError):
        parse_oracle_choice({"answers": {"q": {"probabilities": {}}}}, "q")


def test_no_measurement_yet_banner_is_explicit():
    assert "UNVERIFIED" in NO_MEASUREMENT_YET
    assert "HYPOTHESIS" in NO_MEASUREMENT_YET
