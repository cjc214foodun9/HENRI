"""Tests for the temperature-scaling layer (calibration-physics blueprint, Action 2).

These tests assert the PROPERTIES the blueprint's claim depends on, and they are
written to FAIL if the property does not hold. The load-bearing ones:

  * temperature scaling must not change accuracy or Brier skill (invariance);
    without this the "zero retraining" claim is false.
  * T* must be fitted by NLL, and a hand-picked T from the blueprint's beta* range
    must NOT be assumed to satisfy the gate.
  * the held-out path must be wired: fit on train, report holdout.
  * the reachable ECE floor must be reported honestly, not tuned away.
"""
import math
import pathlib
import sys

import pytest

R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
# Plain import (not spec_from_file_location): the module uses
# `from __future__ import annotations`, so dataclasses resolves its string
# annotations through sys.modules and FAILS if the module is exec'd unregistered.
# Importing normally also exercises the production import path.
import henri_probe_calibration as hpc  # noqa: E402


# --------------------------------------------------------------- scaled probs

def test_scaled_probabilities_sums_to_one_and_is_positive():
    p = hpc.scaled_probabilities([0.1, 0.2, -0.3], 0.5)
    assert abs(sum(p) - 1.0) < 1e-12
    assert all(v > 0.0 for v in p)


def test_scaled_probabilities_rejects_nonpositive_temperature():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            hpc.scaled_probabilities([1.0, 2.0], bad)


def test_temperature_one_matches_plain_softmax():
    s = [0.3411397933959961, 0.36621278524398804, 0.32984939217567444, 0.4725440740585327]
    got = hpc.scaled_probabilities(s, 1.0)
    z = [v - max(s) for v in s]
    e = [math.exp(v) for v in z]
    want = [v / sum(e) for v in e]
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, want))


def test_sharpening_moves_mass_onto_the_argmax():
    s = [0.0, 0.1, 0.05, 0.2]
    coarse = hpc.scaled_probabilities(s, 1.0)
    sharp = hpc.scaled_probabilities(s, 0.05)
    assert max(range(4), key=lambda i: sharp[i]) == max(range(4), key=lambda i: coarse[i])
    assert sharp[3] > coarse[3]
    assert max(sharp) > max(coarse)


# ------------------------------------------------------------------ invariance

def test_argmax_is_invariant_across_temperatures():
    """If this fails, temperature scaling would change decisions, which would
    invalidate the 'zero retraining, calibration-only' premise."""
    score_rows = [[0.3, 0.9, 0.4], [1.0, 0.9, 0.95], [0.2, 0.1, 0.15]]
    base = [max(range(3), key=lambda i: r[i]) for r in score_rows]
    for t in (0.01, 0.05, 0.25, 1.0, 4.0):
        got = [max(range(3), key=lambda i: r[i])
               for r in hpc.scale_score_rows(score_rows, t)]
        assert got == base, f"argmax changed at T={t}"


def test_accuracy_is_invariant_but_brier_score_is_not():
    """CORRECTION (measured): accuracy/argmax IS invariant in T; Brier is NOT.

    An earlier version of this test asserted Brier-skill invariance. That was a
    false claim in the module docstring, caught by this test and corrected. The
    test now pins the true behaviour in both directions.
    """
    score_rows = [[0.4, 0.1, 0.2], [0.1, 0.5, 0.2], [0.2, 0.2, 0.6],
                  [0.5, 0.5, 0.0], [0.3, 0.3, 0.35]]
    y = [0, 1, 2, 0, 2]
    p1 = hpc.scale_score_rows(score_rows, 1.0)
    a1 = sum(1 for pr, yy in zip(p1, y) if max(range(3), key=lambda i: pr[i]) == yy)
    b1 = hpc.brier_skill_score(p1, y)
    moved = False
    for t in (0.02, 0.1, 0.5, 2.0):
        pt = hpc.scale_score_rows(score_rows, t)
        at = sum(1 for pr, yy in zip(pt, y) if max(range(3), key=lambda i: pr[i]) == yy)
        assert at == a1, f"accuracy moved at T={t} (must be invariant)"
        if abs(hpc.brier_skill_score(pt, y) - b1) > 1e-9:
            moved = True
    assert moved, (
        "Brier skill did NOT move under temperature scaling. If this ever holds, "
        "the module docstring correction must be revisited."
    )


def test_brier_worsens_when_temperature_is_pushed_the_wrong_way():
    """A correct answer at high confidence must not be flattened into Brier loss."""
    score_rows = [[9.0, 0.0], [8.0, 0.1], [9.5, 0.2], [9.0, 0.3]]
    y = [0, 0, 0, 0]
    b_sharp = hpc.brier_skill_score(hpc.scale_score_rows(score_rows, 0.05), y)
    b_flat = hpc.brier_skill_score(hpc.scale_score_rows(score_rows, 3.0), y)
    assert b_sharp > b_flat


# -------------------------------------------------------------------- fitting

def test_fit_temperature_recovers_a_sharpening_when_model_is_underconfident():
    """Underconfident data: correct answer has the top score but a small margin."""
    score_rows = [[0.20, 0.18], [0.22, 0.19], [0.21, 0.185], [0.23, 0.20]]
    y = [0, 0, 0, 0]
    t, nll = hpc.fit_temperature_nll(score_rows, y, grid=hpc.temperature_grid())
    assert t < 1.0, "an underconfident readout must be sharpened"
    assert nll < hpc.mean_nll(score_rows, y, 1.0)


def test_fit_temperature_recovers_a_flattening_when_model_is_overconfident():
    score_rows = [[9.0, 0.0], [0.0, 8.0], [9.5, 0.1], [0.1, 9.0]]
    y = [0, 1, 0, 0]  # last one is WRONG at high confidence
    t, _ = hpc.fit_temperature_nll(score_rows, y, grid=hpc.temperature_grid())
    assert t > 1.0, "an overconfident readout must be flattened"


def test_fit_temperature_is_deterministic():
    score_rows = [[0.3, 0.2, 0.25], [0.1, 0.4, 0.2]]
    y = [0, 1]
    a = hpc.fit_temperature_nll(score_rows, y, grid=hpc.temperature_grid())
    b = hpc.fit_temperature_nll(score_rows, y, grid=hpc.temperature_grid())
    assert a == b


def test_mean_nll_rejects_a_label_outside_the_row():
    with pytest.raises(ValueError):
        hpc.mean_nll([[0.1, 0.2]], [5], 1.0)


def test_mean_nll_rejects_length_mismatch():
    with pytest.raises(ValueError):
        hpc.mean_nll([[0.1, 0.2]], [0, 1], 1.0)


def test_temperature_grid_is_increasing_and_positive():
    g = hpc.temperature_grid()
    assert len(g) == hpc.TEMPERATURE_GRID_SIZE
    assert all(v > 0.0 for v in g)
    assert all(b > a for a, b in zip(g, g[1:]))


# ---------------------------------------------------------------------- floor

def test_temperature_floor_reports_an_unreachable_gate_honestly():
    """A near-uniform readout cannot reach ECE <= 0.05 by temperature alone when
    its accuracy is low: sharpening drives it to confidence 1.0 while accuracy
    stays low. The floor must therefore exceed 0.05 and be reported as such."""
    score_rows = [[0.0100, 0.0099], [0.0101, 0.0098],
                  [0.0099, 0.0102], [0.0102, 0.0098]]
    y = [1, 1, 0, 0]  # accuracy 0.5, so T->0 gives ECE -> 0.5
    ece, t = hpc.temperature_floor(score_rows, y, n_bins=10,
                                   grid=hpc.temperature_grid())
    assert ece > 0.05, "floor must not be claimed below the gate when it is not"
    assert t > 0.0


def test_temperature_floor_is_reachable_for_a_sharpenable_case():
    """Separable and already-correct data: low T should reach a small ECE."""
    score_rows = [[0.30, 0.10, 0.11, 0.12], [0.11, 0.30, 0.12, 0.10],
                  [0.12, 0.11, 0.30, 0.10], [0.10, 0.12, 0.11, 0.30]] * 3
    y = [0, 1, 2, 3] * 3
    ece, _ = hpc.temperature_floor(score_rows, y, n_bins=10,
                                   grid=hpc.temperature_grid())
    assert ece < 0.05


# --------------------------------------------------------------------- splits

def test_split_report_reports_holdout_and_flags_overfit_gap():
    score_rows = [[0.30, 0.10, 0.11, 0.12], [0.11, 0.30, 0.12, 0.10],
                  [0.12, 0.11, 0.30, 0.10], [0.10, 0.12, 0.11, 0.30],
                  [0.29, 0.11, 0.12, 0.10], [0.10, 0.31, 0.12, 0.11],
                  [0.13, 0.10, 0.30, 0.11], [0.10, 0.13, 0.11, 0.29]]
    y = [0, 1, 2, 3, 0, 1, 2, 3]
    rep = hpc.split_temperature_report(score_rows, y, n_bins=10, split_seed=7)
    assert rep["n_train"] + rep["n_holdout"] == len(score_rows)
    assert "holdout_ece_T_star" in rep and "train_ece_T_star" in rep
    assert rep["accuracy_invariant_holdout"] is True
    assert "bss_improved_holdout" in rep
    assert isinstance(rep["bss_improved_holdout"], bool)
    assert len(rep["holdout_indices"]) == rep["n_holdout"]


def test_split_report_rejects_tiny_input():
    with pytest.raises(ValueError):
        hpc.split_temperature_report([[0.5, 0.5]], [0])


def test_multi_split_aggregate_is_reported_when_requested():
    score_rows = [[0.30, 0.10, 0.11, 0.12], [0.11, 0.30, 0.12, 0.10],
                  [0.12, 0.11, 0.30, 0.10], [0.10, 0.12, 0.11, 0.30],
                  [0.29, 0.11, 0.12, 0.10], [0.10, 0.31, 0.12, 0.11],
                  [0.13, 0.10, 0.30, 0.11], [0.10, 0.13, 0.11, 0.29]]
    y = [0, 1, 2, 3, 0, 1, 2, 3]
    rep = hpc.split_temperature_report(score_rows, y, n_random_splits=5, split_seed=1)
    ms = rep["multi_split"]
    assert ms["n_splits"] == 5
    assert ms["holdout_ece_min"] <= ms["holdout_ece_mean"] <= ms["holdout_ece_max"]


# ------------------------------------------------------------------ blueprint

def test_blueprint_beta_range_is_not_assumed_to_pass_the_gate():
    """The blueprint proposes beta* in 4..8. On underconfident data with a 4-way
    choice and a modest margin, that range does NOT reach ECE <= 0.05. This test
    pins the measured reality: the range must not be treated as verified."""
    score_rows = [[0.20, 0.18, 0.19, 0.17]] * 8
    y = [0] * 8
    for beta in (4.0, 8.0):
        p = hpc.scale_score_rows(score_rows, 1.0 / beta)
        ece = hpc.evaluate_calibration(p, y, n_bins=10).ece
        assert ece > 0.05, (
            f"beta*={beta} unexpectedly reached the gate; the blueprint claim "
            f"would then need re-testing against the real receipt"
        )


def test_mechanics_of_the_fix_are_monotone():
    """The blueprint's REASONING is that sharpening raises stated confidence
    toward accuracy. That mechanism is testable and must hold, even while its
    numeric beta* range is falsified."""
    score_rows = [[0.30, 0.10, 0.11, 0.12]] * 6
    y = [0] * 6
    peaks = [hpc.evaluate_calibration(
        hpc.scale_score_rows(score_rows, t), y, n_bins=10).mean_confidence
        for t in (1.0, 0.5, 0.25, 0.1, 0.05)]
    assert all(b > a for a, b in zip(peaks, peaks[1:])), "sharpening must be monotone"
