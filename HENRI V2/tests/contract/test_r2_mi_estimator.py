"""Contract tests for the R2-successor conditional-MI estimator
(SpecContract SPEC-2026-08-28-R2SUCC, sealed #577b54d3).

Estimator contract (frozen):
- A = action label (str(game_action) at the live selection boundary).
- S = pre-action state stratum id (tuple of bins over (n_nonzero_cells,
  n_distinct_colors, grid_shape) computed from the PRE-action frame only).
- dS = outcome delta vector DISCRETIZED with FIXED semantic bins declared
  below (no data-dependent quantile binning): (changed_cells_bin,
  frame_diff_mean_bin, delta_levels_bin).
- Support gates BEFORE MI (per env): effective N >= 20, per-stratum N_s >= 4,
  per-action count within stratum >= 4. Any violation -> IDENTIFIABILITY_BLOCKED.
- I(A; dS | S) = sum_s P(s) * I_mm(A; dS | S=s), Miller-Madow bias-corrected
  plugin + (|X|-1)(|Y|-1)/(2N_s).
- Controls: episode-cluster bootstrap 95% CI (resample EPISODES, >= 200
  resamples); action-shuffle null <= 0.05; mismatched-state negative control
  ~ 0 (within noise of the shuffled null).
- Aggregate: EVERY env >= 0.15 (pooled value may not conceal an env below).
"""

import numpy as np

from henri_r2_mi_estimator import (
    DISCRETE_BINS,
    bin_delta,
    conditional_mutual_information,
    estimate_conditional_mi_pipeline,
    identify_env_support,
    episode_cluster_bootstrap_ci,
    action_shuffle_null,
    mismatched_state_control,
    IDENTIFIABILITY_BLOCKED,
)

# Fixed semantic bins (frozen in the SpecContract; never data-derived).
# changed_cells:      [0], [1], [2,5), [5,10), [10,inf) -> 5 bins
# frame_diff_mean:    [0], (0,0.5), [0.5,2), [2,inf)   -> 4 bins
# delta_levels:       [0], [1], [2,inf)                -> 3 bins
assert DISCRETE_BINS["changed_cells"] == [0, 1, 2, 5, 10]
assert DISCRETE_BINS["frame_diff_mean"] == [0.0, 0.5, 2.0]
assert DISCRETE_BINS["delta_levels"] == [0, 1, 2]


def _rows(n, rng, episodes=5):
    """Synthetic telemetry rows: (episode_id, S_id, action, changed_cells,
    frame_diff_mean, delta_levels). Dependence is deliberately STRONG (the
    Miller-Madow correction is ~0.05 nats at N_s=200, so the signal must
    exceed it comfortably): (s==1, a==1) forces changed_cells into bin 4
    (value 12) and frame_diff_mean into bin 3 (value 3.0)."""
    rows = []
    for ep in range(episodes):
        for i in range(n // episodes):
            s = int(rng.integers(0, 3))
            a = int(rng.integers(0, 2))
            if s == 1 and a == 1:
                cc = 12
                fdm = 3.0
            else:
                cc = int(rng.integers(0, 5))
                fdm = float(rng.uniform(0.0, 0.4))
            dl = int(rng.integers(0, 2))
            rows.append((f"ep{ep}", s, a, cc, fdm, dl))
    return rows


def test_support_gate_below_thresholds_blocks():
    rng = np.random.default_rng(7)
    rows = _rows(10, rng, episodes=2)  # N=10 < 20
    verdict = identify_env_support(rows, min_n=20, min_stratum=4, min_action=4)
    assert verdict == IDENTIFIABILITY_BLOCKED, verdict


def test_support_gate_missing_action_in_stratum_blocks():
    rng = np.random.default_rng(11)
    # Deterministic construction: stratum 2 contains ONLY action 0 (a missing
    # (s==2, a==1) cell by construction — a filter on random rows can be
    # vacuous when no such row exists by chance).
    rows = []
    for ep in range(5):
        for i in range(8):
            s = i % 3
            a = 0 if s == 2 else int(rng.integers(0, 2))
            cc = int(rng.integers(0, 5))
            fdm = float(rng.uniform(0.0, 0.4))
            dl = int(rng.integers(0, 2))
            rows.append((f"ep{ep}", s, a, cc, fdm, dl))
    verdict = identify_env_support(rows, min_n=20, min_stratum=4, min_action=4)
    assert verdict == IDENTIFIABILITY_BLOCKED, verdict


def test_support_gate_single_action_stratum_blocks():
    # A stratum containing 10 rows of ONE action passes the count gates but
    # carries zero conditional MI by construction -> must BLOCK.
    rows = []
    for ep in range(5):
        for i in range(8):
            s = i % 3
            a = 0 if s == 1 else int(np.random.default_rng(ep * 10 + i).integers(0, 2))
            cc = int(np.random.default_rng(ep * 10 + i + 1).integers(0, 5))
            rows.append((f"ep{ep}", s, a, cc, 0.1, 0))
    verdict = identify_env_support(rows, min_n=20, min_stratum=4, min_action=4)
    assert verdict == IDENTIFIABILITY_BLOCKED, verdict


def test_plugin_mi_is_biased_high_and_mm_corrects_down():
    rng = np.random.default_rng(13)
    rows = _rows(600, rng, episodes=20)
    mi_mm = conditional_mutual_information(rows)
    mi_plugin = conditional_mutual_information(rows, bias_correction=False)
    assert mi_plugin >= mi_mm, (mi_plugin, mi_mm)
    assert mi_mm >= 0.0 - 1e-9


def test_mi_positive_when_s_conditions_action_outcome():
    rng = np.random.default_rng(17)
    rows = _rows(600, rng, episodes=20)
    mi = conditional_mutual_information(rows)
    # Dependent construction: s=1 & a=1 systematically changes dS.
    assert mi > 0.01, mi


def test_mi_near_zero_when_action_irrelevant():
    rng = np.random.default_rng(19)
    rows = []
    for ep in range(20):
        for i in range(30):
            s = int(rng.integers(0, 3))
            a = int(rng.integers(0, 2))  # irrelevant
            cc = int(rng.integers(0, 6))
            fdm = float(rng.uniform(0.0, 2.5))
            dl = int(rng.integers(0, 2))
            rows.append((f"ep{ep}", s, a, cc, fdm, dl))
    mi = conditional_mutual_information(rows)
    # The Miller-Madow correction is asymptotic, not a calibrated null. The
    # contract's null is the ACTION-SHUFFLE null: the point estimate must lie
    # within noise of the shuffled distribution, and both must sit under the
    # pre-registered 0.05 gate.
    null = action_shuffle_null(rows, n_shuffles=50, seed=3)
    assert mi <= null + 0.01, (mi, null)
    assert mi <= 0.05, mi


def test_bin_delta_fixed_edges():
    assert bin_delta({"changed_cells": 0, "frame_diff_mean": 0.0, "delta_levels": 0}) == (0, 0, 0)
    assert bin_delta({"changed_cells": 3, "frame_diff_mean": 0.3, "delta_levels": 1}) == (2, 1, 1)
    assert bin_delta({"changed_cells": 17, "frame_diff_mean": 5.0, "delta_levels": 3}) == (4, 3, 2)


def test_episode_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(23)
    rows = _rows(600, rng, episodes=20)
    mi = conditional_mutual_information(rows)
    lo, hi = episode_cluster_bootstrap_ci(rows, n_resamples=200, seed=5)
    assert lo <= mi + 1e-9 <= hi, (lo, mi, hi)


def test_action_shuffle_null_below_005():
    rng = np.random.default_rng(29)
    rows = _rows(600, rng, episodes=20)
    null = action_shuffle_null(rows, n_shuffles=50, seed=3)
    assert null <= 0.05, null


def test_mismatched_state_control_near_zero():
    rng = np.random.default_rng(31)
    rows = _rows(600, rng, episodes=20)
    ctrl = mismatched_state_control(rows, seed=9)
    assert ctrl <= 0.05, ctrl


def test_pipeline_returns_full_receipt():
    rng = np.random.default_rng(37)
    rows = _rows(600, rng, episodes=20)
    receipt = estimate_conditional_mi_pipeline(rows, seed=1)
    assert receipt["status"] == "MEASURED"
    assert receipt["mi_nats"] > 0.0
    assert receipt["bootstrap_ci"][0] <= receipt["mi_nats"] <= receipt["bootstrap_ci"][1]
    assert receipt["shuffle_null"] <= 0.05
    assert receipt["mismatch_control"] <= 0.05
    assert receipt["n_rows"] == 600
