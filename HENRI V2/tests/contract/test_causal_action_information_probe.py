"""Contract tests for the causal action-information probe (pure math, CPU-only).

Tests the pre-registered gates on synthetic data:
  - binned MI estimator: dependent data > independent data; recovery of known MI;
  - permutation null: independent data falls inside the null;
  - eta_C variance ratio: action-independent outcomes -> ~0; action-conditioned
    outcomes -> above the floor;
  - verdict logic: all pre-registered combinations;
  - frame_delta_nu / frame_signature / change_signature determinism.
"""

import numpy as np
import pytest

from causal_action_information_probe import (  # noqa: E402  (module on PYTHONPATH="HENRI V2")
    DELTA_NU_BIN_EDGES,
    ETA_C_FLOOR,
    MI_FLOOR_NATS,
    VERDICT_COMPLETE,
    VERDICT_INSUFFICIENT,
    VERDICT_NO_TRANSITIONS,
    VERDICT_SETUP_BLOCKED,
    bin_delta,
    causal_variance_ratio,
    change_signature,
    frame_delta_nu,
    frame_signature,
    mutual_information,
    permutation_null,
    verdict_from_stats,
)


def test_bin_delta_maps_edges():
    assert bin_delta(0) == 0
    assert bin_delta(1) == 1
    assert bin_delta(2) == 2
    assert bin_delta(5) == 5
    assert bin_delta(10_000) == len(DELTA_NU_BIN_EDGES) - 2


def test_frame_delta_nu_counts_cells():
    before = [[0, 0], [0, 0]]
    after = [[0, 1], [0, 2]]
    assert frame_delta_nu(before, after) == 2
    assert frame_delta_nu(before, before) == 0
    assert frame_delta_nu([[0]], [[0, 0]]) == 0  # shape mismatch -> 0


def test_frame_signature_deterministic_and_distinct():
    f1 = [[0, 1], [2, 3]]
    f2 = [[0, 1], [2, 4]]
    assert frame_signature(f1) == frame_signature(f1)
    assert frame_signature(f1) != frame_signature(f2)


def test_change_signature_orders():
    b = [[0, 0], [0, 0]]
    a1 = [[1, 0], [0, 0]]
    a2 = [[0, 0], [0, 1]]
    assert change_signature(b, a1) != change_signature(b, a2)
    assert change_signature(b, b) == change_signature(b, b)


def test_mi_dependent_beats_independent():
    rng = np.random.default_rng(0)
    n = 400
    ind_actions = rng.integers(0, 4, size=n)
    ind_deltas = rng.integers(0, 6, size=n)
    dep_actions = rng.integers(0, 4, size=n)
    dep_deltas = (dep_actions + rng.integers(0, 2, size=n)) % 6
    mi_ind = mutual_information(ind_actions, ind_deltas, 4, 6)
    mi_dep = mutual_information(dep_actions, dep_deltas, 4, 6)
    assert mi_dep > mi_ind
    # independent MI should be small (near 0)
    assert mi_ind < 0.05


def test_mi_known_value_deterministic_map():
    # perfect dependence: delta = action -> MI near ln(4) nats (uniform marginals);
    # with n=2000 the Laplacian smoothing (alpha=0.5 over a 4x6 table) dilutes
    # the estimate by ~0.03 nats, so a 0.08 band around the theoretical maximum
    # is both tight and robust.
    n = 2000
    actions = np.tile(np.arange(4), n // 4)
    deltas = actions.copy()
    mi = mutual_information(actions, deltas, 4, 6)
    assert mi == pytest.approx(np.log(4.0), abs=0.08)


def test_permutation_null_independent_contains_observed():
    rng = np.random.default_rng(7)
    n = 300
    actions = rng.integers(0, 4, size=n)
    deltas = rng.integers(0, 6, size=n)
    mi = mutual_information(actions, deltas, 4, 6)
    mu, sd = permutation_null(actions, deltas, 4, 6, n_perm=100, seed=3)
    assert mi <= mu + 3.0 * sd  # inside null


def test_eta_c_zero_when_env_static():
    # Static environment: every action produces zero delta -> both variances 0
    # -> eta_C = 0 < floor (correct inert classification). NOTE: i.i.d. noise
    # yields eta_C ~ 1 (both variances estimate the same population variance);
    # that case is caught by the MI gate, not the eta_C gate.
    rounds = np.repeat(np.arange(50), 4)
    actions = np.tile(np.arange(4), 50)
    deltas = np.zeros(200, dtype=np.float64)
    eta, a_var, e_var = causal_variance_ratio(rounds, actions, deltas, 4)
    assert eta < ETA_C_FLOOR  # action-independent outcomes (static env)


def test_eta_c_above_floor_when_actions_matter():
    rounds = np.repeat(np.arange(50), 4)
    actions = np.tile(np.arange(4), 50)
    # action 0 -> delta ~ 0, action 1 -> delta ~ 10, etc. (deterministic effect)
    deltas = np.asarray([0.0, 10.0, 20.0, 30.0] * 50, dtype=np.float64)
    eta, a_var, e_var = causal_variance_ratio(rounds, actions, deltas, 4)
    assert eta > ETA_C_FLOOR


def test_verdict_gates():
    # complete
    v, r = verdict_from_stats(0.2, 0.01, 0.005, 0.9, 100, 0, 100, 0, 200)
    assert v == VERDICT_COMPLETE
    # MI below floor
    v, r = verdict_from_stats(0.03, 0.01, 0.005, 0.9, 100, 0, 100, 0, 200)
    assert v == VERDICT_INSUFFICIENT
    # eta_C below floor
    v, r = verdict_from_stats(0.2, 0.01, 0.005, 0.1, 100, 0, 100, 0, 200)
    assert v == VERDICT_INSUFFICIENT
    # permutation not significant
    v, r = verdict_from_stats(0.06, 0.05, 0.01, 0.9, 100, 0, 100, 0, 200)
    assert v == VERDICT_INSUFFICIENT
    # zero transitions
    v, r = verdict_from_stats(0.2, 0.01, 0.005, 0.9, 0, 0, 0, 0, 0)
    assert v == VERDICT_NO_TRANSITIONS
    # setup blocked: high env error rate
    v, r = verdict_from_stats(0.2, 0.01, 0.005, 0.9, 100, 30, 100, 0, 200)
    assert v == VERDICT_SETUP_BLOCKED
    # setup blocked: high reset failure rate
    v, r = verdict_from_stats(0.2, 0.01, 0.005, 0.9, 100, 0, 100, 50, 200)
    assert v == VERDICT_SETUP_BLOCKED
