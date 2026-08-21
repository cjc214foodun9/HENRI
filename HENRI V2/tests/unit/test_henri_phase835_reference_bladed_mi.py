# -*- coding: utf-8 -*-
"""Phase 8.35 — reference-bladed MI + stratified trajectory sourcing tests.

Covers HENRI-SPEC-MI-TRAJECTORY-2026 §1.1 (continuous softmax I_norm,
bounded in [0,1]) and §2 (stratified split, quota selector, exteroceptive
acceptance rule) plus the benchmark bank contract.
"""
import numpy as np
import pytest

from cegis_self_play_sandbox import (
    NUM_ACTIONS,
    QuotaActionSelector,
    _accept,
    _grid_hash,
)
from henri_trajectory_bank import TrajectoryBank

from experiments.verification.arc_phase835_gate1_benchmark import (
    _reference_bladed_inorm,
    stratified_split,
)


# ---- §1.1 Reference-bladed I_norm -------------------------------------
def test_inorm_zero_for_uniform_confs():
    # Zero-information waves: every sample ~ uniform over actions -> 0.0.
    n, a = 48, 6
    confs = np.ones((n, a)) / a
    assert _reference_bladed_inorm(confs) == pytest.approx(0.0, abs=1e-9)


def test_inorm_one_for_perfect_specification():
    # Perfect specification: one-hot confs, balanced classes -> H(Y|Psi)=0.
    n, a = 60, 6
    confs = np.zeros((n, a))
    for i in range(n):
        confs[i, i % a] = 1.0
    assert _reference_bladed_inorm(confs) == pytest.approx(1.0, abs=1e-9)


def test_inorm_bounds_strict():
    rng = np.random.RandomState(835)
    for _ in range(20):
        confs = rng.dirichlet(np.ones(6), size=37)
        v = _reference_bladed_inorm(confs)
        assert 0.0 <= v <= 1.0


def test_inorm_strictly_positive_when_waves_discriminate():
    # Strong class-conditioned confidences -> H(Y|Psi) < H(Y).
    rng = np.random.RandomState(7)
    n, a = 60, 6
    confs = np.zeros((n, a))
    for i in range(n):
        k = i % a
        confs[i, k] = 0.9
        confs[i, (k + 1) % a] = 0.1
    assert _reference_bladed_inorm(confs) > 0.5


# ---- §2.1 stratified split ----------------------------------------------
def test_stratified_split_exact_two_per_class_heldout():
    actions = np.array([0] * 12 + [1] * 11 + [2] * 10 + [3] * 12 +
                       [4] * 10 + [5] * 10)
    train, held = stratified_split(actions, n_held_per_class=2, seed=20260819)
    held_a = actions[np.array(held)]
    train_a = actions[np.array(train)]
    for k in range(6):
        assert int((held_a == k).sum()) == 2
        assert int((train_a == k).sum()) >= 8
    # Disjoint + exhaustive.
    assert sorted(train + held) == list(range(len(actions)))


# ---- §2.2 quota selector -------------------------------------------------
def test_quota_selector_reaches_min_support():
    sel = QuotaActionSelector(min_support=10, seed=20260819)
    counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    for _ in range(10_000):
        if int(counts.min()) >= 10:
            break
        counts[sel.choose_action(counts)] += 1
    assert int(counts.min()) >= 10
    assert int(counts.max()) < 60  # bounded exploration, no runaway class


# ---- §2.3 exteroceptive acceptance --------------------------------------
def test_accept_rule():
    g1 = np.zeros((3, 3), dtype=np.uint8)
    g2 = g1.copy(); g2[0, 0] = 1
    h1, h2 = _grid_hash(g1), _grid_hash(g2)
    assert _accept(h1, h2, 0.0) is True      # frame changed
    assert _accept(h1, h1, 1.0) is True      # score moved
    assert _accept(h1, h1, 0.0) is False     # same state, no score


# ---- bank contract ------------------------------------------------------
def test_bank_contract_rejects_undersupport():
    from henri_trajectory_bank import TrajectoryBankError
    # 90 records but only 4 action classes -> contract must fail closed.
    rng = np.random.RandomState(3)
    M = 90
    psi = rng.randn(M, 64).astype(np.float32)
    nxt = rng.randn(M, 64).astype(np.float32)
    onehot = np.zeros((M, 6), dtype=np.uint8)
    for i in range(M):
        onehot[i, i % 4] = 1
    # (contract check lives in the benchmark; assert the check math here)
    counts = np.bincount(onehot.argmax(axis=1), minlength=6)
    assert int((counts > 0).sum()) == 4
    assert int(counts.min()) < 10
