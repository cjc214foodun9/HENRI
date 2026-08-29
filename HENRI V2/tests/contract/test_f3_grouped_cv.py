"""F3 grouped-CV contract tests (SPEC-2026-08-29-F3-BROAD-BANK §8).

RED-first contract suite for the F3 capture/seal/gates pipeline:
  - fold disjointness: every env in exactly one fold; train ∩ heldout = ∅
  - entropy math: uniform 7-class -> ln 7 nats; single class -> 0
  - per-action accounting: one-hot column sums
  - non-adaptive selection: fold = lexicographic env index mod 4
  - no-tune guard: gates harness has no beta/ridge CLI; constants frozen
  - CV_diff math on a known vector
Runs on numpy + source inspection only (no torch, no GPU).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIF = REPO_ROOT / "HENRI V2" / "experiments" / "verification"


def _load(name: str) -> object:
    path = VERIF / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sealer = _load("f3_split_seal")
finalizer = _load("f3_capture_finalize")


# --- fold disjointness -----------------------------------------------------
def test_fold_assignment_each_env_one_fold():
    envs = [f"e{i:02d}" for i in range(12)]
    assign = sealer.fold_assignment(envs, 4)
    assert sorted(assign.values()) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert len(set(assign.values())) == 4
    assert len(set(assign)) == 12  # every env mapped exactly once


def test_fold_assignment_lexicographic_mod():
    envs = ["wa30", "sb26", "cd82", "ka59"]
    assign = sealer.fold_assignment(envs, 4)
    # sorted: cd82(0) ka59(1) sb26(2) wa30(3)
    assert assign["cd82"] == 0
    assert assign["ka59"] == 1
    assert assign["sb26"] == 2
    assert assign["wa30"] == 3


def test_folds_train_heldout_disjoint():
    envs = [f"e{i:02d}" for i in range(12)]
    assign = sealer.fold_assignment(envs, 4)
    for f in range(4):
        held = {e for e in envs if assign[e] == f}
        train = {e for e in envs if assign[e] != f}
        assert held & train == set()
        assert held | train == set(envs)
        assert len(held) == 3


# --- entropy math -----------------------------------------------------------
def test_entropy_uniform_7():
    h = finalizer.compute_entropy_nats([100] * 7)
    assert abs(h - math.log(7)) < 1e-9


def test_entropy_single_class_zero():
    assert finalizer.compute_entropy_nats([500]) == 0.0
    assert finalizer.compute_entropy_nats([]) == 0.0


def test_entropy_half_half():
    h = finalizer.compute_entropy_nats([3, 3])
    assert abs(h - math.log(2)) < 1e-9


# --- per-action accounting ---------------------------------------------------
def test_per_action_counts():
    onehot = np.zeros((10, 7), dtype=np.uint8)
    onehot[:4, 0] = 1
    onehot[4:7, 2] = 1
    onehot[7:, 6] = 1
    counts = finalizer.per_action_counts(onehot)
    assert counts["ACTION1"] == 4
    assert counts["ACTION3"] == 3
    assert counts["ACTION7"] == 3
    assert counts["ACTION2"] == 0


# --- CV_diff math ------------------------------------------------------------
def test_cv_diff_known():
    means = {"a": 1.0, "b": 2.0, "c": 3.0}
    cv = finalizer.cv_diff(means)
    mu = 2.0
    std = float(np.std([1.0, 2.0, 3.0]))
    assert abs(cv - std / mu) < 1e-9
    assert finalizer.cv_diff({}) == 0.0
    assert finalizer.cv_diff({"a": 1.0}) == 0.0


# --- non-adaptive + record floor validation ----------------------------------
def test_per_env_counts_and_floor_rule():
    meta = [{"env": "a"}, {"env": "a"}, {"env": "b"}]
    counts = finalizer.per_env_counts(meta)
    assert counts == {"a": 2, "b": 1}
    assert any(c < 100 for c in counts.values())  # floor rule would fire


# --- no-tune guard (K6): frozen hyperparameters, no CLI overrides -------------
def test_gates_harness_frozen_hyperparams():
    src = (VERIF / "f3_egress_gates.py").read_text(encoding="utf-8")
    assert "FROZEN_BETA = 8.0" in src
    assert "FROZEN_RIDGE = 1e-3" in src
    assert '--beta' not in src and '--ridge' not in src
    assert "--ridge" not in src
    assert "add_argument" in src  # sanity: we are parsing the real harness


def test_gates_harness_pins_frozen_n_folds():
    src = (VERIF / "f3_egress_gates.py").read_text(encoding="utf-8")
    assert "FROZEN_N_FOLDS = 4" in src
