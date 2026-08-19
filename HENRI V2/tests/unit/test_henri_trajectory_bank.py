# -*- coding: utf-8 -*-
"""Phase 8.32 tests — authorized trajectory bank.

Software verification only; never capability evidence.
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch

from henri_trajectory_bank import (
    TrajectoryBank,
    TrajectoryBankError,
    bank_enabled_from_env,
    filter_onehot_to_vocab,
)

D = 4096  # reduced dim for tests; prod staging is float16 at D=65536


@pytest.fixture()
def bank_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _bank(bank_dir, run_id="test-run"):
    return TrajectoryBank(bank_dir, run_id=run_id,
                          provenance="unit-test-authorized")


def test_bank_enabled_only_exact_one():
    assert bank_enabled_from_env({"HENRI_ARC_TRAJECTORY_BANK": "1"}) is True
    assert bank_enabled_from_env({}) is False
    assert bank_enabled_from_env({"HENRI_ARC_TRAJECTORY_BANK": "true"}) is False
    assert bank_enabled_from_env({"HENRI_ARC_TRAJECTORY_BANK": "0"}) is False


def test_record_flush_roundtrip_and_digest(bank_dir):
    b = _bank(bank_dir)
    g = torch.Generator().manual_seed(7)
    for i in range(8):
        w = torch.randn(D, generator=g)
        w = w / w.norm()
        b.record(w, "ACTION1" if i % 2 == 0 else "ACTION2",
                 meta={"env": "sp80", "step": i})
    r = b.flush()
    assert r["records"] == 8
    assert os.path.isfile(r["npz_path"])
    assert os.path.isfile(r["jsonl_path"])
    assert os.path.isfile(r["manifest_path"])

    data = TrajectoryBank.load(r["npz_path"], r["manifest_path"],
                               verify_digest=True)
    assert data["psi"].shape == (8, D)
    # Bank emits onehot over the FULL default vocab (6 actions); calibration
    # subsets via filter_onehot_to_vocab.
    assert data["actions_onehot"].shape == (8, 6)
    assert data["action_vocab"] == ["ACTION1", "ACTION2", "ACTION3",
                                    "ACTION4", "ACTION5", "ACTION6"]
    assert data["manifest"]["data_source"] == "authorized"
    assert data["manifest"]["dataset_digest"] == r["dataset_digest"]


def test_digest_mismatch_raises(bank_dir):
    b = _bank(bank_dir)
    w = torch.randn(D) / D ** 0.5
    b.record(w, "ACTION1", meta={"env": "cn04", "step": 0})
    r = b.flush()
    z = np.load(r["npz_path"])
    z2 = z["psi"].copy()
    z2[0, 0] += 1.0
    np.savez(r["npz_path"], psi=z2, next_wave=z["next_wave"],
             actions_onehot=z["actions_onehot"],
             action_names=z["action_names"])
    with pytest.raises(TrajectoryBankError):
        TrajectoryBank.load(r["npz_path"], r["manifest_path"],
                            verify_digest=True)


def test_float16_staging_memory_safe():
    # At D=65536 each wave is 128 KB in float16; bank must keep that dtype.
    b = TrajectoryBank(tempfile.gettempdir(), run_id="mem-probe")
    w = torch.randn(65536, dtype=torch.float32)
    w = w / w.norm()
    b.record(w, "ACTION6", meta={"env": "tn36", "step": 0})
    stored = b._waves[0]
    assert stored.dtype == np.float16
    assert stored.nbytes == 65536 * 2


def test_filter_onehot_to_vocab_subset(bank_dir):
    b = _bank(bank_dir)
    for i in range(10):
        w = torch.randn(D)
        b.record(w, f"ACTION{1 + (i % 6)}", meta={"env": "x", "step": i})
    r = b.flush()
    data = TrajectoryBank.load(r["npz_path"], r["manifest_path"])
    sub, kept = filter_onehot_to_vocab(
        data["actions_onehot"], data["action_vocab"],
        ["ACTION2", "ACTION5"])
    assert sub.shape[1] == 2
    # Records cycle ACTION1..ACTION6; only those with action 2 or 5 survive.
    # i -> action: 1,2,3,4,5,6,1,2,3,4  =>  action2 at i=1,7; action5 at i=4.
    assert kept.sum() == 3
    assert bool(np.all(sub.sum(axis=1) == 1))  # each kept row exactly one action


def test_empty_bank_flush_raises(bank_dir):
    b = _bank(bank_dir)
    with pytest.raises(TrajectoryBankError):
        b.flush()


def test_max_records_truncates(bank_dir):
    b = TrajectoryBank(bank_dir, run_id="cap", max_records=3)
    for i in range(5):
        w = torch.randn(D)
        b.record(w, "ACTION1", meta={"env": "e", "step": i})
    r = b.flush()
    assert r["records"] == 3
    assert b._truncated is True
