# -*- coding: utf-8 -*-
"""Phase 8.31 unit tests — algebraic (no-BPTT) semantic action head.

Software verification ONLY on synthetic fixtures. No ARC capability claim.
"""
import json
import os
import tempfile

import pytest
import torch

from algebraic_action_head import (
    AlgebraicActionHeadCalibrator,
    AlgebraicActionHeadError,
    load_algebraic_head_artifact,
    algebraic_head_eligible,
    SCHEMA_ID,
)


def _engrams(d=512, a=6, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(a, d, generator=g, device=device)
    return e / e.norm(dim=1, keepdim=True)


def _waves_like(engrams, n=40, seed=1, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, engrams.shape[0], (n,), generator=g)
    noise = 0.01 * torch.randn(n, engrams.shape[1], generator=g, device=device)
    w = engrams[idx] + noise
    return w / w.norm(dim=1, keepdim=True), idx


@pytest.fixture()
def cal():
    return AlgebraicActionHeadCalibrator(d_model=512, r_rank=64, seed=0)


def test_default_off_noop(cal):
    # No-op until calibrate/store called; no production wiring exists.
    assert cal._engrams is None and cal._w_task is None


def test_compile_operator_shape_and_orthogonality(cal):
    e = _engrams(cal.d_model, 6)
    x = e[:4]
    y = e[:4]
    w = cal.compile_task_operator(x, y, basis_digest="b1", calibration_digest="c1")
    # effective r = min(64, 4, 512) = 4
    assert w.shape == (cal.d_model, 4)
    err = torch.linalg.matrix_norm(w.T @ w - torch.eye(4, device=w.device))
    assert err.item() < 1e-5  # float32 SVD orthogonality (measured 1.03e-06)


def test_transduce_factorized_matches_dense(cal):
    # Algebraic check on small dims: factored application == dense W @ x.
    d = 128
    cal2 = AlgebraicActionHeadCalibrator(d_model=d, r_rank=32, seed=0)
    e = _engrams(d, 6, seed=5)
    cal2.compile_task_operator(e, e, basis_digest="b", calibration_digest="c")
    # dense low-rank reconstruction: W = V_r S_r^-1 P
    dense = cal2._w_task @ (cal2._inv_scale[:, None] * cal2._proj)  # [D, r] @ [r, D]
    x = _engrams(d, 1, seed=7)[0]
    out_fact = cal2.transduce(x)
    out_dense = dense @ x
    assert torch.allclose(out_fact, out_dense, atol=1e-4)


def test_transduce_shape_and_device(cal):
    e = _engrams(cal.d_model, 6)
    cal.compile_task_operator(e, e, basis_digest="b", calibration_digest="c")
    w = _engrams(cal.d_model, 1, seed=2)[0]
    out = cal.transduce(w)
    assert out.shape == w.shape
    assert out.device == w.device


def test_store_and_snap(cal):
    e = _engrams(cal.d_model, 6)
    cal.store_action_engrams(e, ["ACTION%d" % (i + 1) for i in range(6)])
    w, true_idx = _waves_like(e, n=20, seed=3)
    idx, sim, margin = cal.snap(w[0])
    assert idx == int(true_idx[0])
    assert sim > 0.95


def test_snap_margin_positive_on_clean(cal):
    e = _engrams(cal.d_model, 6)
    cal.store_action_engrams(e, ["A%d" % i for i in range(6)])
    w, _ = _waves_like(e, n=1, seed=4)
    _, _, margin = cal.snap(w[0])
    assert margin >= 0.0


def test_snap_requires_engrams(cal):
    with pytest.raises(AlgebraicActionHeadError):
        cal.snap(torch.zeros(cal.d_model))


def test_artifact_roundtrip(cal, tmp_path):
    e = _engrams(cal.d_model, 6)
    cal.store_action_engrams(e, ["ACTION%d" % (i + 1) for i in range(6)])
    cal.compile_task_operator(e, e, basis_digest="b1", calibration_digest="c1")
    p = str(tmp_path / "head.json")
    art = cal.save_artifact(
        p, split_identity="synthetic-fixture",
        held_out_metrics={"true_rank": 1, "margin": 0.12, "accuracy": 0.9, "n_heldout": 40})
    loaded = load_algebraic_head_artifact(p, expected_d_model=512, expected_actions=6)
    assert loaded.schema_id == SCHEMA_ID
    assert loaded.action_names == art.action_names
    assert loaded.action_engrams_sha256 == art.action_engrams_sha256


def test_artifact_wrong_schema(cal, tmp_path):
    p = str(tmp_path / "bad.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"schema_id": "other.v9"}, f)
    with pytest.raises(AlgebraicActionHeadError):
        load_algebraic_head_artifact(p)


def test_artifact_tamper_detected(cal, tmp_path):
    e = _engrams(cal.d_model, 6)
    cal.store_action_engrams(e, ["A%d" % i for i in range(6)])
    cal.compile_task_operator(e, e, basis_digest="b1", calibration_digest="c1")
    p = str(tmp_path / "head.json")
    cal.save_artifact(p)
    raw = json.load(open(p, encoding="utf-8"))
    raw["d_model"] = 513  # tamper
    with open(p, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    with pytest.raises(AlgebraicActionHeadError):
        load_algebraic_head_artifact(p)


def test_artifact_missing_file(cal, tmp_path):
    with pytest.raises(AlgebraicActionHeadError):
        load_algebraic_head_artifact(str(tmp_path / "nope.json"))


def test_eligibility_gate():
    from algebraic_action_head import AlgebraicHeadArtifact
    ok = AlgebraicHeadArtifact(
        action_names=["A1"], held_out_metrics={
            "true_rank": 1, "margin": 0.10, "accuracy": 0.8, "n_heldout": 40})
    ok.finalize()
    elig, reason = algebraic_head_eligible(ok)
    assert elig and reason == ""

    weak = AlgebraicHeadArtifact(
        action_names=["A1"], held_out_metrics={
            "true_rank": 5, "margin": 0.01, "accuracy": 0.2, "n_heldout": 40})
    weak.finalize()
    elig, reason = algebraic_head_eligible(weak)
    assert not elig and reason == "ACTION_HEAD_HELD_OUT_GATES_FAIL"

    empty = AlgebraicHeadArtifact(action_names=["A1"])
    empty.finalize()
    elig, reason = algebraic_head_eligible(empty)
    assert not elig and reason == "ACTION_HEAD_NOT_CALIBRATED"


def test_no_dense_d_matrix_alloc(cal):
    # Compile path must never allocate [D, D]; effective r = min(16, 4, 4096).
    cal2 = AlgebraicActionHeadCalibrator(d_model=4096, r_rank=16, seed=0)
    e = _engrams(cal2.d_model, 4, seed=9)
    w = cal2.compile_task_operator(e, e, basis_digest="b", calibration_digest="c")
    assert w.shape == (4096, 4)
    assert cal2._proj.shape == (4, 4096)  # [r, D] projection, never [D, D]


def test_production_dim_factor_shapes():
    # D=65,536 production dimension: factor shapes only, no [D,D].
    d = 65536
    cal2 = AlgebraicActionHeadCalibrator(d_model=d, r_rank=128, seed=0)
    g = torch.Generator().manual_seed(11)
    e = torch.randn(4, d, generator=g)
    e = e / e.norm(dim=1, keepdim=True)
    w = cal2.compile_task_operator(e, e, basis_digest="b", calibration_digest="c")
    assert w.shape == (d, 4)          # effective r = min(128, 4, 65536)
    assert cal2._proj.shape == (4, d)  # [r, D]
    assert cal2._inv_scale.shape == (4,)
    # transduce on production shape [65536] -> same shape, correct dtype
    out = cal2.transduce(e[0])
    assert out.shape == (d,)
    assert out.dtype == e.dtype


def test_import_no_production_side_effect():
    # Default-OFF: importing the calibrator must not load/patch production
    # modules (no wiring by import).
    import sys
    before = {m for m in sys.modules if "production_arc_run" in m or "henri_decoder" in m}
    import algebraic_action_head  # noqa: F401
    after = {m for m in sys.modules if "production_arc_run" in m or "henri_decoder" in m}
    assert after == before, f"import pulled production modules: {after - before}"


def test_decoder_on_head_off_remains_ineligible():
    # Contract G6: generic decoder LOADED + algebraic head OFF -> still
    # ineligible. Mirrors arc_score_gate dominance rule (Phase 7.4).
    from arc_score_gate import arc_score_eligibility
    res = arc_score_eligibility(
        learned_component_on_action_path=True,
        checkpoint_policy="required",
        checkpoint_load_status="LOADED",
        trained_decoder_active=True,
        checkpoint_sha256="c" * 64,
        state_dict_sha256="d" * 64,
        trained_action_head_active=False,
    )
    assert res["score_eligible"] is False
    assert "ACTION_HEAD_NOT_CALIBRATED" in res["score_block_reason"]
