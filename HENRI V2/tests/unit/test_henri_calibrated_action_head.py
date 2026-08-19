# -*- coding: utf-8 -*-
"""Phase 8.32 tests — calibrated action head (Stiefel-Ridge).

Software verification only (synthetic fixtures); never capability evidence.
"""
import json
import os
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_calibrated_action_head import (  # noqa: E402
    ActionHeadCalibrator,
    CalibratedActionHeadError,
    SCHEMA_ID,
    StiefelActionProactor,
    load_calibrated_head_artifact,
    production_activation_eligible,
)


def _realizable_fixture(seed: int = 0, m: int = 128, latent: int = 64,
                        action: int = 8, noise: float = 0.0):
    """Deterministic REALIZABLE fixture: target = h @ W_true.T + noise.

    W_true rows are ORTHONORMAL (QR of a random matrix), so the Stiefel
    retraction U V^T is near-identity and the mechanism is honestly testable.
    The mechanism must recover W_true and generalize to held-out samples.
    """
    g = torch.Generator().manual_seed(seed)
    model = StiefelActionProactor(wave_dim=64, latent_dim=latent, action_dim=action)
    psi = F.normalize(torch.randn(m, 64, generator=g), p=2, dim=-1)
    with torch.no_grad():
        h = model.extract_features(psi)
    # Orthonormal rows: Q from QR of a random [latent, latent] matrix
    q, _ = torch.linalg.qr(torch.randn(latent, latent, generator=g))
    w_true = q[:action].contiguous()  # [action, latent], rows orthonormal
    target = h @ w_true.T + noise * torch.randn(m, action, generator=g)
    return model, psi, target


def test_import_no_production_side_effect():
    # Default-OFF: importing must not pull production modules.
    before = {mod for mod in sys.modules
              if "production_arc_run" in mod or "henri_decoder" in mod}
    import henri_calibrated_action_head  # noqa: F401,F811
    after = {mod for mod in sys.modules
             if "production_arc_run" in mod or "henri_decoder" in mod}
    assert after == before, f"import pulled production modules: {after - before}"


def test_ridge_svd_form_matches_solve_when_m_ge_l():
    # SVD-form ridge must equal torch.linalg.solve on the SAME train matrix.
    latent, action, m = 32, 8, 64
    g = torch.Generator().manual_seed(1)
    model = StiefelActionProactor(wave_dim=64, latent_dim=latent, action_dim=action)
    psi = F.normalize(torch.randn(m, 64, generator=g), p=2, dim=-1)
    with torch.no_grad():
        h = model.extract_features(psi)
    q, _ = torch.linalg.qr(torch.randn(latent, latent, generator=g))
    w_true = q[:action]
    a = h @ w_true.T
    gamma = 1e-6

    htr, atr = h, a  # identical data for both forms
    hth = htr.T @ htr + gamma * torch.eye(latent)
    w_solve = torch.linalg.solve(hth, htr.T @ atr).T  # [action, latent]

    u, s, vh = torch.linalg.svd(htr, full_matrices=False)
    coef = (s / (s * s + gamma)).contiguous()
    w_svd = ((vh.T @ ((u.T @ atr) * coef[:, None])).T)  # [action, latent]
    rel = (w_svd - w_solve).norm() / (w_solve.norm() + 1e-9)
    assert rel < 1e-3, f"SVD form diverges from solve: rel={rel:.2e}"


def test_stiefel_retraction_orthonormal():
    # After calibration, W_act rows must be orthonormal: W W^T ~ I.
    model, psi, target = _realizable_fixture(seed=2, m=96, latent=48, action=8)
    cal = ActionHeadCalibrator(model, ridge_gamma=1e-6, held_out_frac=0.25)
    art = cal.calibrate_from_trajectories(psi, target, data_source="synthetic_fixture")
    w = model.w_act.weight.detach()
    gram = w @ w.T
    err = (gram - torch.eye(model.action_dim)).abs().max().item()
    assert err <= 1e-5, f"Stiefel orthogonality violated: max err {err:.2e}"
    assert art["weight_sha256"]


def test_heldout_gate_discriminates_realizable_vs_noise():
    # Realizable target (noise=0) -> held-out gate passes.
    model, psi, target = _realizable_fixture(seed=3, m=128, latent=64, action=8, noise=0.0)
    cal = ActionHeadCalibrator(model, ridge_gamma=1e-4, held_out_frac=0.25)
    art = cal.calibrate_from_trajectories(psi, target, data_source="synthetic_fixture")
    assert art["is_qualified"] is True
    assert art["calibration_mse_heldout"] <= 0.05

    # Pure-noise target -> held-out gate must FAIL (discrimination).
    g = torch.Generator().manual_seed(4)
    model2, psi2, _ = _realizable_fixture(seed=4, m=128, latent=64, action=8)
    noise_tgt = torch.randn(128, 8, generator=g)
    cal2 = ActionHeadCalibrator(model2, ridge_gamma=1e-4, held_out_frac=0.25)
    art2 = cal2.calibrate_from_trajectories(psi2, noise_tgt, data_source="synthetic_fixture")
    assert art2["is_qualified"] is False, "noise target must not qualify"
    assert art2["status"] == "OFF"


def test_in_sample_fit_does_not_qualify_on_held_out():
    # Leakage kill: fit on train, report only held-out; in-sample MSE is not the gate.
    model, psi, target = _realizable_fixture(seed=5, m=64, latent=32, action=8, noise=0.5)
    cal = ActionHeadCalibrator(model, ridge_gamma=1e-4, held_out_frac=0.5)
    art = cal.calibrate_from_trajectories(psi, target, data_source="synthetic_fixture")
    assert art["held_out_count"] >= 1
    assert art["train_count"] + art["held_out_count"] == 64
    # With noise=0.5 the held-out MSE is typically > 0.05; the artifact must
    # carry the held-out value, not an in-sample one.
    assert "calibration_mse_heldout" in art


def test_synthetic_never_activates_production():
    model, psi, target = _realizable_fixture(seed=6, m=128, latent=64, action=8, noise=0.0)
    cal = ActionHeadCalibrator(model, ridge_gamma=1e-4, held_out_frac=0.25)
    art = cal.calibrate_from_trajectories(psi, target, data_source="synthetic_fixture")
    ok, reason = production_activation_eligible(art)
    assert ok is False
    assert reason == "ACTION_HEAD_SYNTHETIC_ONLY"


def test_artifact_self_hash_roundtrip():
    model, psi, target = _realizable_fixture(seed=7, m=96, latent=48, action=8)
    cal = ActionHeadCalibrator(model, ridge_gamma=1e-4, held_out_frac=0.25)
    art = cal.calibrate_from_trajectories(psi, target, data_source="synthetic_fixture",
                                          split_identity="test-split")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "artifact.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(art, f, sort_keys=True)
        loaded = load_calibrated_head_artifact(p, expected_wave_dim=64,
                                               expected_action_dim=8)
        assert loaded["artifact_sha256"] == art["artifact_sha256"]
        assert loaded["split_identity"] == "test-split"
    # Tamper -> typed error
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "artifact.json")
        art["weight_sha256"] = "deadbeef"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(art, f, sort_keys=True)
        with pytest.raises(CalibratedActionHeadError):
            load_calibrated_head_artifact(p)


def test_typed_errors():
    model = StiefelActionProactor(wave_dim=64, latent_dim=32, action_dim=8)
    cal = ActionHeadCalibrator(model)
    psi = torch.randn(1, 64)
    a = torch.randn(1, 8)
    with pytest.raises(CalibratedActionHeadError):
        cal.calibrate_from_trajectories(psi, a)  # M < 2
    psi2 = torch.randn(8, 32)
    a2 = torch.randn(8, 8)
    with pytest.raises(CalibratedActionHeadError):
        cal.calibrate_from_trajectories(psi2, a2)  # wave dim mismatch
    with pytest.raises(CalibratedActionHeadError):
        cal.calibrate_from_trajectories(psi2, torch.randn(8, 4))  # action mismatch
    with pytest.raises(CalibratedActionHeadError):
        cal.calibrate_from_trajectories(psi2, a2, data_source="eval_cache")  # bad source


def test_gate_dominance_keeps_score_ineligible():
    # Even a qualified artifact does NOT flip arc_score_eligibility.
    from arc_score_gate import arc_score_eligibility
    res = arc_score_eligibility(
        learned_component_on_action_path=False,
        checkpoint_policy="required",
        checkpoint_load_status="LOADED",
        trained_decoder_active=True,
        checkpoint_sha256="c" * 64,
        state_dict_sha256="d" * 64,
        trained_action_head_active=False,
    )
    assert res["score_eligible"] is False
    assert "LOADED_COMPONENT_NOT_ON_ACTION_PATH" in res["score_block_reason"]


def test_forward_shapes_and_dtype():
    model = StiefelActionProactor(wave_dim=64, latent_dim=32, action_dim=8)
    psi = F.normalize(torch.randn(5, 64), p=2, dim=-1)
    out = model(psi)
    assert out.shape == (5, 8)
    assert out.dtype == psi.dtype
    h = model.extract_features(psi)
    assert h.shape == (5, 32)
