# -*- coding: utf-8 -*-
"""
Phase 8.32 — Calibrated Action Head (Stiefel-Ridge) — from Drive inbox
`Calibrated_Action_Head_Module.py` (sha256 db5c91df7d72fc5345edf658cf03f6aaef02620edf057b5e0a9806e87e4517ce).

Mechanism (faithful to directive):
  UWE_Ingress: Psi_t in S^(D-1), D=65,536
  h_t = GELU(LayerNorm(W_down Psi_t)) in R^2048
  W_act* = argmin ||A - W H||_F^2 + gamma ||W||_F^2  (Ridge)
  W_retracted = U V^T  (Stiefel retraction)
  Gate: E_cal <= 0.05 AND Delta_sagnac <= 0.20 -> default-ON, else default-OFF.

Honest corrections over the inbox module (audit 2026-08-19):
  1. HELD-OUT gate: the inbox __main__ fits and evaluates on the same samples
     and asserts ON -> mock loop + leakage. This implementation evaluates the
     gate ONLY on a deterministic held-out split. In-sample fit must NOT qualify.
  2. SVD-form ridge: torch.linalg.solve(HtH + gamma I, HtA) is unstable when
     M < L (128 < 2048). Equivalent SVD form W = V diag(s/(s^2+gamma)) U^T A
     is stable for M < L and matches solve() on well-conditioned M >= L fixtures.
  3. Sagnac naming: the module's "sagnac_stress" is action-space L2
     (||a_pred - a_true||_2 mean), NOT the wave-space homodyne
     0.5*(1+dot(Psi_pred, Psi_true)) used elsewhere in HENRI. It is labeled
     `sagnac_stress_proxy` here; a true homodyne check requires wave targets.
  4. No mock self-qualification: synthetic fixtures produce data_source=
     "synthetic_fixture" artifacts. Production activation additionally
     requires data_source=="authorized" + production wiring + task validation.
  5. Name: `henri_calibrated_action_head.py` (matches the inbox docstring);
     distinct from arc_action_head.py / algebraic_action_head.py.

Boundary: this commit is DEFAULT-OFF, zero production wiring. ARC arcade
exposes no authorized (o_t, a_t, o_t+1) trajectories (run5 BLOCKED_NO_DEMOS x19)
-> production transition = BLOCKED_NO_ACTION_TRAJECTORIES until authorized data
exists. `trained_action_head_active` stays false; score_eligible stays false.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMA_ID = "henri.calibrated-action-head.v1"
WAVE_DIMENSION = 65536
LATENT_DIMENSION = 2048
DEFAULT_ACTION_DIM = 8
CALIBRATION_THRESHOLD_MSE = 0.05
SAGNAC_THRESHOLD_VETO = 0.20


class CalibratedActionHeadError(RuntimeError):
    """Typed fail-closed error for calibration/artifact failures."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_sha256(t: torch.Tensor) -> str:
    tt = t.detach().cpu().contiguous()
    return _sha256_bytes(tt.numpy().tobytes())


class StiefelActionProactor(nn.Module):
    """Action projection network on the Stiefel manifold.

    Psi in S^(D-1) -> h in R^2048 -> a in R^A.
    """

    def __init__(
        self,
        wave_dim: int = WAVE_DIMENSION,
        latent_dim: int = LATENT_DIMENSION,
        action_dim: int = DEFAULT_ACTION_DIM,
    ) -> None:
        super().__init__()
        if wave_dim <= 0 or latent_dim <= 0 or action_dim <= 0:
            raise CalibratedActionHeadError("dims must be positive")
        self.wave_dim = int(wave_dim)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        # W_down [latent, wave] (536.9 MiB fp32 at D=65,536; by design)
        self.w_down = nn.Linear(self.wave_dim, self.latent_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.latent_dim)
        self.w_act = nn.Linear(self.latent_dim, self.action_dim, bias=False)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.w_down.weight, gain=1.0 / math.sqrt(2.0))
        nn.init.xavier_uniform_(self.w_act.weight, gain=1.0)

    def extract_features(self, psi_states: torch.Tensor) -> torch.Tensor:
        """Unit-normalized latent features h in [B, latent_dim] (input device)."""
        psi_norm = F.normalize(psi_states, p=2, dim=-1)
        h_raw = self.w_down(psi_norm)
        return F.gelu(self.layer_norm(h_raw))

    def forward(self, psi_states: torch.Tensor) -> torch.Tensor:
        return self.w_act(self.extract_features(psi_states))


class ActionHeadCalibrator:
    """Ridge-SVD Stiefel calibration + qualification gate + artifact sealing.

    All tensor ops follow the input device. No [D, D] intermediates.
    """

    def __init__(
        self,
        model: StiefelActionProactor,
        ridge_gamma: float = 1e-3,
        mse_threshold: float = CALIBRATION_THRESHOLD_MSE,
        sagnac_threshold: float = SAGNAC_THRESHOLD_VETO,
        held_out_frac: float = 0.25,
        seed: int = 0,
    ) -> None:
        if not (0.0 < held_out_frac < 1.0):
            raise CalibratedActionHeadError("held_out_frac must be in (0, 1)")
        self.model = model
        self.ridge_gamma = float(ridge_gamma)
        self.mse_threshold = float(mse_threshold)
        self.sagnac_threshold = float(sagnac_threshold)
        self.held_out_frac = float(held_out_frac)
        self.seed = int(seed)

    def calibrate_from_trajectories(
        self,
        psi_states: torch.Tensor,
        target_actions: torch.Tensor,
        data_source: str = "authorized",
        split_identity: str = "",
        no_eval_cache_provenance: str = "no_arc_solutions_or_eval_caches_used",
    ) -> Dict[str, Any]:
        """Compute Stiefel-retracted Ridge weights and a HELD-OUT qualification.

        psi_states [M, wave_dim]; target_actions [M, action_dim].
        Returns a sealed artifact dict (schema henri.calibrated-action-head.v1).
        """
        if psi_states.ndim != 2 or target_actions.ndim != 2:
            raise CalibratedActionHeadError("inputs must be [M, dim]")
        if psi_states.shape[0] != target_actions.shape[0]:
            raise CalibratedActionHeadError("M mismatch between psi and actions")
        if psi_states.shape[1] != self.model.wave_dim:
            raise CalibratedActionHeadError(
                f"wave dim {psi_states.shape[1]} != model {self.model.wave_dim}")
        if target_actions.shape[1] != self.model.action_dim:
            raise CalibratedActionHeadError(
                f"action dim {target_actions.shape[1]} != model {self.model.action_dim}")
        if data_source not in ("authorized", "synthetic_fixture"):
            raise CalibratedActionHeadError(f"unknown data_source {data_source!r}")

        device = psi_states.device
        M = psi_states.shape[0]
        if M < 2:
            raise CalibratedActionHeadError("calibration requires M >= 2")

        # Deterministic held-out split (permutation, seeded)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        perm = torch.randperm(M, generator=gen)
        n_test = max(1, int(round(M * self.held_out_frac)))
        n_test = min(n_test, M - 1)  # keep >= 1 train sample
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        psi_tr, a_tr = psi_states[train_idx], target_actions[train_idx]
        psi_te, a_te = psi_states[test_idx], target_actions[test_idx]

        # 1. Latent features (eval mode, no grad)
        self.model.eval()
        with torch.no_grad():
            H = self.model.extract_features(psi_tr)  # [N_tr, latent]

        # 2. Ridge via SVD form (stable for M < L):
        #    W_opt = V diag(s/(s^2+gamma)) U^T A   -> [action, latent]
        u, s, vh = torch.linalg.svd(H, full_matrices=False)  # [N,k],[k],[k,L]
        coef = (s / (s * s + self.ridge_gamma)).contiguous()  # [k]
        tmp = (u.T @ a_tr) * coef[:, None]                     # [k, action]
        w_t = (vh.T @ tmp).contiguous()                        # [latent, action]
        w_opt = w_t.T                                          # [action, latent]

        # 3. Stiefel retraction W = U2 V2^T (rows orthonormal when A < L)
        u2, _s2, vh2 = torch.linalg.svd(w_opt, full_matrices=False)
        w_stiefel = (u2 @ vh2).contiguous()

        # 4. Assign calibrated weights
        with torch.no_grad():
            self.model.w_act.weight.copy_(w_stiefel)

        # 5. HELD-OUT qualification only
        with torch.no_grad():
            pred_te = self.model(psi_te)
            mse_te = float(F.mse_loss(pred_te, a_te).item())
            # Proxy Sagnac stress: action-space L2 (see module docstring note 3)
            sagnac_proxy = float(torch.norm(pred_te - a_te, dim=-1).mean().item())

        is_qualified = (mse_te <= self.mse_threshold) and (
            sagnac_proxy <= self.sagnac_threshold)
        status = "ON" if is_qualified else "OFF"

        # 6. Artifact sealing (pre-registered schema)
        dataset_digest = _sha256_bytes(
            psi_states.detach().cpu().numpy().tobytes()
            + target_actions.detach().cpu().numpy().tobytes()
        )
        artifact: Dict[str, Any] = {
            "schema_id": SCHEMA_ID,
            "version": "1",
            "status": status,
            "is_qualified": is_qualified,
            "data_source": data_source,
            "split_identity": split_identity,
            "calibration_mse_heldout": mse_te,
            "sagnac_stress_proxy_action_l2": sagnac_proxy,
            "train_count": int(train_idx.numel()),
            "held_out_count": int(n_test),
            "ridge_gamma": self.ridge_gamma,
            "wave_dim": self.model.wave_dim,
            "latent_dim": self.model.latent_dim,
            "action_dim": self.model.action_dim,
            "action_ordering": [f"ACTION{i+1}" for i in range(self.model.action_dim)],
            "weight_sha256": _tensor_sha256(w_stiefel),
            "w_down_sha256": _tensor_sha256(self.model.w_down.weight.detach()),
            "dataset_digest": dataset_digest,
            "no_eval_cache_provenance": no_eval_cache_provenance,
            "timestamp": time.time(),
            "artifact_sha256": "",
        }
        blob = json.dumps({k: v for k, v in artifact.items() if k != "artifact_sha256"},
                          sort_keys=True).encode("utf-8")
        artifact["artifact_sha256"] = _sha256_bytes(blob)
        return artifact


def load_calibrated_head_artifact(
    path: str,
    expected_wave_dim: Optional[int] = None,
    expected_action_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Strict validator: ANY mismatch -> typed CalibratedActionHeadError."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        raise CalibratedActionHeadError(f"artifact unreadable: {exc}") from exc
    if raw.get("schema_id") != SCHEMA_ID:
        raise CalibratedActionHeadError(
            f"schema {raw.get('schema_id')} != {SCHEMA_ID}")
    blob = json.dumps({k: v for k, v in raw.items() if k != "artifact_sha256"},
                      sort_keys=True).encode("utf-8")
    if _sha256_bytes(blob) != raw.get("artifact_sha256", ""):
        raise CalibratedActionHeadError("artifact self-hash mismatch")
    if expected_wave_dim is not None and raw.get("wave_dim") != expected_wave_dim:
        raise CalibratedActionHeadError(
            f"wave_dim {raw.get('wave_dim')} != {expected_wave_dim}")
    if expected_action_dim is not None and raw.get("action_dim") != expected_action_dim:
        raise CalibratedActionHeadError(
            f"action_dim {raw.get('action_dim')} != {expected_action_dim}")
    if not raw.get("weight_sha256"):
        raise CalibratedActionHeadError("artifact missing weight signature")
    return raw


def production_activation_eligible(artifact: Dict[str, Any]) -> Tuple[bool, str]:
    """Dominance rule: synthetic fixtures can NEVER activate production.

    Artifact alone is insufficient even when qualified: production wiring +
    task validation are required (checked by the caller in the live path).
    """
    if not artifact.get("is_qualified"):
        return False, "ACTION_HEAD_NOT_QUALIFIED"
    if artifact.get("data_source") != "authorized":
        return False, "ACTION_HEAD_SYNTHETIC_ONLY"
    return True, ""


if __name__ == "__main__":
    # Verification fixture — SYNTHETIC ONLY, never capability evidence.
    print("=" * 76)
    print("  HENRI: CALIBRATED ACTION HEAD — SYNTHETIC FIXTURE VERIFICATION")
    print("  NOTE: synthetic fixture => NOT capability evidence; production")
    print("  activation requires authorized trajectories (BLOCKED_NO_ACTION_TRAJECTORIES).")
    print("=" * 76)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Substrate] device={device}")

    proactor = StiefelActionProactor(
        wave_dim=WAVE_DIMENSION, latent_dim=LATENT_DIMENSION,
        action_dim=DEFAULT_ACTION_DIM).to(device)
    num_samples = 256
    g = torch.Generator(device="cpu").manual_seed(20260819)
    raw = torch.randn(num_samples, WAVE_DIMENSION, generator=g)
    psi = F.normalize(raw, p=2, dim=-1).to(device)
    # Structured target: linear function of leading latents (realizable fit)
    target = torch.sin(psi[:, :DEFAULT_ACTION_DIM] * math.pi).to(device)

    calibrator = ActionHeadCalibrator(proactor, ridge_gamma=1e-3)
    art = calibrator.calibrate_from_trajectories(
        psi, target, data_source="synthetic_fixture",
        split_identity="fixture-256-seed20260819")
    for k in ("status", "is_qualified", "calibration_mse_heldout",
              "sagnac_stress_proxy_action_l2", "train_count", "held_out_count"):
        print(f"  {k:32s}: {art[k]}")
    print(f"  weight_sha256            : {art['weight_sha256'][:16]}...")
    ok, reason = production_activation_eligible(art)
    print(f"  production_activation    : {ok} ({reason})")
    assert ok is False and reason == "ACTION_HEAD_SYNTHETIC_ONLY"
    print("  [PASSED] fixture mechanics verified; production activation correctly blocked.")
