# -*- coding: utf-8 -*-
"""
Phase 8.31 — Algebraic (No-BPTT) Semantic Action Head Calibrator.

Implements the three native wave-algebraic mechanisms from `8.31.pdf`
(sha256 b0eb8084c528217072b0f3dfd2604cac74dea1317f161e1b08f3a8434b52dafa)
as a DEFAULT-OFF, additive component. NO production wiring in this commit.

Mechanisms (audited against live code):
  A. Closed-form low-rank Procrustes task operator
     W_task = Y X^dag via thin-SVD on [D, r] factors (never [D, D]).
     Target: universal_data_transducer.py (currently has NO Procrustes).
  B. Hopfield "lexical snap" (ContinuousHopfieldCleanup in hopfield_cleanup.py,
     already wired into henri_egress.py) — reuse its energy/retrieve.
  C. In-situ SGLD creep lives in henri_decoder.py:133 adapt_in_context_sgld
     (PDF says wave_jepa.py — module-name correction recorded).

Honest boundary:
  - ARC-AGI-3 arcade games expose NO public demo pairs (BLOCKED_NO_DEMOS
    observed). Calibration from pairs therefore reports
    BLOCKED_NO_ACTION_TRAJECTORIES in production; software verification uses
    synthetic fixtures ONLY (never capability evidence).
  - An artifact passes `trained_action_head_active=true` only when the strict
    validator below passes every gate (schema, digests, dims, ordering,
    held-out rank/margin/accuracy, no-eval-cache provenance).
  - Until then: trained_action_head_active=false, score_eligible=false,
    diagnostic_only=true, terminal reason ACTION_HEAD_NOT_CALIBRATED.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


SCHEMA_ID = "henri.algebraic-action-head.v1"
DEFAULT_RANK = 128          # low-rank Procrustes cap ([D, r] factors only)
DEFAULT_SEED = 0


class AlgebraicActionHeadError(RuntimeError):
    """Typed fail-closed error for missing/corrupt/incompatible artifacts."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_sha256(t: torch.Tensor) -> str:
    tt = t.detach().cpu().contiguous()
    return _sha256_bytes(tt.numpy().tobytes())


@dataclass
class AlgebraicHeadArtifact:
    """Strict provenance artifact for the algebraically calibrated head.

    Mirrors the ActionHeadState contract (arc_action_head.py) plus the
    algebraic-specific fields required by the Phase 8.31 pre-registration.
    """
    schema_id: str = SCHEMA_ID
    version: str = "1"
    action_names: List[str] = field(default_factory=list)      # exact ordering
    d_model: int = 0
    r_rank: int = DEFAULT_RANK
    basis_digest: str = ""                                     # wave-basis sha
    source_wave_family: str = ""                               # e.g. "uwe_clifford_num_blocks_8"
    calibration_dataset_digest: str = ""                       # pair-source digest
    split_identity: str = ""                                   # train/held-out ids
    action_engrams_sha256: str = ""                            # [A, D] engram bytes
    w_task_sha256: str = ""                                    # [D, r] factor bytes
    per_env_action_map: Dict[str, List[str]] = field(default_factory=dict)
    held_out_metrics: Dict[str, Any] = field(default_factory=dict)  # rank, margin, acc, conf
    action6_payload_metrics: Dict[str, Any] = field(default_factory=dict)
    no_eval_cache_provenance: str = ""                         # statement of boundary
    artifact_sha256: str = ""

    def finalize(self) -> "AlgebraicHeadArtifact":
        payload = asdict(self)
        payload.pop("artifact_sha256", None)
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.artifact_sha256 = _sha256_bytes(blob)
        return self


class AlgebraicActionHeadCalibrator:
    """Default-OFF algebraic calibrator.

    All methods are deterministic (seed-pinned) and D=65,536-safe:
    no [D, D] allocations; low-rank factors [D, r]; engram store [A, D]
    complex64/float32 as provided.
    """

    def __init__(
        self,
        d_model: int,
        r_rank: int = DEFAULT_RANK,
        beta: float = 8.0,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if d_model <= 0 or r_rank <= 0:
            raise AlgebraicActionHeadError("d_model and r_rank must be positive")
        self.d_model = int(d_model)
        self.r_rank = int(min(r_rank, d_model))
        self.beta = float(beta)
        self.seed = int(seed)
        self._engrams: Optional[torch.Tensor] = None          # [A, D] unit-norm
        self._w_task: Optional[torch.Tensor] = None           # [D, r] orthonormal
        self.action_names: List[str] = []
        self._gen = torch.Generator().manual_seed(self.seed)

    # -- Mechanism A: low-rank Procrustes -----------------------------------
    def compile_task_operator(
        self,
        x_pairs: torch.Tensor,
        y_pairs: torch.Tensor,
        basis_digest: str,
        calibration_digest: str,
    ) -> torch.Tensor:
        """W_task = argmin ||Y - X W||_F over rank-r W, applied factorized.

        NEVER allocates [D, D]. Thin SVD of X ([N, D]) gives
            U_r [N, r], S_r [r], V_r [D, r];
        the rank-r Procrustes operator is
            W = V_r S_r^{-1} (U_r^T Y),   applied as  Psi_goal = V_r * (S_r^{-1} * (P @ Psi_in))
        with P = U_r^T Y stored as [r, D] (33.5 MiB at r=128, D=65536).
        Total transient memory: O(ND + rD).
        """
        if x_pairs.ndim != 2 or y_pairs.ndim != 2:
            raise AlgebraicActionHeadError("pairs must be [N, D]")
        if x_pairs.shape != y_pairs.shape:
            raise AlgebraicActionHeadError("pair shapes must match")
        n, d = x_pairs.shape
        if d != self.d_model:
            raise AlgebraicActionHeadError(
                f"pair d_model {d} != calibrator {self.d_model}")
        r = min(self.r_rank, n, d)
        dev = x_pairs.device
        # Thin SVD of X: [N, D] -> U [N, min], S [min], Vh [min, D].
        u, s, vh = torch.linalg.svd(x_pairs, full_matrices=False)
        u_r = u[:, :r].contiguous()          # [N, r]
        s_r = s[:r].contiguous()             # [r]
        v_r = vh[:r, :].T.contiguous()       # [D, r]
        p = u_r.T @ y_pairs                  # [r, N] @ [N, D] -> [r, D]
        self._w_task = v_r                   # orthonormal [D, r] (Stiefel)
        self._proj = p                       # [r, D]
        self._inv_scale = (1.0 / (s_r + 1e-9)).contiguous()  # [r]
        self.basis_digest = basis_digest
        self.calibration_digest = calibration_digest
        return self._w_task

    def transduce(self, wave: torch.Tensor) -> torch.Tensor:
        """Psi_goal = W_task Psi_in, factorized: O(rD), no [D, D].

        wave: [D] or [*, D]; returns same shape on the input device.
        """
        if self._w_task is None or self._proj is None:
            raise AlgebraicActionHeadError("compile_task_operator() first")
        flat = wave.reshape(-1, self.d_model)          # [B, D]
        mid = flat @ self._proj.T                      # [B, r]
        mid = mid * self._inv_scale                    # [B, r]
        out = mid @ self._w_task.T                     # [B, D]
        return out.reshape(wave.shape)

    # -- Mechanism B: Hopfield snap (reuse hopfield_cleanup if available) ----
    def store_action_engrams(
        self,
        engrams: torch.Tensor,
        action_names: Sequence[str],
    ) -> int:
        """Store [A, D] unit-norm action engrams + exact action ordering."""
        if engrams.ndim != 2 or engrams.shape[1] != self.d_model:
            raise AlgebraicActionHeadError("engrams must be [A, d_model]")
        if len(action_names) != engrams.shape[0]:
            raise AlgebraicActionHeadError("action_names length != engrams rows")
        norm = engrams.norm(dim=1, keepdim=True)
        self._engrams = (engrams / (norm + 1e-9)).contiguous()
        self.action_names = list(action_names)
        return self._engrams.shape[0]

    def snap(self, wave: torch.Tensor) -> Tuple[int, float, float]:
        """Energy-based snap: returns (index, top_sim, margin).

        margin = top_sim - second_sim; used by held-out gate G4.
        """
        if self._engrams is None:
            raise AlgebraicActionHeadError("no engrams stored")
        flat = wave.reshape(-1, self.d_model)
        sims = torch.nn.functional.cosine_similarity(
            flat[0].unsqueeze(0), self._engrams, dim=1)
        top2 = torch.topk(sims, k=min(2, sims.numel()))
        idx = int(top2.indices[0])
        top = float(top2.values[0])
        margin = float(top2.values[0] - top2.values[1]) if top2.values.numel() > 1 else float(top)
        return idx, top, margin

    # -- Artifact I/O --------------------------------------------------------
    def save_artifact(self, path: str, **extra: Any) -> AlgebraicHeadArtifact:
        if self._engrams is None or self._w_task is None:
            raise AlgebraicActionHeadError("calibrate() and store() before save")
        art = AlgebraicHeadArtifact(
            action_names=list(self.action_names),
            d_model=self.d_model,
            r_rank=self.r_rank,
            basis_digest=getattr(self, "basis_digest", ""),
            source_wave_family="uwe_clifford_num_blocks_8",
            calibration_dataset_digest=getattr(self, "calibration_digest", ""),
            split_identity=extra.get("split_identity", ""),
            action_engrams_sha256=_tensor_sha256(self._engrams),
            w_task_sha256=_tensor_sha256(self._w_task),
            per_env_action_map=extra.get("per_env_action_map", {}),
            held_out_metrics=extra.get("held_out_metrics", {}),
            action6_payload_metrics=extra.get("action6_payload_metrics", {}),
            no_eval_cache_provenance=extra.get(
                "no_eval_cache_provenance",
                "no_arc_solutions_or_eval_caches_used"),
        )
        art.finalize()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(art), f, sort_keys=True, indent=2)
        return art


def load_algebraic_head_artifact(
    path: str,
    expected_d_model: Optional[int] = None,
    expected_actions: Optional[int] = None,
    expected_basis_digest: Optional[str] = None,
) -> AlgebraicHeadArtifact:
    """Strict validator. ANY mismatch -> typed AlgebraicActionHeadError."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        raise AlgebraicActionHeadError(f"artifact unreadable: {exc}") from exc
    if raw.get("schema_id") != SCHEMA_ID:
        raise AlgebraicActionHeadError(
            f"schema {raw.get('schema_id')} != {SCHEMA_ID}")
    art = AlgebraicHeadArtifact(**{k: v for k, v in raw.items() if k != "artifact_sha256"})
    # Recompute self-hash over the canonical payload (artifact_sha256 popped,
    # matching finalize() exactly).
    payload = asdict(art)
    payload.pop("artifact_sha256", None)
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    if _sha256_bytes(blob) != raw.get("artifact_sha256", ""):
        raise AlgebraicActionHeadError("artifact self-hash mismatch")
    if expected_d_model is not None and art.d_model != expected_d_model:
        raise AlgebraicActionHeadError(
            f"d_model {art.d_model} != expected {expected_d_model}")
    if expected_actions is not None and len(art.action_names) != expected_actions:
        raise AlgebraicActionHeadError(
            f"action count {len(art.action_names)} != expected {expected_actions}")
    if expected_basis_digest is not None and art.basis_digest != expected_basis_digest:
        raise AlgebraicActionHeadError("basis digest mismatch")
    if not art.action_engrams_sha256 or not art.w_task_sha256:
        raise AlgebraicActionHeadError("artifact missing tensor digests")
    return art


def algebraic_head_eligible(art: AlgebraicHeadArtifact) -> Tuple[bool, str]:
    """Gate: artifact fully calibrated + held-out metrics meet G3/G4/G5."""
    if not art.artifact_sha256:
        return False, "ACTION_HEAD_NOT_CALIBRATED"
    m = art.held_out_metrics or {}
    rank = m.get("true_rank")
    margin = m.get("margin")
    acc = m.get("accuracy")
    n = m.get("n_heldout", 0)
    if rank is None or margin is None or acc is None or n < 20:
        return False, "ACTION_HEAD_NOT_CALIBRATED"
    if not (int(rank) <= 2 and float(margin) >= 0.05 and float(acc) > 0.0):
        return False, "ACTION_HEAD_HELD_OUT_GATES_FAIL"
    return True, ""
