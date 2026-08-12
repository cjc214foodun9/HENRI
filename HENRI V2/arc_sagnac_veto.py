"""Phase 7.5 CONN Module A: advisory Sagnac dual-channel veto sidecar.

Corpus grounding (bank ca4bb787, convo 3179135d): the dual-channel veto is the
prescribed design - hard axiom channel (epsilon_hard = 0.35, Q -> -inf) plus
soft epistemic channel (valid exploration). The veto is ADVISORY: it re-ranks
EFE candidates (first non-vetoed wins); it NEVER replaces EFEPlanner.select_action.

FALSIFIED assumption (OBSERVED 2026-08-12, worktree fdb7fd3): direct reuse of
SagnacMCTSPlanner.dual_channel_sagnac_veto is NOT valid for the ARC path. The
production method computes delta = 1 - |mean(w_cand . w_ax)|, which for the
real unit-norm UWE family (encode_grid -> F.normalize, ||w||_2 = 1) is bounded
by 1 - 1/D (Cauchy-Schwarz). Identical waves still fire: delta ~ 0.984 at
D=64, ~ 0.99998 at D=65,536. The method is calibrated for the complex
unit-modulus qFHRR family; reusing it as-is produces false vetoes on every
valid move and an always-flagged / never-re-ranking dead channel (the
pre-registered null-stream leakage failure). This sidecar therefore computes
the dual channels with the canonical norm-consistent metric used across the
ARC path (HENRIVisionEncoder.compute_sagnac_similarity: S = 0.5 * (1 + <a,b>),
delta = 1 - S): identical -> 0, orthogonal -> 1. The dual-channel structure,
epsilon_hard = 0.35 semantics, advisory role, and fail-open typing are
unchanged from the approved design.

FAIL-OPEN is deliberate: the veto is an ADVISORY candidate-ranking sidecar,
NOT a safety gate. When the sidecar is unavailable, the default EFE path is
byte-identical. When it fires, it re-ranks candidates (best non-vetoed wins);
if every candidate is vetoed, the original best is kept (no deadlock).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

VETO_OK = "SAGNAC_VETO_OK"
VETO_UNAVAILABLE = "SAGNAC_VETO_UNAVAILABLE"

DEFAULT_EPSILON_HARD = 0.35


def _sagnac_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Canonical Sagnac homodyne similarity, norm-consistent for the UWE family.

    Real unit-norm waves: S = 0.5 * (1 + <a, b>) in [0, 1] (identical -> 1).
    Complex unit-modulus waves: S = |mean(a.conj() * b)| (identical -> 1).
    """
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.is_complex() or b.is_complex():
        return torch.abs(torch.mean(a.conj() * b))
    return 0.5 * (1.0 + torch.dot(a, b))


def evaluate_veto(
    candidate_wave: torch.Tensor,
    axiom_wave: torch.Tensor,
    world_wave: torch.Tensor,
    epsilon_hard: Optional[float] = None,
) -> Tuple[float, float, bool, str]:
    """Evaluate the dual-channel Sagnac veto on a candidate wave.

    Args:
        candidate_wave: proposed trajectory wave (chosen candidate).
        axiom_wave: Zone C axiom baseplate wave (hard channel reference).
        world_wave: current observed world wave (epistemic channel reference).
        epsilon_hard: hard-channel threshold; None -> DEFAULT_EPSILON_HARD.

    Returns:
        (delta_axiom, delta_epistemic, hard_veto_triggered, status).
        status VETO_OK on a clean evaluation, VETO_UNAVAILABLE on any anomaly
        (None input, tensor mismatch, exception). Unavailable NEVER triggers.
    """
    try:
        if candidate_wave is None or axiom_wave is None or world_wave is None:
            return 0.0, 0.0, False, VETO_UNAVAILABLE
        eps = DEFAULT_EPSILON_HARD if epsilon_hard is None else float(epsilon_hard)
        s_ax = _sagnac_similarity(candidate_wave, axiom_wave)
        s_wrld = _sagnac_similarity(candidate_wave, world_wave)
        s_ax = float(torch.clamp(s_ax, 0.0, 1.0).item())
        s_wrld = float(torch.clamp(s_wrld, 0.0, 1.0).item())
        delta_axiom = 1.0 - s_ax
        delta_epistemic = 1.0 - s_wrld
        triggered = delta_axiom > eps
        return delta_axiom, delta_epistemic, bool(triggered), VETO_OK
    except Exception:
        return 0.0, 0.0, False, VETO_UNAVAILABLE


def rerank_with_veto(
    ranked: list,
    vetoed_flags: list,
) -> list:
    """Re-rank candidates: first non-vetoed candidate wins.

    Args:
        ranked: list of candidate dicts already sorted by ascending EFE.
        vetoed_flags: parallel list of bool (True = hard-vetoed).

    Returns:
        re-ranked list (vetoed candidates moved behind non-vetoed ones).
        If ALL candidates are vetoed, the original order is preserved
        (advisory: no deadlock).
    """
    if not ranked or not vetoed_flags:
        return ranked
    if len(ranked) != len(vetoed_flags):
        return ranked
    if all(vetoed_flags):
        return ranked
    clean = [r for r, v in zip(ranked, vetoed_flags) if not v]
    vetoed = [r for r, v in zip(ranked, vetoed_flags) if v]
    return clean + vetoed
