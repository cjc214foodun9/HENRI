"""HENRI Goal Subspace Projection — Arm E (default OFF).

Pre-registration: Project_HENRI__Arm-D_Forensic_Audit___Arm-E_Subspace_Projection_Pre-Registration.md
(SHA-256 1839da60fee4c90292f367a411d7ee27ebe7f187ae54ef497765c877e4bdf19f)

Mechanism (Section 3.1 of the pre-registration):
    Psi_tilde = V V^dag Psi_goal + (1/|A|) sum_a R_block(a)^dag Psi_goal
    Psi_goal_proj = Psi_tilde / (||Psi_tilde||_2 + 1e-9)

Live-operator derivation note (audited against efe_planner.py:70-178,
2026-08-27): LowRankCoupledTransition holds ONE action-agnostic
block_residual [num_blocks, 8, 8] (the action enters through the FHRR bind,
not through per-action residuals). With identical R_a for all a, the
pre-registered action mean collapses exactly:
    (1/|A|) * sum_a R_a^dag = R^dag
so the implementation applies the single residual adjoint once. No block
averaging is applied (that would destroy the per-block wave structure).

Hardware invariants:
  - No dense [d, d] tensor is ever allocated. V V^dag is applied as
    V @ (V^dag @ x) (left-associative). R^dag is applied per block as a
    batch [B, 8, 8] matmul (einsum, no [d, d] materialization).
  - The operator is zero-trainable and read-only: it detaches the transition
    factors and never mutates them.

Failure mode (fail-closed): if any factor is missing, the shapes mismatch,
or the projection is non-finite/degenerate, project_goal returns the ORIGINAL
goal wave with projected=False and a reason string. A failed projection never
fabricates a goal.
"""

from __future__ import annotations

import torch


def project_goal(
    goal_wave: torch.Tensor,
    field_V: torch.Tensor,
    block_residual: torch.Tensor,
) -> dict:
    """Project an ambient goal wave into the transition operator's reachable
    subspace (Arm E, Section 3.1).

    Args:
        goal_wave: [num_blocks, block_dim] real goal wave.
        field_V: [d, r] column-semi-unitary field basis (QR-retracted).
        block_residual: [num_blocks, block_dim, block_dim] per-block unitary
            residual (the R factor; action-agnostic in the live operator).

    Returns dict with:
        goal_wave: projected [B, D] tensor (unit norm), or the ORIGINAL goal
            on any failure.
        projected: bool — True when the projection was applied.
        projected_norm: float | None — ||Psi_tilde|| before normalization.
        reason: str | None — failure reason when projected=False.
    """
    if goal_wave is None:
        return {"goal_wave": goal_wave, "projected": False,
                "projected_norm": None, "reason": "goal_wave_none"}
    if field_V is None or block_residual is None:
        return {"goal_wave": goal_wave, "projected": False,
                "projected_norm": None, "reason": "missing_factor"}
    try:
        g = goal_wave
        B, D = g.shape[0], g.shape[1]
        if block_residual.shape != (B, D, D):
            return {"goal_wave": goal_wave, "projected": False,
                    "projected_norm": None,
                    "reason": f"block_residual_shape_{tuple(block_residual.shape)}"}
        if field_V.shape[0] != B * D:
            return {"goal_wave": goal_wave, "projected": False,
                    "projected_norm": None,
                    "reason": f"field_V_shape_{tuple(field_V.shape)}"}
        # Left-associative V V^dag: [d, r] @ ([r, d] @ [d]) — no [d, d] tensor.
        g_flat = g.reshape(-1)
        field_term = field_V @ (field_V.T @ g_flat)  # [d] real
        # Per-block residual adjoint: conj(R).transpose(-1,-2) @ g_rows.
        # R is COMPLEX in the live operator (efe_planner.py:117, complex64)
        # while the goal wave is float32; einsum does NOT auto-promote
        # (OBSERVED RuntimeError: expected scalar type ComplexFloat but
        # found Float on CUDA probe, 2026-08-27). Cast the goal rows to the
        # adjoint's complex dtype, apply, then take the real part (the
        # planner boundary is REAL [num_blocks, 8]; mirrors the transition's
        # own `local.real + field`).
        adjoint = block_residual.conj().transpose(-1, -2)  # [B, D, D] complex
        g_c = g.reshape(B, D).to(adjoint.dtype)            # [B, D] complex
        residual_term = torch.einsum(
            "bij,bj->bi", adjoint, g_c).real.reshape(-1)
        tilde = field_term + residual_term
        tilde_norm = float(torch.norm(tilde, p=2).item())
        if not torch.isfinite(tilde).all() or tilde_norm <= 1e-12:
            return {"goal_wave": goal_wave, "projected": False,
                    "projected_norm": tilde_norm,
                    "reason": "non_finite_or_degenerate"}
        projected = tilde / (tilde_norm + 1e-9)
        projected = projected.reshape_as(g)
        if torch.is_complex(projected):
            projected = projected.real
        return {"goal_wave": projected, "projected": True,
                "projected_norm": round(tilde_norm, 6), "reason": None}
    except Exception as exc:  # fail-closed: keep the original goal
        return {"goal_wave": goal_wave, "projected": False,
                "projected_norm": None, "reason": type(exc).__name__}
