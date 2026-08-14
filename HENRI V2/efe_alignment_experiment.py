"""Phase 8.3r — EFE Alignment experiment (default OFF).

Packet: `8.3 remediation.pdf` — Master Architectural Handoff & Continuity
Brief + Epistemological Audit / Phase 8.3 R1 Postmortem
(SHA-256 75e86eca759bffc426163adb5f9a0f0184c79f82fa99966e2490a6b61b6b7551).

Hypothesis under test (packet Lens C): the production EFE cost function
ranked a Rank-1-similarity true transform LAST in Phase 8.3 R1 (sealed
d1edd9b3) because (i) the epistemic term overpowers pragmatic attraction,
(ii) the R1 probe used a single-pixel diagnostic boundary instead of the
canonical 11-axiom Zone C baseplate, and (iii) a sign-convention conflict
turns the pragmatic goal attractor into an EFE repeller. The actionable:
re-calibrate the EFE cost function so Rank-1 similarity candidates achieve
Rank-1 minimal EFE.

This module is the DEFAULT-OFF experiment harness (flag
HENRI_ARC_EFE_ALIGNMENT). It reuses ONLY production kernels: the masked-ramp
HENRIVisionEncoder, compile_functor_wave, goal_bind, options_from_grid /
option_waves, and the EFEPlanner score_actions + select_action path (the
COMPLETE production selection path including the T4 accuracy-gated explore
gate). No surrogate planner, no duplicated EFE math, no R1-branch import.

Never steps an environment: no game.step, no SANS rows, no rollout.
score_eligible=false, diagnostic_only=true, authorizes_rollout=false on
every return path (single-source eligibility rule).

Reflection parity is OUT OF SCOPE (packet: separating mirror parity
requires non-commutative Clifford algebra).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

FEATURE_FLAG = "HENRI_ARC_EFE_ALIGNMENT"
STATUS_DISABLED = "FEATURE_DISABLED"
STATUS_OK = "OK"
SCHEMA_ID = "henri.efe-alignment-arm.v1"


def _base_out(task_id: str, **extra: Any) -> Dict[str, Any]:
    """Eligibility telemetry carried by EVERY return path (single source)."""
    out = {
        "schema_id": SCHEMA_ID,
        "transform": task_id,
        "score_eligible": False,
        "diagnostic_only": True,
        "authorizes_rollout": False,
    }
    out.update(extra)
    return out


class EfeAlignmentProbe:
    """Default-OFF EFE-alignment arm harness (reuses production kernels)."""

    def __init__(
        self,
        planner: Any,
        tokenizer: Any,
        psg: Any,
        device: str = "cpu",
        num_blocks: int = 8192,
        block_dim: int = 8,
    ) -> None:
        self.planner = planner
        self.tokenizer = tokenizer
        self.psg = psg
        self.device = device
        self.num_blocks = num_blocks
        self.block_dim = block_dim

    @property
    def enabled(self) -> bool:
        return os.environ.get(FEATURE_FLAG, "0") == "1"

    def status(self) -> Dict[str, Any]:
        return {
            "schema_id": "henri.efe-alignment.status.v1",
            "status": STATUS_OK if self.enabled else STATUS_DISABLED,
            "feature_gate": FEATURE_FLAG,
            "device": self.device,
            "score_eligible": False,
            "diagnostic_only": True,
            "authorizes_rollout": False,
        }

    def run_arm(
        self,
        arm_id: str,
        grid: List[List[int]],
        pairs: Sequence[Tuple[List[List[int]], List[List[int]]]],
        true_label: str,
        task_id: str,
        boundary_batch: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run ONE arm on ONE transform through the FULL production path."""
        if not self.enabled:
            return _base_out(task_id, status=STATUS_DISABLED, arm=arm_id)
        if self.psg is None or self.planner is None or self.tokenizer is None:
            return _base_out(task_id, status="BLOCKED_MISSING_MACHINERY",
                             arm=arm_id)

        res = self.psg.compile_task_functor(pairs, task_id=f"{arm_id}:{task_id}")
        if getattr(res, "status", STATUS_OK) != STATUS_OK:
            return _base_out(
                task_id,
                arm=arm_id,
                status=getattr(res, "status", "COMPILE_FAILED"),
                reason=getattr(res, "reason", ""),
            )

        state_wave = self.tokenizer.encode_spatial_grid(grid).squeeze(0).to(self.device)
        goal_wave = self.psg.goal_bind(state_wave)  # functor-bound goal (R1 parity)

        options = self.psg.options_from_grid(grid)
        # Pre-registered bound: identity + 4 translations + 3 rotations (8).
        # Reflection EXCLUDED (packet out-of-scope; main's builder has no
        # reflection kind).
        keep = [o for o in options if o.kind in ("translate", "rotate")]
        waves, labels = self.psg.option_waves(grid, keep)
        true_idx = next((i for i, lb in enumerate(labels) if lb == true_label), -1)
        if true_idx < 0:
            return _base_out(
                task_id, arm=arm_id, status="TRUE_LABEL_ABSENT",
                labels=labels[:20],
            )

        candidates = list(zip(labels, [w for w in waves]))

        # G2/G3: decomposition + loop/vmap identity from production kernels.
        efe_loop = self.psg.score(state_wave, waves, boundary_batch, goal_wave)
        efe_bat = self.psg.score_batched(state_wave, waves, boundary_batch, goal_wave)
        vmap_agreement = float((efe_loop - efe_bat).abs().max().item())

        # COMPLETE production selection path (score_actions + select_action
        # incl. the T4 explore gate). Raw EFE argmin is NOT the production
        # path.
        ranked = self.planner.score_actions(
            state_wave, candidates, boundary_batch, goal_wave=goal_wave)
        action, predicted, table, chosen = self.planner.select_action(
            state_wave, candidates, boundary_batch, goal_wave=goal_wave)

        efe_by_label = {r["action"]: r["efe"] for r in ranked}
        true_efe = efe_by_label.get(true_label)
        efe_order = [r["action"] for r in ranked]
        efe_true_rank = (
            efe_order.index(true_label) + 1 if true_label in efe_order else None
        )
        sorted_efe = sorted(efe_by_label.values())
        efe_margin = (
            float(sorted_efe[1] - sorted_efe[0]) if len(sorted_efe) >= 2 else None
        )

        # G2 decomposition consistency: recompute EFE from logged terms.
        lam = self.planner._constraint_lambda()
        recomputed = {}
        for r in ranked:
            recomputed[r["action"]] = (
                self.planner.pragmatic_weight * r["pragmatic"]
                - self.planner.epistemic_weight * r["epistemic"]
                + lam * r["constraint_penalty"]
            )
        max_diff = max(
            abs(recomputed[a] - efe_by_label[a]) for a in efe_by_label
        )

        chosen_action = str(chosen.get("action")) if isinstance(chosen, dict) else str(action)
        return _base_out(
            task_id,
            arm=arm_id,
            status=STATUS_OK,
            functor_status=res.status,
            held_out_cos=getattr(res, "held_out_cos", None),
            identity_cos=getattr(res, "identity_cos", None),
            num_options=len(labels),
            true_label=true_label,
            true_efe=float(true_efe) if true_efe is not None else None,
            efe_true_rank=efe_true_rank,
            efe_margin=efe_margin,
            chosen_action=chosen_action,
            chosen_was_true=(chosen_action == true_label),
            explored=bool(chosen.get("explored", False)) if isinstance(chosen, dict) else False,
            decomposition=[
                {k: r[k] for k in (
                    "action", "efe", "pragmatic", "epistemic",
                    "constraint_penalty", "goal_distance", "rejected",
                )}
                for r in ranked
            ],
            vmap_agreement=vmap_agreement,
            decomposition_max_diff=max_diff,
            boundary_shape=list(boundary_batch.shape),
        )
