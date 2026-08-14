"""Phase 8.2 In-Context Functor Grounding engine (ICFG).

Implements Phase8.2.pdf (SHA-256 a41433e5..., extracted text 52242308...):

    W_task   = Normalize( sum_i conj(Psi_Xi) * Psi_Yi )   (Hadamard unbinding)
    Psi_goal = W_task (x) Psi_test                        (directional goal)

Pre-flight K1 gate (PDF Lens B): Rank(T(Psi_test, a_true) | Psi_goal) <= 2
AND Margin = Sim(true) - max_a!=true Sim(a) >= +0.05.

REUSE, not duplication: compiles via the production PSG engine's
`compile_functor_wave` / `goal_bind` / `option_waves` / `score` /
`score_batched` kernels (progressive_semantic_grounding_engine.py).

Discipline:
- default-OFF flag `HENRI_ARC_IN_CONTEXT_FUNCTOR`;
- demo boundary enforced: BLOCKED_NO_DEMONSTRATIONS / 
  BLOCKED_INSUFFICIENT_HOLDOUT_PAIRS; never fabricate pairs;
- diagnostic-only: score_eligible=false, never steps an environment.

Kills K1-K5 pre-registered in experiments/sweeps/phase82_in_context_functor_design.md.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from progressive_semantic_grounding_engine import (
    ProgressiveSemanticGroundingEngine,
    STATUS_FEATURE_DISABLED,
    STATUS_NO_DEMOS,
    STATUS_FALSIFIED,
    STATUS_EMPTY,
    STATUS_OK,
    MacroOption,
)

FEATURE_FLAG = "HENRI_ARC_IN_CONTEXT_FUNCTOR"
SCHEMA_ID = "henri.ic-functor.v1"
STATUS_INSUFFICIENT = "BLOCKED_INSUFFICIENT_HOLDOUT_PAIRS"
K1_PASS = "K1_PASS"
K1_FAIL = "K1_FAIL"
K1_TRUE_RANK_MAX = 2
K1_MARGIN_MIN = 0.05
K1_AGREEMENT_MAX = 1e-6


class InContextFunctorGroundingEngine:
    """Phase 8.2 in-context functor grounding (default-OFF, diagnostic-only)."""

    def __init__(
        self,
        planner: Any,
        tokenizer: Any,
        device: Optional[str] = None,
        num_blocks: int = 8192,
        block_dim: int = 8,
        max_options: int = 128,
        goal_lambda: float = 0.5,
    ):
        self._psg = ProgressiveSemanticGroundingEngine(
            planner=planner, tokenizer=tokenizer, device=device,
            num_blocks=num_blocks, block_dim=block_dim,
            max_options=max_options, goal_lambda=goal_lambda,
        )
        self.device = self._psg.device
        self.feature_flag = FEATURE_FLAG

    # -- Feature gate -----------------------------------------------------
    @property
    def enabled(self) -> bool:
        return os.environ.get(self.feature_flag, "0") == "1"

    def compile(self, demo_pairs: Sequence[Tuple[Any, Any]],
                task_id: str = ""):
        """Compile W_task from AUTHORIZED in-context pairs (leave-one-out).

        Returns (result, status); the result object is retained even on
        FUNCTOR_FALSIFIED so telemetry keeps held_out_cos/identity_cos.
        """
        if not demo_pairs:
            return None, STATUS_NO_DEMOS
        if len(demo_pairs) < 2:
            return None, STATUS_INSUFFICIENT
        res = self._psg.compile_task_functor(demo_pairs, task_id=task_id)
        if self._psg.w_task is None:
            return None, (res.status or STATUS_FALSIFIED)
        return res, res.status

    def goal_sim(self, goal_wave: torch.Tensor,
                 obs_wave: torch.Tensor) -> float:
        return float(F.cosine_similarity(
            goal_wave.reshape(-1), obs_wave.reshape(-1), dim=0).item())

    def k1_gate(self, option_waves: torch.Tensor, goal_wave: torch.Tensor,
                labels: List[str], true_label: str) -> Dict[str, Any]:
        """Sim-based pre-flight K1 ranking gate (PDF Lens B)."""
        B = option_waves.shape[0]
        g = goal_wave.reshape(-1)
        sims = torch.stack([
            F.cosine_similarity(option_waves[i].reshape(-1), g, dim=0)
            for i in range(B)
        ])
        order = torch.argsort(sims, descending=True).tolist()
        true_idx = next((i for i, lb in enumerate(labels)
                         if lb == true_label), -1)
        if true_idx < 0:
            return {"status": "TRUE_OPTION_MISSING", "true_rank": None,
                    "true_margin": None, "pass": False}
        true_rank = order.index(true_idx) + 1  # 1-indexed
        false_sims = [float(sims[i].item()) for i in range(B)
                      if i != true_idx]
        margin = float(sims[true_idx].item()) - (max(false_sims) if false_sims else 0.0)
        ok = (true_rank <= K1_TRUE_RANK_MAX and margin >= K1_MARGIN_MIN)
        return {
            "status": K1_PASS if ok else K1_FAIL,
            "true_rank": true_rank,
            "true_margin": margin,
            "pass": ok,
            "sims": [float(s.item()) for s in sims],
        }

    def plan(
        self, grid: List[List[int]], demo_pairs: Optional[Sequence[Tuple[Any, Any]]],
        boundary_batch: Optional[torch.Tensor], task_id: str = "",
        top_k: int = 1, use_batched: bool = True,
    ) -> Dict[str, Any]:
        """Full in-context functor pipeline (fail-closed ordering)."""
        out: Dict[str, Any] = {
            "schema_id": SCHEMA_ID,
            "feature_gate": self.feature_flag,
            "status": STATUS_FEATURE_DISABLED,
            "diagnostic_only": True,
            "score_eligible": False,
            "reason": "",
            "functor_status": None,
            "held_out_cos": None,
            "identity_cos": None,
            "w_task_sha256": None,
            "pairs_digest": None,
            "goal_sim_obs": None,
            "num_objects": 0,
            "num_options": 0,
            "agreement_max_abs_diff": None,
            "ranked": [],
        }
        if not self.enabled:
            out["reason"] = f"{self.feature_flag} != 1; engine did not allocate"
            return out
        if not demo_pairs:
            out["status"] = STATUS_NO_DEMOS
            out["reason"] = "no authorized demonstration pairs; never fabricate demos"
            return out
        if len(demo_pairs) < 2:
            out["status"] = STATUS_INSUFFICIENT
            out["reason"] = "fewer than 2 pairs: no valid leave-one-out hold-out"
            return out

        res, compile_status = self.compile(demo_pairs, task_id=task_id)
        out["functor_status"] = res.status if res is not None else compile_status
        if res is not None:
            out["held_out_cos"] = res.held_out_cos
            out["identity_cos"] = res.identity_cos
            out["w_task_sha256"] = res.w_task_sha256
            out["pairs_digest"] = res.pairs_digest
        if compile_status != STATUS_OK:
            out["status"] = compile_status
            out["reason"] = (f"functor gate blocked: {out['functor_status']} "
                             f"(held {res.held_out_cos:.4f} vs identity "
                             f"{res.identity_cos:.4f})" if res else "functor unavailable")
            return out

        objects = self._psg.segment(grid)
        out["num_objects"] = len(objects)
        options = self._psg.options_from_grid(grid)
        out["num_options"] = len(options)
        if not options:
            out["status"] = STATUS_EMPTY
            out["reason"] = "no objects segmented in grid"
            return out
        if boundary_batch is None:
            out["status"] = "BLOCKED_BOUNDARY"
            out["reason"] = "boundary_batch is required (fail-closed)"
            return out

        state_wave = self._psg.tokenizer.encode_spatial_grid(
            grid).squeeze(0).to(self.device)
        goal_wave = self._psg.goal_bind(state_wave)  # W_task (x) Psi_test
        out["goal_sim_obs"] = self.goal_sim(goal_wave, state_wave)

        option_waves, _labels = self._psg.option_waves(grid, options)
        efe_loop = self._psg.score(state_wave, option_waves, boundary_batch,
                                   goal_wave)
        efe_bat = self._psg.score_batched(state_wave, option_waves,
                                          boundary_batch, goal_wave)
        agreement = float((efe_loop - efe_bat).abs().max().item())
        out["agreement_max_abs_diff"] = agreement

        order = torch.argsort(efe_bat).tolist()
        ranked = []
        for i, idx in enumerate(order[:top_k]):
            ranked.append({
                "rank": i,
                "option": options[idx].to_dict(),
                "efe": float(efe_bat[idx].item()),
                "payload": options[idx].to_payload(),
            })
        out["status"] = STATUS_OK  # functor gate OK + options ranked
        out["ranked"] = ranked
        out["reason"] = (f"in-context functor OK; top-{top_k} macro-options "
                         f"ranked; vmap-loop agreement {agreement:.2e}")
        return out
