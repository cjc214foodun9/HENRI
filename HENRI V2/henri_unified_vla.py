"""
HENRI Unified Vision-Language-Action runtime assembly (Carrier U1).

Consolidation of the live, verified HENRI V2 components into one importable
production entry point. This module adds NO new mathematics, NO new kernels,
and NO new action vocabulary. It composes:

  Ingress : HENRIVisionEncoder            (henri_vision_encoder.py)
  Task    : compile_task_functor          (arc_task_functor.py, held-out gate)
  Planner : HenriSwarmOrchestrator        (darwinian_phase_swarm.py, EFE + Hopfield)
  Gate    : TypedActionGate               (henri_action_gate.py, fail-closed)
  Egress  : HENRIUnifiedEgressTransducer  (henri_decoder.py, checkpoint policy)
  Ledger  : TemporalTransitionLedger      (temporal_transition_ledger.py)

Default-OFF: the factory `get_unified_vla()` returns None unless the
environment flag HENRI_UNIFIED_VLA=1 is set. With the flag absent, this
module's runtime classes are never constructed and no consumer changes.

Evidence boundary: this carrier proves COMPOSITION and CONSUMPTION of live
components only. It grants no score eligibility and no benchmark claim.
Task outcomes are measured exclusively through the real environment and the
temporal ledger (external outcome meta), never through an internal coherence
signal. See docs/unified-vla-reconciliation.md (Carrier U1 section).

Reconciliation source: HENRI_Inbox/HENRI_Unified_Vision-Language-Action_Engine.py
sha256 12c4cc57c842247c5858c021a540d69878a8dbf8852dde651077916bff5915b0.
That blueprint's synthetic gauntlet (run_synthetic_benchmark) is REJECTED as a
mock loop and is intentionally NOT reproduced here.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

FLAG = "HENRI_UNIFIED_VLA"


# ---------------------------------------------------------------------------
# Typed result containers (no new action vocabulary; live types only)
# ---------------------------------------------------------------------------

@dataclass
class UnifiedVLAResult:
    """Compact per-step result from the unified runtime.

    Never a score. The action is executed by the caller ONLY when
    action is not None; otherwise action_rejection names the fail-closed
    reason (ACTION_NOT_LEGAL | PAYLOAD_MALFORMED | DECODE_FAILED).
    """

    step: int
    action: Any = None                      # GameAction (arcengine) or None
    action_rejection: Optional[str] = None  # TypedActionRejection reason
    action_name: str = ""
    confidence: float = 0.0
    predicted_wave: Optional[torch.Tensor] = None
    efe_chosen: float = 0.0
    explored: bool = False
    sagnac_delta: float = 1.0
    ledger_status: str = "LEDGER_NOT_WIRED"
    egress_status: str = "EGRESS_NOT_WIRED"
    telemetry: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Unified runtime assembly (composition only, zero trainable parameters added)
# ---------------------------------------------------------------------------

class HENRIUnifiedVLAModel:
    """Unified VLA runtime composing live HENRI components.

    Construction is pure composition: every subsystem is the live production
    class. No parameters are introduced by this class itself.

    Parameters mirror the production runner's construction sites:
      - tokenizer: HENRIVisionEncoder(d_model, k_blocks, device,
                    spatial_basis_kind, bg_mask)
      - orchestrator: HenriSwarmOrchestrator(..., action_enum_class=GameAction)
      - action_gate: TypedActionGate(decoder, complex_action_names, seed,
                    confidence_threshold)
      - egress: HENRIUnifiedEgressTransducer(..., checkpoint_policy)
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        orchestrator: Any,
        action_gate: Any,
        egress_transducer: Optional[Any] = None,
        temporal_ledger: Optional[Any] = None,
        boundary_axioms: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.orch = orchestrator
        self.gate = action_gate
        self.egress = egress_transducer
        self.ledger = temporal_ledger
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.boundary_axioms = boundary_axioms

    # -- Ingress ------------------------------------------------------------

    def perceive(self, grid: Any) -> Tuple[torch.Tensor, str]:
        """Encode a real environment grid into a [num_blocks, 8] wave + digest."""
        wave_blocks = self.tokenizer.encode_spatial_grid(grid)  # [1, K, 8]
        wave = wave_blocks.squeeze(0)                            # [K, 8]
        digest = _wave_digest(wave)
        return wave, digest

    # -- Task compilation (held-out falsifiable gate) -----------------------

    def compile_task(
        self,
        demo_pairs: Sequence[Tuple],
        task_id: str = "",
        hold_out_index: int = -1,
    ) -> Any:
        """Compile W_task + goal anchor via arc_task_functor (FUNCTOR_OK / FALSIFIED)."""
        from arc_task_functor import compile_task_functor

        return compile_task_functor(
            demo_pairs,
            tokenizer=self.tokenizer,
            device=self.dev,
            task_id=task_id,
            hold_out_index=hold_out_index,
        )

    # -- Planning + fail-closed action gate --------------------------------

    def act(
        self,
        state_wave: torch.Tensor,
        grid: Any,
        allowed_actions: Sequence[Any],
        goal_wave: Optional[torch.Tensor] = None,
        step: int = 0,
        efe_penalties: Optional[Dict[str, float]] = None,
    ) -> UnifiedVLAResult:
        """Plan via live EFE, then snap through the live TypedActionGate.

        The caller executes ONLY when result.action is not None. A rejection
        is a fail-closed No-Op (live gate contract), never a fallback.
        """
        result = UnifiedVLAResult(step=step)

        # 1. EFE planning through the live orchestrator.
        action, predicted_wave, efe_table, chosen = self.orch.plan_action(
            state_wave,
            self.boundary_axioms,
            top_k=4,
            return_chosen=True,
            goal_wave=goal_wave,
            grid_dist=None,
            allowed_actions=list(allowed_actions),
            efe_penalties=efe_penalties,
        )
        result.predicted_wave = predicted_wave
        result.explored = bool(chosen.get("explored", False))
        result.efe_chosen = float(chosen.get("efe", 0.0))

        # 2. Fail-closed typed gate on the planner's policy wave.
        policy_wave = chosen.get("predicted_wave", predicted_wave)
        gated = self.gate.gate(policy_wave, grid, allowed_actions)
        if getattr(gated, "rejection_reason", None):
            result.action_rejection = gated.rejection_reason
            result.action_name = "NOOP"
            return result

        result.action = gated.action
        result.action_name = getattr(gated.action, "name", str(gated.action))
        result.confidence = float(getattr(gated, "confidence", 0.0))
        return result

    # -- Egress (language/tool head, checkpoint-policy guarded) -------------

    def egress_decode(
        self,
        wave: torch.Tensor,
        prompt_text: str,
        w_task: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Decode wave -> response text via HENRIUnifiedEgressTransducer.

        Returns (None, telemetry) when the egress transducer is absent or its
        checkpoint is not LOADED (fail-closed: never emit untrained-decoder
        logits as output).
        """
        if self.egress is None:
            return None, {"egress_status": "EGRESS_NOT_WIRED"}
        tele = self.egress.checkpoint_telemetry()
        if tele.get("checkpoint_load_status") != "LOADED":
            return None, {"egress_status": "EGRESS_BLOCKED_CHECKPOINT", **tele}
        text, meta = self.egress.decode_wave_to_response(wave, prompt_text, w_task=w_task)
        return text, {"egress_status": "EGRESS_DECODED", **meta}

    # -- Temporal ledger (real transitions, external outcome meta) ----------

    def record_transition(
        self,
        *,
        pre_grid: Any,
        game_action: Any,
        obs_next: Any,
        episode_id: str,
        step: int,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist one real (s_t, a_t, s_{t+1}) triple via the live bridge.

        Raises LEDGER_FAIL_CLOSED on any defect (live bridge contract).
        Returns the ledger status string on success.
        """
        if self.ledger is None:
            return "LEDGER_NOT_WIRED"
        from temporal_ledger_bridge import record_temporal_transition

        record_temporal_transition(
            self.ledger,
            pre_grid=pre_grid,
            game_action=game_action,
            obs_next=obs_next,
            episode_id=episode_id,
            step=step,
            extra_meta=extra_meta,
        )
        return "LEDGER_RECORDED"


# ---------------------------------------------------------------------------
# Flag-gated factory (Default-OFF differential)
# ---------------------------------------------------------------------------

def get_unified_vla(
    *,
    tokenizer: Any,
    orchestrator: Any,
    action_gate: Any,
    egress_transducer: Optional[Any] = None,
    temporal_ledger: Optional[Any] = None,
    boundary_axioms: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
) -> Optional[HENRIUnifiedVLAModel]:
    """Return the unified runtime ONLY when HENRI_UNIFIED_VLA=1.

    Flag absent => None (module classes never constructed, no consumer
    change, byte-identical existing path).
    """
    if os.environ.get(FLAG, "0") != "1":
        return None
    return HENRIUnifiedVLAModel(
        tokenizer=tokenizer,
        orchestrator=orchestrator,
        action_gate=action_gate,
        egress_transducer=egress_transducer,
        temporal_ledger=temporal_ledger,
        boundary_axioms=boundary_axioms,
        device=device,
    )


def _wave_digest(wave: torch.Tensor) -> str:
    """Deterministic SHA-256 over the real float32 wave bytes (CPU)."""
    flat = wave.detach().to("cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(flat).hexdigest()
