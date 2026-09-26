"""Zone A steering core — THIN FACADE over verified live modules.

Ratified decision (user, 2026-09-26): build a THIN FACADE over modules that
already exist and already have consumers. It adds NO new mathematics.

LIVE SURFACE (probed 2026-09-26, tools/probe_zone_a_surface.py — an earlier
draft targeted classes that DO NOT EXIST):
  DarwinianPhaseSwarm.GapJunctionSwarmSyncytium(num_experts, d_model, r_rank)
  DarwinianPhaseSwarm.HenriSwarmOrchestrator(num_experts, d_model, r_rank,
      num_blocks, action_enum_class) -> attrs [syncytium, clifford, decoder, planner]
      .candidate_action_waves(top_k, allowed_actions) -> [(action, [K,8] wave)]
      .sagnac_coherence(active, target) -> tensor in [-1,1]
      .plan_action(...) -> the production action path
  henri_vision_encoder.HENRIVisionEncoder(d_model, k_blocks, block_dim, ...)
      .encode_spatial_grid(grid) -> [1, K, 8], unit norm
  wave_jepa.WaveJEPA(d_model, num_blocks, r_rank, device)
      .predict_future_latent(state_wave, action_wave)   (NO `.rollout` exists)

DIFFERENTIABILITY CONTRACT (this is the load-bearing part)
----------------------------------------------------------
An earlier revision of this file converted the live coherence to a Python float:

    coh = float(self.orch.sagnac_coherence(wave, reference_wave))   # WRONG

That severs the autograd graph, so any downstream dream/SGLD loss built from it
has `requires_grad=False` and `autograd.grad` raises
"element 0 of tensors does not require grad". The live `sagnac_coherence` IS
differentiable with respect to its TARGET argument (measured: grad norm
6.59e-01), so the facade must PRESERVE the tensor.

`CandidateSet.deltas` therefore returns a differentiable `[K]` tensor.
`delta_floats()` exists for reporting only and is never used to build a loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

__all__ = ["ZoneACore", "CandidateSet"]

DEFAULT_D_MODEL = 65536
DEFAULT_NUM_BLOCKS = 8192
DEFAULT_NUM_EXPERTS = 1024
DEFAULT_R_RANK = 16


@dataclass
class CandidateSet:
    """Live candidate waves with a DIFFERENTIABLE delta tensor.

    Attributes:
        actions: the action objects, in candidate order.
        waves: the [K, 8] real candidate waves.
        deltas: [K] tensor, differentiable w.r.t. the reference wave. This is
            the CLIFFORD-COHERENCE family: delta = 1 - Re(<cand,tgt>)/(|c| |t|),
            bounded [0, 2]. It is the SAME metric `plan_action` selects by.
    """

    actions: List[Any]
    waves: List[torch.Tensor]
    deltas: torch.Tensor

    def __len__(self) -> int:
        return len(self.actions)

    def delta_floats(self) -> List[float]:
        """Reporting only. NEVER build a loss from these (they detach)."""
        with torch.no_grad():
            return [float(x) for x in self.deltas.detach()]


class ZoneACore:
    """Thin facade over the live Zone A stack. Zero new math."""

    def __init__(
        self,
        *,
        d_model: int = DEFAULT_D_MODEL,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        num_experts: int = DEFAULT_NUM_EXPERTS,
        r_rank: int = DEFAULT_R_RANK,
        action_enum_class: Any = None,
        device: Optional[str] = None,
    ) -> None:
        from darwinian_phase_swarm import HenriSwarmOrchestrator
        from henri_vision_encoder import HENRIVisionEncoder
        from wave_jepa import WaveJEPA

        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_experts = num_experts
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.orch = HenriSwarmOrchestrator(
            num_experts=num_experts,
            d_model=d_model,
            r_rank=r_rank,
            num_blocks=num_blocks,
            action_enum_class=action_enum_class,
        )
        self.encoder = HENRIVisionEncoder(
            d_model=d_model, k_blocks=num_blocks, device=self.dev
        )
        self.jepa = WaveJEPA(
            d_model=d_model, num_blocks=num_blocks, r_rank=r_rank, device=self.dev
        )

    # ------------------------------------------------------------------ ingress
    def encode(self, grid: Any) -> torch.Tensor:
        """Encode a real grid into a [num_blocks, 8] unit-norm wave."""
        return self.encoder.encode_spatial_grid(grid).squeeze(0)

    # --------------------------------------------------------------- candidates
    def candidate_set(
        self,
        active_wave: torch.Tensor,
        reference_wave: torch.Tensor,
        *,
        top_k: int = 4,
        allowed_actions: Optional[Sequence[Any]] = None,
    ) -> CandidateSet:
        """Enumerate live candidates; score with the LIVE, DIFFERENTIABLE metric.

        `reference_wave` is kept in the autograd graph, so the returned
        `deltas` tensor carries a gradient path to any tensor that produced it.
        """
        actions: List[Any] = []
        waves: List[torch.Tensor] = []
        deltas: List[torch.Tensor] = []

        for action, wave in self.orch.candidate_action_waves(
            top_k=top_k, allowed_actions=allowed_actions
        ):
            actions.append(action)
            waves.append(wave)
            # NO float() here: keep the tensor so the graph survives.
            coh = self.orch.sagnac_coherence(wave, reference_wave)
            deltas.append(1.0 - torch.as_tensor(coh))

        if not deltas:
            empty = torch.zeros(0)
            return CandidateSet(actions=actions, waves=waves, deltas=empty)

        return CandidateSet(actions=actions, waves=waves, deltas=torch.stack(deltas))

    # --------------------------------------------------------------- transition
    def predict_future_latent(
        self, state_wave: torch.Tensor, action_wave: torch.Tensor
    ) -> torch.Tensor:
        """Latent transition via the live Wave-JEPA operator."""
        return self.jepa.predict_future_latent(state_wave, action_wave)

    # ------------------------------------------------------------- action path
    def plan(
        self,
        active_wave: torch.Tensor,
        reference_wave: torch.Tensor,
        *,
        top_k: int = 4,
        allowed_actions: Optional[Sequence[Any]] = None,
        goal_wave: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor, Any, Any]:
        """The production action path: live EFE planning, delegating to the
        orchestrator. The dreamer must GATE this, never replace it."""
        axioms = reference_wave.unsqueeze(0)
        return self.orch.plan_action(
            active_wave,
            axioms,
            top_k=top_k,
            return_chosen=True,
            goal_wave=goal_wave,
            allowed_actions=list(allowed_actions) if allowed_actions is not None else None,
        )

    # ------------------------------------------------------------------- utils
    def trainable_parameters(self):
        return [p for p in self.orch.parameters() if p.requires_grad]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.orch.parameters())
