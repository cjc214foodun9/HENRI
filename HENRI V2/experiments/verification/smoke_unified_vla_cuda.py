"""Carrier U1 remote CUDA construction smoke (Vast 47411800).

Proves HENRIUnifiedVLAModel composes the LIVE components on the CUDA target
with the real checkpoint overlay: perceive -> [8192,8] wave; egress transducer
LOADED; factory flag-gated. NO score claim; composition evidence only.
"""
import os
import sys

os.environ["HENRI_UNIFIED_VLA"] = "1"

import torch

from henri_vision_encoder import HENRIVisionEncoder
from darwinian_phase_swarm import HenriSwarmOrchestrator
from henri_action_gate import TypedActionGate
from henri_decoder import HENRIUnifiedEgressTransducer
from henri_unified_vla import get_unified_vla, HENRIUnifiedVLAModel
from arcengine import GameAction


def main() -> int:
    dev = "cuda"
    print("cuda available:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    tokenizer = HENRIVisionEncoder(
        d_model=65536, k_blocks=8192, device=dev,
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction, d_model=65536, num_blocks=8192,
        num_experts=1024, r_rank=16,
    ).to(dev)
    gate = TypedActionGate(orch.decoder, seed=0)
    egress = HENRIUnifiedEgressTransducer(
        d_model=65536, hidden_dim=2048, vocab_size=32000,
        device=dev, checkpoint_policy="required",
    )

    vla = get_unified_vla(
        tokenizer=tokenizer, orchestrator=orch, action_gate=gate,
        egress_transducer=egress, device=dev,
    )
    assert isinstance(vla, HENRIUnifiedVLAModel), "factory returned wrong type"

    grid = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    wave, digest = vla.perceive(grid)
    print("perceive shape:", tuple(wave.shape), "digest:", digest[:12])
    assert tuple(wave.shape) == (8192, 8), "unexpected wave shape"
    assert torch.isfinite(wave).all().item()

    text, tele = vla.egress_decode(wave, "test prompt")
    print("egress_status:", tele.get("egress_status"),
          "checkpoint:", tele.get("checkpoint_load_status"))

    # Compose one act() call through the live gate (allowed actions = legal set).
    allowed = list(GameAction)
    result = vla.act(wave, grid, allowed, step=0)
    print("act action:", result.action_name,
          "rejection:", result.action_rejection,
          "efe:", round(result.efe_chosen, 4),
          "explored:", result.explored)
    assert result.action is not None or result.action_rejection is not None

    print("UNIFIED_VLA_CUDA_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
