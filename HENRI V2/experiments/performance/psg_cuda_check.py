"""Focused CUDA verification of the PSG vectorized EFE macro-kernel.

Runs at PRODUCTION scale (num_experts=1024, d_model=65536, num_blocks=8192)
on the RTX 5090: constructs the PSG engine, compiles a synthetic functor,
segments a synthetic grid, scores options via loop vs torch.vmap, and
reports max-abs-diff agreement + per-option latency.

Diagnostic-only: no env stepping, no scorecard, score_eligible=false.
"""
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "HENRI V2") not in sys.path:
    sys.path.insert(0, str(_REPO / "HENRI V2"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

os.environ["HENRI_ARC_PSG"] = "1"

from progressive_semantic_grounding_engine import (  # noqa: E402
    ProgressiveSemanticGroundingEngine,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402
from darwinian_phase_swarm import HenriSwarmOrchestrator  # noqa: E402
from arcengine import GameAction  # noqa: E402


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    SCALE = dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
    torch.manual_seed(20260814)
    t0 = time.time()
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=5.0,
        constraint_reject_thresh=0.38,
        beta_pragmatic=1.0,
        lambda_goal=0.5,
        learnable_actions=False,
        chimera_mode=False,
        chimera_alpha=1.4,
        chimera_explorer_fraction=0.25,
        happy_tensor_cut=False,
        external_outcome_efe=False,
        external_eig_weight=0.25,
        external_task_weight=1.0,
        task_weighted_eig=False,
        task_eig_gamma=4.0,
        **SCALE,
    ).to(device)
    orch.eval()
    tokenizer = HENRIVisionEncoder(
        d_model=SCALE["d_model"], k_blocks=SCALE["num_blocks"],
        block_dim=8, max_grid_dim=30, device=device)
    eng = ProgressiveSemanticGroundingEngine(
        orch.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8)
    print(f"[psg-cuda] orchestrator+encoder built in {time.time()-t0:.1f}s", flush=True)

    # Synthetic grid with objects (deterministic; not an ARC task grid).
    g = torch.Generator().manual_seed(20260814)
    grid = torch.randint(0, 10, (10, 10), generator=g).tolist()

    # Synthetic demo pairs (diagnostic; no real ARC demos).
    grid2 = [[(v + 1) % 10 for v in row] for row in grid]
    pairs = [(grid, grid2), (grid2, grid)]
    res = eng.compile_task_functor(pairs, task_id="psg-cuda-check")
    print(f"[psg-cuda] functor: {res.status} (held {res.held_out_cos:.4f} "
          f"vs identity {res.identity_cos:.4f})", flush=True)

    opts = eng.options_from_grid(grid)
    print(f"[psg-cuda] objects/options: {eng.segment(grid).__len__()}/{len(opts)}",
          flush=True)
    if not opts:
        print("[psg-cuda] STATUS EMPTY_OBJECTS — nothing to score")
        return 0
    waves, _labels = eng.option_waves(grid, opts)
    waves = waves[:64]  # bounded option set for the CUDA check
    opts = opts[:64]
    obs = tokenizer.encode_spatial_grid(grid).squeeze(0).to(device)
    goal = eng.goal_bind(obs)
    bnd = F.normalize(torch.randn(1, SCALE["num_blocks"], 8, device=device),
                      p=2, dim=-1)

    # Warmup.
    eng.score_batched(obs, waves, bnd, goal_wave=goal)
    torch.cuda.synchronize()

    t1 = time.time()
    efe_bat = eng.score_batched(obs, waves, bnd, goal_wave=goal)
    torch.cuda.synchronize()
    t_bat = time.time() - t1

    t2 = time.time()
    efe_loop = eng.score(obs, waves, bnd, goal_wave=goal)
    torch.cuda.synchronize()
    t_loop = time.time() - t2

    agree = float((efe_loop - efe_bat).abs().max().item())
    print(f"[psg-cuda] options={len(opts)} B={waves.shape[0]}")
    print(f"[psg-cuda] vmap: {t_bat*1000:.2f} ms total, "
          f"{t_bat/len(opts)*1000:.4f} ms/option")
    print(f"[psg-cuda] loop: {t_loop*1000:.2f} ms total, "
          f"{t_loop/len(opts)*1000:.4f} ms/option")
    print(f"[psg-cuda] agreement_max_abs_diff={agree:.3e}")
    print(f"[psg-cuda] VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print(f"[psg-cuda] score_eligible=false diagnostic_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
