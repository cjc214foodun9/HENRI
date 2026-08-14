"""Phase 8.1 zero-shot CUDA probe (production scale, RTX 5090).

Runs the D4-orbit goal + vmap EFE ranking at D=65,536:
- K1: synthetic single-object grid with KNOWN translate(1,0) option present;
      reports true-rank, margin, cell accuracy of the top option's grid.
- K2: EFE score spread over candidates.
- K4: vmap-loop agreement (<= 1e-6).
- goal_sim_obs + orbit_norm_raw (vacuous-gate telemetry).

Diagnostic-only: no env stepping, no scorecard, score_eligible=false.
Writes JSON to HENRI_PSG_ZS_OUT or /tmp/psg_zero_shot_result.json.
"""

import json
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

os.environ["HENRI_ARC_PSG_ZERO_SHOT"] = "1"

from progressive_semantic_grounding_engine import (  # noqa: E402
    ProgressiveSemanticGroundingEngine,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402
from darwinian_phase_swarm import HenriSwarmOrchestrator  # noqa: E402
from arcengine import GameAction  # noqa: E402


def _cell_acc(a, b) -> float:
    n = len(a) * len(a[0])
    same = sum(1 for r in range(len(a)) for c in range(len(a[0]))
               if a[r][c] == b[r][c])
    return same / n


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    SCALE = dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
    torch.manual_seed(20260814)
    t0 = time.time()
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=5.0, constraint_reject_thresh=0.38,
        beta_pragmatic=1.0, lambda_goal=0.5,
        learnable_actions=False, chimera_mode=False, chimera_alpha=1.4,
        chimera_explorer_fraction=0.25, happy_tensor_cut=False,
        external_outcome_efe=False, external_eig_weight=0.25,
        external_task_weight=1.0, task_weighted_eig=False,
        task_eig_gamma=4.0, **SCALE,
    ).to(device)
    orch.eval()
    tokenizer = HENRIVisionEncoder(
        d_model=SCALE["d_model"], k_blocks=SCALE["num_blocks"],
        block_dim=8, max_grid_dim=30, device=device)
    eng = ProgressiveSemanticGroundingEngine(
        orch.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8)
    print(f"[zs-cuda] built in {time.time()-t0:.1f}s", flush=True)

    # K1 synthetic grid: object at (1,1)-(2,2); true transform translate(dx=1).
    grid = [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]]
    boundary = torch.stack([
        tokenizer.encode_spatial_grid([[0] * 4 for _ in range(4)]).squeeze(0),
    ]).to(device)  # production boundary shape [1, num_blocks, 8]

    t1 = time.time()
    r = eng.zero_shot_plan(grid, boundary, task_id="synthetic-k1",
                           top_k=1, use_batched=True)
    plan_ms = (time.time() - t1) * 1000.0
    top = r["ranked"][0] if r["ranked"] else {}
    top_opt = top.get("option", {})
    top_payload = top.get("payload", {})

    # Recompute full ranking for K2 spread + K1 true-rank.
    state = tokenizer.encode_spatial_grid(grid).squeeze(0).to(device)
    goal_wave, orbit, orbit_size = eng.d4_orbit_goal(grid)
    opts = eng.options_from_grid(grid)
    waves, labels = eng.option_waves(grid, opts)
    efes = eng.score_batched(state, waves, boundary, goal_wave)
    order = torch.argsort(efes).tolist()
    spread = float((efes.max() - efes.min()).item())

    # True transform = translate(dx=1) description match.
    true_label = "translate(dx=1, dy=0)"
    true_rank = None
    for i, idx in enumerate(order):
        if labels[idx] == true_label:
            true_rank = i
            break
    margin = None
    if len(order) > 1 and true_rank is not None:
        margin = float((efes[order[1 if true_rank == 0 else 0]] - efes[order[true_rank]]).item())

    # Cell accuracy: apply top option to grid, compare with true target.
    cell_acc = None
    if top_opt:
        from progressive_semantic_grounding_engine import (
            MacroOption, _apply_option_to_grid,
        )
        mo = MacroOption.from_dict(top_opt)
        pred_grid = _apply_option_to_grid(grid, mo)
        true_target = [[0, 0, 0, 0],
                       [0, 0, 1, 1],
                       [0, 0, 1, 0],
                       [0, 0, 0, 0]]
        cell_acc = _cell_acc(pred_grid, true_target)

    result = {
        "schema_id": "henri.psg-zero-shot-cuda.v1",
        "status": r["status"],
        "goal_source": r["goal_source"],
        "goal_sim_obs": r["goal_sim_obs"],
        "orbit_norm_raw": r["orbit_norm_raw"],
        "orbit_size": orbit_size,
        "plan_ms": round(plan_ms, 3),
        "agreement_max_abs_diff": r["agreement_max_abs_diff"],
        "num_options": r["num_options"],
        "efe_spread": spread,
        "true_rank": true_rank,
        "true_margin": margin,
        "top_option": top_opt,
        "top_payload": top_payload,
        "cell_acc_top_vs_true": cell_acc,
        "score_eligible": False,
        "diagnostic_only": True,
    }
    out = os.environ.get("HENRI_PSG_ZS_OUT", "/tmp/psg_zero_shot_result.json")
    with open(out, "w") as fp:
        json.dump(result, fp, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
