"""Phase 8.2 in-context functor K1 pre-flight probe (production scale, RTX 5090).

Software-integrity verification ONLY (PDF Stage 2): synthetic task with
KNOWN translation rule, W_task compiled from synthetic prompt pairs, goal
bound, options ranked by sim-K1 gate AND by EFE. Reports true_rank /
true_margin / agreement.

A synthetic PASS proves the harness executes correctly; it does NOT
authorize environment rollout (authentic demo pairs remain the gate).

Writes JSON to HENRI_ICF_OUT or /tmp/icf_k1_result.json.
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

os.environ["HENRI_ARC_IN_CONTEXT_FUNCTOR"] = "1"

from in_context_functor_grounding_engine import (  # noqa: E402
    InContextFunctorGroundingEngine,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402
from darwinian_phase_swarm import HenriSwarmOrchestrator  # noqa: E402
from arcengine import GameAction  # noqa: E402


def _shift_right(grid, k=1):
    return [[0] * k + row[:-k] for row in grid]


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
    eng = InContextFunctorGroundingEngine(
        orch.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8)
    print(f"[icf-cuda] built in {time.time()-t0:.1f}s", flush=True)

    # Synthetic task: shift every column right by 1 (dx=+1), 3 prompt pairs.
    pairs = []
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(1, 10, (6, 6), generator=g).tolist()
        y = _shift_right(x, 1)
        pairs.append((x, y))
    test_grid = pairs[0][0]

    boundary = torch.stack([
        tokenizer.encode_spatial_grid([[0] * 6 for _ in range(6)]).squeeze(0),
    ]).to(device)

    # Compile W_task from the synthetic pairs (leave-one-out internally).
    res, cstatus = eng.compile(pairs, task_id="synthetic-shift")
    print(f"[icf-cuda] compile status={cstatus} "
          f"held={getattr(res, 'held_out_cos', None)} "
          f"identity={getattr(res, 'identity_cos', None)}", flush=True)

    state_wave = tokenizer.encode_spatial_grid(test_grid).squeeze(0).to(device)
    goal_wave = eng._psg.goal_bind(state_wave)
    goal_sim_obs = eng.goal_sim(goal_wave, state_wave)

    # Option waves: segment test grid, build macro-options, label the true
    # translate(dx=1, dy=0) option explicitly.
    options = eng._psg.options_from_grid(test_grid)
    option_waves, labels = eng._psg.option_waves(test_grid, options)
    true_label = "translate(dx=1, dy=0)"

    # K1 sim gate (PDF Lens B).
    k1 = eng.k1_gate(option_waves, goal_wave, labels, true_label)

    # EFE ranking + vmap-loop agreement.
    efe_loop = eng._psg.score(state_wave, option_waves, boundary, goal_wave)
    efe_bat = eng._psg.score_batched(state_wave, option_waves, boundary, goal_wave)
    agreement = float((efe_loop - efe_bat).abs().max().item())
    order = torch.argsort(efe_bat).tolist()
    true_idx = next((i for i, lb in enumerate(labels) if lb == true_label), -1)
    efe_true_rank = (order.index(true_idx) + 1) if true_idx >= 0 else None

    result = {
        "schema_id": "henri.icf-k1-cuda.v1",
        "compile_status": cstatus,
        "functor_status": res.status if res is not None else None,
        "held_out_cos": res.held_out_cos if res is not None else None,
        "identity_cos": res.identity_cos if res is not None else None,
        "goal_sim_obs": goal_sim_obs,
        "num_options": len(options),
        "true_label": true_label,
        "k1": k1,
        "efe_true_rank": efe_true_rank,
        "efe_agreement_max_abs_diff": agreement,
        "score_eligible": False,
        "diagnostic_only": True,
        "authorizes_rollout": False,
    }
    out = os.environ.get("HENRI_ICF_OUT", "/tmp/icf_k1_result.json")
    with open(out, "w") as fp:
        json.dump(result, fp, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
