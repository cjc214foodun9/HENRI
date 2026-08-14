"""Phase 8.3 R1 — K1 pre-flight probe (production scale, RTX 5090).

Runs the FULL production ranking path (option waves -> functor goal bind ->
sim-K1 gate -> vmap EFE ranking) for the 3 required transforms:

    translation, rotation, reflection

against the K1 gate: true_rank <= 2 AND true_margin >= +0.05 (per transform,
independent). The FULL mask+ramp variant is the only candidate arm. Ablation
arms (legacy / mask-only / ramps-only) are separate probes if attribution is
needed; this file tests the candidate.

Synthetic single-object grids with KNOWN true transforms; W_task compiled from
synthetic in-context pairs (leave-one-out) using the PRODUCTION functor.

A pass proves representation discrimination ONLY — it does NOT authorize
environment rollout (authentic demo pairs remain the gate; 20/20 BLOCKED).

Writes JSON to HENRI_R1_OUT or /tmp/r1_k1_result.json.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

os.environ["HENRI_ARC_REPRESENTATION_R1"] = "1"

from representation_discrimination_engine import (  # noqa: E402
    RepresentationDiscriminationEngine,
    _masked_ramp_encoder,
    apply_reflection,
    build_transform_options,
)
from progressive_semantic_grounding_engine import (  # noqa: E402
    MacroOption,
    _apply_option_to_grid,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402
from darwinian_phase_swarm import HenriSwarmOrchestrator  # noqa: E402
from arcengine import GameAction  # noqa: E402


def _object_color(grid):
    for row in grid:
        for v in row:
            if v != 0:
                return v
    return 1


def _transform_via_option(grid, name):
    """Apply the SAME transform construction the harness scores, so pair
    semantics == option semantics (bbox full-grid, pixel color match)."""
    color = _object_color(grid)
    H, W = len(grid), len(grid[0])
    if name == "translation":
        m = MacroOption(object_id=0, kind="translate", dx=1, dy=0,
                        bbox=(0, 0, H - 1, W - 1), area=0, color=color,
                        description="translate(dx=1, dy=0)")
        return _apply_option_to_grid(grid, m)
    if name == "rotation":
        m = MacroOption(object_id=0, kind="rotate", angle_deg=90,
                        bbox=(0, 0, H - 1, W - 1), area=0, color=color,
                        description="rotate(90)")
        return _apply_option_to_grid(grid, m)
    if name == "reflection":
        return apply_reflection(grid, "h")
    raise ValueError(name)


def _single_object_grid(size=8, color=1):
    """Grid with ONE asymmetric foreground object.

    L-shape cells (2,3),(2,4),(3,4) — asymmetric under 90/180/270 rotation,
    under horizontal AND vertical reflection, and under translation.
    """
    g = [[0] * size for _ in range(size)]
    g[2][3] = g[2][4] = g[3][4] = color
    return g


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

    # CANDIDATE arm: full mask+ramp (production default variant).
    tokenizer = _masked_ramp_encoder(
        d_model=SCALE["d_model"], num_blocks=SCALE["num_blocks"],
        block_dim=8, max_grid_dim=30, device=device,
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    eng = RepresentationDiscriminationEngine(
        orch.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )
    print(f"[r1-cuda] built in {time.time()-t0:.1f}s", flush=True)

    base = _single_object_grid(size=8, color=1)

    transforms = {
        "translation": "translate(dx=1, dy=0)",
        "rotation": "rotate(90)",
        "reflection": "reflect_h",
    }

    results = {}
    all_pass = True
    for name, true_label in transforms.items():
        # 3 synthetic prompt pairs: random grids + the SAME transform
        # (constructed via the identical option path the harness scores).
        pairs = []
        for seed in range(3):
            g = _single_object_grid(size=8, color=1 + (seed % 3))
            pairs.append((g, _transform_via_option(g, name)))

        out = eng.k1_harness(
            base, pairs, true_label=true_label, task_id=name, color=1)
        results[name] = out
        k1 = out.get("k1", {})
        passed = k1.get("status") == "K1_PASS"
        all_pass = all_pass and passed
        print(f"[r1-cuda] {name}: status={k1.get('status')} "
              f"rank={k1.get('true_rank')} margin={k1.get('true_margin')} "
              f"efe_rank={out.get('efe_true_rank')} "
              f"goal_sim={out.get('goal_sim_obs')}",
              flush=True)

    result = {
        "schema_id": "henri.r1-k1-cuda.v1",
        "candidate_arm": "mask_ramp_incommensurate",
        "device": device,
        "d_model": SCALE["d_model"],
        "k1_gate": {"true_rank_max": 2, "true_margin_min": 0.05},
        "transforms": results,
        "verdict": "R1_ACCEPT" if all_pass else "R1_FALSIFIED",
        "score_eligible": False,
        "diagnostic_only": True,
        "authorizes_rollout": False,
    }
    out = os.environ.get("HENRI_R1_OUT", "/tmp/r1_k1_result.json")
    with open(out, "w") as fp:
        json.dump(result, fp, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
