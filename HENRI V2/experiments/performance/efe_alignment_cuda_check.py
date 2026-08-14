"""Phase 8.3r — EFE Alignment CUDA matrix (production scale, RTX 5090).

Packet `8.3 remediation.pdf` (SHA-256 75e86eca...). Runs the 4-arm matrix
through the COMPLETE production selection path:
  A: canonical 11-axiom Zone C boundary + lambda_goal=0.0 (PRODUCTION cfg)
  B: canonical 11-axiom Zone C boundary + lambda_goal=0.5 (R1 probe cfg)
  C: single-pixel diagnostic boundary + lambda_goal=0.0
  D: single-pixel diagnostic boundary + lambda_goal=0.5 (R1 reproduction)

Per arm, per transform (translation, rotation): 3 synthetic prompt pairs via
the same option path; functor compile; goal_bind; 8 macro-options; production
score_actions decomposition + select_action (incl. T4 explore gate) + loop/
vmap agreement.

Gates (pre-registered, NO threshold tuning):
  G1 boundary integrity  G2 decomposition consistency  G3 loop/vmap identity
  G4 ALIGNMENT (arm A): true EFE rank == 1 AND margin >= 1e-3 for BOTH
     transforms -> EFE_ALIGNMENT_PASS; else EFE_ALIGNMENT_FALSIFIED
  G5 production selection consistency (reported, not gating)
  G6-G8 enforced by contract tests (flag OFF identity, eligibility fields,
     no game.step in source).

Fail-closed: canonical boundary load failure -> BLOCKED_INFRASTRUCTURE (no
single-pixel fallback for arms A/B). Reflection excluded (packet out-of-scope).

Run from repo root with Zone C prod env sourced:
  set -a && . /workspace/zonec_prod.env && set +a
  env HENRI_ARC_EFE_ALIGNMENT=1 HENRI_EFE_OUT=/tmp/efe_alignment_result.json \
      /venv/main/bin/python 'HENRI V2/experiments/performance/efe_alignment_cuda_check.py'
"""

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.abspath("HENRI V2"))

from efe_planner import EFEPlanner  # noqa: E402
from darwinian_phase_swarm import HenriSwarmOrchestrator  # noqa: E402
from arcengine import GameAction  # noqa: E402
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402
from progressive_semantic_grounding_engine import (  # noqa: E402
    ProgressiveSemanticGroundingEngine, MacroOption, _apply_option_to_grid,
)
from zone_c_boundary_axiom_loader import (  # noqa: E402
    BoundaryAxiomLoadError, load_boundary_axioms,
)
from efe_alignment_experiment import EfeAlignmentProbe  # noqa: E402

SCALE = dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
GATE = {"true_efe_rank_max": 1, "efe_margin_min": 1e-3}


def _masked_ramp_encoder(device: str) -> Any:
    return HENRIVisionEncoder(
        d_model=SCALE["d_model"], k_blocks=SCALE["num_blocks"], block_dim=8,
        max_grid_dim=30, device=device,
        spatial_basis_kind="incommensurate", bg_mask=True,
    )


def _single_object_grid(size: int = 8, color: int = 1) -> List[List[int]]:
    g = [[0] * size for _ in range(size)]
    g[2][3] = g[2][4] = g[3][4] = color
    return g


def _object_color(grid: List[List[int]]) -> int:
    for row in grid:
        for v in row:
            if v != 0:
                return v
    return 1


def _transform_via_option(grid: List[List[int]], name: str) -> List[List[int]]:
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
    raise ValueError(name)


def _single_pixel_boundary(tokenizer: Any, device: str) -> torch.Tensor:
    bg = [[0] * 8 for _ in range(8)]
    bg[0][0] = 1
    return tokenizer.encode_spatial_grid(bg).squeeze(0).unsqueeze(0).to(device)


def _canonical_receipt(axioms: torch.Tensor) -> Dict[str, Any]:
    """Per-block (8-dim) unit-norm integrity receipt for the 11 axioms.

    Storage contract (zone_c loader docstring + skill invariant): each
    [num_blocks, 8] row is PER-BLOCK unit normalized, so the flat D=65,536
    row norm is sqrt(num_blocks) ~= 90.5, NOT 1.0. The integrity check must
    reshape to [N*num_blocks, 8] and verify each 8-dim block has norm ~1.
    """
    flat = axioms.detach().cpu().float()  # [N, num_blocks, 8]
    per_block = flat.reshape(-1, 8)       # [N*num_blocks, 8]
    norms = torch.norm(per_block, p=2, dim=-1)
    payload_sha = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
    return {
        "num_axioms": int(flat.shape[0]),
        "shape": list(flat.shape),
        "per_block_norm_min": float(norms.min().item()),
        "per_block_norm_max": float(norms.max().item()),
        "flat_row_norm_reference": float(
            torch.norm(flat.view(flat.shape[0], -1), p=2, dim=-1).mean().item()),
        "payload_sha256": payload_sha,
    }


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    assert os.environ.get("HENRI_ARC_EFE_ALIGNMENT", "0") == "1", "flag must be ON"
    out_path = os.environ.get("HENRI_EFE_OUT", "/tmp/efe_alignment_result.json")
    torch.manual_seed(20260814)
    t0 = time.time()

    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=5.0, constraint_reject_thresh=0.38,
        beta_pragmatic=1.0, lambda_goal=0.0,
        learnable_actions=False, chimera_mode=False, chimera_alpha=1.4,
        chimera_explorer_fraction=0.25, happy_tensor_cut=False,
        external_outcome_efe=False, external_eig_weight=0.25,
        external_task_weight=1.0, task_weighted_eig=False,
        task_eig_gamma=4.0, **SCALE,
    ).to(device)
    orch.eval()
    tokenizer = _masked_ramp_encoder(device)
    psg = ProgressiveSemanticGroundingEngine(
        orch.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )
    probe = EfeAlignmentProbe(
        orch.planner, tokenizer, psg, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )
    print(f"[efe-alignment] built in {time.time()-t0:.1f}s", flush=True)

    # G1: canonical 11-axiom boundary (fail-closed; NO fallback).
    try:
        axioms, axiom_summary = load_boundary_axioms(
            num_blocks=SCALE["num_blocks"])
    except BoundaryAxiomLoadError as exc:
        result = {
            "schema_id": "henri.efe-alignment-matrix.v1",
            "verdict": "BLOCKED_INFRASTRUCTURE",
            "reason": f"canonical boundary load failed: {exc}",
            "score_eligible": False, "diagnostic_only": True,
            "authorizes_rollout": False,
        }
        _write(out_path, result)
        print(f"BLOCKED_INFRASTRUCTURE: {exc}", flush=True)
        sys.exit(2)
    axioms = axioms.to(device=device, dtype=torch.float32)
    receipt = _canonical_receipt(axioms)
    norms_min, norms_max = receipt["per_block_norm_min"], receipt["per_block_norm_max"]
    g1_ok = (norms_min >= 1.0 - 1e-6) and (norms_max <= 1.0 + 1e-6)
    print(f"[efe-alignment] axioms {receipt['num_axioms']} "
          f"norm[{norms_min:.7f},{norms_max:.7f}] sha={receipt['payload_sha256'][:16]}",
          flush=True)

    single_b = _single_pixel_boundary(tokenizer, device)

    arms = {
        "A": {"boundary": axioms, "lambda_goal": 0.0},
        "B": {"boundary": axioms, "lambda_goal": 0.5},
        "C": {"boundary": single_b, "lambda_goal": 0.0},
        "D": {"boundary": single_b, "lambda_goal": 0.5},
    }
    transforms = {"translation": "translate(dx=1, dy=0)",
                  "rotation": "rotate(90)"}

    results: Dict[str, Any] = {}
    for arm_id, cfg in arms.items():
        orch.planner.lambda_goal = cfg["lambda_goal"]
        results[arm_id] = {}
        for name, true_label in transforms.items():
            pairs: List[Tuple[List[List[int]], List[List[int]]]] = []
            for seed in range(3):
                g = _single_object_grid(size=8, color=1 + (seed % 3))
                pairs.append((g, _transform_via_option(g, name)))
            out = probe.run_arm(
                arm_id, _single_object_grid(size=8, color=1), pairs,
                true_label=true_label, task_id=name,
                boundary_batch=cfg["boundary"],
            )
            results[arm_id][name] = out
            print(f"[efe-alignment] arm={arm_id} {name}: "
                  f"efe_rank={out.get('efe_true_rank')} "
                  f"margin={out.get('efe_margin')} "
                  f"chosen={out.get('chosen_action')} "
                  f"explored={out.get('explored')} "
                  f"vmap={out.get('vmap_agreement'):.2e} "
                  f"decomp={out.get('decomposition_max_diff'):.2e}",
                  flush=True)

    # Gates.
    g2_ok = all(
        results[a][t].get("decomposition_max_diff", 1.0) <= 1e-4
        for a in arms for t in transforms)
    g3_ok = all(
        results[a][t].get("vmap_agreement", 1.0) <= 1e-5
        for a in arms for t in transforms)
    arm_a = results["A"]
    g4_ok = all(
        arm_a[t].get("efe_true_rank") == 1
        and arm_a[t].get("efe_margin", -1.0) >= GATE["efe_margin_min"]
        for t in transforms)
    g5 = {
        "A": {t: {"chosen_was_true": results["A"][t]["chosen_was_true"],
                  "explored": results["A"][t]["explored"]} for t in transforms}
    }
    if not g1_ok:
        verdict = "BLOCKED_INFRASTRUCTURE"
    elif not (g2_ok and g3_ok):
        verdict = "BLOCKED_INFRASTRUCTURE"
    elif g4_ok:
        verdict = "EFE_ALIGNMENT_PASS"
    else:
        verdict = "EFE_ALIGNMENT_FALSIFIED"

    result = {
        "schema_id": "henri.efe-alignment-matrix.v1",
        "verdict": verdict,
        "gates": {
            "G1_boundary_integrity": g1_ok,
            "G2_decomposition_consistency": g2_ok,
            "G3_loop_vmap_identity": g3_ok,
            "G4_alignment_armA": g4_ok,
            "G5_production_selection": g5,
            "gate_spec": GATE,
        },
        "canonical_boundary_receipt": receipt,
        "arms": results,
        "config": {
            "d_model": SCALE["d_model"], "num_blocks": SCALE["num_blocks"],
            "device": device, "flag": "HENRI_ARC_EFE_ALIGNMENT",
            "encoder": "mask_ramp_incommensurate", "seed": 20260814,
        },
        "score_eligible": False,
        "diagnostic_only": True,
        "authorizes_rollout": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _write(out_path, result)
    print(f"[efe-alignment] VERDICT={verdict} elapsed={result['elapsed_s']}s",
          flush=True)


def _write(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
