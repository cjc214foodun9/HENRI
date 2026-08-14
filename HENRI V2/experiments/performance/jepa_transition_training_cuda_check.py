"""Phase 8.4 — JEPA Transition Training CUDA matrix (production scale, RTX 5090).

Protocol: `R3 Transition Training Specification.pdf` (sha256 ecbc1a5a...) +
`Wavejepatrain.txt` (sha256 5820021f...) + `Synthesis Report.pdf`
(sha256 861a3845...), reconciled against live code (see
experiments/sweeps/phase84_jepa_transition_training_design.md).

Runs the production-scale matrix through the COMPLETE production selection
path (score_actions + select_action incl. the T4 explore gate):

  Training: 9 known-transform pairs (3 kinds x 3 seeds), real encoder waves,
  via EFEPlanner.train_transition_step (Sagnac loss, Wirtinger grads, QR
  retraction), K=200 steps, lr=0.05, valence=1.0.

  Arm A: TRAINED transition, canonical 11-axiom Zone C boundary, lambda_goal=0.0
  Arm B: UNTRAINED control (identical seed twin), canonical boundary

Gates (pre-registered, NO post-hoc threshold tuning, NO rerun on fire):
  G0 boundary integrity + decomposition consistency (<=1e-4) + loop/vmap
     identity (<=1e-5)
  G1 loss_ema after K=200 < 0.30   (secondary: raw Sagnac loss <= 0.15)
  G2 ranking ladder (arm A, both transforms):
       JEPA_ALIGNMENT_PASS  : G1 AND true-EFE rank == 1 for BOTH transforms
       JEPA_PARTIAL_SIGNAL  : G1 AND rank <= 2 for BOTH AND strict
                              improvement vs arm B on >= 1 transform
       JEPA_TRAINING_INERT  : G1 fails OR no improvement vs control

Eligibility: score_eligible=false, diagnostic_only=true,
authorizes_rollout=false on every return path. Never steps an environment.
Synthetic known-transform pairs are diagnostic-only, NOT ARC demo
fabrication. The Wavejepatrain verify_gate_g4 (self-generated target, mock
loop) is REJECTED; ranking is measured through the real production path.

Run from repo root with Zone C prod env sourced:
  set -a && . /workspace/zonec_prod.env && set +a
  env HENRI_ARC_JEPA_TRANSITION_TRAINING=1 HENRI_JEPA_OUT=/tmp/jepa_training_result.json \
      /venv/main/bin/python 'HENRI V2/experiments/performance/jepa_transition_training_cuda_check.py'
"""

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

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
from jepa_transition_training_experiment import (  # noqa: E402
    JepaTransitionTrainingProbe, TRAIN_KINDS,
)

SCALE = dict(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192)
GATE_LOSS = 0.30          # G1 (Synthesis Report acceptance test)
SPEC_LOSS = 0.15          # secondary marker (spec), NOT gating
TRAIN_K = 200
TRAIN_LR = 0.05
RANK_PASS = 1
RANK_PARTIAL = 2


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


def _option_for(kind: str, grid: List[List[int]]) -> MacroOption:
    color = _object_color(grid)
    H, W = len(grid), len(grid[0])
    bbox = (0, 0, H - 1, W - 1)
    if kind == "translate_dx":
        return MacroOption(object_id=0, kind="translate", dx=1, dy=0,
                           bbox=bbox, area=0, color=color,
                           description="translate(dx=1, dy=0)")
    if kind == "translate_dy":
        return MacroOption(object_id=0, kind="translate", dx=0, dy=1,
                           bbox=bbox, area=0, color=color,
                           description="translate(dx=0, dy=1)")
    if kind == "rotate90":
        return MacroOption(object_id=0, kind="rotate", angle_deg=90,
                           bbox=bbox, area=0, color=color,
                           description="rotate(90)")
    raise ValueError(kind)


def _transform_via_option(grid: List[List[int]], name: str) -> List[List[int]]:
    if name == "translation":
        return _apply_option_to_grid(grid, _option_for("translate_dx", grid))
    if name == "rotation":
        return _apply_option_to_grid(grid, _option_for("rotate90", grid))
    raise ValueError(name)


def _eval_pairs(name: str, seeds: Sequence[int] = (0, 1, 2)) -> List[Tuple[List[List[int]], List[List[int]]]]:
    pairs: List[Tuple[List[List[int]], List[List[int]]]] = []
    for seed in seeds:
        g = _single_object_grid(size=8, color=1 + (seed % 3))
        pairs.append((g, _transform_via_option(g, name)))
    return pairs


def _canonical_receipt(axioms: torch.Tensor) -> Dict[str, Any]:
    """Per-block (8-dim) unit-norm integrity receipt for the 11 axioms."""
    flat = axioms.detach().cpu().float()  # [N, num_blocks, 8]
    per_block = flat.reshape(-1, 8)
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


def _build_orch(device: str, seed: int) -> HenriSwarmOrchestrator:
    """Build the production orchestrator with a pinned RNG seed.

    Called with the same seed twice -> identical untrained twins; the
    treatment then trains its transition, the control does not.
    """
    torch.manual_seed(seed)
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
    return orch


def _write(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    assert os.environ.get("HENRI_ARC_JEPA_TRANSITION_TRAINING", "0") == "1", \
        "flag must be ON"
    out_path = os.environ.get("HENRI_JEPA_OUT", "/tmp/jepa_training_result.json")
    torch.manual_seed(20260814)
    t0 = time.time()

    orch_t = _build_orch(device, seed=20260814)   # treatment (trained)
    orch_c = _build_orch(device, seed=20260814)   # control (untrained twin)
    tokenizer = _masked_ramp_encoder(device)
    psg_t = ProgressiveSemanticGroundingEngine(
        orch_t.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )
    psg_c = ProgressiveSemanticGroundingEngine(
        orch_c.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )
    probe_t = JepaTransitionTrainingProbe(
        orch_t.planner, tokenizer, psg_t, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )
    probe_c = JepaTransitionTrainingProbe(
        orch_c.planner, tokenizer, psg_c, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )
    print(f"[jepa-train] built in {time.time()-t0:.1f}s", flush=True)

    # G0: canonical 11-axiom boundary (fail-closed; NO fallback).
    try:
        axioms, axiom_summary = load_boundary_axioms(
            num_blocks=SCALE["num_blocks"])
    except BoundaryAxiomLoadError as exc:
        result = {
            "schema_id": "henri.jepa-transition-training-matrix.v1",
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
    g1_boundary_ok = (
        receipt["per_block_norm_min"] >= 1.0 - 1e-6
        and receipt["per_block_norm_max"] <= 1.0 + 1e-6
    )
    print(f"[jepa-train] axioms {receipt['num_axioms']} "
          f"norm[{receipt['per_block_norm_min']:.7f},{receipt['per_block_norm_max']:.7f}] "
          f"sha={receipt['payload_sha256'][:16]}", flush=True)

    # Training set (real encoder waves, production option path).
    pairs_res = probe_t.build_pairs(seeds=(0, 1, 2))
    print(f"[jepa-train] pairs={len(pairs_res['pairs'])} skipped={pairs_res['skipped']}",
          flush=True)

    # Train the TREATMENT transition (production API, Sagnac loss).
    train = probe_t.train_transition(pairs_res["pairs"], k_steps=TRAIN_K, lr=TRAIN_LR)
    print(f"[jepa-train] trained K={TRAIN_K} "
          f"loss_final={train['loss_final']:.6f} "
          f"loss_min={train['loss_min']:.6f} "
          f"loss_ema_final={train['loss_ema_final']:.6f} "
          f"ema_cps={ {k: round(v,4) for k,v in train['loss_ema_checkpoints'].items()} }",
          flush=True)

    # Evaluation through the COMPLETE production path (both arms).
    transforms = {"translation": "translate(dx=1, dy=0)",
                  "rotation": "rotate(90)"}
    eval_cache = {name: _eval_pairs(name, seeds=(0, 1, 2)) for name in transforms}
    grid = _single_object_grid(size=8, color=1)

    results: Dict[str, Any] = {}
    for arm_id, probe in (("A", probe_t), ("B", probe_c)):
        results[arm_id] = {}
        for name, true_label in transforms.items():
            out = probe.run_arm(
                arm_id, grid, eval_cache[name],
                true_label=true_label, task_id=name,
                boundary_batch=axioms,
            )
            results[arm_id][name] = out
            print(f"[jepa-train] arm={arm_id} {name}: "
                  f"efe_rank={out.get('efe_true_rank')} "
                  f"margin={out.get('efe_margin')} "
                  f"chosen={out.get('chosen_action')} "
                  f"explored={out.get('explored')} "
                  f"vmap={out.get('vmap_agreement'):.2e} "
                  f"decomp={out.get('decomposition_max_diff'):.2e}",
                  flush=True)

    # Gates.
    g0_decomp = all(
        results[a][t].get("decomposition_max_diff", 1.0) <= 1e-4
        for a in ("A", "B") for t in transforms)
    g0_vmap = all(
        results[a][t].get("vmap_agreement", 1.0) <= 1e-5
        for a in ("A", "B") for t in transforms)
    g0 = g1_boundary_ok and g0_decomp and g0_vmap

    g1 = train.get("loss_ema_final", 1.0) < GATE_LOSS
    spec_secondary = train.get("loss_final", 1.0) <= SPEC_LOSS

    a_ranks = {t: results["A"][t].get("efe_true_rank") for t in transforms}
    b_ranks = {t: results["B"][t].get("efe_true_rank") for t in transforms}
    rank_pass = g1 and all(r == RANK_PASS for r in a_ranks.values())
    rank_partial = (
        g1
        and all(r is not None and r <= RANK_PARTIAL for r in a_ranks.values())
        and any(a_ranks[t] < b_ranks[t] for t in transforms)
    )

    if not g0:
        verdict = "BLOCKED_INFRASTRUCTURE"
    elif rank_pass:
        verdict = "JEPA_ALIGNMENT_PASS"
    elif rank_partial:
        verdict = "JEPA_PARTIAL_SIGNAL"
    else:
        verdict = "JEPA_TRAINING_INERT"

    result = {
        "schema_id": "henri.jepa-transition-training-matrix.v1",
        "verdict": verdict,
        "gates": {
            "G0_infra": g0,
            "G0_boundary_integrity": g1_boundary_ok,
            "G0_decomposition_consistency": g0_decomp,
            "G0_loop_vmap_identity": g0_vmap,
            "G1_loss_ema_lt_030": g1,
            "G1_loss_ema_final": train.get("loss_ema_final"),
            "G1_spec_secondary_loss_le_015": spec_secondary,
            "G2_rank_pass_armA": rank_pass,
            "G2_rank_partial_armA": rank_partial,
            "G2_ranks_armA": a_ranks,
            "G2_ranks_armB": b_ranks,
            "gate_spec": {
                "loss_ema_threshold": GATE_LOSS,
                "spec_loss_marker": SPEC_LOSS,
                "rank_pass": RANK_PASS,
                "rank_partial": RANK_PARTIAL,
                "k_steps": TRAIN_K, "lr": TRAIN_LR,
            },
        },
        "training": train,
        "training_pairs": {
            "num_pairs": len(pairs_res["pairs"]),
            "kinds": list(TRAIN_KINDS),
            "skipped": pairs_res["skipped"],
        },
        "canonical_boundary_receipt": receipt,
        "arms": results,
        "config": {
            "d_model": SCALE["d_model"], "num_blocks": SCALE["num_blocks"],
            "device": device, "flag": "HENRI_ARC_JEPA_TRANSITION_TRAINING",
            "encoder": "mask_ramp_incommensurate", "seed": 20260814,
            "protocol_sources": {
                "r3_spec_sha256": "ecbc1a5a5de5e5aa291a8b42cfc0ff10341a5d7c231ea84003bf97076cff85a4",
                "wavejepa_txt_sha256": "5820021ff94a63b0c530a69fc01de331b85d560a06f2dfbc49dc9cd2027e8f89",
                "synthesis_report_sha256": "861a3845ba5030fc069b460807e386b2c603997cbb7feeb3bf79f256401790af",
            },
        },
        "score_eligible": False,
        "diagnostic_only": True,
        "authorizes_rollout": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _write(out_path, result)
    print(f"[jepa-train] VERDICT={verdict} elapsed={result['elapsed_s']}s",
          flush=True)


if __name__ == "__main__":
    main()
