"""Phase 8.3 R1 — Representation Discrimination engine (default OFF).

Implements Phase8.2.pdf Lens A/B/C (packet SHA-256 a41433e5...): foreground
zero-power masking + independent incommensurate X/Y phase ramps, tested
against the K1 pre-flight ranking gate (true_rank <= 2, true_margin >= +0.05)
for translation, rotation, AND reflection at production scale.

Flag: HENRI_ARC_REPRESENTATION_R1=1 (default OFF -> byte-identical legacy).
Reuses production kernels (compile_functor_wave, goal_bind, options_from_grid,
option_waves, score_batched) from ProgressiveSemanticGroundingEngine; adds a
masked-ramp encoder constructor and a 3-transform K1 harness.

Engine never steps an environment: no game.step, no SANS rows, held-out
untouched, score_eligible=false, diagnostic_only=true, no rollout.
"""

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from progressive_semantic_grounding_engine import (
        ProgressiveSemanticGroundingEngine,
        MacroOption,
        _apply_option_to_grid,
    )
    _IMPORT_OK = True
except Exception as _e:  # pragma: no cover - import guard for reduced tests
    _IMPORT_OK = False
    _IMPORT_ERR = str(_e)


FEATURE_FLAG = "HENRI_ARC_REPRESENTATION_R1"
STATUS_DISABLED = "FEATURE_DISABLED"
STATUS_OK = "OK"
STATUS_NO_DEMOS = "BLOCKED_NO_DEMONSTRATIONS"
STATUS_FALSIFIED = "FUNCTOR_FALSIFIED"
STATUS_EMPTY = "BLOCKED_EMPTY_FOREGROUND"
K1_PASS = "K1_PASS"
K1_FAIL = "K1_FAIL"
MAX_TRANSFORMS = ("translation", "rotation", "reflection")


def _masked_ramp_encoder(
    d_model: int,
    num_blocks: int,
    block_dim: int,
    max_grid_dim: int,
    device: str,
    spatial_basis_kind: str = "incommensurate",
    bg_mask: bool = True,
) -> Any:
    """Construct the production HENRIVisionEncoder with mask+ramp variant.

    Uses the REAL production encoder (henri_vision_encoder.HENRIVisionEncoder)
    with spatial_basis_kind='incommensurate' and bg_mask=True, matching the
    production default from arc_spatial_basis.resolve_spatial_basis().
    """
    from henri_vision_encoder import HENRIVisionEncoder

    return HENRIVisionEncoder(
        d_model=d_model, k_blocks=num_blocks, block_dim=block_dim,
        max_grid_dim=max_grid_dim, device=device,
        spatial_basis_kind=spatial_basis_kind, bg_mask=bg_mask,
    )


def _legacy_encoder(
    d_model: int,
    num_blocks: int,
    block_dim: int,
    max_grid_dim: int,
    device: str,
) -> Any:
    """Legacy collinear unmasked encoder (constructor defaults) — control arm."""
    from henri_vision_encoder import HENRIVisionEncoder

    return HENRIVisionEncoder(
        d_model=d_model, k_blocks=num_blocks, block_dim=block_dim,
        max_grid_dim=max_grid_dim, device=device,
    )


def apply_reflection(grid: List[List[int]], axis: str) -> List[List[int]]:
    """Mirror grid columns ('h') or rows ('v'); returns a fresh copy."""
    out = copy.deepcopy(grid)
    if axis == "h":
        return [row[::-1] for row in out]
    return out[::-1]


def build_transform_options(
    grid: List[List[int]],
    tokenizer: Any,
    device: str = "cpu",
    color: int = 1,
) -> Tuple[torch.Tensor, List[str]]:
    """Controlled candidate set for the R1 K1 gate.

    identity + 4 translations + 3 rotations + 2 reflections (the 3 required
    R1 transforms) applied with the SAME production helpers the functor pairs
    are generated with. Returns (waves [B, num_blocks, block_dim], labels).
    """
    H, W = len(grid), len(grid[0])
    candidates: List[Tuple[List[List[int]], str]] = [(grid, "identity")]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        m = MacroOption(
            object_id=0, kind="translate", dx=dx, dy=dy,
            bbox=(0, 0, H - 1, W - 1), area=0, color=color,
            description=f"translate(dx={dx}, dy={dy})")
        candidates.append((_apply_option_to_grid(grid, m), m.description))
    for ang in (90, 180, 270):
        m = MacroOption(
            object_id=0, kind="rotate", angle_deg=ang,
            bbox=(0, 0, H - 1, W - 1), area=0, color=color,
            description=f"rotate({ang})")
        candidates.append((_apply_option_to_grid(grid, m), m.description))
    for axis, label in (("h", "reflect_h"), ("v", "reflect_v")):
        candidates.append((apply_reflection(grid, axis), label))
    waves, labels = [], []
    for g, lb in candidates:
        w = tokenizer.encode_spatial_grid(g).squeeze(0).to(device)
        waves.append(w)
        labels.append(lb)
    return torch.stack(waves), labels


class RepresentationDiscriminationEngine:
    """R1 kill-gate harness. Default OFF. Never steps an environment."""

    FEATURE_FLAG = FEATURE_FLAG

    def __init__(
        self,
        planner: Any,
        tokenizer: Any,
        device: str = "cpu",
        num_blocks: int = 8192,
        block_dim: int = 8,
        max_options: int = 128,
    ):
        self.device = torch.device(device)
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.max_options = max_options
        self._psg = ProgressiveSemanticGroundingEngine(
            planner, tokenizer, device=self.device,
            num_blocks=num_blocks, block_dim=block_dim,
            max_options=max_options,
        )
        self._k1_results: Dict[str, Dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return os.environ.get(FEATURE_FLAG, "0") == "1"

    # -- API -------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "flag": FEATURE_FLAG,
            "enabled": self.enabled,
            "status": STATUS_DISABLED if not self.enabled else STATUS_OK,
            "score_eligible": False,
            "diagnostic_only": True,
            "rollout_authorized": False,
        }

    def compile_functor(self, demo_pairs, task_id: str = ""):
        """Compile W_task from AUTHORIZED in-context pairs (leave-one-out)."""
        return self._psg.compile_task_functor(demo_pairs, task_id=task_id)

    def k1_gate(self, option_waves, goal_wave, labels, true_label):
        """K1 ranking gate: true_rank <= 2 AND true_margin >= +0.05.

        true_margin = Sim(true, goal) - max_{a != true} Sim(a, goal).
        Fail-closed: returns K1_FAIL dict on any anomaly.
        """
        B = option_waves.shape[0]
        if B == 0:
            return {"status": K1_FAIL, "reason": "no options"}
        sims = F.cosine_similarity(option_waves.reshape(B, -1),
                                   goal_wave.reshape(-1).unsqueeze(0),
                                   dim=1)
        true_idx = next((i for i, lb in enumerate(labels)
                         if lb == true_label), -1)
        if true_idx < 0:
            return {"status": K1_FAIL, "reason": "true label absent",
                    "sims": [float(s) for s in sims]}
        order = torch.argsort(sims, descending=True).tolist()
        true_rank = order.index(true_idx) + 1  # 1-indexed
        false_sims = [float(sims[i].item()) for i in range(B)
                      if i != true_idx]
        margin = float(sims[true_idx].item()) - max(false_sims)
        passed = (true_rank <= 2) and (margin >= 0.05)
        return {
            "status": K1_PASS if passed else K1_FAIL,
            "true_rank": true_rank,
            "true_margin": margin,
            "sim_true": float(sims[true_idx].item()),
            "sim_best_false": max(false_sims),
            "num_options": B,
            "pass": passed,
            "sims": [float(s) for s in sims],
        }

    def goal_sim(self, goal_wave, state_wave) -> float:
        return float(F.cosine_similarity(
            goal_wave.reshape(1, -1), state_wave.reshape(1, -1)).item())

    def _base_out(self, task_id: str, **extra) -> Dict[str, Any]:
        """Discipline telemetry carried by EVERY return path (single source)."""
        out = {
            "transform": task_id,
            "score_eligible": False,
            "diagnostic_only": True,
            "authorizes_rollout": False,
        }
        out.update(extra)
        return out

    def k1_harness(
        self,
        grid: List[List[int]],
        pairs: Sequence[Tuple[Any, Any]],
        true_label: str,
        boundary_batch: Optional[torch.Tensor] = None,
        task_id: str = "",
        color: int = 1,
    ) -> Dict[str, Any]:
        """Run the full production K1 ranking path for one transform.

        Option waves, goal bind, sim-K1 gate, AND vmap-EFE ranking all use the
        production kernels on the SAME encoder instance.
        """
        if not self.enabled:
            return self._base_out(task_id, status=STATUS_DISABLED)
        res = self.compile_functor(pairs, task_id=task_id)
        if res.status != STATUS_OK:
            return self._base_out(task_id, status=res.status,
                                  reason=res.reason)
        state_wave = self._psg.tokenizer.encode_spatial_grid(
            grid).squeeze(0).to(self.device)
        goal_wave = self._psg.goal_bind(state_wave)
        goal_sim_obs = self.goal_sim(goal_wave, state_wave)

        waves, labels = build_transform_options(
            grid, self._psg.tokenizer, str(self.device), color=color)
        true_idx = next((i for i, lb in enumerate(labels)
                         if lb == true_label), -1)
        if true_idx < 0:
            return self._base_out(task_id, status=K1_FAIL,
                                  reason="true label absent",
                                  labels=labels[:20])

        # Boundary: masked encoder REJECTS all-zero grids (fail-closed), so
        # the diagnostic boundary uses a single colored pixel.
        if boundary_batch is None:
            H, W = len(grid), len(grid[0])
            bg = [[0] * W for _ in range(H)]
            bg[0][0] = 1
            boundary_batch = self._psg.tokenizer.encode_spatial_grid(
                bg).squeeze(0).unsqueeze(0).to(self.device)

        # K1 sim gate (PDF Lens B).
        k1 = self.k1_gate(waves, goal_wave, labels, true_label)

        # Full production ranking path: vmap EFE + loop agreement.
        efe_loop = self._psg.score(state_wave, waves, boundary_batch, goal_wave)
        efe_bat = self._psg.score_batched(state_wave, waves, boundary_batch,
                                          goal_wave)
        agreement = float((efe_loop - efe_bat).abs().max().item())
        order = torch.argsort(efe_bat).tolist()
        efe_true_rank = (order.index(true_idx) + 1) if true_idx >= 0 else None

        out = {
            "transform": task_id,
            "functor_status": res.status,
            "held_out_cos": getattr(res, "held_out_cos", None),
            "identity_cos": getattr(res, "identity_cos", None),
            "goal_sim_obs": goal_sim_obs,
            "num_options": len(labels),
            "k1": k1,
            "efe_true_rank": efe_true_rank,
            "efe_agreement_max_abs_diff": agreement,
            "score_eligible": False,
            "diagnostic_only": True,
            "authorizes_rollout": False,
        }
        self._k1_results[task_id] = out
        return out


def _self_check() -> int:
    """Deterministic CPU self-check (reduced scale)."""
    if not _IMPORT_OK:
        raise RuntimeError(f"PSG import failed: {_IMPORT_ERR}")
    torch.manual_seed(20260814)
    from efe_planner import EFEPlanner
    from darwinian_phase_swarm import HenriSwarmOrchestrator
    from arcengine import GameAction  # production import

    SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)
    device = "cpu"
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
    tokenizer = _masked_ramp_encoder(
        d_model=SCALE["d_model"], num_blocks=SCALE["num_blocks"],
        block_dim=8, max_grid_dim=30, device=device,
    )
    eng = RepresentationDiscriminationEngine(
        orch.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )
    assert not eng.enabled, "flag must default OFF"
    s = eng.status()
    assert s["status"] == STATUS_DISABLED, s
    assert s["score_eligible"] is False
    assert s["diagnostic_only"] is True
    assert s["rollout_authorized"] is False
    print("[r1] self-check OK (feature OFF by default; no allocation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
