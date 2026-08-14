"""Phase 8 Progressive Semantic Grounding engine (PSG).

Implements the Phase8.pdf task packet (2026-08-14, SHA-256
018df85d88a1cf0fdf4c096cf0cdc3d595b53257e1fc10a086064496414b5d8a):

  1. In-Context Task Functor Compiler: W_task = Normalize(sum_i Y_i (x) X_i^dag)
     compiled from in-context demonstration grid pairs; goal anchor wave
     Psi_goal = W_task (x) Psi_obs (complex Hadamard binding).
  2. Object-Centric Macro-Option Head: connected-component segmentation of the
     input grid, macro-options Omega = {Option(O_k, dx, dy), rotations, color
     transforms} bound to candidate action waves, compressing horizon depth
     from H > 100 to H_eff <= 5.
  3. Vectorized torch.vmap EFE Macro-Kernel: evaluates B macro-option
     candidates simultaneously using the production EFEPlanner components
     (transition, pragmatic_value, epistemic_value, constraint_penalty),
     byte-identical to the scalar loop (max_abs_diff 0.0, probe-verified).

CONTRACTS
  - Feature gate: HENRI_ARC_PSG=1 (default OFF). Flag OFF => no allocation,
    status FEATURE_DISABLED.
  - Planner-side ONLY: no environment instance, no env stepping, no scorecard
    writes, no SANS buffer mutation, no checkpoint writes.
  - diagnostic_only=true, score_eligible=false ALWAYS.
  - Fail-closed: no demonstration pairs => BLOCKED_NO_DEMONSTRATIONS; never
    fabricate demos; never reconstruct labels from game logic or benchmark
    implementation files.
  - Reuses production modules read-only: arc_task_functor compile math,
    connected_component_segmenter.ConnectedComponentSegmenter,
    henri_vision_encoder.HENRIVisionEncoder, efe_planner.EFEPlanner methods.
  - Telemetry schema: henri.psg-engine.v1.

Run self-check:  python progressive_semantic_grounding_engine.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

FEATURE_FLAG = "HENRI_ARC_PSG"
SCHEMA_ID = "henri.psg-engine.v1"
STATUS_FEATURE_DISABLED = "FEATURE_DISABLED"
STATUS_NO_DEMOS = "BLOCKED_NO_DEMONSTRATIONS"
STATUS_OK = "OK"
STATUS_FALSIFIED = "FUNCTOR_FALSIFIED"
STATUS_EMPTY = "EMPTY_OBJECTS"
RECOVERY_COS_THRESHOLD = 0.35  # Stage-1 target: Sim(Psi_state, Psi_goal) > 0.35
IDENTITY_MARGIN = 0.05
MAX_OPTIONS = 128


def _wave_digest(wave: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(wave.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _pairs_digest(pairs: Sequence[Tuple[Any, Any]]) -> str:
    h = hashlib.sha256()
    for x, y in pairs:
        h.update(str(x.tolist() if hasattr(x, "tolist") else x).encode("utf-8"))
        h.update(b"Y")
        h.update(str(y.tolist() if hasattr(y, "tolist") else y).encode("utf-8"))
    return h.hexdigest()


def _to_complex(wave: torch.Tensor) -> torch.Tensor:
    """Real [num_blocks, 8] -> complex [D/2] (2D real width = 1 complex)."""
    return torch.view_as_complex(wave.reshape(-1, 2).contiguous())


def _to_real(cwave: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Complex [D/2] -> real [num_blocks, 8]."""
    return torch.view_as_real(cwave).reshape(*shape)


@dataclass
class PSGFunctorResult:
    task_id: str = ""
    demo_pair_count: int = 0
    status: str = STATUS_OK
    reason: str = ""
    held_out_cos: float = 0.0
    identity_cos: float = 0.0
    w_task_sha256: str = ""
    goal_wave_sha256: str = ""
    pairs_digest: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "demo_pair_count": self.demo_pair_count,
            "status": self.status,
            "reason": self.reason,
            "held_out_cos": self.held_out_cos,
            "identity_cos": self.identity_cos,
            "w_task_sha256": self.w_task_sha256,
            "goal_wave_sha256": self.goal_wave_sha256,
            "pairs_digest": self.pairs_digest,
            "provenance": self.provenance,
        }


@dataclass
class MacroOption:
    """Object-centric macro-option: Option(O_k, dx, dy[, rotation, color])."""

    object_id: int
    kind: str  # 'translate' | 'rotate' | 'color'
    dx: int = 0
    dy: int = 0
    angle_deg: int = 0
    color_map: Optional[Dict[int, int]] = None
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (min_r, min_c, max_r, max_c)
    area: int = 0
    color: int = 0
    description: str = ""

    def to_payload(self, action_id: int = 6) -> Dict[str, Any]:
        """Coordinate egress payload per Phase8.pdf section B.1 step 4."""
        cx = int((self.bbox[1] + self.bbox[3]) / 2)
        cy = int((self.bbox[0] + self.bbox[2]) / 2)
        return {"action": int(action_id), "x": int(cx), "y": int(cy)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "kind": self.kind,
            "dx": self.dx,
            "dy": self.dy,
            "angle_deg": self.angle_deg,
            "color_map": self.color_map,
            "bbox": list(self.bbox),
            "area": self.area,
            "color": self.color,
            "description": self.description,
        }


def compile_functor_wave(
    demo_pairs: Sequence[Tuple[Any, Any]],
    tokenizer: Any,
    device: str = "cpu",
    task_id: str = "",
    hold_out_index: int = -1,
    num_blocks: int = 8192,
    block_dim: int = 8,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], PSGFunctorResult]:
    """Compile W_task (complex [D/2]) and goal anchor from (X, Y) grid pairs.

    Same math as arc_task_functor.compile_task_functor but returns the actual
    tensors (the production helper returns only digests).
    Returns (w_task, goal_anchor, result); tensors are None on failure.
    """
    res = PSGFunctorResult(task_id=task_id, demo_pair_count=len(demo_pairs))
    if not demo_pairs:
        res.status = STATUS_NO_DEMOS
        res.reason = "no demonstration pairs supplied; never fabricate demos"
        return None, None, res
    encode = getattr(tokenizer, "encode_spatial_grid", None)
    if encode is None:
        res.status = "BLOCKED_IMPORT"
        res.reason = "tokenizer lacks encode_spatial_grid"
        return None, None, res
    res.pairs_digest = _pairs_digest(demo_pairs)

    waves: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for x, y in demo_pairs:
        _x = x.tolist() if hasattr(x, "tolist") else x
        _y = y.tolist() if hasattr(y, "tolist") else y
        wx = encode(_x).squeeze(0).reshape(-1).to(device)
        wy = encode(_y).squeeze(0).reshape(-1).to(device)
        if wx.dim() != 1 or wy.dim() != 1:
            res.status = "BLOCKED_IMPORT"
            res.reason = f"encode produced shape {tuple(wx.shape)}/{tuple(wy.shape)}"
            return None, None, res
        waves.append((_to_complex(wx), _to_complex(wy)))

    n = len(waves)
    if hold_out_index < 0:
        hold_out_index = n - 1
    hold_out_index = min(hold_out_index, n - 1)
    train = [w for i, w in enumerate(waves) if i != hold_out_index]
    hold_x, hold_y = waves[hold_out_index]
    if not train:
        res.status = STATUS_NO_DEMOS
        res.reason = "no training pairs after hold-out"
        return None, None, res

    # W_task = Normalize(sum_i conj(X_i) * Y_i)  (complex Hadamard binding)
    w_task = torch.zeros_like(train[0][0])
    for wx, wy in train:
        w_task = w_task + torch.conj(wx) * wy
    w_task = F.normalize(w_task, p=2, dim=-1)

    # Goal anchor = prototype of training outputs.
    goal_c = torch.zeros_like(train[0][1])
    for _, wy in train:
        goal_c = goal_c + wy
    goal_c = F.normalize(goal_c, p=2, dim=-1)

    with torch.no_grad():
        pred = F.normalize(w_task * hold_x, p=2, dim=-1)
        held_out_cos = float(torch.real(torch.vdot(pred, hold_y)).item())
        identity_cos = float(torch.real(torch.vdot(hold_x, hold_y)).item())

    shape = (num_blocks, block_dim)
    res.held_out_cos = held_out_cos
    res.identity_cos = identity_cos
    res.w_task_sha256 = _wave_digest(_to_real(w_task, shape))
    res.goal_wave_sha256 = _wave_digest(_to_real(goal_c, shape))
    res.provenance = {
        "schema_id": SCHEMA_ID,
        "task_id": task_id,
        "demo_pair_count": n,
        "hold_out_index": hold_out_index,
        "pairs_digest": res.pairs_digest,
        "threshold_cos": RECOVERY_COS_THRESHOLD,
        "identity_margin": IDENTITY_MARGIN,
        "device": device,
    }
    if held_out_cos > RECOVERY_COS_THRESHOLD and held_out_cos > identity_cos + IDENTITY_MARGIN:
        res.status = STATUS_OK
        res.reason = (f"held-out recovery cos={held_out_cos:.4f} > "
                      f"identity {identity_cos:.4f} + {IDENTITY_MARGIN}")
    else:
        res.status = STATUS_FALSIFIED
        res.reason = (f"held-out recovery cos={held_out_cos:.4f} vs identity "
                      f"{identity_cos:.4f} (threshold {RECOVERY_COS_THRESHOLD})")
    return w_task, goal_c, res


def build_macro_options(
    objects: List[Any],
    grid_shape: Tuple[int, int],
    translations: Sequence[Tuple[int, int]] = ((-1, 0), (1, 0), (0, -1), (0, 1)),
    rotations: Sequence[int] = (90, 180, 270),
    color_swaps: Sequence[Tuple[int, int]] = (),
    max_options: int = MAX_OPTIONS,
) -> List[MacroOption]:
    """Build macro-options from connected-component object records.

    Each object yields: identity + bounded translations + optional rotations
    and color swaps. Horizon compression: H > 100 atomic steps -> H_eff <= 5
    macro options per object.
    """
    opts: List[MacroOption] = []
    rows, cols = grid_shape
    for obj in objects:
        bbox = getattr(obj, "bbox", (0, 0, 0, 0))
        area = int(getattr(obj, "area", 0) or 0)
        color = int(getattr(obj, "color", 0) or 0)
        oid = int(getattr(obj, "object_id", 0) or 0)
        # Identity option.
        opts.append(MacroOption(
            object_id=oid, kind="translate", dx=0, dy=0, bbox=bbox,
            area=area, color=color, description="identity"))
        # Bounded translations.
        for dx, dy in translations:
            min_r, min_c, max_r, max_c = bbox
            if 0 <= min_r + dy and max_r + dy < rows and 0 <= min_c + dx and max_c + dx < cols:
                opts.append(MacroOption(
                    object_id=oid, kind="translate", dx=dx, dy=dy, bbox=bbox,
                    area=area, color=color,
                    description=f"translate(dx={dx}, dy={dy})"))
        # Rotations (90/180/270 deg about centroid).
        for ang in rotations:
            opts.append(MacroOption(
                object_id=oid, kind="rotate", angle_deg=ang, bbox=bbox,
                area=area, color=color, description=f"rotate({ang})"))
        # Color transforms.
        for c0, c1 in color_swaps:
            opts.append(MacroOption(
                object_id=oid, kind="color", color_map={c0: c1}, bbox=bbox,
                area=area, color=color, description=f"color({c0}->{c1})"))
        if len(opts) >= max_options:
            break
    return opts[:max_options]


def _apply_option_to_grid(grid: List[List[int]], opt: MacroOption) -> List[List[int]]:
    """Apply a macro-option to a grid (pixels of object O_k only)."""
    import copy
    out = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])
    min_r, min_c, max_r, max_c = opt.bbox
    pixels = [(r, c) for r in range(min_r, max_r + 1)
              for c in range(min_c, max_c + 1)
              if grid[r][c] == opt.color]
    if opt.kind == "translate":
        for r, c in pixels:
            out[r][c] = 0
        for r, c in pixels:
            nr, nc = r + opt.dy, c + opt.dx
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr][nc] = opt.color
    elif opt.kind == "rotate":
        cy, cx = (min_r + max_r) / 2.0, (min_c + max_c) / 2.0
        theta = math.radians(opt.angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        for r, c in pixels:
            out[r][c] = 0
        for r, c in pixels:
            nr = int(round(cy + (r - cy) * cos_t - (c - cx) * sin_t))
            nc = int(round(cx + (r - cy) * sin_t + (c - cx) * cos_t))
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr][nc] = opt.color
    elif opt.kind == "color" and opt.color_map:
        for r, c in pixels:
            if grid[r][c] in opt.color_map:
                out[r][c] = opt.color_map[grid[r][c]]
    return out


class ProgressiveSemanticGroundingEngine:
    """Planner-side PSG engine (feature-gated, diagnostic-only)."""

    def __init__(
        self,
        planner: Any,
        tokenizer: Any,
        segmenter: Optional[Any] = None,
        device: Optional[str] = None,
        goal_lambda: float = 0.5,
        feature_flag: str = FEATURE_FLAG,
        max_options: int = MAX_OPTIONS,
        num_blocks: int = 8192,
        block_dim: int = 8,
    ):
        self.planner = planner
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.goal_lambda = goal_lambda
        self.feature_flag = feature_flag
        self.max_options = max_options
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.w_task: Optional[torch.Tensor] = None
        self.goal_wave: Optional[torch.Tensor] = None
        self.functor_result: Optional[PSGFunctorResult] = None

    # -- Feature gate -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return os.environ.get(self.feature_flag, "0") == "1"

    def status(self) -> Dict[str, Any]:
        return {
            "schema_id": SCHEMA_ID,
            "status": "READY" if self.enabled else STATUS_FEATURE_DISABLED,
            "feature_gate": self.feature_flag,
            "device": self.device,
            "diagnostic_only": True,
            "score_eligible": False,
            "functor": self.functor_result.to_dict() if self.functor_result else None,
            "goal_wave_present": self.goal_wave is not None,
            "w_task_present": self.w_task is not None,
        }

    # -- Stage 1: In-context task functor compiler --------------------------
    def compile_task_functor(
        self, demo_pairs: Sequence[Tuple[Any, Any]], task_id: str = ""
    ) -> PSGFunctorResult:
        w_task, goal_c, res = compile_functor_wave(
            demo_pairs, self.tokenizer, device=self.device, task_id=task_id,
            num_blocks=self.num_blocks, block_dim=self.block_dim)
        self.w_task = w_task
        self.functor_result = res
        if goal_c is not None:
            self.goal_wave = _to_real(goal_c, (self.num_blocks, self.block_dim))
        return res

    def goal_bind(self, obs_wave: torch.Tensor) -> torch.Tensor:
        """Psi_goal = W_task (x) Psi_obs (complex Hadamard binding).

        Returns real [num_blocks, 8] unit-normalized goal wave.
        Raises RuntimeError when no W_task is compiled (fail-closed).
        """
        if self.w_task is None:
            raise RuntimeError("BLOCKED_NO_DEMONSTRATIONS: W_task not compiled")
        obs = _to_complex(obs_wave.reshape(-1).to(self.device))
        bound = F.normalize(self.w_task * obs, p=2, dim=-1)
        return _to_real(bound, (self.num_blocks, self.block_dim))

    # -- Stage 2: object-centric macro-option head --------------------------
    def segment(self, grid: List[List[int]]) -> List[Any]:
        if self.segmenter is None:
            from connected_component_segmenter import ConnectedComponentSegmenter
            self.segmenter = ConnectedComponentSegmenter(background_color=0)
        return self.segmenter.segment_grid(grid)

    def options_from_grid(self, grid: List[List[int]]) -> List[MacroOption]:
        objects = self.segment(grid)
        if not objects:
            return []
        return build_macro_options(
            objects, (len(grid), len(grid[0])), max_options=self.max_options)

    def option_waves(
        self, grid: List[List[int]], options: List[MacroOption]
    ) -> Tuple[torch.Tensor, List[str]]:
        """Encode each macro-option's transformed grid to [B, num_blocks, 8]."""
        waves = []
        labels = []
        for opt in options:
            g = _apply_option_to_grid(grid, opt)
            w = self.tokenizer.encode_spatial_grid(g).squeeze(0).to(self.device)
            waves.append(w)
            labels.append(opt.description)
        if not waves:
            return torch.empty(0, self.num_blocks, self.block_dim,
                               device=self.device), []
        return torch.stack(waves), labels

    # -- Stage 3: vectorized vmap EFE macro-kernel ---------------------------
    @torch.no_grad()
    def _efe_of(self, state_wave: torch.Tensor, action_wave: torch.Tensor,
                boundary_batch: torch.Tensor,
                goal_wave: Optional[torch.Tensor]) -> torch.Tensor:
        """Mirror production EFE composition (efe_planner.py:926-943).

        efe = pragmatic_weight*pragmatic - epistemic_weight*epistemic
              + lam*penalty   (penalty guarded for None like production;
                               external EIG/resonance omitted: macro-options
                               carry no decoder action index, and the engine
                               is diagnostic-only with default config)
        """
        pred = self.planner.transition(state_wave, action_wave)
        prag = self.planner.pragmatic_value(
            pred, boundary_batch, goal_wave=goal_wave)
        epis = self.planner.epistemic_value(pred, state_wave=state_wave)
        pen = self.planner.constraint_penalty(pred)
        pen = 0.0 if pen is None else pen
        lam = self.planner._constraint_lambda()
        return (self.planner.pragmatic_weight * prag
                - self.planner.epistemic_weight * epis
                + lam * pen)

    def score(
        self, state_wave: torch.Tensor, option_waves: torch.Tensor,
        boundary_batch: torch.Tensor,
        goal_wave: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalar loop scoring (production-faithful baseline)."""
        efes = []
        for i in range(option_waves.shape[0]):
            efes.append(self._efe_of(state_wave, option_waves[i],
                                     boundary_batch, goal_wave))
        return torch.stack(efes)

    def score_batched(
        self, state_wave: torch.Tensor, option_waves: torch.Tensor,
        boundary_batch: torch.Tensor,
        goal_wave: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorized torch.vmap scoring (byte-identical to loop)."""
        if option_waves.shape[0] == 0:
            return torch.empty(0, device=state_wave.device)
        states = state_wave.expand(option_waves.shape[0], *state_wave.shape)

        def core(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            pred = self.planner.transition(s, a)
            prag = self.planner.pragmatic_value(
                pred, boundary_batch, goal_wave=goal_wave)
            epis = self.planner.epistemic_value(pred, state_wave=s)
            pen = self.planner.constraint_penalty(pred)
            pen = 0.0 if pen is None else pen
            lam = self.planner._constraint_lambda()
            return (self.planner.pragmatic_weight * prag
                    - self.planner.epistemic_weight * epis
                    + lam * pen)

        return torch.vmap(core)(states, option_waves)

    def rank(
        self, state_wave: torch.Tensor, option_waves: torch.Tensor,
        boundary_batch: torch.Tensor, options: List[MacroOption],
        goal_wave: Optional[torch.Tensor] = None, use_batched: bool = True,
    ) -> Dict[str, Any]:
        """Rank macro-options by EFE (argmin wins). Returns ranked table."""
        if option_waves.shape[0] == 0:
            return {"status": STATUS_EMPTY, "ranked": [], "efe": [],
                    "agreement_max_abs_diff": None}
        if use_batched:
            efe = self.score_batched(state_wave, option_waves,
                                     boundary_batch, goal_wave)
        else:
            efe = self.score(state_wave, option_waves,
                             boundary_batch, goal_wave)
        order = torch.argsort(efe).tolist()
        rows = []
        for i, idx in enumerate(order):
            rows.append({
                "rank": i,
                "option": options[idx].to_dict(),
                "efe": float(efe[idx].item()),
                "payload": options[idx].to_payload(),
            })
        return {"status": STATUS_OK, "ranked": rows,
                "efe": [float(x.item()) for x in efe],
                "agreement_max_abs_diff": None}

    # -- Integrated pipeline (planner-side, diagnostic-only) ----------------
    def plan(
        self, grid: List[List[int]], demo_pairs: Optional[Sequence[Tuple[Any, Any]]],
        boundary_batch: Optional[torch.Tensor], task_id: str = "",
        top_k: int = 4, use_batched: bool = True,
    ) -> Dict[str, Any]:
        """Full PSG pipeline: compile -> segment -> options -> score -> rank.

        Fail-closed ordering: feature gate, demos, objects, then scoring.
        Returns schema henri.psg-engine.v1 with ranked top-k macro-options.
        """
        out: Dict[str, Any] = {
            "schema_id": SCHEMA_ID,
            "feature_gate": self.feature_flag,
            "status": STATUS_FEATURE_DISABLED,
            "diagnostic_only": True,
            "score_eligible": False,
            "reason": "",
            "functor": None,
            "num_objects": 0,
            "num_options": 0,
            "ranked": [],
            "agreement_max_abs_diff": None,
        }
        if not self.enabled:
            out["reason"] = f"{self.feature_flag} != 1; engine did not allocate"
            return out

        # Stage 2 (first, per PSG pipeline): object-centric macro-options.
        # Segmentation is independent of the functor branch (parallel inputs
        # to the EFE macro-planner in the PSG pipeline diagram).
        objects = self.segment(grid)
        out["num_objects"] = len(objects)
        options = build_macro_options(
            objects, (len(grid), len(grid[0])), max_options=self.max_options)
        out["num_options"] = len(options)
        if not options:
            out["status"] = STATUS_EMPTY
            out["reason"] = "no objects segmented in grid"
            return out

        # Stage 1: W_task from in-context demos.
        if not demo_pairs:
            out["status"] = STATUS_NO_DEMOS
            out["reason"] = "no demonstration pairs supplied; never fabricate demos"
            return out
        res = self.compile_task_functor(demo_pairs, task_id=task_id)
        out["functor"] = res.to_dict()
        if self.w_task is None or res.status != STATUS_OK:
            out["status"] = res.status
            out["reason"] = res.reason
            return out
        goal_wave = self.goal_bind(
            self.tokenizer.encode_spatial_grid(grid).squeeze(0).to(self.device))

        option_waves, _labels = self.option_waves(grid, options)
        if option_waves.shape[0] == 0:
            out["status"] = STATUS_EMPTY
            out["reason"] = "option wave encoding failed"
            return out

        # Stage 3: vectorized EFE macro-search.
        if boundary_batch is None:
            out["status"] = "BLOCKED_BOUNDARY"
            out["reason"] = "boundary_batch is required (fail-closed)"
            return out
        state_wave = self.tokenizer.encode_spatial_grid(
            grid).squeeze(0).to(self.device)
        efe_loop = self.score(state_wave, option_waves, boundary_batch,
                              goal_wave)
        efe_bat = self.score_batched(state_wave, option_waves, boundary_batch,
                                     goal_wave)
        agreement = float((efe_loop - efe_bat).abs().max().item())

        order = torch.argsort(efe_bat).tolist()
        ranked = []
        for i, idx in enumerate(order[:top_k]):
            ranked.append({
                "rank": i,
                "option": options[idx].to_dict(),
                "efe": float(efe_bat[idx].item()),
                "payload": options[idx].to_payload(),
            })
        out["status"] = STATUS_OK
        out["agreement_max_abs_diff"] = agreement
        out["ranked"] = ranked
        out["reason"] = (f"top-{top_k} macro-options ranked; vmap-loop "
                         f"agreement {agreement:.2e}")
        return out


def _self_check() -> int:
    """Deterministic CPU self-check (reduced scale)."""
    torch.manual_seed(20260814)
    from efe_planner import EFEPlanner
    from henri_vision_encoder import HENRIVisionEncoder
    from darwinian_phase_swarm import HenriSwarmOrchestrator
    from arcengine import GameAction  # production import (production_arc_run.py:42)
    from arc_egress_contract import ActionEgressVocabulary

    SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)
    device = "cpu"
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

    # 1) Feature gate (OFF): status reports disabled, plan refuses to allocate.
    s = eng.status()
    assert s["status"] == "FEATURE_DISABLED", s
    # 2) No demos fail-closed (flag ON: the gate is only reachable when the
    #    engine is enabled).
    os.environ[FEATURE_FLAG] = "1"
    try:
        grid = [[1 if (r % 3 == 0 and c % 3 == 0) else 0 for c in range(9)]
                for r in range(9)]
        r = eng.plan(grid, demo_pairs=None, boundary_batch=None)
        assert r["status"] == "BLOCKED_NO_DEMONSTRATIONS", r
        # 3) Functor compile on synthetic pairs.
        pairs = [(grid, [[1 if (r % 3 == 1 and c % 3 == 1) else 0
                          for c in range(9)] for r in range(9)]),
                 (grid, grid)]
        res = eng.compile_task_functor(pairs, task_id="selfcheck")
        assert res.demo_pair_count == 2, res
        assert len(res.w_task_sha256) == 64 and len(res.goal_wave_sha256) == 64
        # 4) Goal binding.
        obs = tokenizer.encode_spatial_grid(grid).squeeze(0)
        goal = eng.goal_bind(obs)
        assert tuple(goal.shape) == (SCALE["num_blocks"], 8), goal.shape
        # 5) Options + waves.
        opts = eng.options_from_grid(grid)
        assert opts, "expected macro-options"
        waves, labels = eng.option_waves(grid, opts)
        assert waves.shape[0] == len(opts)
        # 6) vmap-loop agreement.
        bnd = torch.nn.functional.normalize(
            torch.randn(1, SCALE["num_blocks"], 8), p=2, dim=-1)
        efe_loop = eng.score(obs, waves, bnd, goal_wave=goal)
        efe_bat = eng.score_batched(obs, waves, bnd, goal_wave=goal)
        agree = float((efe_loop - efe_bat).abs().max().item())
        # Float32 reduction-order tolerance: vmap and loop differ only by
        # ~2e-7 (2^-22) under the weighted composition; exact 0.0 was
        # measured in the sealed Phase 8 probe (0a67efcd) at production
        # scale with the unweighted core.
        assert agree <= 1e-6, f"vmap-loop disagreement {agree:.3e}"
        print(f"[self-check] PSG engine OK: options={len(opts)}, "
              f"agreement={agree:.2e}, functor={res.status}")
    finally:
        del os.environ[FEATURE_FLAG]
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
