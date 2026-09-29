"""SpatialMatrixAdapter — typed ingress for 2D integer grids (Directive 1).

WHY THIS EXISTS
===============
`stage0_seeding_run.py` (line 581) states its learner consumes "byte sequences,
not grids": `TapeLearner` takes `[B, T]` integers over `VOCAB = 257`. ARC-style
tasks arrive as 2D matrices `X in {0..15}^{H x W}`, `H, W <= 30`. The two have no
common type. This module supplies the MISSING TYPED ADAPTER between them.

WHAT IT IS NOT
==============
It does NOT wire grids into the byte learner. MEASURED 2026-09-28 (Directive 4,
`d4_optimizer_sweep.json`, HARNESS_OK): raising capacity 8x made held-out loss
WORSE, LR/schedule changes moved it < 0.0012 nats, and the control sits 0.00136
above the matched-data bigram floor -- while a capacity-matched WITH-CONTEXT
model improved by 0.0115. The plateau is therefore NOT an input-representation
problem, and NOT an optimiser problem. Claiming that this adapter "fixes" the
plateau would be a false claim. The adapter is delivered as a correct, typed,
default-off interface; no such claim is made.

It also does NOT re-implement grid encoding. `HENRIVisionEncoder.encode_grid`
already maps a grid to a real D=65,536 unit wave with parity-contour weighting.
This adapter VALIDATES and DISPATCHES to that encoder; it does not duplicate it.

CONTRACT
========
  * input   : 2D integer matrix (list/np.ndarray/torch.Tensor), values 0..15
  * output  : flat real wave [D] on S^{D-1}, plus a [num_blocks, 8] planner view
  * errors  : fail-closed `SpatialIngressError` on any malformed input
  * mixing  : `collate_mixed` proves sequential strings and 2D grids coexist in
              one batch with zero type errors and no silent coercion
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

__all__ = ["SpatialIngressError", "SpatialMatrixAdapter", "MixedBatch"]


class SpatialIngressError(RuntimeError):
    """Raised for any malformed spatial or sequential ingress item."""


# ARC-AGI palette: 16 discrete colours (0..15 inclusive).
N_COLOURS = 16
MAX_GRID_DIM = 30


def _as_int_matrix(item: Any) -> np.ndarray:
    """Coerce to a validated [H, W] int64 matrix. Fail-closed on anything else."""
    if isinstance(item, torch.Tensor):
        if item.is_complex():
            raise SpatialIngressError("grid must be real-valued, got complex")
        arr = item.detach().cpu().numpy()
    elif isinstance(item, np.ndarray):
        arr = item
    elif isinstance(item, (list, tuple)):
        try:
            arr = np.asarray(item)
        except Exception as exc:                                   # noqa: BLE001
            raise SpatialIngressError(f"grid not array-like: {exc}") from exc
    else:
        raise SpatialIngressError(
            f"grid must be list/tuple/ndarray/Tensor, got {type(item).__name__}")

    if arr.ndim != 2:
        raise SpatialIngressError(
            f"grid must be 2-D [H,W]; got ndim={arr.ndim} shape={arr.shape}")
    h, w = arr.shape
    if h < 1 or w < 1:
        raise SpatialIngressError(f"grid must be non-empty; got shape={(h, w)}")
    if h > MAX_GRID_DIM or w > MAX_GRID_DIM:
        raise SpatialIngressError(
            f"grid {h}x{w} exceeds the {MAX_GRID_DIM}x{MAX_GRID_DIM} ARC bound")
    if not np.issubdtype(arr.dtype, np.integer):
        if np.issubdtype(arr.dtype, np.floating):
            if not np.all(np.isfinite(arr)):
                raise SpatialIngressError("grid contains non-finite values")
            if not np.all(arr == np.floor(arr)):
                raise SpatialIngressError(
                    "grid must be integral; got fractional colour values")
            arr = arr.astype(np.int64)
        else:
            raise SpatialIngressError(f"grid dtype {arr.dtype} is not integral")
    if arr.min() < 0 or arr.max() >= N_COLOURS:
        raise SpatialIngressError(
            f"grid colours must lie in [0,{N_COLOURS - 1}]; "
            f"got min={int(arr.min())} max={int(arr.max())}")
    return np.ascontiguousarray(arr, dtype=np.int64)


def _as_token_ids(item: Any) -> torch.Tensor:
    """Coerce to a validated 1-D int64 token sequence. Fail-closed."""
    if isinstance(item, torch.Tensor):
        t = item.detach().cpu()
    elif isinstance(item, np.ndarray):
        t = torch.as_tensor(item)
    elif isinstance(item, (list, tuple)):
        if item and isinstance(item[0], (list, tuple, np.ndarray, torch.Tensor)):
            t = torch.as_tensor(np.asarray(item))
        else:
            t = torch.as_tensor(list(item))
    else:
        raise SpatialIngressError(
            f"sequence must be list/tuple/ndarray/Tensor, got {type(item).__name__}")
    if t.is_complex():
        raise SpatialIngressError("sequence must be real-valued, got complex")
    if t.ndim != 1:
        raise SpatialIngressError(
            f"sequence must be 1-D [T]; got ndim={t.ndim} shape={tuple(t.shape)}")
    if t.numel() == 0:
        raise SpatialIngressError("sequence must be non-empty")
    if t.dtype not in (torch.int64, torch.int32, torch.uint8, torch.int16):
        if t.dtype.is_floating_point:
            if not torch.isfinite(t).all():
                raise SpatialIngressError("sequence contains non-finite values")
            if not torch.all(t == t.floor()):
                raise SpatialIngressError("sequence must be integral token ids")
        t = t.to(torch.int64)
    -1  # explicit no-op; keeps the int64 guarantee below
    t = t.to(torch.int64)
    if int(t.min()) < 0:
        raise SpatialIngressError(
            f"token ids must be non-negative; got min={int(t.min())}")
    return t


class MixedBatch:
    """A batch holding BOTH sequential strings and 2D spatial matrices.

    `kinds[i]` is "seq" or "grid". No item is ever coerced across kinds, so a
    consumer cannot silently receive a grid where it expected a token sequence.
    """

    def __init__(self, seqs: List[torch.Tensor], grids: List[np.ndarray],
                 waves: List[torch.Tensor], kinds: List[str]):
        self.seqs = seqs
        self.grids = grids
        self.waves = waves
        self.kinds = kinds

    def __len__(self) -> int:
        return len(self.kinds)

    def counts(self) -> Dict[str, int]:
        return {"seq": self.kinds.count("seq"), "grid": self.kinds.count("grid")}

    def to_token_batch(self, pad_id: int = 0) -> torch.Tensor:
        """Right-pad ONLY the sequential items into [n_seq, T_max]."""
        if not self.seqs:
            raise SpatialIngressError("no sequential items in this batch")
        tmax = max(int(s.numel()) for s in self.seqs)
        out = torch.full((len(self.seqs), tmax), int(pad_id), dtype=torch.int64)
        for i, s in enumerate(self.seqs):
            out[i, : s.numel()] = s
        return out

    def to_wave_batch(self) -> torch.Tensor:
        """Stack the encoded waves into [n_grid, D]."""
        if not self.waves:
            raise SpatialIngressError("no spatial items in this batch")
        return torch.stack(self.waves, dim=0)


class SpatialMatrixAdapter:
    """Typed 2D-grid -> high-dimensional phase-vector ingress.

    Reuses `HENRIVisionEncoder` for the actual wave construction (no duplicated
    grid mathematics). The adapter owns validation, shaping, and the mixed-batch
    contract.
    """

    def __init__(self, d_model: int = 65536, num_blocks: int = 8192,
                 device: str = "cpu", encoder: Any = None):
        # MEASURED GEOMETRY (henri_vision_encoder.py: `encode_grid` returns a
        # real [d_model] wave, built from d_model//2 complex phases; then
        # `encode_spatial_grid` does `.view(1, k_blocks, block_dim)` with
        # block_dim=8. So the planner boundary contract is
        #     d_model == num_blocks * 8
        # (checked at 65536 = 8192*8). An earlier draft of this file asserted
        # `num_blocks * 8 * 2`, which was wrong and was caught by running the
        # production-scale test.
        if d_model % 2 != 0:
            raise SpatialIngressError("d_model must be even (real/imag halves)")
        if d_model != num_blocks * 8:
            raise SpatialIngressError(
                f"d_model {d_model} must equal num_blocks*8 = {num_blocks * 8}")
        self.d_model = int(d_model)
        self.num_blocks = int(num_blocks)
        self.device = str(device)
        if encoder is None:
            from henri_vision_encoder import HENRIVisionEncoder
            encoder = HENRIVisionEncoder(d_model=self.d_model,
                                         k_blocks=self.num_blocks,
                                         device=self.device)
        self.encoder = encoder

    # ------------------------------------------------------------------ single
    def to_wave(self, grid: Any) -> torch.Tensor:
        """Validated grid -> flat unit wave [d_model]."""
        arr = _as_int_matrix(grid)
        wave = self.encoder.encode_grid(arr)
        wave = torch.as_tensor(wave).to(torch.float32).reshape(-1)
        if wave.numel() != self.d_model:
            raise SpatialIngressError(
                f"encoder returned width {wave.numel()}, expected {self.d_model}")
        if not torch.isfinite(wave).all():
            raise SpatialIngressError("encoder produced non-finite wave")
        return wave

    def to_planner_wave(self, grid: Any) -> torch.Tensor:
        """Validated grid -> [num_blocks, 8] view for the planner boundary."""
        return self.to_wave(grid).reshape(self.num_blocks, 8)

    # ------------------------------------------------------------------- batch
    def collate_mixed(self, items: Iterable[Any]) -> MixedBatch:
        """Build a batch from a MIXED iterable of token sequences and 2D grids.

        Dispatch is by shape and dtype, and it is EXPLICIT: a 2-D item is a grid,
        a 1-D item is a sequence. Anything else raises. Nothing is coerced across
        kinds, which is exactly the "zero type errors" property Directive 1 asks
        for -- enforced, not assumed.
        """
        seqs: List[torch.Tensor] = []
        grids: List[np.ndarray] = []
        waves: List[torch.Tensor] = []
        kinds: List[str] = []
        for i, item in enumerate(items):
            try:
                arr = np.asarray(item) if not isinstance(
                    item, (torch.Tensor, np.ndarray)) else (
                    item.detach().cpu().numpy()
                    if isinstance(item, torch.Tensor) else item)
            except Exception as exc:                                # noqa: BLE001
                raise SpatialIngressError(f"item {i}: unreadable ({exc})") from exc
            ndim = arr.ndim
            if ndim == 2:
                g = _as_int_matrix(item)
                grids.append(g)
                waves.append(self.to_wave(item))
                kinds.append("grid")
            elif ndim == 1:
                seqs.append(_as_token_ids(item))
                kinds.append("seq")
            else:
                raise SpatialIngressError(
                    f"item {i}: ndim={ndim} is neither a 1-D sequence nor a 2-D grid")
        return MixedBatch(seqs=seqs, grids=grids, waves=waves, kinds=kinds)

    def describe(self) -> Dict[str, object]:
        return {
            "schema": "henri.spatial-matrix-adapter.v1",
            "d_model": self.d_model,
            "num_blocks": self.num_blocks,
            "device": self.device,
            "grid_kind": "2-D integer matrix, values in [0,15]",
            "sequence_kind": "1-D integer token ids, VOCAB-compatible",
            "dispatch_rule": "ndim==2 -> grid; ndim==1 -> sequence; else raise",
            "encoder": type(self.encoder).__name__,
            "does_not_claim": ("this adapter does NOT raise the Stage-0 loss "
                               "ceiling; Directive 4 measured the ceiling to be "
                               "the absence of context, not the input encoding"),
        }
