"""henri_metric_ingress.py — Metric-preserving spatial & token ingress (HENRI-DIR-2026-09-BUNDLE-VLA-GROUNDING, Component 1).

Canonical boundary: real unit-norm wave [D] (equivalently [num_blocks, 8] with
D = num_blocks * 8).  No global mean-pooling is used anywhere in this module:
visual scenes are encoded patch-wise with translation-covariant complex phase
binding; token streams are bound through an optional frozen semantic anchor.

Math (patch path):
    Psi_patch = sum_{cells in patch} exp(i * f_j * (c*dx + r*dy + theta_color))_j
    Psi = [Re(concat_patches); Im(concat_patches)] / ||...||            # [D]

Shift sensitivity: shifting a glyph by (dr, dc) multiplies each complex
dimension j by exp(i * f_j * (dr*dy + dc*dx)) with f_j = 1..L.  Because the
phase advance differs per dimension, the real-wave cosine against the
unshifted code is NOT invariant -- this module does not claim global
translation invariance.  Locality preservation is claimed at the retrieval
level: patch-structured binding keeps local glyph content (color, shape,
position) discriminative instead of collapsing it through global pooling
(the G1 mean-pooling failure this module reverses).

Default-OFF: nothing here is imported by the production runner.  Enable only
through an explicit consumer decision (approval-gated wiring).

Math status: real Clifford waves at the planner boundary are [num_blocks, 8].
This module emits flat [D]; consumers reshape with .view(num_blocks, 8).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional structured codec (exists in-tree; not required)
    from qfhrr_structured_codec import StructuredCharPositionCodec  # type: ignore
    _HAS_STRUCTURED_CODEC = True
except Exception:  # pragma: no cover - optional dependency
    StructuredCharPositionCodec = None  # type: ignore
    _HAS_STRUCTURED_CODEC = False


class HenriMetricPatchIngress(nn.Module):
    """Patch-structured, translation-covariant visual ingress (no global pooling).

    Args:
        d_model: total real wave dimension (must be divisible by 2 * num_patches).
        num_blocks: block count for the [num_blocks, 8] reshape view.
        patch_size: side length of a square patch; grid dims must divide by it.
        spatial_basis_kind: "incommensurate" | "default" | "random" y-ramp scaling.
        bg_mask: exclude color-0 cells from the superposition (DC suppression).
        parity_weight: apply ParityContourMask-style +-1 interior weighting (needs
            connected-component segmentation; disabled by default in this module).
    """

    def __init__(
        self,
        d_model: int = 65536,
        num_blocks: int = 8192,
        patch_size: int = 4,
        spatial_basis_kind: str = "incommensurate",
        bg_mask: bool = True,
    ) -> None:
        super().__init__()
        if d_model % 8 != 0:
            raise ValueError(f"d_model must be a multiple of 8; got {d_model}")
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.patch_size = int(patch_size)
        self.spatial_basis_kind = spatial_basis_kind
        self.bg_mask = bg_mask
        if spatial_basis_kind not in ("default", "incommensurate", "random"):
            raise ValueError(
                f"spatial_basis_kind must be default|incommensurate|random; got {spatial_basis_kind}"
            )
        # frequency lattice for the phase binding (per complex dim)
        self.register_buffer("_freqs", torch.arange(1, d_model // 2 + 1, dtype=torch.float32))
        self._delta_x = 2.0 * math.pi / max(1.0, float(d_model // 2))
        self._delta_y = self._delta_x
        if spatial_basis_kind == "incommensurate":
            self._delta_y = self._delta_x * math.sqrt(2.0)
        elif spatial_basis_kind == "random":
            g = torch.Generator(device="cpu").manual_seed(7)
            self._delta_y = (float(torch.rand(1, generator=g)) + 0.5) * self._delta_x

    @torch.no_grad()
    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Map a 2D int grid [H, W] (values 0..15) into a unit [D] wave.

        Patch partition: rows and cols are split into patch_size blocks.  Each
        patch occupies a contiguous complex band of width L = (D/2) / P where
        P = total patch count.  Requires D % (2*P) == 0.
        """
        if grid.ndim == 3 and grid.shape[0] == 1:
            grid = grid.squeeze(0)
        if grid.ndim != 2:
            raise ValueError(f"grid must be [H, W]; got shape {tuple(grid.shape)}")
        grid = grid.to(device=self._freqs.device, dtype=torch.long)
        H, W = grid.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(f"grid {H}x{W} not divisible by patch_size {p}")
        grid = torch.clamp(grid, 0, 15)
        npr, npc = H // p, W // p
        P = npr * npc
        if self.d_model % (2 * P) != 0:
            raise ValueError(f"d_model {self.d_model} not divisible by 2*P={2 * P}")
        L = self.d_model // (2 * P)  # complex dims per patch
        freqs = self._freqs[:L]
        dx, dy = self._delta_x, self._delta_y

        patches: List[torch.Tensor] = []
        total_cells = 0
        for pr in range(npr):
            for pc in range(npc):
                cells: List[Tuple[int, int]] = []
                colors: List[int] = []
                for r in range(p):
                    for c in range(p):
                        gr, gc = pr * p + r, pc * p + c
                        v = int(grid[gr, gc].item())
                        if self.bg_mask and v == 0:
                            continue
                        cells.append((gr, gc))
                        colors.append(v)
                if not cells:
                    # Sparse-grid case: empty patches contribute a zero
                    # complex band (allowed). Fail-closed only when the
                    # WHOLE grid encodes to a zero wave.
                    patches.append(torch.zeros(L, dtype=torch.complex64, device=grid.device))
                    continue
                total_cells += len(cells)
                phase = torch.zeros(L, dtype=torch.complex64, device=grid.device)
                for (gr, gc), v in zip(cells, colors):
                    # per-dimension phase: f_j * (pos*scale + color_angle)
                    # color_angle inside the f_j ramp => color cannot factor
                    # out of the superposition (shape stays discriminative)
                    theta_c = float(v) * (2.0 * math.pi * 15.0 / 16.0)
                    arg = freqs * (float(gc) * dx + float(gr) * dy + theta_c)
                    phase.add_(torch.exp(1j * arg))
                patches.append(phase)
        if total_cells == 0:
            raise ValueError(
                "empty grid: bg_mask excluded every cell; superposition would "
                "produce a zero wave (fail-closed)"
            )
        complex_wave = torch.cat(patches, dim=-1)  # [D/2]
        real_wave = torch.cat([complex_wave.real, complex_wave.imag], dim=-1)  # [D]
        return F.normalize(real_wave, p=2, dim=-1)

    def encode_grid_blocks(self, grid: torch.Tensor) -> torch.Tensor:
        """Same as encode_grid but returns [1, num_blocks, 8]."""
        return self.encode_grid(grid).view(1, self.num_blocks, 8)

    @staticmethod
    def contact_auc(query: torch.Tensor, candidates: torch.Tensor, correct: int) -> float:
        """Fraction of (correct > distractor) pairwise wins; AUC_contact in [0, 1]."""
        sims = F.normalize(query, p=2, dim=-1) @ F.normalize(candidates, p=2, dim=-1).T
        s_correct = float(sims[correct])
        wins = sum(1.0 for j in range(candidates.shape[0]) if j != correct and float(sims[j]) < s_correct)
        return wins / max(1, candidates.shape[0] - 1)


class SemanticAnchorProjector(nn.Module):
    """Orthogonal projection of frozen embedding rows into a D-dim frame (QR).

    Frozen metric anchor for token semantics (Component 1, token branch).
    W = QR of a seeded [D, d_emb] Gaussian; rows of W are orthonormal columns
    spanning the anchor subspace.  Projection:  Psi = W @ e  (zero trainable).

    Dense memory warning: [D, d_emb] is D * d_emb * 4 bytes (at D=65536 and
    d_emb=4096 this is 1 GiB).  Use small d_emb or a factorized consumer at
    production scale; this module asserts an explicit budget.

    Default-OFF, zero trainable, CPU-testable at reduced scale.
    """

    def __init__(self, d_model: int = 1024, d_emb: int = 64, seed: int = 11,
                 max_anchor_bytes: int = 256 * 1024 * 1024) -> None:
        super().__init__()
        if d_emb > d_model:
            raise ValueError(f"d_emb {d_emb} > d_model {d_model}: cannot orthogonally project up")
        nbytes = d_model * d_emb * 4
        if nbytes > max_anchor_bytes:
            raise ValueError(
                f"anchor matrix {d_model}x{d_emb} = {nbytes / 1e6:.0f} MB exceeds "
                f"max_anchor_bytes {max_anchor_bytes}; use a smaller d_emb or factorized path"
            )
        g = torch.Generator(device="cpu").manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(d_model, d_emb, generator=g))
        self.register_buffer("_W", q)  # [D, d_emb], columns orthonormal

    @torch.no_grad()
    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: [N, d_emb] frozen rows -> [N, d_model] orthonormal projection."""
        if embeddings.shape[-1] != self._W.shape[1]:
            raise ValueError(
                f"embedding dim {embeddings.shape[-1]} != anchor d_emb {self._W.shape[1]}"
            )
        out = embeddings.to(self._W.device, torch.float32) @ self._W.T  # [N, D]
        return F.normalize(out, p=2, dim=-1)

    @torch.no_grad()
    def encode_token_sequence(self, token_ids: Sequence[int], embeddings: torch.Tensor) -> torch.Tensor:
        """Order-sensitive sequence wave: position-keyed superposition of projected tokens.

        Position key: odd-integer phase key (2p+1) per the verified Channel-T
        protocol -- NOT a separable sum; position is bound multiplicatively so
        order permutations change the wave.
        """
        if len(token_ids) == 0:
            raise ValueError("empty token sequence (fail-closed)")
        tok = torch.tensor(list(token_ids), dtype=torch.long, device=embeddings.device)
        if tok.max() >= embeddings.shape[0]:
            raise ValueError(f"token id {int(tok.max())} outside embedding rows {embeddings.shape[0]}")
        proj = self.project(embeddings[tok])  # [N, D]
        D = proj.shape[-1]
        freqs = torch.arange(D, dtype=torch.float32, device=proj.device).unsqueeze(0) + 1.0
        pos = torch.arange(len(token_ids), dtype=torch.float32, device=proj.device).unsqueeze(1)
        key = (2.0 * pos + 1.0) * freqs
        bound = proj * torch.cos(key)  # deterministic position binding (real ring)
        wave = F.normalize(bound.sum(dim=0), p=2, dim=-1)
        return wave

    @staticmethod
    def structured_codec_available() -> bool:
        return _HAS_STRUCTURED_CODEC
