"""Pillar 2 candidate: a STRUCTURED, ITERATED predictor over psi (replaces flat maps).

DIRECTIVE: HENRI-ARCH-2026-CRITICAL-DIRECTIVE-V1 Lens C Pillar 2 + MVP-LAB
Phase 2: "Implement a deep continuous wave propagator (Wave-JEPA) or an iterative
Tripartite Resonator Network that decomposes operators into three decoupled
factors: Psi_transformed = Psi_spatial (x) Psi_mask (x) Psi_rotor".

WHY STRUCTURE, NOT ANOTHER FLAT MAP
-----------------------------------
Measured on this fixture (receipts committed in `e728934`), test 3-step cosine:

    linear reference    0.146
    mlp  (flat)         0.226   memorisation gap +0.298  (guard FIRED)
    mlp+skip (flat)     0.226   gap grew +0.008 -> +0.219

A flat map has enough capacity to fit four training trajectories and not enough
structure to generalise: loss keeps falling while the gap grows. So "more
capacity / more steps" is ruled out as the next move. The untested variable is
STRUCTURE -- the transition being a composition of operations the ARC domain
actually contains (rotate a value, mask a support, shift spatially).

WHAT IS IMPLEMENTED (read before citing)
----------------------------------------
Three factors, composed in a fixed order, with the composition ITERATED and the
factors SHARED across iterations (this is the "recursive" part of the namesake
mandate, and it is what makes the parameter count independent of depth):

    x_0 = combined wave [nb, 8]
    x_{k+1} = x_k + alpha_k * T( Pi( R( x_k ) ) )
    return normalize(x_K)

    R  ROTOR   : shared 8x8 orthogonal matrix on the real slot axis, applied as
                 `x @ R`. Parameterised as expm(S - S^T) from a skew generator S,
                 so R is EXACTLY orthogonal and `x @ R` is EXACTLY
                 norm-preserving per block. This is the Cl(3,0) slot algebra the
                 repository already uses (arc_tripartite_resonator.REAL_SLOTS = 8;
                 `apply_rotor` applies `x @ R` the same way).
    Pi MASK    : elementwise gate in (0,1) on (block, slot) pairs -- magnitude
                 only, phase relationship untouched. Initialised near 1 so the
                 identity is reachable.
    T  SPATIAL : circular shift of BLOCKS on the [gb, gb] block ring, where
                 gb = isqrt(nb). A block permutation is exactly norm-preserving.
                 Learned as a distribution over candidate (dr, dc) with a
                 straight-through hard pick.

NORM HONESTY
------------
R and T are exactly norm-preserving; Pi is not (it is a contraction). The final
`normalize` restores the unit norm globally, exactly as the existing
`WavePropagator` arm does, so the unit-norm contract cannot be the source of any
gain. Do NOT cite this module as "norm-preserving throughout" -- it is
norm-restoring at the boundary, by construction.

SCOPE LIMITS
------------
* CPU-scale research arm (d=512 fixture). NOT the production D=65,536 path.
* No benchmark / ARC / score claim. All gates belong to receipts.
* Rank-r factorisation at production D is NOT exercised here; at d=512 the
  parameters are small enough that no factorisation is needed. The D=65,536
  memory argument (16 GiB per action dense) is a SEPARATE, unmeasured claim.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _isqrt(n: int) -> int:
    r = int(math.isqrt(n))
    return r


class _SlotRotor(nn.Module):
    """Exactly-orthogonal 8x8 rotor on the real slot axis, applied as `x @ R`."""

    def __init__(self, slots: int = 8) -> None:
        super().__init__()
        self.slots = int(slots)
        # small init -> R ~ I, so the arm starts near identity
        self.S = nn.Parameter(torch.randn(slots, slots) * 0.01)

    def matrix(self) -> torch.Tensor:
        A = self.S - self.S.t()
        return torch.matrix_exp(A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix()


class _BlockShift(nn.Module):
    """Learned circular shift over the [gb, gb] block ring (exact permutation)."""

    def __init__(self, nb: int, gb: int) -> None:
        super().__init__()
        self.nb, self.gb = int(nb), int(gb)
        self.cands = [(dr, dc) for dr in range(gb) for dc in range(gb)]
        self.logits = nn.Parameter(torch.zeros(len(self.cands)))
        # PRECOMPUTE all candidate index grids once. An earlier revision rebuilt
        # them inside forward() on every iteration (64 torch.roll calls x 3 iters
        # per pass), which made a 12000-step run minutes slower for no benefit.
        self.register_buffer("S", torch.stack(
            [self._index_grid(dr, dc) for dr, dc in self.cands]), persistent=False)

    def _index_grid(self, dr: int, dc: int) -> torch.Tensor:
        gb = self.gb
        idx = torch.arange(self.nb).reshape(gb, gb)
        return torch.roll(idx, shifts=(dr, dc), dims=(0, 1)).reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, nb, slots]; the permutation acts on axis 1 (blocks), NOT axis 0.
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        S = self.S.to(x.device)
        w = torch.softmax(self.logits, dim=0)
        k = int(torch.argmax(self.logits).item())
        xs = x[:, S, :]                          # [B, C, nb, slots]
        soft = torch.einsum('c,bcns->bns', w.to(x.dtype), xs)
        hard = xs[:, k, :, :]
        out = hard + soft - soft.detach()        # hard forward, soft gradient
        return out.squeeze(0) if squeeze else out


class StructuredResonator(nn.Module):
    """Iterated tripartite (rotor -> mask -> shift) predictor with shared factors.

    Drop-in for the probe's arm contract: `forward(x) -> y`, same shape, unit
    norm at the boundary. `x` is the COMBINED wave (normalize(psi + action_wave)),
    exactly what `WavePropagator` receives, so the comparison is like-for-like.
    """

    def __init__(self, d_model: int, num_blocks: int, grid: int = 16,
                 slots: int = 8, iters: int = 3, mask_init: float = 2.0) -> None:
        super().__init__()
        if d_model != num_blocks * slots:
            raise ValueError(
                f"d_model {d_model} != num_blocks {num_blocks} * slots {slots}")
        self.d_model, self.nb, self.slots, self.iters = (
            int(d_model), int(num_blocks), int(slots), int(iters))
        gb = _isqrt(num_blocks)
        if gb * gb != num_blocks:
            raise ValueError(f"num_blocks {num_blocks} is not a perfect square")
        self.gb = gb

        self.rotor = _SlotRotor(slots)
        self.shift = _BlockShift(num_blocks, gb)
        # gate logits initialised >0 -> sigmoid ~0.88, near-identity at start
        self.mask_logits = nn.Parameter(torch.full((num_blocks, slots), float(mask_init)))
        self.alpha = nn.Parameter(torch.full((iters,), 0.3))

    def factors(self):
        return ("R_rotor", "Pi_mask", "T_blockshift")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        h = x.reshape(-1, self.nb, self.slots)
        for k in range(self.iters):
            r = self.rotor(h)
            m = r * torch.sigmoid(self.mask_logits).unsqueeze(0)
            t = self.shift(m)
            h = h + self.alpha[k] * t
        return F.normalize(h.reshape(*lead, self.d_model), p=2, dim=-1)


def build_arm(d_model: int, num_blocks: int, grid: int = 16, iters: int = 3):
    """Factory for the probe's arm registry."""
    return StructuredResonator(d_model, num_blocks, grid=grid, iters=iters)
