"""Phase 1: rank-r factorized transition operator (memory-wall removal).

SPECIFICATION
-------------
Refinement spec Gap 3 / Action item 3: transition operators must be
parameterized strictly as low-rank factorizations,

    K_a = U_a Sigma_a V_a^dagger,    U_a, V_a in C^{D x r},   r <= 256

so the parameter overhead is O(2 r D) instead of O(D^2). The specification
states the per-action footprint at D=65,536, r=256 as 268.4 MB and asks for
"parameter footprints under 300 MB".

MEMORY-BOUND INTERPRETATION (resolved explicitly, not silently)
---------------------------------------------------------------
The specification's own arithmetic gives 268.4 MB = 2 * 256 * 65,536 * 8 bytes.
That is PER ACTION. With |A| = 8 the total is ~2.15 GB. The specification
asserts a 300 MB bound without stating the aggregation.

DECISION: the 300 MB bound is enforced PER ACTION and tested per action.
`footprint_bytes(dim, rank, num_actions=1)` is the bounded quantity;
`footprint_bytes(..., num_actions=8)` is REPORTED separately and is NOT
claimed to satisfy the 300 MB bound. Both numbers are emitted by the gate
runner so neither can be quoted in isolation.

NO DENSE OPERATOR, EVER
-----------------------
`K_a` is never materialized. Application is two skinny products:

    y = U @ (sigma * (V^H @ x))

A `[D, D]` allocation would be 34.36 GB at complex64 / D=65,536. The module
exposes `dense_equivalent()` ONLY for small-D correctness verification
(explicitly guarded), and `assert_no_dense_allocation()` to let the test suite
fail closed if a dense tensor is ever introduced.

RELATION TO EXISTING CODE
-------------------------
`efe_planner.LowRankCoupledTransition` already avoids `[D, D]` on the
production path (verified in the Phase 1 premise audit). This module is the
specification-named, standalone, unit-tested artifact for the rank-r contract;
it does NOT replace or rewire the planner.

HONEST LIMITS
-------------
1. No GPU on the authoring host. Footprint numbers at D=65,536 are ARITHMETIC
   (DERIVED). Real allocated bytes are OBSERVED at reduced D and the scaling is
   verified, not assumed.
2. This is a representation/contract artifact. It carries no task-capability
   claim.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

DEFAULT_RANK = 256
DEFAULT_NUM_ACTIONS = 8
BYTES_PER_COMPLEX64 = 8


def footprint_bytes(dim: int, rank: int, num_actions: int = 1) -> int:
    """ARITHMETIC footprint of the factorized operator.

    Two complex64 factors [dim, rank]. The rank-length sigma is float32 and is
    counted separately by `footprint_with_sigma_bytes`, because the
    specification's own stated figure (268.4 MB) is exactly 2*r*D*8 and
    excludes it. num_actions=1 is the specification's bounded quantity.
    """
    per_action = 2 * int(dim) * int(rank) * BYTES_PER_COMPLEX64
    return per_action * int(num_actions)


def footprint_with_sigma_bytes(dim: int, rank: int, num_actions: int = 1) -> int:
    """Factor footprint INCLUDING the rank-length float32 sigma vector."""
    per_action = footprint_bytes(dim, rank, 1) + int(rank) * 4
    return per_action * int(num_actions)


def dense_footprint_bytes(dim: int) -> int:
    """The rejected dense alternative, for the comparison table."""
    return int(dim) * int(dim) * BYTES_PER_COMPLEX64


class FactorizedTransitionKernel(nn.Module):
    """Action-conditioned rank-r transition operator over C^D.

    Structure (per action a):
        K_a = U_a diag(sigma_a) V_a^H
        apply_a(x) = U_a @ (sigma_a * (V_a^H @ x))

    Shapes:
        U:     [A, D, r]  complex64
        V:     [A, D, r]  complex64
        sigma: [A, r]     float32
    """

    def __init__(
        self,
        dim: int,
        rank: int = DEFAULT_RANK,
        num_actions: int = DEFAULT_NUM_ACTIONS,
        num_blocks: int | None = None,
        block_slots: int = 8,
        seed: int = 20261001,
        device: str = "cpu",
        project: bool = True,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if dim < 1 or num_actions < 1:
            raise ValueError("dim and num_actions must be >= 1")
        if rank > dim:
            raise ValueError(f"rank {rank} cannot exceed dim {dim}")

        self.dim = int(dim)
        self.rank = int(rank)
        self.num_actions = int(num_actions)
        self.num_blocks = int(num_blocks) if num_blocks else None
        self.block_slots = int(block_slots)
        self.seed = int(seed)
        self.device = torch.device(device)

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        scale_u = 1.0 / math.sqrt(self.dim * self.rank)
        # Complex parameters: interleave (re, im) views so the module works
        # without relying on complex-aware optimizers downstream.
        self.U = nn.Parameter(
            torch.randn(self.num_actions, self.dim, self.rank, 2, generator=g) * scale_u
        )
        self.V = nn.Parameter(
            torch.randn(self.num_actions, self.dim, self.rank, 2, generator=g) * scale_u
        )
        self.sigma = nn.Parameter(torch.ones(self.num_actions, self.rank))

        # `project=False` skips the Stiefel projection. It exists for the
        # production-dimension MEMORY tests, which must not pay an O(D r^2)
        # Gram cost, and it is explicit so that an unprojected kernel can never
        # be mistaken for a manifold-constrained one.
        if project:
            self._qr_orthonormalize(full=True)

    # ------------------------------------------------------------ internal
    @staticmethod
    def _as_complex(t: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(t.contiguous())

    def _u(self) -> torch.Tensor:
        return self._as_complex(self.U)

    def _v(self) -> torch.Tensor:
        return self._as_complex(self.V)

    @torch.no_grad()
    def _qr_orthonormalize(self, full: bool = False) -> None:
        """Project U (and optionally V) onto the complex Stiefel manifold.

        U_a -> column-orthonormal [D, r]: U^H U = I_r exactly. With V also
        orthonormal and sigma = 1, K_a = U_a U_a^H is an orthogonal projector
        (P1-G8).

        WHY NOT torch.linalg.qr — two measured failures (2026-10-01):
        1. On a COMPLEX [D, r] with r < D, `torch.linalg.qr(m, mode="reduced")`
           IGNORES mode and returns the complete factorization, Q of shape
           [D, D]. Copying that into a [D, r] parameter raises a size mismatch.
        2. Substituting the real block embedding [[Re, -Im], [Im, Re]] and
           un-stacking afterwards yields [2D, r], not [D, r], and the recovered
           complex matrix satisfies only Re(U^H U) = I -- the imaginary part is
           left free, so K is not actually a projector. The G8 gate failed on
           exactly that (error 0.169, not the 1e-4 tolerance).

        The exact projection is the inverse square root of the r x r Gram
        matrix, which is cheap because r << D:

            G = U^H U  (r x r, Hermitian)  ->  U <- U G^{-1/2}
            then U^H U = G^{-1/2} G G^{-1/2} = I.

        Cost is O(D r^2) for the Gram and O(r^3) for the eigh, versus O(D^3)
        for a full QR. Guarded for large D by the `project` constructor flag.
        """
        for param in ((self.U, self.V) if full else (self.U,)):
            w = self._as_complex(param)                      # [A, D, r]
            gram = w.conj().transpose(-2, -1) @ w            # [A, r, r] Hermitian
            evals, evecs = torch.linalg.eigh(gram)           # evals REAL, evecs complex
            # Broadcast the real scale against the complex eigenvectors BEFORE
            # the matmul. torch does not promote a real diagonal into a complex
            # product implicitly: `complex_matrix @ real_diag` raises
            # "expected scalar type Float but found ComplexFloat" (measured
            # 2026-10-01). Scaling the columns promotes correctly.
            d = evals.clamp_min(1e-12).rsqrt().unsqueeze(-2)  # [A, 1, r]
            inv_sqrt = (evecs * d) @ evecs.conj().transpose(-2, -1)
            # inv_sqrt is Hermitian, so G^{-1/2} G G^{-1/2} = I exactly.
            proj = w @ inv_sqrt
            param.copy_(torch.view_as_real(proj.contiguous()))

    # ---------------------------------------------------------- arithmetic
    def apply(self, action: int, x: torch.Tensor) -> torch.Tensor:
        """Apply K_a to x [..., D] WITHOUT materializing K_a. O(r D) work."""
        a = int(action)
        if not 0 <= a < self.num_actions:
            raise IndexError(f"action {a} out of range [0, {self.num_actions})")
        vh = self._v()[a].conj().transpose(0, 1)          # [r, D]
        u = self._u()[a]                                   # [D, r]
        sig = self.sigma[a].to(u.dtype)                    # [r]
        # x must be a VECTOR [D] here: reshaping to (-1, dim) would produce a
        # [1, D] row matrix and the [r, D] @ [1, D] product raises. Measured
        # 2026-10-01. Batch users go through apply_batch.
        xv = x.reshape(-1)
        if xv.numel() != self.dim:
            raise ValueError(f"x has {xv.numel()} elements, expected {self.dim}")
        return u @ (sig * (vh @ xv))

    def apply_batch(self, actions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply one operator per row. actions [B] long, x [B, D]."""
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must be [B, {self.dim}], got {tuple(x.shape)}")
        if actions.numel() != x.shape[0]:
            raise ValueError("actions length must equal batch size")
        return torch.stack([self.apply(int(a), x[i]) for i, a in enumerate(actions)], 0)

    @torch.no_grad()
    def share_basis(self, action: int) -> None:
        """Set V_a = U_a, making K_a = U_a U_a^H an orthogonal PROJECTOR.

        With V != U, K = U V^H is not idempotent; only K K^H = U U^H is. The
        idempotence gate P1-G8 tests BOTH forms explicitly so the algebra is
        pinned rather than assumed.
        """
        a = int(action)
        self.V[a].copy_(self.U[a])

    # ------------------------------------------------------- verification
    def dense_equivalent(self, action: int, max_dim: int = 256) -> torch.Tensor:
        """Materialize K_a. GUARDED: small-D correctness probing only."""
        if self.dim > max_dim:
            raise ValueError(
                f"dense_equivalent is a verification-only probe; dim={self.dim} "
                f"exceeds the guard max_dim={max_dim}. The dense operator at "
                f"D=65,536 would be {dense_footprint_bytes(65536) / 2**30:.1f} GiB."
            )
        a = int(action)
        u = self._u()[a]
        vh = self._v()[a].conj().transpose(0, 1)
        return u @ (self.sigma[a].to(u.dtype)[:, None] * vh)

    def assert_no_dense_allocation(self) -> None:
        """Fail closed if any parameter/buffer reaches [D, D] scale."""
        limit = self.dim * self.dim
        for name, t in list(self.named_parameters()) + list(self.named_buffers()):
            if t.numel() >= limit:
                raise AssertionError(
                    f"dense-scale allocation detected in {name}: numel="
                    f"{t.numel()} >= dim^2={limit}"
                )
        if self._u().numel() >= limit or self._v().numel() >= limit:
            raise AssertionError("dense-scale complex factor detected")

    def footprint_report(self) -> Dict[str, object]:
        """Per-action and total footprint, both stated. DERIVED (arithmetic)."""
        per_action = footprint_bytes(self.dim, self.rank, 1)
        total = footprint_bytes(self.dim, self.rank, self.num_actions)
        return {
            "dim": self.dim,
            "rank": self.rank,
            "num_actions": self.num_actions,
            "per_action_bytes": per_action,
            "per_action_MB": per_action / 2**20,
            "per_action_with_sigma_bytes": footprint_with_sigma_bytes(self.dim, self.rank, 1),
            "total_bytes": total,
            "total_MB": total / 2**20,
            "dense_alternative_bytes": dense_footprint_bytes(self.dim),
            "dense_alternative_GiB": dense_footprint_bytes(self.dim) / 2**30,
            "spec_300MB_bound_applies_to": "per_action",
            "evidence_class": "DERIVED_ARITHMETIC",
        }

    def observed_bytes(self) -> int:
        """OBSERVED allocated bytes of the parameters."""
        return sum(
            t.numel() * t.element_size() for t in self.parameters()
        )
