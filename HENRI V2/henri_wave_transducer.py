"""Fused Unitary Wave Transducer (FUWT) — polar soft-prompt prefix.

Spec: SPEC-2026-09-24-FUWT-EGRESS-V1 (committed at d97ddd9).

PURPOSE
-------
Convert a wave on S^(D-1) into a CONTINUOUS polar prefix that a language-model
style backbone can consume as KV-cache prefix, removing the need for a rigid AST
grammar mask. The prior measured ceiling
(`UHR05_egress_emittable_ceiling.md`, 48 chars; `UHR05_no_henri_code_generator.md`,
106 chars) is the problem this module addresses.

TENSOR CONTRACT (spec, enforced here)
    wave_input        [Batch, 65536]  complex64   <- 65536 COMPLEX components
    polar_features    [Batch, 32, 6144] float32   (r, cos, sin per 2048-packet)
    prefix_embeddings [Batch, 32, 2048] bfloat16

Note the spec's wave_input is COMPLEX with last dim 65536 = D (not D/2). D = 65536
complex components = 32 orthogonal packets x 2048. A previous revision of this file
required `shape[-1] * 2 == d_model`, which demanded 32768 and then failed to
reshape into 32 x 2048. That defect is fixed and is covered by a test.

INVARIANTS (each has a failing test)
    * ||psi||_2 == 1.0. A non-unitary or NaN input RAISES TransducerNormViolation
      (fail closed -- never emit untrained-decoder content from a broken wave).
    * The polar decomposition z_k = r_k * exp(i*theta_k) preserves L2 energy, so
      sum_m ||r_m||_2^2 == 1.0 over the M=32 orthogonal packets.
    * Reconstruction from (r, theta) reproduces psi to float tolerance.

MEASURED UPSTREAM DEFECT THIS MODULE MUST NOT INHERIT
-----------------------------------------------------
`hopfield_cleanup.ContinuousHopfieldCleanup.store_engrams` accepts a rank-3 input
such as [1, 8, 8] WITHOUT validation (measured: engrams.shape becomes (1,8,8) and
silently mis-treats the last axis as `dim`). This module therefore FLATTENS
explicitly at every Hopfield boundary and rejects non-2D codebooks itself, rather
than relying on the upstream buffer to catch it.

HONEST BOUNDARY
---------------
This module is the PREFIX PRODUCER. It does NOT wire into any backbone here: the
local code backbone is ABSENT from this worktree, so KV-cache wiring and any
SciCode re-run remain BLOCKED. This module makes no score claim and no ICL claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "TransducerNormViolation",
    "TransducerShapeError",
    "FUWTConfig",
    "PolarFeatures",
    "FusedUnitaryWaveTransducer",
    "real_wave_to_complex",
    "spec_wave_to_real",
]

D_MODEL_DEFAULT = 65536
N_PACKETS_DEFAULT = 32
PACKET_WIDTH_DEFAULT = 2048                    # 65536 / 32
POLAR_FEATURE_WIDTH = 3 * PACKET_WIDTH_DEFAULT  # r, cos, sin -> 6144
NORM_TOL_DEFAULT = 1e-4                        # spec: ||psi|| != 1.0 +- 1e-4 raises


class TransducerNormViolation(RuntimeError):
    """Raised when the input wave is not unitary or contains non-finite values."""


class TransducerShapeError(ValueError):
    """Raised when a tensor does not satisfy the declared contract."""


@dataclass(frozen=True)
class FUWTConfig:
    d_model: int = D_MODEL_DEFAULT
    n_packets: int = N_PACKETS_DEFAULT
    prefix_dim: int = 2048
    norm_tol: float = NORM_TOL_DEFAULT
    max_new_tokens: int = 4096
    # bf16 prefix output is the spec contract; the projection accumulates in fp32
    # because bf16 has ~8 mantissa bits and LayerNorm statistics need more.
    output_dtype: torch.dtype = torch.bfloat16
    accum_dtype: torch.dtype = torch.float32

    def validate(self) -> "FUWTConfig":
        if self.n_packets * PACKET_WIDTH_DEFAULT != self.d_model:
            raise TransducerShapeError(
                f"n_packets ({self.n_packets}) x {PACKET_WIDTH_DEFAULT} must equal "
                f"d_model ({self.d_model})"
            )
        if self.norm_tol <= 0.0:
            raise TransducerShapeError("norm_tol must be > 0")
        if self.prefix_dim < 1 or self.max_new_tokens < 1:
            raise TransducerShapeError("prefix_dim and max_new_tokens must be >= 1")
        return self


@dataclass
class PolarFeatures:
    """Result of the polar mode decomposition.

    radius:   [B, M, W]   magnitudes; sum_m ||r_m||^2 == 1.0 for a unit psi
    angle:    [B, M, W]   phases in [-pi, pi]
    features: [B, M, 3W]  the spec's u_p = [r_p, cos(theta_p), sin(theta_p)]
    """

    radius: torch.Tensor
    angle: torch.Tensor
    features: torch.Tensor

    def packet_energy(self) -> torch.Tensor:
        """Per-packet L2 energy [B, M]; the spec requires sum == 1.0."""
        return self.radius.pow(2).sum(dim=-1)


def real_wave_to_complex(wave: torch.Tensor) -> torch.Tensor:
    """Adapter: LIVE real [B, D] -> complex64 [B, D/2].

    The live HENRI wave family is real (a flat [D] real wave, or [num_blocks, 8]).
    ``torch.view_as_complex`` interprets the last axis as (Re, Im) pairs, which is
    exactly the storage convention `arc_task_functor._to_complex` uses.

    WARNING (explicit, not silent): this halves the last dimension. The spec's
    transducer input is complex with last dim D, so the result of this adapter does
    NOT satisfy the transducer contract unless D/2 == d_model. Callers must state
    which representation they hold; the transducer rejects a mismatched width.
    """
    if wave.is_complex():
        return wave
    if wave.shape[-1] % 2 != 0:
        raise TransducerShapeError(
            f"real wave last dim must be even to pair into complex, got {wave.shape[-1]}"
        )
    return torch.view_as_complex(wave.contiguous().reshape(*wave.shape[:-1], -1, 2))


def spec_wave_to_real(psi: torch.Tensor) -> torch.Tensor:
    """Explicit interleave: complex [B, D] -> real [B, 2D].

    Used only at the Hopfield boundary, where the cleanup engine needs a real
    tensor. The doubling is stated in the name so a caller cannot mistake it for a
    no-op.
    """
    if not psi.is_complex():
        return psi
    return torch.view_as_real(psi).reshape(psi.shape[0], -1)


class FusedUnitaryWaveTransducer(nn.Module):
    """Polar soft-prompt transducer. Zero new mathematics; one bounded projection."""

    def __init__(self, config: Optional[FUWTConfig] = None, seed: int = 0) -> None:
        super().__init__()
        self.cfg = (config or FUWTConfig()).validate()
        g = torch.Generator().manual_seed(seed)
        # W_transduce: R^6144 -> R^2048 (spec: 2048 x 6144, ~25.2 MB in bf16)
        self.transduce = nn.Parameter(
            torch.randn(self.cfg.prefix_dim, POLAR_FEATURE_WIDTH,
                        dtype=self.cfg.accum_dtype, generator=g)
            * (1.0 / POLAR_FEATURE_WIDTH ** 0.5)
        )
        self.bias = nn.Parameter(torch.zeros(self.cfg.prefix_dim, dtype=self.cfg.accum_dtype))
        self.ln = nn.LayerNorm(self.cfg.prefix_dim)

    # ---------------------------------------------------------------- validation
    def _check_wave(self, psi: torch.Tensor) -> None:
        if not torch.is_complex(psi):
            raise TransducerShapeError(
                f"wave must be complex64 per spec, got {psi.dtype}"
            )
        if psi.dim() != 2 or psi.shape[-1] != self.cfg.d_model:
            raise TransducerShapeError(
                f"wave must be [Batch, {self.cfg.d_model}] complex64 per spec "
                f"(65536 COMPLEX components), got {tuple(psi.shape)}"
            )
        if not torch.isfinite(psi.real).all() or not torch.isfinite(psi.imag).all():
            raise TransducerNormViolation("wave contains non-finite values")

    def _check_unitary(self, psi: torch.Tensor) -> torch.Tensor:
        """Return the per-batch L2 norm; RAISE if it deviates beyond norm_tol.

        DEFECT FIXED 2026-09-27 (found by this module's own test suite, 9 failures):
            A previous revision computed
                torch.linalg.vector_norm(psi.to(torch.float32), dim=-1)
            Casting a COMPLEX tensor to a real dtype SILENTLY DISCARDS THE IMAGINARY
            PART (torch emits only a UserWarning). For a unit complex vector the real
            half alone has norm ~1/sqrt(2), so every unitary input "deviated" by
            ~0.293 and the module rejected everything -- including its own valid
            fixtures. The norm MUST be taken on the complex tensor, then cast.
        """
        # take the norm on the COMPLEX tensor (returns a real tensor), THEN cast
        norm = torch.linalg.vector_norm(psi, dim=-1).to(self.cfg.accum_dtype)  # [B]
        bad = (norm - 1.0).abs() > self.cfg.norm_tol
        if bool(bad.any()):
            worst = float((norm - 1.0).abs().max())
            raise TransducerNormViolation(
                f"non-unitary wave: ||psi||_2 deviates by {worst:.3e} "
                f"(tolerance {self.cfg.norm_tol:g}). Fail closed."
            )
        return norm

    # ------------------------------------------------------------- decomposition
    def polar_decompose(self, psi: torch.Tensor) -> PolarFeatures:
        """z_k = r_k * exp(i*theta_k) over M orthogonal packets of width W.

        Orthogonal partitioning uses contiguous slices of the D complex dimensions,
        so sum_m ||r_m||^2 == ||psi||^2 == 1.0 exactly.
        """
        self._check_wave(psi)
        self._check_unitary(psi)

        B = psi.shape[0]
        M, W = self.cfg.n_packets, PACKET_WIDTH_DEFAULT
        acc = self.cfg.accum_dtype

        packets = psi.reshape(B, M, W)          # [B, 32, 2048] complex
        radius = packets.abs().to(acc)          # [B, M, W]
        angle = torch.angle(packets).to(acc)    # [B, M, W] in [-pi, pi]

        features = torch.cat(
            [radius, torch.cos(angle), torch.sin(angle)], dim=-1
        )  # [B, M, 3W] = [B, M, 6144]
        return PolarFeatures(radius=radius, angle=angle, features=features)

    def reconstruct(self, radius: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Inverse of the polar decomposition: r * exp(i*theta). Proves no energy loss."""
        z = torch.polar(radius.to(torch.float32), angle.to(torch.float32))
        return z.reshape(z.shape[0], -1)

    # ------------------------------------------------------------------- prefix
    def prefix_embeddings(self, psi: torch.Tensor) -> Tuple[torch.Tensor, PolarFeatures]:
        """[B, D] complex -> [B, 32, 2048] prefix embeddings (bfloat16 per spec).

        h[b, m, d] = LayerNorm( sum_f W[d, f] * u[b, m, f] + bias[d] )
        """
        pol = self.polar_decompose(psi)
        # W is [prefix_dim, feature_dim]; contract over the FEATURE axis f.
        h = torch.einsum("bmf,df->bmd", pol.features, self.transduce) + self.bias
        h = self.ln(h)
        return h.to(self.cfg.output_dtype), pol

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """Public path: wave -> prefix embeddings. Shape/dtype per spec."""
        h, _ = self.prefix_embeddings(psi)
        return h

    # ------------------------------------------------------------- Hopfield snap
    def lexical_snap(
        self,
        psi: torch.Tensor,
        codebook: torch.Tensor,
        *,
        beta: float = 8.0,
        top_k: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Snap a continuous wave to discrete codebook entries via Hopfield cleanup.

        CONTRACT (stated so it cannot be misused):
          * `codebook` must be RANK 2: [V, dim] real. Rank-3 is REJECTED here even
            though the upstream `store_engrams` silently accepts it -- that upstream
            leniency is a measured defect and this module refuses to inherit it.
          * `psi` complex [B, D] is explicitly interleaved to real [B, 2D]; the
            codebook must then be [V, 2D]. If instead the caller holds a real
            [B, 2D] query, the widths must agree. Mismatch RAISES.
          * beta = 8.0 is the spec value (`henri_architecture` Hopfield contract).

        Returns (snapped_wave, confidence).
        """
        from hopfield_cleanup import ContinuousHopfieldCleanup

        if codebook.dim() != 2:
            raise TransducerShapeError(
                f"codebook must be rank 2 [V, dim]; got rank {codebook.dim()} "
                f"{tuple(codebook.shape)}. (Upstream store_engrams accepts rank 3 "
                f"without validation -- do not rely on it.)"
            )

        query = spec_wave_to_real(psi).to(torch.float32)
        cb = codebook.to(torch.float32)
        if cb.shape[-1] != query.shape[-1]:
            raise TransducerShapeError(
                f"codebook width {cb.shape[-1]} != query width {query.shape[-1]}. "
                f"A complex [B, {self.cfg.d_model}] wave interleaves to real "
                f"[B, {2 * self.cfg.d_model}], so the codebook must be "
                f"[V, {2 * self.cfg.d_model}]."
            )

        engine = ContinuousHopfieldCleanup(dim=int(query.shape[-1]), beta=beta).to(query.device)
        n = engine.store_engrams(cb.to(query.device))
        if n != cb.shape[0]:
            raise TransducerShapeError(
                f"store_engrams kept {n} of {cb.shape[0]} rows - refusing to continue"
            )
        return engine.lexical_snap(query, top_k=top_k)
