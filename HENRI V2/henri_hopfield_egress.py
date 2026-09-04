"""henri_hopfield_egress.py — syntax-validated Modern Hopfield egress (HENRI-DIR-2026-09-BUNDLE-VLA-GROUNDING, Component 3).

ALREADY-IMPLEMENTED base (audited @ b31f873): henri_egress.py provides
TextEgress / ToolEgress / UniversalEgress as zero-entropy Modern Hopfield
snapping engines (beta = 8.0) over ContinuousHopfieldCleanup (hopfield_cleanup.py,
Ramsauer-style energy E = -tau*logsumexp(beta*<r, M_k>)).

This module does NOT duplicate the Hopfield core.  It adds the missing
fail-closed layer the live engine lacks:
  1. codebook registration with a canonical-id allowlist and a validator;
  2. decode that returns REJECTED (never a fabricated <token_N> fallback) when
     the snapped prototype is not canonical / fails the validator;
  3. strict mode raising EgressSyntaxRejectedError (fail-closed for score paths);
  4. the dimension-aware noise floor sigma_elem = eps / sqrt(D) helper
     (directive §2.3: eps = 0.15, D = 65536 -> 5.86e-4).

Default-OFF: never imported by the production runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from hopfield_cleanup import ContinuousHopfieldCleanup

EPSILON_DEFAULT = 0.15  # directive §2.3 perturbation tolerance


class EgressSyntaxRejectedError(RuntimeError):
    """Raised in strict mode when a snapped prototype is not canonical/valid."""


@dataclass
class ValidatedEgressResult:
    status: str  # "SNAPPED" | "REJECTED"
    snapped_index: int = -1
    similarity: float = 0.0
    reason: str = ""


def noise_floor_sigma(dim: int, eps: float = EPSILON_DEFAULT) -> float:
    """sigma_elem = eps / sqrt(D); at D=65536, eps=0.15 -> 5.86e-4."""
    if dim <= 0:
        raise ValueError(f"dim must be positive; got {dim}")
    return eps / math.sqrt(float(dim))


class CanonicalCodebookEgress:
    """Modern Hopfield codebook snapping with fail-closed syntax validation.

    Reuses the live ContinuousHopfieldCleanup (beta = 8.0 default) for the
    associative retrieval core; enforces canonical-vocabulary validity on the
    snapped index.
    """

    def __init__(self, dim: int, beta: float = 8.0) -> None:
        self.dim = dim
        self.cleanup = ContinuousHopfieldCleanup(dim=dim, beta=beta)
        self.canonical_ids: List[int] = []
        self._validators: Dict[int, Callable[[int], bool]] = {}

    @torch.no_grad()
    def register(
        self,
        code_waves: torch.Tensor,
        canonical_ids: Sequence[int],
        validator: Optional[Callable[[int], bool]] = None,
    ) -> int:
        """Register codebook rows [M, D] with their canonical ids.

        A prototype whose canonical id is absent from the allowlist is
        rejected at decode time (REJECTED), never emitted.
        """
        if code_waves.ndim != 2 or code_waves.shape[-1] != self.dim:
            raise ValueError(
                f"code_waves must be [M, {self.dim}]; got shape {tuple(code_waves.shape)}"
            )
        ids = list(canonical_ids)
        if len(ids) != code_waves.shape[0]:
            raise ValueError(
                f"canonical_ids count {len(ids)} != code rows {code_waves.shape[0]}"
            )
        self.cleanup.store_engrams(code_waves)
        base = len(self.canonical_ids)
        for k, cid in enumerate(ids):
            self.canonical_ids.append(int(cid))
            if validator is not None:
                self._validators[base + k] = validator
        return self.cleanup.num_engrams()

    def _is_valid(self, mem_idx: int) -> Tuple[bool, str]:
        if mem_idx < 0 or mem_idx >= len(self.canonical_ids):
            return False, "index outside canonical codebook"
        cid = self.canonical_ids[mem_idx]
        v = self._validators.get(mem_idx)
        if v is not None and not v(cid):
            return False, f"validator rejected canonical id {cid}"
        return True, ""

    def decode(self, wave: torch.Tensor) -> ValidatedEgressResult:
        """Snap wave to nearest prototype; REJECTED when non-canonical.

        Never fabricates text or a fallback payload.
        """
        if self.cleanup.num_engrams() == 0:
            return ValidatedEgressResult(
                status="REJECTED", reason="empty codebook (fail-closed)"
            )
        _, idx, sim = self.cleanup.hard_retrieve(wave)
        mem_idx = int(idx)
        ok, reason = self._is_valid(mem_idx)
        if not ok:
            return ValidatedEgressResult(
                status="REJECTED", snapped_index=mem_idx,
                similarity=float(sim), reason=reason,
            )
        return ValidatedEgressResult(
            status="SNAPPED", snapped_index=self.canonical_ids[mem_idx],
            similarity=float(sim), reason="canonical",
        )

    def decode_valid(self, wave: torch.Tensor) -> Tuple[int, float]:
        """Strict decode for score paths: raises on any rejection (fail-closed)."""
        res = self.decode(wave)
        if res.status != "SNAPPED":
            raise EgressSyntaxRejectedError(
                f"egress rejected: {res.reason} (index {res.snapped_index})"
            )
        return res.snapped_index, res.similarity
