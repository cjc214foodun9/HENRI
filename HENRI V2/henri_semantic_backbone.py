"""Frozen semantic-backbone boundary for the HENRI hybrid architecture.

This module does not download a model, choose a model, or claim world
knowledge. A caller supplies an already-loaded backbone and a provenance record.
The adapter is zero-trainable and maps backbone features into the canonical
real wave boundary [batch, num_blocks, 8].

The projection is factorized:
    z = normalize((e @ A) @ B.T)
where A has shape [semantic_dim, r] and B has shape [D, r]. It is a bounded
representation adapter, not a global semantic isometry. Components outside the
chosen rank are discarded and must be measured on real data before promotion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F


class SemanticContractError(ValueError):
    """Raised when the frozen semantic boundary cannot satisfy its contract."""


@dataclass(frozen=True)
class FrozenBackboneProvenance:
    """Immutable identity record for a frozen external backbone artifact."""

    model_id: str
    revision: str
    artifact_sha256: str
    artifact_bytes: int
    hidden_size: int
    source: str

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise SemanticContractError("model_id must be non-empty")
        if not self.revision.strip() or self.revision.lower() in {"main", "latest", "master"}:
            raise SemanticContractError("revision must be an immutable pinned identifier")
        digest = self.artifact_sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SemanticContractError("artifact_sha256 must be a 64-character hex digest")
        if self.artifact_bytes <= 0:
            raise SemanticContractError("artifact_bytes must be positive")
        if self.hidden_size <= 0:
            raise SemanticContractError("hidden_size must be positive")
        if not self.source.strip():
            raise SemanticContractError("source must be non-empty")
        object.__setattr__(self, "artifact_sha256", digest)

    def verify_artifact(self, path: str | Path) -> bool:
        """Verify exact bytes and digest for the declared external artifact."""
        p = Path(path)
        if not p.is_file():
            raise SemanticContractError(f"backbone artifact missing: {p}")
        size = p.stat().st_size
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if size != self.artifact_bytes or digest != self.artifact_sha256:
            raise SemanticContractError(
                f"backbone artifact mismatch: path={p} bytes={size} sha256={digest}"
            )
        return True


class FactorizedSemanticWaveAdapter(nn.Module):
    """Map frozen semantic features to canonical HENRI real wave packets.

    No dense [D, semantic_dim] matrix is allocated. The two frozen factors are
    generated on CPU with a pinned seed and can be moved with the module to the
    caller's device. The output has global unit Frobenius norm per sample.
    """

    def __init__(
        self,
        semantic_dim: int,
        num_blocks: int = 8192,
        block_dim: int = 8,
        rank: int = 64,
        seed: int = 20260903,
        max_projection_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        super().__init__()
        if semantic_dim <= 0:
            raise SemanticContractError("semantic_dim must be positive")
        if num_blocks <= 0 or block_dim != 8:
            raise SemanticContractError(
                "canonical wave boundary requires num_blocks > 0 and block_dim == 8"
            )
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise SemanticContractError("rank must be a positive integer")
        if max_projection_bytes <= 0:
            raise SemanticContractError("max_projection_bytes must be positive")

        self.semantic_dim = int(semantic_dim)
        self.num_blocks = int(num_blocks)
        self.block_dim = int(block_dim)
        self.requested_rank = int(rank)
        self.effective_rank = min(rank, semantic_dim, num_blocks * block_dim)
        self.seed = int(seed)
        self.projection_bytes = (
            self.semantic_dim * self.effective_rank
            + (self.num_blocks * self.block_dim) * self.effective_rank
        ) * 4
        if self.projection_bytes > max_projection_bytes:
            raise SemanticContractError(
                f"factorized projection uses {self.projection_bytes} bytes; "
                f"limit is {max_projection_bytes}"
            )

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        left_raw = torch.randn(
            self.semantic_dim, self.effective_rank, generator=generator, device="cpu"
        )
        right_raw = torch.randn(
            self.num_blocks * self.block_dim,
            self.effective_rank,
            generator=generator,
            device="cpu",
        )
        left, _ = torch.linalg.qr(left_raw, mode="reduced")
        right, _ = torch.linalg.qr(right_raw, mode="reduced")
        self.register_buffer("left_basis", left.contiguous())
        self.register_buffer("right_basis", right.contiguous())

    @property
    def output_dim(self) -> int:
        return self.num_blocks * self.block_dim

    @torch.no_grad()
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return [B, num_blocks, 8] real unit-norm wave packets."""
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.semantic_dim:
            raise SemanticContractError(
                f"embeddings must be [B, {self.semantic_dim}], got {tuple(embeddings.shape)}"
            )
        if embeddings.device != self.left_basis.device:
            raise SemanticContractError(
                f"device mismatch: embeddings={embeddings.device}, adapter={self.left_basis.device}"
            )
        features = embeddings.to(dtype=torch.float32)
        flat = (features @ self.left_basis) @ self.right_basis.T
        flat = F.normalize(flat, p=2, dim=-1)
        return flat.view(-1, self.num_blocks, self.block_dim)


class FrozenSemanticBackbone(nn.Module):
    """Wrap a caller-supplied frozen backbone and the factorized wave adapter."""

    def __init__(
        self,
        backbone: nn.Module,
        provenance: FrozenBackboneProvenance,
        wave_rank: int = 64,
        num_blocks: int = 8192,
        block_dim: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise SemanticContractError("backbone must be a torch.nn.Module")
        params = tuple(backbone.parameters())
        if any(p.requires_grad for p in params):
            raise SemanticContractError("backbone contains trainable parameters")
        self.backbone = backbone.eval()
        self.provenance = provenance
        self.wave_adapter = FactorizedSemanticWaveAdapter(
            semantic_dim=provenance.hidden_size,
            num_blocks=num_blocks,
            block_dim=block_dim,
            rank=wave_rank,
        )

    def train(self, mode: bool = True) -> "FrozenSemanticBackbone":
        """Keep the supplied backbone in eval mode even if the wrapper is trained."""
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _extract_features(
        outputs: Any, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            hidden = outputs
        elif isinstance(outputs, dict):
            hidden = outputs.get("pooler_output")
            if hidden is None:
                hidden = outputs.get("last_hidden_state")
        else:
            hidden = getattr(outputs, "pooler_output", None)
            if hidden is None:
                hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor):
            raise SemanticContractError(
                "backbone output must expose a tensor, pooler_output, or last_hidden_state"
            )
        if hidden.ndim == 2:
            return hidden
        if hidden.ndim != 3:
            raise SemanticContractError(
                f"backbone features must be [B,H] or [B,T,H], got {tuple(hidden.shape)}"
            )
        if attention_mask is None:
            return hidden.mean(dim=1)
        if attention_mask.ndim != 2 or attention_mask.shape[:2] != hidden.shape[:2]:
            raise SemanticContractError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"hidden shape {tuple(hidden.shape)}"
            )
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    @torch.no_grad()
    def forward(
        self,
        *args: Any,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run frozen inference and emit canonical real wave packets."""
        call_kwargs = dict(kwargs)
        if attention_mask is not None:
            call_kwargs["attention_mask"] = attention_mask
        outputs = self.backbone(*args, **call_kwargs)
        features = self._extract_features(outputs, attention_mask)
        if features.shape[-1] != self.provenance.hidden_size:
            raise SemanticContractError(
                f"backbone hidden size {features.shape[-1]} != declared "
                f"{self.provenance.hidden_size}"
            )
        return self.wave_adapter(features)
