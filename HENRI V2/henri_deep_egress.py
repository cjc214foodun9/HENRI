"""Phase 10.0 deep egress proposal head (default-OFF identity carrier).

Four-Phase Reconciliation Report (HENRI-ARCH-2026-TEMPORAL-GROUNDING-
RECONCILIATION, inbox sha 7313359b) Phase 10.0 repair spec:

    z_out = (1 - beta) * z_linear + beta * z_deep,  beta init 0.0
  - beta = 0 => output is byte-identical to the verified linear decoder
    baseline (and the deep path is never computed).
  - factorized low-rank structure: block-diagonal projection
    R^{8192x8} -> R^{8192x2} followed by structured aggregation to hidden,
    then the shared vocab head. No dense [65536, 65536] intermediates
    (34 GiB forbidden).
  - gradient-flow contract: every trainable deep parameter receives a
    nonzero gradient when beta > 0.

Default-OFF: HENRI_DEEP_EGRESS=1 must be set or get_deep_egress_head()
returns None (factory contract; production code never imports this module
without the flag).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

FLAG = "HENRI_DEEP_EGRESS"
MAX_ACTIVATION_BYTES = int(1.5e9)  # report bound: < 1.5 GB activation VRAM


class DeepEgressDisabledError(RuntimeError):
    """Raised on dense-allocation or activation-budget violations."""


class DeepEgressProposalHead(nn.Module):
    """Parallel deep egress path blended against the linear baseline.

    The caller supplies ``linear_logits`` from the ACTUAL checkpoint-loaded
    baseline forward pass (e.g. HENRINeuralEgressUnbinder.forward). At
    beta=0 this module returns that tensor verbatim — it never recomputes
    or perturbs the baseline.
    """

    def __init__(
        self,
        d_model: int = 65536,
        num_blocks: int = 8192,
        block_dim: int = 8,
        proj_dim: int = 2,
        d_hidden: int = 2048,
        vocab_size: int = 32000,
        beta: float = 0.0,
        lm_head: Optional[nn.Linear] = None,
    ):
        super().__init__()
        if d_model != num_blocks * block_dim:
            raise ValueError(
                f"d_model {d_model} != num_blocks {num_blocks} * block_dim {block_dim}")
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.proj_dim = proj_dim
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        agg_dim = num_blocks * proj_dim  # 16384 at production scale

        # Factorized block-diagonal projection R^{8192x8} -> R^{8192x2}
        # (shared per-block map; no dense [D,D] intermediate).
        self.block_proj = nn.Linear(block_dim, proj_dim, bias=False)
        # Structured aggregation to hidden, then the shared vocab head.
        self.deep_down = nn.Linear(agg_dim, d_hidden, bias=False)
        self.layer_norm = nn.LayerNorm(d_hidden)
        self.act = nn.GELU()
        # Shared vocab head: the baseline's lm_head when supplied (same
        # egress space); otherwise a fresh head.
        if lm_head is None:
            lm_head = nn.Linear(d_hidden, vocab_size, bias=False)
        self.lm_head = lm_head
        # Blending coefficient, init 0.0 (report Phase 10.0 constraint 1).
        self.register_buffer("beta", torch.tensor(float(beta)))
        self._assert_no_dense_allocation()

    def _assert_no_dense_allocation(self) -> None:
        """Reject any [D,D] parameter and any activation budget > 1.5 GB."""
        for name, p in self.named_parameters():
            if len(p.shape) == 2 and p.shape[0] == self.d_model and p.shape[1] == self.d_model:
                raise DeepEgressDisabledError(
                    f"dense [D,D] parameter forbidden: {name} {tuple(p.shape)}")
        # B=1 peak activation bytes: block output + aggregated input +
        # hidden + logits (all fp32).
        act_bytes = (self.num_blocks * self.proj_dim * 4
                     + self.num_blocks * self.proj_dim * 4
                     + self.d_hidden * 4
                     + self.vocab_size * 4)
        if act_bytes > MAX_ACTIVATION_BYTES:
            raise DeepEgressDisabledError(
                f"activation budget {act_bytes} B > {MAX_ACTIVATION_BYTES} B")

    def deep_logits(self, unit_wave: torch.Tensor) -> torch.Tensor:
        """Factorized deep path: [B,D] -> [B,8192,2] -> [B,16384] -> [B,2048] -> [B,V]."""
        if unit_wave.dim() == 1:
            unit_wave = unit_wave.unsqueeze(0)
        B = unit_wave.shape[0]
        x = unit_wave.view(B, self.num_blocks, self.block_dim)
        h = self.block_proj(x)          # [B, num_blocks, proj_dim]
        h = h.reshape(B, -1)            # [B, num_blocks*proj_dim]
        h = self.deep_down(h)           # [B, d_hidden]
        h = self.layer_norm(h)
        h = self.act(h)
        return self.lm_head(h)          # [B, vocab_size]

    def forward(
        self,
        unit_wave: torch.Tensor,
        linear_logits: torch.Tensor,
        beta: Optional[float] = None,
    ) -> torch.Tensor:
        """Blend: z_out = (1-beta) z_linear + beta z_deep.

        beta == 0 (the default) returns ``linear_logits`` VERBATIM — the
        deep path is never computed, so the output is byte-identical to the
        verified linear decoder baseline.
        """
        b = float(self.beta) if beta is None else float(beta)
        if b <= 0.0:
            return linear_logits
        z_deep = self.deep_logits(unit_wave)
        return (1.0 - b) * linear_logits + b * z_deep


def get_deep_egress_head(
    d_model: int = 65536,
    num_blocks: int = 8192,
    block_dim: int = 8,
    d_hidden: int = 2048,
    vocab_size: int = 32000,
    lm_head: Optional[nn.Linear] = None,
) -> Optional[DeepEgressProposalHead]:
    """Flag-gated factory: returns None unless HENRI_DEEP_EGRESS=1."""
    if os.environ.get(FLAG, "0") != "1":
        return None
    return DeepEgressProposalHead(
        d_model=d_model, num_blocks=num_blocks, block_dim=block_dim,
        d_hidden=d_hidden, vocab_size=vocab_size, lm_head=lm_head)
