"""E2 — k-NN softmax lexical probe (default-OFF, zero trainable).

Carrier: carrier/e2-egress-knn-scale @ base c8fc94a0 (E1 tip).
Prereg: experiments/verification/e2_egress_knn_prereg.md (sealed).
Geometry only: these helpers operate on d_target features (896). The
wave->feature head and corpus pipeline live in e2_calibrate.py (reuses the E1
factorized Stiefel head + frozen hash-pinned Qwen teacher).

Contract (from prereg section 2):
  s_j = cos(z, T_j)/tau                     j = 1..V
  w   = softmax(s_topk)                     k = 16, tau = 0.07 (frozen)
  z_hat = sum_j w_j * T_j                   cluster centroid readout
  G3 margin = align_trained - max(align_untrained, align_random) >= +0.05

Zero trainable parameters (frozen k/tau buffers). Never imported by
production_arc_run.py / henri_decoder.py / henri_egress.py (enforced by
tests/contract/test_e2_egress_knn.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

TEACHER_VOCAB = 151936
TEACHER_DIM = 896
NUM_BLOCKS = 8192
BLOCK_DIM = 8
WAVE_DIM = NUM_BLOCKS * BLOCK_DIM


@dataclass
class E2Config:
    d_model: int = WAVE_DIM
    num_blocks: int = NUM_BLOCKS
    block_dim: int = BLOCK_DIM
    d_bottleneck: int = 256
    d_target: int = TEACHER_DIM
    vocab_size: int = TEACHER_VOCAB
    k: int = 16
    tau: float = 0.07
    seed: int = 20260909
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


def knn_softmax_probe(feats: torch.Tensor, teacher: torch.Tensor,
                      k: int = 16, tau: float = 0.07
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """z_hat = sum_j softmax(cos(z,T_j)/tau)_{top-k} * T_j.

    feats:   [B, d_target]   (head output, L2-normalized internally)
    teacher: [V, d_target]   (frozen embedding table; any V >= k)
    returns (z_hat [B, d_target], w [B, min(k, V)]). No trainable parameters.
    """
    z = F.normalize(feats.float(), dim=-1)
    t = F.normalize(teacher.float(), dim=-1)
    k = max(1, min(int(k), t.shape[0]))
    sim = z @ t.t() / max(float(tau), 1e-12)          # [B, V]
    vals, idx = sim.topk(k, dim=-1)                   # [B, k]
    w = F.softmax(vals, dim=-1)                       # [B, k], sums to 1
    z_hat = torch.einsum("bk,bkd->bd", w, t[idx])     # [B, d_target]
    return z_hat, w


class KNNSoftmaxProbe(nn.Module):
    """Frozen probe (zero trainable parameters; k/tau are buffers)."""
    def __init__(self, config: E2Config):
        super().__init__()
        self.config = config
        self.register_buffer("_k", torch.tensor(int(config.k)))
        self.register_buffer("_tau", torch.tensor(float(config.tau)))

    def forward(self, feats: torch.Tensor, teacher: torch.Tensor):
        return knn_softmax_probe(feats, teacher,
                                 k=int(self._k.item()), tau=float(self._tau.item()))


def _align(preds: Sequence[torch.Tensor] | torch.Tensor,
           targets: Sequence[torch.Tensor] | torch.Tensor) -> float:
    """Mean cosine similarity between paired feature vectors."""
    z = preds if torch.is_tensor(preds) else torch.stack([p.float() for p in preds], dim=0)
    t = targets if torch.is_tensor(targets) else torch.stack([p.float() for p in targets], dim=0)
    z = F.normalize(z.float(), dim=-1)
    t = F.normalize(t.float(), dim=-1)
    return float((z * t).sum(dim=-1).mean())


def e2_alignment_margin(trained_z, targets, untrained_z, random_score) -> float:
    """Gate G3: align_trained - max(align_untrained, align_random) >= +0.05."""
    a_trained = _align(trained_z, targets)
    a_untrained = _align(untrained_z, targets)
    return float(a_trained - max(a_untrained, float(random_score)))


def e2_negative_control(feats: torch.Tensor, targets: torch.Tensor,
                        teacher: torch.Tensor, k: int = 16, tau: float = 0.07,
                        seed: int = 1) -> float:
    """Shuffled-input alignment; must sit BELOW the aligned margin (negative control)."""
    perm = torch.randperm(feats.shape[0],
                          generator=torch.Generator().manual_seed(seed))
    z_shuf, _ = knn_softmax_probe(feats[perm], teacher, k=k, tau=tau)
    return _align(z_shuf, targets)
