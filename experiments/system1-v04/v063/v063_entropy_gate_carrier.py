"""
System-1 v0.6.3a — Pre-Reasoning Entropy INSTRUMENTATION (default OFF).
======================================================================
NO behavioral change. NO gating. NO learning. Telemetry only.

Mechanism: Shannon entropy of the normalized pre-verifier candidate
score distribution (cosine sims via CandidateRetrievalRanker), plus
H/logK normalization for variable pool sizes.

Corpus #20 (INFERRED): low entropy = score concentration, NOT
correctness. This module never claims correctness; it only measures.

Gate: HENRI_V063A_ENABLE=1 required for any entropy computation.
"""

import math
import os
from typing import List, Optional, Tuple

import torch

_ENABLED = os.environ.get("HENRI_V063A_ENABLE", "0") == "1"


def shannon_entropy_nats(probs: List[float]) -> float:
    """Shannon entropy in nats over a normalized distribution."""
    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0 or not math.isfinite(total):
        return float("nan")
    p = [x / total for x in probs if x > 0 and math.isfinite(x)]
    if not p:
        return float("nan")
    return -sum(x * math.log(x) for x in p)


def normalized_entropy(probs: List[float]) -> Tuple[float, float]:
    """Return (H_nats, H/logK). K = number of positive-mass candidates."""
    h = shannon_entropy_nats(probs)
    k = sum(1 for x in probs if x > 0 and math.isfinite(x))
    h_norm = h / math.log(k) if k > 1 else 0.0
    return h, h_norm


@torch.no_grad()
def candidate_score_distribution(
    sims: torch.Tensor,
) -> Tuple[List[float], float, float, int]:
    """From per-candidate cosine sims (pre-verifier, real signal):

    - returns (softmax_probs, H_nats, H/logK, K).
    - sims may be on GPU; converted to float list.
    - temperature fixed at 1.0 (frozen, pre-registered).
    """
    if not _ENABLED:
        return [], float("nan"), float("nan"), 0
    s = sims.detach().cpu().float()
    if s.numel() == 0:
        return [], float("nan"), float("nan"), 0
    m = s.max()
    e = torch.exp(s - m)  # numerically stable softmax
    probs = (e / e.sum()).tolist()
    h, h_norm = normalized_entropy(probs)
    k = sum(1 for p in probs if p > 0)
    return probs, h, h_norm, k


def entropy_summary(
    probs: List[float], h: float, h_norm: float, k: int
) -> dict:
    """Compact telemetry record (schema-stable)."""
    return {
        "H_nats": round(h, 6) if math.isfinite(h) else None,
        "H_norm": round(h_norm, 6) if math.isfinite(h_norm) else None,
        "K": k,
        "p1": round(probs[0], 6) if probs else None,
        "p1_rank_by_sim": 1 if probs else None,
    }
