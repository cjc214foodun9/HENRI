"""Discriminative membership / novelty statistic for HENRI's wave bank.

WHY THIS EXISTS
    Measured defect (Q4): after the swarm projection, max similarity to the bank
    is ~0.99 for EVERY input. The system cannot reject an input it does not know.
    The projection is the confound: CCCP maps any wave onto the bank manifold.

    The ingress wave, read BEFORE the swarm runs, still carries membership
    information. This module scores that wave.

DESIGN
    S_membership(q) = max_k |<psi_q, psi_k>|      level
    S_margin(q)     = beta*(s_top1 - s_top2)      shape (peaked vs flat)

    Both are pure functions. Neither mutates the model. Neither moves a bound.
    beta=26.10 is the frozen Hopfield temperature from the model, not a tuned knob.

SCOPE
    This is an encoder-membership test. It asks "is this wave near a stored
    wave?", not "does the system understand this text?".
"""
from __future__ import annotations

import torch

BETA_FROZEN = 26.10


def wave_similarity(query: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """|cos| of one query against every bank row.

    Args:
        query: [D] complex (or real) wave.
        bank:  [N, D] complex (or real) waves.
    Returns:
        [N] real tensor in [0, 1].
    """
    q = query.reshape(-1).to(torch.complex64)
    q = q / q.norm().clamp_min(1e-12)
    b = bank.to(torch.complex64)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (b @ q.conj()).abs()


def membership_score(query: torch.Tensor, bank: torch.Tensor) -> float:
    """max_k |<psi_q, psi_k>| in [0, 1]. 1.0 means the query IS a stored wave."""
    if bank.numel() == 0:
        return 0.0
    return float(wave_similarity(query, bank).max())


def margin_score(query: torch.Tensor, bank: torch.Tensor,
                 beta: float = BETA_FROZEN) -> float:
    """beta * (top1 - top2) similarity. Peaked for a known input, flat for an unknown one."""
    s = wave_similarity(query, bank)
    if s.numel() < 2:
        return float("inf")
    top2 = torch.topk(s, k=2).values
    return float(beta * (top2[0] - top2[1]))


def calibrated_threshold(pos, neg) -> tuple:
    """Threshold maximizing balanced accuracy on a CALIBRATION split only.

    Returns (theta, balanced_accuracy). Never call this with eval families.
    """
    pos = [float(x) for x in pos]
    neg = [float(x) for x in neg]
    if not pos or not neg:
        return 0.5, float("nan")
    cand = sorted(set(pos + neg))
    best_t, best_bal = 0.5, -1.0
    for t in cand:
        tpr = sum(1 for x in pos if x >= t) / len(pos)
        tnr = sum(1 for x in neg if x < t) / len(neg)
        bal = 0.5 * (tpr + tnr)
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    return best_t, best_bal


def auc(pos, neg) -> float:
    """P(score(pos) > score(neg)). Ties count 0.5. Chance = 0.5."""
    pos = [float(x) for x in pos]
    neg = [float(x) for x in neg]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the token SETS. Explains a lexical gate's behaviour."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))
