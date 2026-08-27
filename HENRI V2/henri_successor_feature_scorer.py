"""HENRI Successor-Feature Action Scoring (SFAS) — Arm F (default OFF).

Pre-registration: docs/arm_f_sfas_pre_registration.md (SHA recorded at seal).
Approved 2026-08-27 (user: "Approved s1 and the r1-r4 fixes, and bounded
carrier proposal. Please implement and wire in now").

Mechanism (S1): per-action successor features
    psi_a(s) = sum_{k=0}^{H-1} gamma^k * K_a^k * phi(s)
computed MATRIX-FREE by iterating the LIVE action-conditioned transition
operator: K_a x := transition(x, a_wave) (FHRR bind + field channel of
LowRankCoupledTransition, efe_planner.py:158-178). The goal score becomes
candidate-specific:
    score(s, a) = cos(psi_a(s), phi(g))
and the EFE table is re-ranked by the blended score
    efe' = efe + lambda_sfas * (1 - score)
(lower is better, preserving the argmin semantics of EFE selection).

This is the adapter lever (user constraint: no learner or codec tuning):
the scorer READS action-specific discrimination from the transition
operator's own action-conditioned responses instead of a shared projection
(Arm E FALSIFIED: goal_dist ~ 1.0 for every action; a shared-span projector
cannot manufacture discrimination the operator lacks).

Zero trainable: the transition is called read-only (detached inputs), never
mutated. No dense [d, d] tensor: each rollout step is the operator's own
[blocks, 8] forward pass.

Fail-closed: any missing factor, shape mismatch, or non-finite value returns
scores=None; the caller keeps the EFE order byte-identical.

Hardware invariants:
  - horizon H in [1, 4] (pre-registered bound; default 2).
  - gamma in [0, 1); default 0.9.
  - lambda_sfas >= 0; default 1.0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def successor_feature(
    transition: torch.nn.Module,
    state_wave: torch.Tensor,
    action_wave: torch.Tensor,
    horizon: int = 2,
    gamma: float = 0.9,
) -> Optional[torch.Tensor]:
    """Roll the action-conditioned transition H-1 steps and accumulate the
    discounted successor feature.

    psi_a(s) = sum_{k=0}^{H-1} gamma^k * K_a^k * phi(s), where
    K_a x := transition(x, a_wave).

    Args:
        transition: callable (state_wave, action_wave) -> [B, D] real wave.
        state_wave: [B, D] real Clifford wave.
        action_wave: [B, D] real action wave.
        horizon: H in [1, 4].
        gamma: discount in [0, 1).

    Returns [B, D] tensor normalized to unit L2 norm (global, matching the
    goal_distance convention), or None on any failure (fail-closed).
    """
    if horizon < 1 or horizon > 4:
        raise ValueError("horizon must be in [1, 4]")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    if state_wave is None or action_wave is None:
        return None
    try:
        acc = torch.zeros_like(state_wave)
        cur = state_wave
        gk = 1.0
        for _ in range(horizon):
            acc = acc + gk * cur
            cur = transition(cur, action_wave)
            if cur is None or not torch.isfinite(cur).all():
                return None
            gk *= gamma
        norm = torch.norm(acc, p=2)
        if not torch.isfinite(norm) or norm <= 1e-12:
            return None
        return acc / (norm + 1e-9)
    except Exception:
        return None


def compute_sfas_scores(
    state_wave: torch.Tensor,
    goal_wave: torch.Tensor,
    action_waves: Dict[Any, torch.Tensor],
    transition: torch.nn.Module,
    horizon: int = 2,
    gamma: float = 0.9,
) -> Optional[Dict[Any, float]]:
    """Per-action successor-feature goal scores.

    score(s, a) = cos(psi_a(s), phi(g)) in [-1, 1]; higher = closer to goal.

    Args:
        state_wave: [B, D] real wave.
        goal_wave: [B, D] real goal wave.
        action_waves: {action_key: [B, D] real action wave}.
        transition: callable (state, action_wave) -> [B, D] real wave.

    Returns {action_key: float} for every provided action, or None on any
    failure (fail-closed). Missing goal/state returns None.
    """
    if state_wave is None or goal_wave is None or not action_waves:
        return None
    try:
        g = goal_wave.reshape(-1)
        g = g / (torch.norm(g) + 1e-12)
        scores: Dict[Any, float] = {}
        for key, a_wave in action_waves.items():
            psi = successor_feature(
                transition, state_wave, a_wave, horizon=horizon, gamma=gamma)
            if psi is None:
                return None
            p = psi.reshape(-1)
            scores[key] = float(torch.dot(p, g).item())
        if not all(torch.isfinite(torch.tensor(v)).item() for v in scores.values()):
            return None
        return scores
    except Exception:
        return None


def rerank_efe_table(
    efe_table: List[dict],
    scores: Optional[Dict[Any, float]],
    lambda_sfas: float = 1.0,
) -> tuple:
    """Blend SFAS goal scores into the EFE table and re-rank ascending.

    blended = efe + lambda_sfas * (1 - score) when a score exists for the
    row's action; otherwise the row keeps its raw EFE (no score available).
    Stable sort ascending (argmin semantics). Returns (new_table, info):

    info = {
        "reordered": bool,
        "discordance": int (rows whose rank position changed),
        "scores": [float | None per row],
        "blended": [float per row],
        "horizon": int, "gamma": float, "lambda_sfas": float,
    }

    Fail-closed: scores=None leaves the table order byte-identical
    (discordance 0, reordered False).
    """
    if lambda_sfas < 0.0:
        raise ValueError("lambda_sfas must be >= 0.0")
    if not efe_table:
        return [], {
            "reordered": False, "discordance": 0, "scores": [],
            "blended": [], "horizon": None, "gamma": None,
            "lambda_sfas": lambda_sfas,
        }
    if scores is None:
        return list(efe_table), {
            "reordered": False, "discordance": 0,
            "scores": [None] * len(efe_table),
            "blended": [float(r.get("efe", 0.0)) for r in efe_table],
            "horizon": None, "gamma": None, "lambda_sfas": lambda_sfas,
        }
    rows_scores: List[Optional[float]] = []
    rows_blended: List[float] = []
    for r in efe_table:
        key = r.get("action")
        k = int(key.value if hasattr(key, "value") else key) if key is not None else None
        sc = scores.get(k) if k is not None else None
        rows_scores.append(sc)
        base = float(r.get("efe", 0.0))
        rows_blended.append(base + lambda_sfas * (1.0 - sc) if sc is not None else base)
    indexed = list(enumerate(efe_table))
    ranked = sorted(indexed, key=lambda p: rows_blended[p[0]])
    new_table = [r for _, r in ranked]
    discordance = sum(
        1 for new_pos, (old_pos, _) in enumerate(ranked) if new_pos != old_pos)
    return new_table, {
        "reordered": discordance > 0,
        "discordance": discordance,
        "scores": rows_scores,
        "blended": rows_blended,
        "horizon": None,  # filled by the caller's config
        "gamma": None,
        "lambda_sfas": lambda_sfas,
    }
