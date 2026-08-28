"""M2 horizon-8 open-loop coherence diagnostic (SPEC-2026-08-28-M2SUCC,
sealed #bb0be1c9).

Module contract (frozen):
- open_loop_rollout(transition, state_wave, action_wave, horizon=8) returns a
  list of 8 [num_blocks, 8] predictions; None on any non-finite output.
- Rollout is CAUSAL: only the current (already-executed) action wave is used;
  no future action waves enter.
- sagnac_delta(pred, emp) = 1 - cos(pred, emp) in [0, 2] (full-wave cosine,
  clamped for float32 noise).
- per_horizon_means(deltas) -> {k: mean} for k in 1..8.
- m2_engaged(deltas) -> True iff at least one finite delta was emitted.

Default-OFF diagnostic (HENRI_M2_COHERENCE=1); no action-policy influence.
"""

from __future__ import annotations

import numpy as np
import torch

M2_HORIZON = 8


def open_loop_rollout(transition, state_wave, action_wave, horizon=M2_HORIZON):
    """Roll the live transition H steps open-loop.

    pred_{k+1} = T(pred_k, a_t) with pred_0 = state_wave(t). The SAME
    (already-executed) action wave is reused across the roll — no future
    action waves enter (causal). Returns list of `horizon` [num_blocks, 8]
    tensors, or None on any failure (fail-closed).
    """
    if horizon < 1 or horizon > M2_HORIZON:
        raise ValueError(f"horizon must be in [1, {M2_HORIZON}]")
    if state_wave is None or action_wave is None:
        return None
    try:
        preds = []
        cur = state_wave
        for _ in range(horizon):
            cur = transition(cur, action_wave)
            if cur is None or not torch.isfinite(cur).all():
                return None
            preds.append(cur)
        return preds
    except Exception:
        return None


def sagnac_delta(pred, emp):
    """Normalized Sagnac residual 1 - cos(pred, emp) in [0, 2].

    Full-wave cosine over the flattened [num_blocks, 8] tensors. Cosine is
    clamped to [-1, 1] for float32 noise so the delta stays in [0, 2].
    """
    p = pred.reshape(-1).float()
    e = emp.reshape(-1).float()
    num = float((p * e).sum().item())
    den = float(torch.norm(p).item() * torch.norm(e).item())
    if den <= 1e-12:
        return 1.0
    cos = max(-1.0, min(1.0, num / den))
    return 1.0 - cos


def per_horizon_means(deltas):
    """{k: mean over steps of delta_k} for k with any values."""
    return {k: float(np.mean(v)) for k, v in deltas.items() if len(v) > 0}


def m2_engaged(deltas):
    """True iff at least one horizon accumulated at least one delta."""
    return any(len(v) > 0 for v in deltas.values())
