"""Phase 7.5 D3: authoritative irreversible-progress detection via scorecard
levels delta.

Corpus grounding (bank ca4bb787, convo 3179135d, 2026-08-12): the only honest
valence signal is derived from irreversible exteroceptive scorecard deltas
(level_actions / level_scores / levels_completed), NOT frame deltas. RESET
loops are high-entropy traps and must never count as progress.

This module is a pure, fail-closed helper. It NEVER fabricates progress:
any anomaly (missing attribute, empty list, negative count, exception)
returns (progressed=False, current=previous) — the caller falls back to the
pre-existing WIN / observation-levels detection unchanged.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

SCORECARD_DELTA_OK = "SCORECARD_DELTA_OK"
SCORECARD_DELTA_UNAVAILABLE = "SCORECARD_DELTA_UNAVAILABLE"


def detect_level_progress(
    environment_scores: Sequence[Any],
    prev_levels: int,
) -> Tuple[bool, int, str]:
    """Detect irreversible level-completion progress from scorecard data.

    Args:
        environment_scores: iterable of per-environment score objects. Each
            object may expose ``levels_completed`` (int). Missing or
            malformed objects are skipped.
        prev_levels: the last observed levels-completed count (per env).

    Returns:
        (progressed, current_levels, status):
        - progressed: True only when current > prev (strictly irreversible).
        - current_levels: the max levels_completed across environments, or
          prev_levels when the signal is unavailable (fail-closed).
        - status: SCORECARD_DELTA_OK when the signal was read cleanly,
          SCORECARD_DELTA_UNAVAILABLE when it could not be read.
    """
    if not environment_scores:
        return False, int(prev_levels), SCORECARD_DELTA_UNAVAILABLE
    current = 0
    any_read = False
    try:
        for env_score in environment_scores:
            try:
                val = getattr(env_score, "levels_completed", None)
                if val is None:
                    continue
                val_i = int(val)
                if val_i < 0:
                    continue  # malformed: fail-closed per-object
                any_read = True
                current = max(current, val_i)
            except (TypeError, ValueError):
                continue
    except Exception:
        return False, int(prev_levels), SCORECARD_DELTA_UNAVAILABLE

    if not any_read:
        return False, int(prev_levels), SCORECARD_DELTA_UNAVAILABLE

    prev = int(prev_levels)
    progressed = current > prev
    return progressed, current, SCORECARD_DELTA_OK
