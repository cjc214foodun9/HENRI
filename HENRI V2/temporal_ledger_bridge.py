"""Temporal ledger bridge for the live ARC production runner (Carrier 1).

Wires the T0 temporal transition ledger (default-OFF) into
production_arc_run.py: persists REAL (s_t, a_t, s_{t+1}) triples with
external-outcome meta (frame_changed, task_progressed, levels_completed,
terminal_state) and chain continuity. Fail-closed: any ledger defect raises
LEDGER_FAIL_CLOSED and blocks the step — recording is engagement, never a
silent pass (T0 contract).

The ledger and payload-store modules are provenance-pinned from
origin/feat/temporal-navigation-t0 @ 8fe4e7f (T0 ledger, K0 payloads).
This bridge adds NO trainable parameters and NO behavior when the flag is
absent; production_arc_run.py imports it lazily inside the enabled branch
(flag absent => module never imported, differential contract).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from temporal_transition_ledger import TemporalTransitionLedger


def record_temporal_transition(
    ledger: TemporalTransitionLedger,
    pre_grid: List[Any],
    game_action: Any,
    obs_next: Any,
    episode_id: str,
    step: int,
    extra_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Record one real (s_t, a_t, s_{t+1}) transition with outcome meta.

    The post-state and outcome are read from the REAL environment return
    (obs_next), never from the planner's own prediction. extra_meta merges
    runner-side context (executed macro list, reset flag) into the row.
    Raises LEDGER_FAIL_CLOSED (RuntimeError) on any defect or null post-state.
    """
    if obs_next is None or not getattr(obs_next, "frame", None):
        raise RuntimeError("LEDGER_FAIL_CLOSED: null post-state observation")
    frame0 = obs_next.frame[0]
    # Normalize the post-state: the ARC surface can yield a numpy array
    # (.tolist()) or already a plain nested list. Accept both; anything else
    # is a fail-closed defect (never persist an uncanonical post-state).
    if hasattr(frame0, "tolist"):
        post_grid = frame0.tolist()
    elif isinstance(frame0, list):
        post_grid = frame0
    else:
        raise RuntimeError(
            "LEDGER_FAIL_CLOSED: uncanonical post-state frame type "
            f"{type(frame0).__name__}")
    meta: Dict[str, Any] = dict(extra_meta or {})
    try:
        import numpy as np
        pre_arr = np.array(pre_grid)
        post_arr = np.array(post_grid)
        meta["frame_changed"] = bool(
            post_arr.shape == pre_arr.shape and np.any(post_arr != pre_arr))
    except Exception:
        meta["frame_changed"] = None
    try:
        task_progressed = False
        terminal_state = None
        if getattr(obs_next, "state", None):
            terminal_state = obs_next.state.name
        if terminal_state == "WIN":
            task_progressed = True
        if hasattr(obs_next, "levels_completed"):
            try:
                levels = int(obs_next.levels_completed)
                meta["levels_completed"] = levels
                if levels > 0:
                    task_progressed = True
            except Exception:
                pass
        meta["task_progressed"] = task_progressed
        if terminal_state:
            meta["terminal_state"] = terminal_state
    except Exception:
        meta["task_progressed"] = False
    try:
        return ledger.record(
            pre_grid, game_action, post_grid,
            episode_id=episode_id, step=step,
            t_phys=time.time(), meta=meta,
        )
    except Exception as exc:
        raise RuntimeError(
            f"LEDGER_FAIL_CLOSED:{type(exc).__name__}:{exc}") from exc


def ledger_summary(ledger: TemporalTransitionLedger) -> Dict[str, Any]:
    """Continuity check + counts for per-env telemetry (fail-closed)."""
    try:
        cont = ledger.continuity_check()
        return {
            "enabled": True,
            "records": len(ledger),
            "episodes": ledger.episodes(),
            "continuity_ok": bool(cont["ok"]),
            "continuity_violations": len(cont["violations"]),
        }
    except Exception as exc:
        raise RuntimeError(
            f"LEDGER_FAIL_CLOSED:{type(exc).__name__}:{exc}") from exc
