"""P1 carrier: five-step exteroceptive failure trace (instrumentation-only).

Sliding-window temporal trace (k=5) over exteroceptive score deltas,
resolving the retroactive valence nu per the SOTA-Gap required
implementation A, but as TELEMETRY ONLY:

    window_delta = sum(score_delta over the last k records)
    nu = 0.0   if window_delta > 0
    nu = -1.0  if window_delta <= 0

The module performs NO parameter mutation, NO heat injection, and owns
NO trainable state. Coordinate attribution and bounded anisotropic heat
injection are gated behind P2/P3 carriers (sealed preregs). This module
is the causal-consumer prerequisite: it proves the window fires on real
persisted transition rows without touching model state.

Default-OFF: HENRI_FAILURE_TRACE=1 must be set; construction raises
FailureTraceDisabledError otherwise (T0-ledger pattern).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

FLAG = "HENRI_FAILURE_TRACE"
DEFAULT_K = 5


class FailureTraceDisabledError(RuntimeError):
    """Raised when HENRI_FAILURE_TRACE is not set (default-OFF invariant)."""


def resolve_valence(window_delta: float) -> float:
    """Retroactive valence: 0.0 on progress, -1.0 on stall/zero (spec)."""
    return 0.0 if window_delta > 0.0 else -1.0


class FailureTraceWindow:
    """Sliding k-step window resolving nu over exteroceptive score deltas.

    Pure state machine: no nn.Module, no Parameter, no optimizer, no
    mutation of any planner/model state.
    """

    def __init__(self, k: int = DEFAULT_K, *, flag: str = FLAG):
        if os.environ.get(flag, "0") != "1":
            raise FailureTraceDisabledError(
                f"{flag} is not set; the failure trace is default-OFF")
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = k
        self._flag = flag
        self._episode_id: Optional[str] = None
        self._buf: List[Dict[str, Any]] = []
        self._windows_resolved = 0
        self._stall_windows = 0
        self._progress_windows = 0
        self.resolved_windows: List[Dict[str, Any]] = []

    def reset(self, episode_id: str) -> None:
        """Explicit episode boundary: clears the window (T0 reset pattern)."""
        self._episode_id = episode_id
        self._buf.clear()
        self.resolved_windows.clear()

    def observe(self, step: int, action: str, score_delta: float) -> Dict[str, Any]:
        """Append one (step, action, score_delta); resolve at k records.

        Returns PENDING until the window has k records, then RESOLVED with
        window_delta and nu; the window then slides (oldest evicted).
        """
        self._buf.append({"step": step, "action": action,
                          "score_delta": float(score_delta)})
        if len(self._buf) < self.k:
            return {"status": "PENDING", "window_len": len(self._buf), "nu": None}
        window = self._buf[-self.k:]
        window_delta = sum(r["score_delta"] for r in window)
        nu = resolve_valence(window_delta)
        self._windows_resolved += 1
        if nu < 0.0:
            self._stall_windows += 1
        else:
            self._progress_windows += 1
        self.resolved_windows.append({
            "episode_id": self._episode_id,
            "window_delta": window_delta,
            "nu": nu,
            "steps": [r["step"] for r in window],
            "actions": [r["action"] for r in window],
            "score_deltas": [r["score_delta"] for r in window],
        })
        self._buf = self._buf[-(self.k - 1):]
        return {
            "status": "RESOLVED",
            "window_len": self.k,
            "window_delta": window_delta,
            "nu": nu,
            "actions": [r["action"] for r in window],
            "steps": [r["step"] for r in window],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "episode_id": self._episode_id,
            "window_len": len(self._buf),
            "windows_resolved": self._windows_resolved,
            "stall_windows": self._stall_windows,
            "progress_windows": self._progress_windows,
            "trainable_parameters": 0,
        }
