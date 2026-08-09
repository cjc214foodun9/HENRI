"""ARC-AGI-3 action-payload generation (Gate 1, bounded repair).

arc-agi 0.9.9 contract: ``game.step(action, data=None, reasoning=None)``.
Coordinate-bearing actions (arcengine ``ACTION6`` / ComplexAction) require
``data={"x": <screen_x>, "y": <screen_y>}``; a bare enum is a near-no-op
(observed: 1/4096 pixels changed vs 21/4096 with data).

This module derives payload coordinates from the CURRENT OBSERVATION
(connected-component object centroids, deterministic fallback grid) — never
from arbitrary hashes. It is arcengine-agnostic: actions are passed in and
matched by ``.name`` against ``complex_action_names``, so contract tests run
on local CPU without the arcengine package.

Default path (flag OFF) is byte-identical: callers pass ``enabled=False`` and
``step_with_payload`` delegates to the bare ``game.step(action)`` call.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from connected_component_segmenter import ConnectedComponentSegmenter

# arcengine enum name for the coordinate-bearing action (ComplexAction).
DEFAULT_COMPLEX_ACTION_NAMES: Tuple[str, ...] = ("ACTION6",)

# Fallback grid coordinates used only when the observation has no objects.
_FALLBACK_GRID_POINTS = (
    (0, 0),       # top-left
    (0, 0),       # placeholder replaced by (W-1, 0)
    (0, 0),       # placeholder replaced by (0, H-1)
    (0, 0),       # placeholder replaced by (W-1, H-1)
    (0, 0),       # placeholder replaced by center
)


@dataclass(frozen=True)
class ArcActionCandidate:
    """Complete (action, data) candidate for the arcade step contract."""

    action: Any
    data: Optional[dict]
    source: str  # "object_centroid" | "fallback_grid" | "none"
    x: Optional[int]
    y: Optional[int]

    @property
    def payload_complete(self) -> bool:
        return self.data is not None


def _fallback_points(width: int, height: int) -> List[Tuple[int, int]]:
    """Bounded deterministic fallback coordinate set within the frame."""
    cx, cy = width // 2, height // 2
    pts = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
        (cx, cy),
    ]
    # De-duplicate while preserving order.
    seen = set()
    out = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build_payload_candidates(
    grid: Sequence[Sequence[int]],
    allowed_actions: Sequence[Any],
    complex_action_names: Tuple[str, ...] = DEFAULT_COMPLEX_ACTION_NAMES,
    max_candidates: int = 8,
    seed: int = 0,
) -> List[ArcActionCandidate]:
    """Build complete (action, data) candidates from the current observation.

    - Simple actions: one candidate per action, ``data=None`` (no invented
      payloads).
    - Coordinate actions: candidates at connected-component object centroids
      (largest object first, up to ``max_candidates``); when the frame has no
      objects, a bounded deterministic fallback grid is used.

    Deterministic for a fixed ``seed`` and grid.
    """
    arr = [[int(v) for v in row] for row in grid]
    height = len(arr)
    width = len(arr[0]) if height else 0

    rng = random.Random(seed)

    segmenter = ConnectedComponentSegmenter(background_color=0)
    objects = segmenter.segment_grid(arr)
    # Largest objects first (most salient target), deterministic tie-break.
    objects = sorted(
        objects, key=lambda o: (-o.area, o.object_id)
    )

    candidates: List[ArcActionCandidate] = []
    for action in allowed_actions:
        name = getattr(action, "name", str(action))
        if name not in complex_action_names:
            candidates.append(
                ArcActionCandidate(action=action, data=None, source="none",
                                   x=None, y=None)
            )
            continue

        coords: List[Tuple[int, int]] = []
        source = "object_centroid"
        for obj in objects[:max_candidates]:
            cy, cx = obj.tracking_key  # (centroid_y, centroid_x)
            coords.append((int(round(cx)), int(round(cy))))
        if not coords:
            source = "fallback_grid"
            coords = _fallback_points(width, height)[:max_candidates]

        for (x, y) in coords:
            if not (0 <= x < width and 0 <= y < height):
                continue
            candidates.append(
                ArcActionCandidate(
                    action=action,
                    data={"x": x, "y": y},
                    source=source,
                    x=x,
                    y=y,
                )
            )

    # Deterministic shuffle for stable ordering under a fixed seed.
    rng.shuffle(candidates)
    return candidates


def select_payload(
    candidates: Sequence[ArcActionCandidate],
    chosen_action: Any,
) -> Optional[ArcActionCandidate]:
    """Select the payload candidate for the chosen action.

    Prefers object-centroid candidates over the fallback grid. Returns the
    first matching candidate (deterministic ordering), or None when the
    chosen action has no payload requirement or no candidate exists.
    """
    chosen_name = getattr(chosen_action, "name", str(chosen_action))
    matching = [c for c in candidates
                if getattr(c.action, "name", str(c.action)) == chosen_name]
    if not matching:
        return None
    # Priority: object_centroid > fallback_grid (simple actions carry no data).
    matching.sort(key=lambda c: 0 if c.source == "object_centroid" else 1)
    return matching[0]


def step_with_payload(
    game: Any,
    game_action: Any,
    grid: Optional[Sequence[Sequence[int]]],
    enabled: bool,
    complex_action_names: Tuple[str, ...] = DEFAULT_COMPLEX_ACTION_NAMES,
    seed: int = 0,
) -> Tuple[Any, dict]:
    """Execute ``game.step`` with the correct payload contract.

    ``enabled=False`` reproduces the legacy bare-enum call exactly.

    Returns ``(obs_next, payload_info)`` where ``payload_info`` carries the
    telemetry fields: action_enum, payload_present, payload_x, payload_y,
    payload_source, payload_complete.
    """
    info = {
        "action_enum": getattr(game_action, "name", str(game_action)),
        "payload_present": False,
        "payload_x": None,
        "payload_y": None,
        "payload_source": "none",
        "payload_complete": False,
    }
    if not enabled or grid is None:
        obs = game.step(game_action)
        return obs, info

    name = getattr(game_action, "name", str(game_action))
    if name in complex_action_names:
        candidates = build_payload_candidates(grid, [game_action],
                                              complex_action_names, seed=seed)
        selected = select_payload(candidates, game_action)
        if selected is not None and selected.data is not None:
            info.update({
                "payload_present": True,
                "payload_x": selected.x,
                "payload_y": selected.y,
                "payload_source": selected.source,
                "payload_complete": selected.payload_complete,
            })
            obs = game.step(game_action, data=selected.data)
            return obs, info
    # Simple action or no payload available: bare-enum call (no invented data).
    obs = game.step(game_action)
    return obs, info
