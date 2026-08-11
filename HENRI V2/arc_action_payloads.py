"""ARC-AGI-3 action-payload generation (Gate 1 semantic repair).

arc-agi 0.9.9 contract: ``game.step(action, data=None, reasoning=None)``.
Coordinate-bearing actions (arcengine ``ACTION6``) consume SCREEN/display
coordinates: generated envs invert them with
``camera.display_to_grid(data["x"], data["y"])``, and arcengine's own action
producers emit ``screen = grid * scale + offset`` (top-left anchored, no Y
inversion; letterbox padding via camera offsets).

This module:
- transforms object centroids from render space into screen space with a
  pure, deterministic camera transform (``grid_to_display``);
- PREFERS environment-provided valid ``ActionInput`` objects
  (``_get_valid_clickable_actions`` / ``_get_valid_placeble_actions``),
  selecting the one nearest the segmented object in screen space;
- falls back to camera-transformed object centroids, then to a bounded
  deterministic grid fallback;
- labels every payload with its coordinate space in telemetry.

It is arcengine-agnostic: actions are matched by ``.name`` and camera
parameters are passed in, so contract tests run on local CPU without the
arcengine package.

Default path (flag OFF) is byte-identical: callers pass ``enabled=False``
and ``step_with_payload`` delegates to the bare ``game.step(action)`` call.
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
class CameraParams:
    """Camera transform metadata (arcengine-agnostic mirror).

    Mirrors ``Camera._calculate_scale_and_offset()`` output plus the
    viewport (arcengine MAX_DIMENSION = 64).
    """

    scale: int
    x_offset: int = 0
    y_offset: int = 0
    viewport: int = 64

    def validate(self) -> None:
        if self.scale < 1:
            raise ValueError(
                f"invalid camera scale {self.scale}: must be >= 1"
            )
        if self.viewport < 1:
            raise ValueError(
                f"invalid viewport {self.viewport}: must be >= 1"
            )


def grid_to_display(x: int, y: int, camera: CameraParams) -> Tuple[int, int]:
    """Convert render-space grid coordinates to screen/display coordinates.

    Mirrors arcengine's producer convention
    (``screen = grid * scale + offset``, top-left anchor, NO Y inversion)
    and clamps into the viewport. Deterministic and pure.
    """
    camera.validate()
    sx = int(x) * camera.scale + camera.x_offset
    sy = int(y) * camera.scale + camera.y_offset
    vmax = camera.viewport - 1
    return max(0, min(vmax, sx)), max(0, min(vmax, sy))


@dataclass(frozen=True)
class ArcActionCandidate:
    """Complete (action, data) candidate for the arcade step contract."""

    action: Any
    data: Optional[dict]
    source: str  # "object_centroid" | "fallback_grid" | "none"
    x: Optional[int]
    y: Optional[int]
    grid_x: Optional[int] = None
    grid_y: Optional[int] = None
    coordinate_space: str = "grid"  # "grid" | "screen"
    camera_scale: Optional[int] = None
    camera_offset: Optional[Tuple[int, int]] = None

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


def _env_action_inputs(game: Any) -> List[Tuple[int, int]]:
    """Collect environment-provided valid ActionInput screen coordinates.

    Uses the arcengine private producers when exposed; returns an empty
    list when unavailable (bounded, never raises).
    """
    base = getattr(game, "_game", None) or game
    points: List[Tuple[int, int]] = []
    for meth in ("_get_valid_clickable_actions", "_get_valid_placeble_actions"):
        try:
            for ai in getattr(base, meth)() or []:
                d = getattr(ai, "data", None)
                if isinstance(d, dict) and "x" in d and "y" in d:
                    points.append((int(d["x"]), int(d["y"])))
        except Exception:
            continue
    return points


def build_payload_candidates(
    grid: Sequence[Sequence[int]],
    allowed_actions: Sequence[Any],
    complex_action_names: Tuple[str, ...] = DEFAULT_COMPLEX_ACTION_NAMES,
    max_candidates: int = 8,
    seed: int = 0,
    camera: Optional[CameraParams] = None,
) -> List[ArcActionCandidate]:
    """Build complete (action, data) candidates from the current observation.

    - Simple actions: one candidate per action, ``data=None`` (no invented
      payloads).
    - Coordinate actions: candidates at connected-component object centroids
      (largest object first, up to ``max_candidates``), transformed into
      screen space when ``camera`` is provided; when the frame has no
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
            if camera is not None:
                sx, sy = grid_to_display(x, y, camera)
                candidates.append(
                    ArcActionCandidate(
                        action=action,
                        data={"x": sx, "y": sy},
                        source=source,
                        x=sx,
                        y=sy,
                        grid_x=x,
                        grid_y=y,
                        coordinate_space="screen",
                        camera_scale=camera.scale,
                        camera_offset=(camera.x_offset, camera.y_offset),
                    )
                )
            else:
                candidates.append(
                    ArcActionCandidate(
                        action=action,
                        data={"x": x, "y": y},
                        source=source,
                        x=x,
                        y=y,
                        grid_x=x,
                        grid_y=y,
                        coordinate_space="grid",
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
    camera: Optional[CameraParams] = None,
) -> Tuple[Any, dict]:
    """Execute ``game.step`` with the correct payload contract.

    ``enabled=False`` reproduces the legacy bare-enum call exactly.

    For coordinate actions with ``enabled=True``:
    1. environment-provided valid ActionInput (screen-space, nearest to the
       segmented object) when exposed;
    2. otherwise the camera-transformed object centroid (screen space);
    3. otherwise the deterministic fallback grid.

    Returns ``(obs_next, payload_info)`` where ``payload_info`` carries the
    telemetry fields: action_enum, payload_present, payload_x, payload_y,
    payload_source, payload_complete, coordinate_space, grid_x, grid_y,
    camera_scale, camera_offset.
    """
    info = {
        "action_enum": getattr(game_action, "name", str(game_action)),
        "payload_present": False,
        "payload_x": None,
        "payload_y": None,
        "payload_source": "none",
        "payload_complete": False,
        "coordinate_space": None,
        "grid_x": None,
        "grid_y": None,
        "camera_scale": None,
        "camera_offset": None,
    }
    if not enabled or grid is None:
        obs = game.step(game_action)
        return obs, info

    name = getattr(game_action, "name", str(game_action))
    if name in complex_action_names:
        candidates = build_payload_candidates(grid, [game_action],
                                              complex_action_names,
                                              seed=seed, camera=camera)
        selected = select_payload(candidates, game_action)
        if selected is not None and selected.data is not None:
            payload = selected.data
            source = selected.source
            coordinate_space = selected.coordinate_space
            grid_x, grid_y = selected.grid_x, selected.grid_y
            cam_scale = selected.camera_scale
            cam_offset = selected.camera_offset
            # Environment-provided valid ActionInputs are the semantic
            # oracle: prefer the one nearest the segmented object target.
            oracle = _env_action_inputs(game)
            if oracle:
                tx, ty = selected.x, selected.y
                px, py = min(
                    oracle,
                    key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2,
                )
                payload = {"x": px, "y": py}
                source = "env_actioninput"
                coordinate_space = "screen"
            info.update({
                "payload_present": True,
                "payload_x": payload["x"],
                "payload_y": payload["y"],
                "payload_source": source,
                "payload_complete": True,
                "coordinate_space": coordinate_space,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "camera_scale": cam_scale,
                "camera_offset": cam_offset,
            })
            obs = game.step(game_action, data=payload)
            return obs, info
    # Simple action or no payload available: bare-enum call (no invented data).
    obs = game.step(game_action)
    return obs, info
