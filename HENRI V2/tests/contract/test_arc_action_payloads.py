"""Gate 1 contract tests: ARC action-payload completeness (HENRI_ARC_ACTION_PAYLOADS).

Proves, on CPU with fake environments:
1. coordinate actions cannot reach game.step without required payload;
2. coordinates are in bounds;
3. non-coordinate actions receive no invented payloads;
4. deterministic seed -> deterministic candidates;
5. fake env sees the exact action and data;
6. flag-off path is byte-identical to legacy bare-enum behavior.
"""

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "HENRI V2"))

from arc_action_payloads import (  # noqa: E402
    ArcActionCandidate,
    build_payload_candidates,
    select_payload,
    step_with_payload,
)


class _FakeAction:
    def __init__(self, name):
        self.name = name


class _FakeGame:
    """Records every step call; returns a dummy observation."""

    def __init__(self):
        self.calls = []
        self.obs = object()

    def step(self, action, data=None, reasoning=None):
        self.calls.append((action, data))
        return self.obs


GRID_OBJ = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 2, 2],
]
GRID_EMPTY = [[0, 0], [0, 0]]


def _grid6():
    return [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]


# --- 1. coordinate actions cannot reach step without payload -----------------
def test_coordinate_action_requires_payload_candidate():
    acts = [_FakeAction("ACTION6")]
    cands = build_payload_candidates(GRID_OBJ, acts)
    assert cands, "coordinate action must yield payload candidates"
    for c in cands:
        assert c.data is not None
        assert "x" in c.data and "y" in c.data
        assert c.payload_complete


def test_coordinate_action_no_invented_payload_when_flag_off():
    game = _FakeGame()
    act = _FakeAction("ACTION6")
    # flag OFF -> legacy bare-enum call, NO data.
    obs, info = step_with_payload(game, act, GRID_OBJ, enabled=False)
    assert obs is game.obs
    assert game.calls == [(act, None)]
    assert info["payload_present"] is False
    assert info["payload_complete"] is False


def test_coordinate_action_payload_forwarded_when_flag_on():
    game = _FakeGame()
    act = _FakeAction("ACTION6")
    obs, info = step_with_payload(game, act, GRID_OBJ, enabled=True, seed=0)
    assert obs is game.obs
    assert len(game.calls) == 1
    _, data = game.calls[0]
    assert data is not None and "x" in data and "y" in data
    assert info["payload_present"] is True
    assert info["payload_complete"] is True
    assert info["payload_source"] in ("object_centroid", "fallback_grid")


# --- 2. coordinates in bounds -------------------------------------------------
@pytest.mark.parametrize("grid", [GRID_OBJ, GRID_EMPTY, _grid6()])
def test_payload_coordinates_in_bounds(grid):
    acts = [_FakeAction("ACTION6")]
    cands = build_payload_candidates(grid, acts, max_candidates=16)
    assert cands
    h, w = len(grid), len(grid[0])
    for c in cands:
        assert 0 <= c.x < w, f"x={c.x} out of bounds w={w}"
        assert 0 <= c.y < h, f"y={c.y} out of bounds h={h}"


# --- 3. non-coordinate actions receive no invented payloads -------------------
def test_simple_action_no_payload():
    acts = [_FakeAction("ACTION1"), _FakeAction("ACTION7")]
    cands = build_payload_candidates(GRID_OBJ, acts)
    for c in cands:
        assert c.data is None
        assert c.payload_complete is False
        assert c.source == "none"


# --- 4. deterministic seed -> deterministic candidates ------------------------
def test_deterministic_candidates():
    acts = [_FakeAction("ACTION6")]
    a = build_payload_candidates(GRID_OBJ, acts, seed=7)
    b = build_payload_candidates(GRID_OBJ, acts, seed=7)
    assert [(c.x, c.y, c.source) for c in a] == [(c.x, c.y, c.source) for c in b]


# --- 5. fake env sees the exact action and data ------------------------------
def test_fake_env_exact_call():
    game = _FakeGame()
    act = _FakeAction("ACTION6")
    step_with_payload(game, act, GRID_OBJ, enabled=True, seed=1)
    assert len(game.calls) == 1
    seen_action, seen_data = game.calls[0]
    assert seen_action is act
    assert isinstance(seen_data, dict) and set(seen_data) == {"x", "y"}
    assert isinstance(seen_data["x"], int) and isinstance(seen_data["y"], int)


# --- 6. flag-off path byte-identical to legacy bare-enum behavior -------------
def test_flag_off_legacy_identical():
    game = _FakeGame()
    acts = [_FakeAction("ACTION1"), _FakeAction("ACTION6")]
    for act in acts:
        step_with_payload(game, act, GRID_OBJ, enabled=False)
    assert game.calls == [(acts[0], None), (acts[1], None)]


# --- candidate selection priority --------------------------------------------
def test_select_payload_prefers_object_centroid():
    acts = [_FakeAction("ACTION6")]
    cands = build_payload_candidates(GRID_OBJ, acts, seed=0)
    sel = select_payload(cands, acts[0])
    assert sel is not None
    # object-centroid source is available for GRID_OBJ
    assert sel.source == "object_centroid"


def test_select_payload_none_when_no_match():
    cands = build_payload_candidates(GRID_OBJ, [_FakeAction("ACTION6")])
    assert select_payload(cands, _FakeAction("ACTION9")) is None


# --- Gate 1 semantic repair: screen-space transform + oracle preference ------
from arc_action_payloads import CameraParams, grid_to_display  # noqa: E402


def test_grid_to_display_identity():
    assert grid_to_display(3, 4, CameraParams(scale=1, x_offset=0, y_offset=0)) == (3, 4)


def test_grid_to_display_scaled_and_offset():
    assert grid_to_display(2, 3, CameraParams(scale=8, x_offset=4, y_offset=6)) == (20, 30)


def test_grid_to_display_no_y_inversion():
    # arcengine producers: screen = grid*scale + offset (top-left, NO Y flip).
    assert grid_to_display(1, 7, CameraParams(scale=8)) == (8, 56)


def test_grid_to_display_clamps_to_viewport():
    assert grid_to_display(100, -5, CameraParams(scale=8, viewport=64)) == (63, 0)


def test_invalid_camera_scale_fails_closed():
    with pytest.raises(ValueError):
        grid_to_display(1, 1, CameraParams(scale=0))


def test_payload_candidates_screen_space_with_camera():
    acts = [_FakeAction("ACTION6")]
    cands = build_payload_candidates(
        GRID_OBJ, acts, camera=CameraParams(scale=2, x_offset=3, y_offset=5))
    assert cands
    for c in cands:
        assert c.coordinate_space == "screen"
        assert c.data is not None and c.data["x"] == c.x and c.data["y"] == c.y
        assert c.grid_x is not None and c.grid_y is not None
        assert c.data["x"] == c.grid_x * 2 + 3
        assert c.data["y"] == c.grid_y * 2 + 5


def test_env_actioninput_preferred_over_transform():
    class _OracleActionInput:
        data = {"x": 12, "y": 34}

    class _OracleGame:
        def __init__(self):
            self.calls = []
            self.obs = object()

        def step(self, action, data=None, reasoning=None):
            self.calls.append((action, data))
            return self.obs

        class _G:
            @staticmethod
            def _get_valid_clickable_actions():
                return [_OracleActionInput()]

            @staticmethod
            def _get_valid_placeble_actions():
                return []

        _game = _G()

    game = _OracleGame()
    act = _FakeAction("ACTION6")
    obs, info = step_with_payload(game, act, GRID_OBJ, enabled=True, seed=0,
                                  camera=CameraParams(scale=1))
    _, data = game.calls[0]
    assert data == {"x": 12, "y": 34}
    assert info["payload_source"] == "env_actioninput"
    assert info["coordinate_space"] == "screen"
    assert obs is game.obs
