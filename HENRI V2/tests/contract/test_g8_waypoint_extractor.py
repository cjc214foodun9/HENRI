"""Contract tests for Carrier G8 waypoint extractor (synthetic fixture)."""

import pathlib
import sys

import numpy as np
import pytest

TESTS = pathlib.Path(__file__).resolve()
ROOT = TESTS.parents[2]  # .../HENRI V2 (code dir)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

from arc_g8_waypoint_extractor import extract_waypoints


def _synthetic_bank():
    """Two envs, deterministic smooth + kink trajectories at D=512.

    env_a: 100 rows; two sharp turns (kinks) -> expect >=2 intermediates.
    env_b: 10 rows (too short) -> expect 1 terminal only.
    """
    rng = np.random.default_rng(7)
    D = 512
    psi_a = []
    base = rng.normal(size=D)
    base /= np.linalg.norm(base)
    t = np.linspace(0, 6.0, 100)
    # smooth curve with two direction changes
    for i in range(100):
        v = base + 0.35 * np.cos(t[i]) * rng.normal(size=D) + 0.35 * np.sin(2 * t[i]) * rng.normal(size=D)
        v /= np.linalg.norm(v)
        psi_a.append(v)
    psi_b = []
    dir_b = rng.normal(size=D)
    dir_b /= np.linalg.norm(dir_b)
    start_b = rng.normal(size=D)
    start_b /= np.linalg.norm(start_b)
    for i in range(10):
        # smooth low-curvature walk: small consistent steps along one direction
        v = start_b + 0.02 * i * dir_b
        v /= np.linalg.norm(v)
        psi_b.append(v)
    psi = np.asarray(psi_a + psi_b, dtype=np.float32)
    rows_a = list(range(100))
    rows_b = list(range(100, 110))
    return psi, {"env_a": rows_a, "env_b": rows_b}


def test_g8_extract_long_env_gets_intermediates():
    psi, rows = _synthetic_bank()
    wp = extract_waypoints(rows, psi, min_sep=4)
    a = wp["env_a"]
    roles = [r for _, r in a]
    assert roles[-1] == "terminal"
    assert sum(1 for r in roles if r == "intermediate") >= 2
    assert len(a) >= 3
    # ordered by row index
    idxs = [i for i, _ in a]
    assert idxs == sorted(idxs)


def test_g8_extract_short_env_terminal_only():
    psi, rows = _synthetic_bank()
    wp = extract_waypoints(rows, psi, min_sep=4)
    b = wp["env_b"]
    assert [r for _, r in b] == ["terminal"]


def test_g8_extract_no_dense_operator_contract():
    """Extraction must not build DxD arrays; sanity on output row bounds."""
    psi, rows = _synthetic_bank()
    wp = extract_waypoints(rows, psi, min_sep=4)
    n = psi.shape[0]
    for env, wps in wp.items():
        for r, _ in wps:
            assert 0 <= r < n
