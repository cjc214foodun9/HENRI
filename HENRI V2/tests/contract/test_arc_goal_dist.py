"""R2-next carrier: goal_dist_var guard contracts (2026-08-28).

Pre-registered repair for the BLOCKED_INFRASTRUCTURE round(None) crash
(production_arc_run.py:2605, TypeError: type NoneType doesn't define
__round__ method, fired at env 4 vc33 on the first pilot attempt).

Contracts (test-first):
- C0 empty EFE table -> None (never round(None))
- C1 single observation -> None
- C2 two+ observations -> sample variance rounded to 6 decimals
- C3 None goal_distance entries excluded before variance
- C4 malformed entries -> None, never raise
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from arc_goal_dist import compute_goal_dist_var


def test_empty_table_returns_none():
    assert compute_goal_dist_var([]) is None


def test_single_observation_returns_none():
    assert compute_goal_dist_var([{"goal_distance": 0.5}]) is None


def test_two_observations_sample_variance():
    # sample variance of [0.5, 0.7]: mean 0.6, sq devs 0.01+0.01, /(n-1) = 0.02
    assert compute_goal_dist_var(
        [{"goal_distance": 0.5}, {"goal_distance": 0.7}]
    ) == 0.02


def test_none_entries_excluded():
    assert compute_goal_dist_var(
        [
            {"goal_distance": 0.5},
            {"goal_distance": None},
            {"goal_distance": 0.7},
        ]
    ) == 0.02


def test_all_none_returns_none():
    assert compute_goal_dist_var(
        [{"goal_distance": None}, {"goal_distance": None}]
    ) is None


def test_malformed_entries_fail_closed():
    assert compute_goal_dist_var([{"efe": 1.0}, {}]) is None
    assert compute_goal_dist_var([{"goal_distance": "bad"}]) is None
