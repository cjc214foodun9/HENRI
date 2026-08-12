"""Phase 7.5 D3 contracts: scorecard-delta irreversible progress detector.

Pre-registered contracts (manifest_D3_zero_demo_signal.md):
- C1 empty input -> fail-closed (False, prev, UNAVAILABLE)
- C2 equal counts -> no progress (strict increase only)
- C3 strict increase -> progressed True
- C4 malformed/negative values -> fail-closed per object
- C5 missing attribute -> UNAVAILABLE, never fabricated progress
- C6 exception safety -> UNAVAILABLE, no raise
- C7 monotonic rollback (new run) -> no false progress, lower base allowed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from arc_scorecard_delta import (
    SCORECARD_DELTA_OK,
    SCORECARD_DELTA_UNAVAILABLE,
    detect_level_progress,
)


class _FakeEnvScore:
    def __init__(self, levels_completed):
        self.levels_completed = levels_completed


class _NoAttrEnvScore:
    pass


def test_empty_inputs_fail_closed():
    prog, cur, status = detect_level_progress([], 3)
    assert prog is False
    assert cur == 3
    assert status == SCORECARD_DELTA_UNAVAILABLE


def test_none_inputs_fail_closed():
    prog, cur, status = detect_level_progress(None, 3)
    assert prog is False
    assert cur == 3
    assert status == SCORECARD_DELTA_UNAVAILABLE


def test_equal_counts_no_progress():
    envs = [_FakeEnvScore(3)]
    prog, cur, status = detect_level_progress(envs, 3)
    assert prog is False
    assert cur == 3
    assert status == SCORECARD_DELTA_OK


def test_strict_increase_detects_progress():
    envs = [_FakeEnvScore(4)]
    prog, cur, status = detect_level_progress(envs, 3)
    assert prog is True
    assert cur == 4
    assert status == SCORECARD_DELTA_OK


def test_max_across_environments():
    envs = [_FakeEnvScore(1), _FakeEnvScore(5), _FakeEnvScore(2)]
    prog, cur, status = detect_level_progress(envs, 4)
    assert prog is True
    assert cur == 5


def test_negative_value_skipped_fail_closed():
    envs = [_FakeEnvScore(-1), _FakeEnvScore(5)]
    prog, cur, status = detect_level_progress(envs, 4)
    assert prog is True
    assert cur == 5


def test_missing_attribute_unavailable():
    envs = [_NoAttrEnvScore()]
    prog, cur, status = detect_level_progress(envs, 3)
    assert prog is False
    assert cur == 3
    assert status == SCORECARD_DELTA_UNAVAILABLE


def test_all_malformed_unavailable():
    envs = [_FakeEnvScore("not-an-int"), _NoAttrEnvScore()]
    prog, cur, status = detect_level_progress(envs, 3)
    assert prog is False
    assert cur == 3
    assert status == SCORECARD_DELTA_UNAVAILABLE


def test_exception_aborts_scan_fail_closed():
    """Any object raising a non-value exception aborts the scan: fail-closed.

    Contract (manifest_D3): an anomalous scorecard must never fabricate
    progress. The whole scan is UNAVAILABLE; prev levels are preserved.
    """

    class _Boom:
        @property
        def levels_completed(self):
            raise RuntimeError("boom")

    envs = [_Boom(), _FakeEnvScore(4)]
    prog, cur, status = detect_level_progress(envs, 3)
    assert prog is False
    assert cur == 3
    assert status == SCORECARD_DELTA_UNAVAILABLE


def test_rollback_lower_base_no_false_progress():
    # New run reset: levels went backward. Must NOT report progress, but the
    # lower base is accepted so later real completions count.
    envs = [_FakeEnvScore(1)]
    prog, cur, status = detect_level_progress(envs, 5)
    assert prog is False
    assert cur == 1
    assert status == SCORECARD_DELTA_OK


def test_zero_levels_is_valid():
    envs = [_FakeEnvScore(0)]
    prog, cur, status = detect_level_progress(envs, 0)
    assert prog is False
    assert cur == 0
    assert status == SCORECARD_DELTA_OK
