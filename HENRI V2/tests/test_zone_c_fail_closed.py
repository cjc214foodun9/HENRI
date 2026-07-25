import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zone_c_segment_cache import DatabaseConnectionError, InProcessZoneCStore, SegmentCache


def test_explicit_surrogate_is_available_for_offline_tests(monkeypatch):
    monkeypatch.delenv("MOCK_TEST_MODE", raising=False)
    cache = SegmentCache.connect(dsn="offline://surrogate", num_blocks=4)
    assert isinstance(cache.store, InProcessZoneCStore)


def test_unreachable_live_target_fails_closed(monkeypatch):
    monkeypatch.delenv("MOCK_TEST_MODE", raising=False)
    monkeypatch.setenv("ZONE_C_ENV", "dev")
    dsn = "postgres://zonec_dev_user:zonec_dev@127.0.0.1:1/henri_zonec_dev"
    with pytest.raises(DatabaseConnectionError):
        SegmentCache.connect(dsn=dsn, num_blocks=4)


def test_mock_mode_is_explicit(monkeypatch):
    monkeypatch.setenv("MOCK_TEST_MODE", "1")
    monkeypatch.setenv("ZONE_C_ENV", "dev")
    dsn = "postgres://zonec_dev_user:zonec_dev@127.0.0.1:1/henri_zonec_dev"
    cache = SegmentCache.connect(dsn=dsn, num_blocks=4)
    assert isinstance(cache.store, InProcessZoneCStore)
