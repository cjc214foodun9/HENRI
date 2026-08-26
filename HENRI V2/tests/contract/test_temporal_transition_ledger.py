"""Contract tests: T0 temporal transition ledger (default-OFF substrate).

Run on the Vast CUDA target with /venv/main/bin/python:
    PYTHONPATH="$(pwd)" /venv/main/bin/python tests/contract/test_temporal_transition_ledger.py
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from temporal_transition_ledger import (
    FLAG,
    ContinuityViolationError,
    MissingActionError,
    StaleStateError,
    TemporalLedgerDisabledError,
    TemporalTransitionLedger,
    get_ledger,
    wave_digest,
)


def _wave(seed: int, n_blocks: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_blocks, 8, generator=g)
    return w / w.norm(dim=-1, keepdim=True)


@pytest.fixture
def flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    return tmp_path


def test_c1_default_off(tmp_path, monkeypatch):
    """C1: absent flag -> construction and factory fail closed."""
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(TemporalLedgerDisabledError):
        TemporalTransitionLedger(tmp_path / "ledger.jsonl")
    assert get_ledger(tmp_path / "ledger.jsonl") is None


def test_c2_continuity_pass(flag_on):
    """C2: chained transitions record[t].obs_next == record[t+1].obs_t."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl")
    w0, w1, w2 = _wave(1), _wave(2), _wave(3)
    led.record(w0, "a0", w1, episode_id="ep1", step=0)
    led.record(w1, "a1", w2, episode_id="ep1", step=1)
    chk = led.continuity_check()
    assert chk["ok"] is True
    assert chk["episodes"] == {"ep1": 2}
    assert len(led) == 2


def test_c3_reset_boundary(flag_on):
    """C3: explicit reset / episode change breaks the chain deliberately."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl")
    w0, w1 = _wave(1), _wave(2)
    led.record(w0, "a0", w1, episode_id="ep1", step=0)
    led.record(w1, "a1", w0, episode_id="ep1", step=1)
    led.reset("ep2")  # next record starts fresh
    led.record(_wave(9), "a2", _wave(10), episode_id="ep2", step=0)
    chk = led.continuity_check()
    assert chk["ok"] is True
    assert chk["episodes"] == {"ep1": 2, "ep2": 1}


def test_c4_missing_action_fail_closed(flag_on):
    """C4: record with action=None raises."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl")
    with pytest.raises(MissingActionError):
        led.record(_wave(1), None, _wave(2), episode_id="ep1", step=0)


def test_c5_stale_state_fail_closed(flag_on):
    """C5: out-of-order step in the same episode raises (stale state)."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl")
    led.record(_wave(1), "a0", _wave(2), episode_id="ep1", step=0)
    with pytest.raises(StaleStateError):
        led.record(_wave(3), "a1", _wave(4), episode_id="ep1", step=0)


def test_c5b_continuity_violation_strict(flag_on):
    """C5b: within-episode obs_t != previous obs_next raises in strict mode."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl", strict=True)
    led.record(_wave(1), "a0", _wave(2), episode_id="ep1", step=0)
    with pytest.raises(ContinuityViolationError):
        led.record(_wave(9), "a1", _wave(4), episode_id="ep1", step=1)


def test_c6_digest_round_trip():
    """C6: same tensor bytes -> same digest; different -> different."""
    assert wave_digest(_wave(7)) == wave_digest(_wave(7))
    assert wave_digest(_wave(7)) != wave_digest(_wave(8))


def test_c7_incremental_jsonl_reload(flag_on):
    """C7: per-row append before return; reload reconstructs the chain."""
    p = flag_on / "ledger.jsonl"
    led = TemporalTransitionLedger(p)
    led.record(_wave(1), "a0", _wave(2), episode_id="ep1", step=0)
    led.record(_wave(2), "a1", _wave(3), episode_id="ep1", step=1)
    rows = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 2  # incremental: both rows persisted before return
    reloaded = TemporalTransitionLedger.load(p)
    assert reloaded.continuity_check()["ok"] is True
    assert len(reloaded) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
