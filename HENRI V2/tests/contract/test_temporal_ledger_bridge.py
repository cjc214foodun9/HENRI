"""Contract tests: temporal ledger bridge for the live ARC runner (Carrier 1).

Run locally (CPU) from the repo root:
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \\
      PYTHONPATH="HENRI V2" /c/Python314/python.exe -m pytest \\
      "HENRI V2/tests/contract/test_temporal_ledger_bridge.py" -q --tb=short
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from temporal_ledger_bridge import ledger_summary, record_temporal_transition
from temporal_transition_ledger import FLAG, TemporalTransitionLedger


def _grid(seed_val: int, size: int = 4) -> list:
    g = [[0] * size for _ in range(size)]
    g[0][0] = seed_val
    return g


def _act(name: str = "ACTION1"):
    return SimpleNamespace(name=name, data=None)


def _obs(grid, name: str = "RUNNING", levels: int = 0):
    return SimpleNamespace(
        frame=[grid],
        state=SimpleNamespace(name=name),
        levels_completed=levels,
    )


@pytest.fixture
def flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    return tmp_path


def test_c1_records_real_transition_with_outcome(flag_on):
    """C1: a real (pre_grid, action, post_grid) row persists with outcome meta."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl", strict=True)
    rec = record_temporal_transition(led, _grid(1), _act(), _obs(_grid(2)), "ep1", 0)
    assert rec["episode_id"] == "ep1" and rec["step"] == 0
    assert rec["meta"]["frame_changed"] is True
    assert rec["meta"]["task_progressed"] is False
    assert rec["meta"]["terminal_state"] == "RUNNING"
    assert len(led) == 1


def test_c2_win_and_levels_progress(flag_on):
    """C2: WIN / levels_completed>0 set task_progressed (external outcome)."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl", strict=True)
    rec = record_temporal_transition(
        led, _grid(1), _act("ACTION2"), _obs(_grid(2), name="WIN", levels=1), "ep1", 0)
    assert rec["meta"]["task_progressed"] is True
    assert rec["meta"]["levels_completed"] == 1
    assert rec["meta"]["terminal_state"] == "WIN"


def test_c3_chain_continuity(flag_on):
    """C3: record[t].obs_next == record[t+1].obs_t (chain invariant)."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl", strict=True)
    w0, w1, w2 = _grid(1), _grid(2), _grid(3)
    record_temporal_transition(led, w0, _act(), _obs(w1), "ep1", 0)
    record_temporal_transition(led, w1, _act(), _obs(w2), "ep1", 1)
    chk = led.continuity_check()
    assert chk["ok"] is True
    assert chk["episodes"] == {"ep1": 2}
    summ = ledger_summary(led)
    assert summ["continuity_ok"] is True and summ["records"] == 2


def test_c4_null_post_state_fail_closed(flag_on):
    """C4: a null post-state blocks the step (LEDGER_FAIL_CLOSED)."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl", strict=True)
    with pytest.raises(RuntimeError, match="LEDGER_FAIL_CLOSED"):
        record_temporal_transition(led, _grid(1), _act(), None, "ep1", 0)


def test_c5_unchanged_frame(flag_on):
    """C5: identical pre/post grid -> frame_changed False (honest delta)."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl", strict=True)
    w = _grid(5)
    rec = record_temporal_transition(led, w, _act(), _obs(w), "ep1", 0)
    assert rec["meta"]["frame_changed"] is False
    assert rec["meta"]["task_progressed"] is False


def test_c6_extra_meta_merged(flag_on):
    """C6: runner-side context (macro actions, reset flag) merges into meta."""
    led = TemporalTransitionLedger(flag_on / "ledger.jsonl", strict=True)
    rec = record_temporal_transition(
        led, _grid(1), _act("ACTION1"), _obs(_grid(2)), "ep1", 0,
        extra_meta={"macro_actions": ["ACTION1", "ACTION2"],
                    "action_was_reset": False})
    assert rec["meta"]["macro_actions"] == ["ACTION1", "ACTION2"]
    assert rec["meta"]["action_was_reset"] is False
    assert rec["meta"]["frame_changed"] is True
