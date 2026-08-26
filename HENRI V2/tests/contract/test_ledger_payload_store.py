"""Contract tests: K0 content-addressed payload persistence (T0 ledger)."""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from temporal_transition_ledger import (  # noqa: E402
    FLAG as LEDGER_FLAG,
    TemporalTransitionLedger,
    action_digest,
    wave_digest,
)
from ledger_payload_store import (  # noqa: E402
    FLAG,
    LedgerPayloadStore,
    PayloadReferenceError,
    PayloadStoreDisabledError,
)


class _Action:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data


@pytest.fixture
def flags_on(tmp_path, monkeypatch):
    monkeypatch.setenv(LEDGER_FLAG, "1")
    monkeypatch.setenv(FLAG, "1")
    return tmp_path


def test_c1_default_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(PayloadStoreDisabledError):
        LedgerPayloadStore("x")


def test_c2_grid_payload_digest_reproducible(flags_on):
    grid = [[1, 0], [2, 3]]
    store = LedgerPayloadStore(flags_on / "p")
    info = store.put(grid)
    assert info["digest"] == wave_digest(grid)
    data = store.get(info["digest"])
    assert hashlib.sha256(data).hexdigest() == info["digest"]
    assert json.loads(data.decode()) == grid


def test_c3_action_payload_digest_reproducible(flags_on):
    act = _Action("up", data={"dx": 1, "dy": 0})
    store = LedgerPayloadStore(flags_on / "p")
    info = store.put(act)
    assert info["digest"] == action_digest(act)
    assert hashlib.sha256(store.get(info["digest"])).hexdigest() == info["digest"]


def test_c4_ledger_rows_carry_refs_and_chain(flags_on):
    led = TemporalTransitionLedger(
        flags_on / "l.jsonl",
        payload_store=LedgerPayloadStore(flags_on / "p"))
    led.record([[0]], _Action("a"), [[1]], episode_id="e1", step=0)
    led.record([[1]], _Action("b"), [[2]], episode_id="e1", step=1)
    rows = [json.loads(x)
            for x in (flags_on / "l.jsonl").read_text().splitlines()
            if x.strip()]
    assert rows[0]["obs_next_ref"] == rows[1]["obs_t_ref"]
    assert rows[0]["obs_next_kind"] == "grid"
    assert rows[0]["action_kind"] == "action"
    assert rows[0]["payload_schema"] == "payload.v1"
    assert led.continuity_check()["ok"]


def test_c5_differential_no_store_rows_unchanged(flags_on):
    led0 = TemporalTransitionLedger(flags_on / "a.jsonl")
    led0.record([[0]], _Action("a"), [[1]], episode_id="e1", step=0)
    keys = set(json.loads((flags_on / "a.jsonl").read_text().splitlines()[0]).keys())
    assert "obs_t_ref" not in keys
    assert keys == {"episode_id", "step", "t_phys", "obs_t_digest",
                    "action_digest", "obs_next_digest", "meta", "utc"}


def test_c6_reset_boundary_deliberate_break(flags_on):
    led = TemporalTransitionLedger(
        flags_on / "l.jsonl",
        payload_store=LedgerPayloadStore(flags_on / "p"))
    led.record([[0]], _Action("a"), [[1]], episode_id="e1", step=0)
    led.reset("e1")
    led.record([[9]], _Action("b"), [[8]], episode_id="e1", step=0)
    assert led._last[1] == 0


def test_c7_missing_and_corrupt_refs_fail_closed(flags_on):
    store = LedgerPayloadStore(flags_on / "p")
    info = store.put([[1]])
    with pytest.raises(PayloadReferenceError):
        store.get("0" * 64)
    p = flags_on / "p" / f"{info['digest']}.bin"
    p.write_bytes(b"corrupt")
    with pytest.raises(PayloadReferenceError):
        store.get(info["digest"])


def test_c8_incremental_append(flags_on):
    led = TemporalTransitionLedger(
        flags_on / "l.jsonl",
        payload_store=LedgerPayloadStore(flags_on / "p"))
    led.record([[0]], _Action("a"), [[1]], episode_id="e1", step=0)
    led2 = TemporalTransitionLedger(
        flags_on / "l.jsonl",
        payload_store=LedgerPayloadStore(flags_on / "p"))
    led2.record([[1]], _Action("b"), [[2]], episode_id="e1", step=1)
    n = len([x for x in (flags_on / "l.jsonl").read_text().splitlines()
             if x.strip()])
    assert n == 2
