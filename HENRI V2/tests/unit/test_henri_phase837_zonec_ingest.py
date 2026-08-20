# -*- coding: utf-8 -*-
"""Phase 8.37 unit tests — Zone C ingest (dry-run) + retrieval bridge.

- test_ingest_dry_run_rejects_unverified_manifest
- test_ingest_dry_run_counts_records
- test_ingest_dry_run_requires_authorized_source
- test_bridge_default_off_returns_empty
- test_bridge_enabled_fails_closed_without_dsn
- test_bridge_flag_parse
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from henri_trajectory_bank import TrajectoryBank
from zone_c_engram_ingest import ingest, record_id
from zone_c_retrieval_bridge import (
    ZoneCRetrievalBridge,
    bridge_enabled_from_env,
)


def _make_bank(tmpdir: str, n: int = 8, source: str = "authorized") -> dict:
    bank = TrajectoryBank(log_dir=tmpdir, run_id="test837",
                          provenance="test")
    for i in range(n):
        w = torch.randn(65536, dtype=torch.float32)
        w = w / torch.norm(w)
        nw = torch.randn(65536, dtype=torch.float32)
        nw = nw / torch.norm(nw)
        bank.record(w, f"ACTION{(i % 6) + 1}",
                    meta={"env": f"env{i % 3}"}, next_wave=nw)
    flush = bank.flush()
    # Patch manifest source for the authorized test.
    manifest_path = flush["manifest_path"]
    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["data_source"] = source
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    return flush


def test_ingest_dry_run_counts_records():
    with tempfile.TemporaryDirectory() as td:
        flush = _make_bank(td, n=8)
        out = ingest(
            flush["npz_path"], flush["manifest_path"],
            flush["jsonl_path"], dsn_env="/nonexistent.env",
            dry_run=True)
        assert out["status"] == "OK"
        assert out["records_read"] == 8
        assert out["inserted"] == 8
        assert out["dry_run"] is True
        assert out["post_count"] is None


def test_ingest_dry_run_rejects_unverified_manifest():
    with tempfile.TemporaryDirectory() as td:
        flush = _make_bank(td, n=4)
        # Corrupt the DATA inside the npz (appending trailing bytes is
        # ignored by zip readers and leaves the digest intact).
        npz_path = flush["npz_path"]
        z = np.load(npz_path, allow_pickle=False)
        psi = z["psi"].copy()
        psi[0, 0] += 1.0  # flip one value
        onehot = z["actions_onehot"].copy()
        nxt = z["next_wave"].copy()
        names = z["action_names"].copy()
        z.close()
        np.savez(npz_path, psi=psi, next_wave=nxt,
                 actions_onehot=onehot, action_names=names)
        with pytest.raises(Exception):
            ingest(flush["npz_path"], flush["manifest_path"],
                   flush["jsonl_path"], dsn_env="/nonexistent.env",
                   dry_run=True)


def test_ingest_dry_run_requires_authorized_source():
    with tempfile.TemporaryDirectory() as td:
        flush = _make_bank(td, n=4, source="eval_cache")
        with pytest.raises(RuntimeError, match="authorized"):
            ingest(flush["npz_path"], flush["manifest_path"],
                   flush["jsonl_path"], dsn_env="/nonexistent.env",
                   dry_run=True)


def test_record_id_deterministic_and_distinct():
    a = np.zeros(16, dtype=np.float16).tobytes()
    b = np.ones(16, dtype=np.float16).tobytes()
    assert record_id(a, b, None) == record_id(a, b, None)
    assert record_id(a, b, None) != record_id(b, a, None)
    assert record_id(a, b, None) != record_id(a, b, b)


def test_bridge_default_off_returns_empty():
    os.environ.pop("HENRI_ZONEC_BRIDGE", None)
    br = ZoneCRetrievalBridge(dsn="postgres://x", enabled=None)
    assert br.enabled is False
    assert br.retrieve(torch.zeros(65536)) == []


def test_bridge_flag_parse():
    assert bridge_enabled_from_env({"HENRI_ZONEC_BRIDGE": "1"}) is True
    assert bridge_enabled_from_env({"HENRI_ZONEC_BRIDGE": "0"}) is False
    assert bridge_enabled_from_env({}) is False
    assert bridge_enabled_from_env({"HENRI_ZONEC_BRIDGE": "true"}) is False


def test_bridge_enabled_fails_closed_without_db():
    os.environ["HENRI_ZONEC_BRIDGE"] = "1"
    try:
        # Constructor connects eagerly (TimescaleZoneCStore) -> raises.
        with pytest.raises(Exception):
            ZoneCRetrievalBridge(dsn="postgres://nope:nope@127.0.0.1:1/x")
    finally:
        os.environ.pop("HENRI_ZONEC_BRIDGE", None)
