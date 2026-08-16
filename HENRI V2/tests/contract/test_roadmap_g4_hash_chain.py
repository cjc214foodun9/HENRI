"""Roadmap R8.14 — Zone C provenance ledger Gate G4 (hash-chain verification).

Roadmap: Project HENRI V2 Strategic R&D Roadmap.pdf (SHA 0ca9f7a1...),
Phase 8.14 / Gate G4: "100% hash-chain verification across 1,000 updates;
any hash mismatch or out-of-order sequence insertion" = falsification.

This test drives the REAL governance machinery (henri_audit.py from
HERMES_HOME/scripts + scripts/agentic_event_store.py) against a TEMP
ledger (module global patched), so the production governance chain is
never polluted. Deterministic store test — local execution is correct
(no CUDA path).
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

N_UPDATES = 1000

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "")) or (
    Path.home() / "AppData" / "Local" / "hermes"
)
AUDIT_PATH = HERMES_HOME / "scripts" / "henri_audit.py"
STORE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "agentic_event_store.py"


def _load_audit(tmp_path):
    """Import the REAL henri_audit module with the ledger redirected to tmp."""
    assert AUDIT_PATH.exists(), f"henri_audit.py not found at {AUDIT_PATH}"
    spec = importlib.util.spec_from_file_location("henri_audit_g4", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.LEDGER_DIR = tmp_path / "audit"
    module.LEDGER = module.LEDGER_DIR / "henri_audit_chain.jsonl"
    sys.modules["henri_audit"] = module  # store's _audit_module picks this up
    return module


def _load_store():
    spec = importlib.util.spec_from_file_location(
        "agentic_event_store_g4", STORE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_g4_chain_1000_intact(tmp_path):
    """1,000 sealed updates -> chain intact, head hash matches last record."""
    audit = _load_audit(tmp_path)
    last = None
    for i in range(N_UPDATES):
        last = audit.record_event("g4_test", "LEDGER_UPDATE", {"i": i})
    ok, msg = audit.verify_chain()
    assert ok, f"chain broken: {msg}"
    assert "1000 records" in msg
    chain = audit._read_all()
    assert chain[-1]["hash"] == last


def test_g4_tamper_detected(tmp_path):
    """Flip one payload byte in the middle -> verify_chain fails."""
    audit = _load_audit(tmp_path)
    for i in range(N_UPDATES):
        audit.record_event("g4_test", "LEDGER_UPDATE", {"i": i})
    chain = audit._read_all()
    rec = chain[N_UPDATES // 2]
    rec["payload"]["i"] = -1
    path = audit.LEDGER
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[N_UPDATES // 2] = json.dumps(rec, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, msg = audit.verify_chain()
    assert not ok, "tampered chain verified clean"
    assert "hash mismatch" in msg or "prev_hash" in msg


def test_g4_out_of_order_detected(tmp_path):
    """Delete one middle record (index gap) -> verify_chain fails."""
    audit = _load_audit(tmp_path)
    for i in range(N_UPDATES):
        audit.record_event("g4_test", "LEDGER_UPDATE", {"i": i})
    path = audit.LEDGER
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[N_UPDATES // 2]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, msg = audit.verify_chain()
    assert not ok, "index-gapped chain verified clean"
    assert "index gap" in msg


def test_g4_store_path_1000(tmp_path, monkeypatch):
    """1,000 events through the REAL event store -> audit chain + records valid."""
    audit = _load_audit(tmp_path)
    store = _load_store()

    def patched_audit_module():
        return audit

    monkeypatch.setattr(store, "_audit_module", patched_audit_module)
    vault = tmp_path / "vault"
    first = None
    last = None
    for i in range(N_UPDATES):
        last = store.append_event(
            "WAVE_LEDGER_UPDATE",
            {"i": i, "sagnac_delta": 0.01 * (i % 7)},
            stream="zone_c_wave_ledger",
            actor="g4_test",
            causal_status="observed",
            parent_event_id=first["event_id"] if first else None,
            vault_path=vault,
        )
        first = first or last
    records = list(store.iter_events(vault))
    assert len(records) == N_UPDATES
    ok, msg = audit.verify_chain()
    assert ok, f"audit chain broken after store path: {msg}"
    assert len(audit._read_all()) >= N_UPDATES
