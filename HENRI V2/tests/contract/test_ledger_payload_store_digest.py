"""Regression test: K0 payload store digest alignment for data-less actions.

FALSIFIED 2026-08-26 during Gate 1 execution: encode_payload appended
':null' for data=None while action_digest hashes 'Type:name' without the
suffix -> sha256(raw) != digest -> every action payload failed get() as
'corrupt (digest mismatch)' -> load_corpus counted missing_payload=1472.

This test pins the contract sha256(raw) == digest for every kind,
including actions with data=None (the ARC loop's data-less GameActions).
Pure-python: a stub object with .name/.data exercises the action branch
without torch (torch import is required only in the tensor branch).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

HENRI_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HENRI_DIR))

from ledger_payload_store import encode_payload  # noqa: E402


class _StubAction:
    def __init__(self, name, data=None):
        self.name = name
        self.data = data


def test_action_data_none_digest_roundtrip():
    a = _StubAction("ACTION1", None)
    enc = encode_payload(a)
    assert enc["kind"] == "action"
    raw = enc["raw"]
    # The recorded digest must reproduce from the raw bytes (K0 contract).
    assert hashlib.sha256(raw).hexdigest() == enc["digest"]
    # Raw must NOT contain the ':null' suffix that broke action_digest.
    assert b":null" not in raw
    assert raw == b"_StubAction:ACTION1"


def test_action_data_present_digest_roundtrip():
    a = _StubAction("ACTION6", {"x": 1, "y": 2})
    enc = encode_payload(a)
    assert hashlib.sha256(enc["raw"]).hexdigest() == enc["digest"]
    text = enc["raw"].decode("utf-8")
    assert text.startswith("_StubAction:ACTION6:")
    assert '"x": 1' in text


def test_grid_digest_roundtrip():
    g = [[0, 1], [2, 3]]
    enc = encode_payload(g)
    assert enc["kind"] == "grid"
    canonical = json.dumps(g, sort_keys=True, separators=(",", ":")).encode()
    assert enc["raw"] == canonical
    assert hashlib.sha256(canonical).hexdigest() == enc["digest"]


def test_encode_matches_action_digest_function():
    """encode_payload digest must equal the ledger's action_digest."""
    from temporal_transition_ledger import action_digest
    a = _StubAction("ACTION3", None)
    enc = encode_payload(a)
    assert enc["digest"] == action_digest(a)
