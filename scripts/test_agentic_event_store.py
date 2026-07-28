"""Focused tests for the local agentic event protocol.

These tests use a temporary vault and a temporary audit module. They verify the
protocol mechanics, not HENRI task performance or Zone C behavior.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agentic_event_store", ROOT / "scripts" / "agentic_event_store.py"
)
STORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STORE)


class FakeAudit:
    @staticmethod
    def record_event(actor, action, payload):
        body = json.dumps(
            {"actor": actor, "action": action, "payload": payload},
            sort_keys=True,
        ).encode()
        import hashlib
        return hashlib.sha256(body).hexdigest()


def _patch_audit(monkeypatch):
    monkeypatch.setattr(STORE, "_audit_module", lambda: FakeAudit)


def test_append_query_and_projection(tmp_path, monkeypatch):
    _patch_audit(monkeypatch)
    first = STORE.append_event(
        "PAPER_INGESTED",
        {"title": "Temporal Graph Memory", "source": "arxiv:1234"},
        stream="research",
        actor="test",
        source_uri="https://arxiv.org/abs/1234",
        vault_path=tmp_path,
    )
    second = STORE.append_event(
        "APPROVAL_REQUESTED",
        {"change": "graph-projection"},
        stream="approval",
        actor="test",
        causal_status="derived",
        parent_event_id=first["event_id"],
        vault_path=tmp_path,
    )
    edge = STORE.append_edge(
        first["event_id"],
        second["event_id"],
        "REQUIRES_APPROVAL",
        actor="test",
        vault_path=tmp_path,
    )
    assert first["payload_hash"]
    assert second["parent_event_id"] == first["event_id"]
    assert edge["event_type"] == "EDGE_CREATED"
    assert len(STORE.query_events(vault_path=tmp_path, stream="research")) == 1
    projection = STORE.graph_projection(tmp_path)
    assert projection["node_count"] == 3
    assert projection["edge_count"] == 1
    assert projection["projection_hash"]


def test_payload_tampering_is_rejected(tmp_path, monkeypatch):
    _patch_audit(monkeypatch)
    event = STORE.append_event(
        "OBSERVATION",
        {"value": 1},
        stream="telemetry",
        actor="test",
        vault_path=tmp_path,
    )
    path = tmp_path / "_agentic" / "events" / f"{event['event_id']}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["value"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(STORE.EventStoreError, match="payload hash mismatch"):
        list(STORE.iter_events(tmp_path))


def test_audit_failure_does_not_write_event(tmp_path, monkeypatch):
    def fail():
        raise STORE.AuditUnavailable("ledger unavailable")

    monkeypatch.setattr(STORE, "_audit_module", fail)
    with pytest.raises(STORE.AuditUnavailable):
        STORE.append_event(
            "BLOCKED_EVENT",
            {},
            stream="audit",
            actor="test",
            vault_path=tmp_path,
        )
    events = list((tmp_path / "_agentic" / "events").glob("*.json"))
    assert events == []
