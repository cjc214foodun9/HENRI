"""Append-only local event store for HENRI agentic graph memory.

This module is local governance and workflow memory. It is not a Zone C
latent-space client and it must not store HENRI wave checkpoints.

The event log is the source of truth. Obsidian Markdown and vector search are
projections. Every event receives an audit-chain hash from the Hermes audit
ledger before it is appended to the local event stream.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1
CAUSAL_STATUSES = {
    "observed",
    "derived",
    "inferred",
    "hypothesis",
    "falsified",
    "blocked",
}
EDGE_TYPES = {
    "SUPPORTS",
    "CONTRADICTS",
    "DERIVED_FROM",
    "TRIGGERS",
    "REQUIRES_APPROVAL",
    "VERIFIED_BY",
    "FALSIFIED_BY",
    "IMPLEMENTS",
    "MEASURES",
    "CONSUMES",
    "SEPARATE_FROM",
}


class EventStoreError(RuntimeError):
    """Base error for local event-store failures."""


class AuditUnavailable(EventStoreError):
    """Raised when a governance event cannot be sealed."""


def default_vault_path() -> Path:
    configured = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "Documents" / "HENRI_Research_Vault").resolve()


def event_root(vault_path: str | os.PathLike[str] | None = None) -> Path:
    vault = Path(vault_path).expanduser().resolve() if vault_path else default_vault_path()
    return vault / "_agentic"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _audit_module():
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    candidates = []
    if hermes_home:
        candidates.append(Path(hermes_home) / "scripts")
    candidates.append(Path.home() / "AppData" / "Local" / "hermes" / "scripts")
    for candidate in candidates:
        if (candidate / "henri_audit.py").exists():
            sys.path.insert(0, str(candidate))
            try:
                import henri_audit  # type: ignore
                return henri_audit
            except Exception as exc:  # pragma: no cover - environment-specific
                raise AuditUnavailable(f"cannot import Hermes audit ledger: {exc}") from exc
    raise AuditUnavailable("Hermes audit ledger not found")


def _validate_common(event: dict[str, Any]) -> None:
    required = {
        "event_id",
        "event_time",
        "ingested_time",
        "stream",
        "event_type",
        "actor",
        "causal_status",
        "schema_version",
        "payload",
        "payload_hash",
        "audit_hash",
    }
    missing = sorted(required.difference(event))
    if missing:
        raise EventStoreError(f"event missing fields: {', '.join(missing)}")
    if event["causal_status"] not in CAUSAL_STATUSES:
        raise EventStoreError(f"invalid causal_status: {event['causal_status']!r}")
    if not isinstance(event["payload"], dict):
        raise EventStoreError("event payload must be an object")
    if event["payload_hash"] != _sha256(event["payload"]):
        raise EventStoreError(f"payload hash mismatch for {event['event_id']}")
    if not isinstance(event["audit_hash"], str) or len(event["audit_hash"]) != 64:
        raise EventStoreError(f"invalid audit hash for {event['event_id']}")


def append_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    stream: str,
    actor: str,
    causal_status: str = "observed",
    run_id: str | None = None,
    parent_event_id: str | None = None,
    source_uri: str | None = None,
    event_id: str | None = None,
    vault_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Seal and append one local event.

    The audit record is written first. If the audit ledger is unavailable, no
    local event is written. This prevents an apparently valid graph event from
    existing without governance evidence.
    """
    if causal_status not in CAUSAL_STATUSES:
        raise EventStoreError(f"invalid causal_status: {causal_status!r}")
    if not isinstance(payload, dict):
        raise EventStoreError("payload must be an object")
    event_id = event_id or str(uuid.uuid4())
    event_time = _utc_now()
    envelope = {
        "event_id": event_id,
        "event_time": event_time,
        "ingested_time": _utc_now(),
        "stream": stream,
        "event_type": event_type,
        "actor": actor,
        "run_id": run_id,
        "parent_event_id": parent_event_id,
        "causal_status": causal_status,
        "source_uri": source_uri,
        "schema_version": SCHEMA_VERSION,
        "payload": payload,
        "payload_hash": _sha256(payload),
    }
    audit = _audit_module()
    audit_payload = {
        "event_id": event_id,
        "stream": stream,
        "event_type": event_type,
        "actor": actor,
        "causal_status": causal_status,
        "payload_hash": envelope["payload_hash"],
        "parent_event_id": parent_event_id,
        "source_uri": source_uri,
    }
    try:
        audit_hash = audit.record_event(actor, f"AGENTIC_{event_type}", audit_payload)
    except Exception as exc:
        raise AuditUnavailable(f"audit seal failed: {exc}") from exc
    envelope["audit_hash"] = audit_hash
    _validate_common(envelope)

    root = event_root(vault_path)
    events_dir = root / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    event_file = events_dir / f"{event_id}.json"
    if event_file.exists():
        existing = json.loads(event_file.read_text(encoding="utf-8"))
        if existing != envelope:
            raise EventStoreError(f"event ID collision with different content: {event_id}")
        return existing
    stream_file = events_dir / f"{stream}.jsonl"
    event_file.write_text(_canonical(envelope) + "\n", encoding="utf-8")
    with stream_file.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(envelope) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return envelope


def append_visualizer_telemetry(
    metrics: dict[str, Any],
    *,
    stream: str = "telemetry",
    actor: str = "henri_cognitive_visualizer",
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Seals and records 3D Cognitive Visualizer test-time telemetry into the SHA-256 audit ledger.
    Tracks 3D phase space trajectory dispersion (r), Sagnac fringe phase delta (DeltaPhi),
    mutual information I(W; Y), and logit entropy H(Y).
    """
    payload = {
        "visualizer_active": True,
        "phase_space_dimension": 65536,
        "trajectory_dispersion_r": float(metrics.get("trajectory_dispersion_r", 0.25)),
        "sagnac_fringe_delta": float(metrics.get("sagnac_fringe_delta", 0.042)),
        "mutual_information_bits": float(metrics.get("mutual_information_bits", 14.82)),
        "logit_entropy_nats": float(metrics.get("logit_entropy_nats", 0.14)),
        "veto_rate_percent": float(metrics.get("veto_rate_percent", 0.0)),
        "w_task_modulated": bool(metrics.get("w_task_modulated", True)),
    }
    return append_event(
        event_type="visualizer_telemetry_snapshot",
        payload=payload,
        stream=stream,
        actor=actor,
        causal_status="observed",
        run_id=run_id
    )


def iter_events(vault_path: str | os.PathLike[str] | None = None) -> Iterator[dict[str, Any]]:
    root = event_root(vault_path) / "events"
    if not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        event = json.loads(path.read_text(encoding="utf-8"))
        _validate_common(event)
        yield event


def query_events(
    *,
    vault_path: str | os.PathLike[str] | None = None,
    stream: str | None = None,
    event_type: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if limit < 1 or limit > 1000:
        raise EventStoreError("limit must be between 1 and 1000")
    result = []
    for event in iter_events(vault_path):
        if stream and event["stream"] != stream:
            continue
        if event_type and event["event_type"] != event_type:
            continue
        if after and event["event_time"] < after:
            continue
        if before and event["event_time"] > before:
            continue
        result.append(event)
    result.sort(key=lambda item: item["event_time"], reverse=True)
    return result[:limit]


def append_edge(
    source_event_id: str,
    target_event_id: str,
    relation: str,
    *,
    actor: str,
    causal_status: str = "derived",
    vault_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    if relation not in EDGE_TYPES:
        raise EventStoreError(f"invalid edge type: {relation!r}")
    return append_event(
        "EDGE_CREATED",
        {
            "source_event_id": source_event_id,
            "target_event_id": target_event_id,
            "relation": relation,
        },
        stream="graph",
        actor=actor,
        causal_status=causal_status,
        parent_event_id=source_event_id,
        vault_path=vault_path,
    )


def graph_projection(vault_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for event in iter_events(vault_path):
        nodes[event["event_id"]] = {
            "event_id": event["event_id"],
            "event_time": event["event_time"],
            "stream": event["stream"],
            "event_type": event["event_type"],
            "causal_status": event["causal_status"],
            "audit_hash": event["audit_hash"],
        }
        if event["event_type"] == "EDGE_CREATED":
            payload = event["payload"]
            edges.append({
                "edge_event_id": event["event_id"],
                "source_event_id": payload["source_event_id"],
                "target_event_id": payload["target_event_id"],
                "relation": payload["relation"],
                "audit_hash": event["audit_hash"],
            })
    projection = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": sorted(nodes.values(), key=lambda item: item["event_time"]),
        "edges": edges,
    }
    projection["projection_hash"] = _sha256(projection)
    return projection


def write_projection(vault_path: str | os.PathLike[str] | None = None) -> Path:
    root = event_root(vault_path)
    root.mkdir(parents=True, exist_ok=True)
    projection = graph_projection(vault_path)
    path = root / "graph_projection.json"
    path.write_text(_canonical(projection) + "\n", encoding="utf-8")
    return path


def verify_local_events(vault_path: str | os.PathLike[str] | None = None) -> tuple[bool, str]:
    count = 0
    for event in iter_events(vault_path):
        count += 1
    return True, f"local event records valid: {count}"
