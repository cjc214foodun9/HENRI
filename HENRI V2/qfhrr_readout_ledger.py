"""
Project HENRI V2 — qFHRR Hash-Chained Audit Ledger & Typed Readout Adapter.

Implements Phase B & Phase C of the Egress Boundary Architecture:
  1. ReadoutPacket Schema: Auditable contract carrying run metadata, tensor shapes, qFHRR phase codes, Sagnac metrics, actions, and external outcomes.
  2. qFHRRAuditLedger: Hash-chained event ledger in PostgreSQL/TimescaleDB and JSONL mirror with SHA-256 parent-hash verification.
"""

import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

try:
    import psycopg
except ImportError:
    psycopg = None


@dataclass
class ReadoutPacket:
    """Typed Egress Readout Packet Contract."""
    schema_version: str
    run_id: str
    step_id: int
    environment_id: str
    source_commit: str
    state_kind: str
    tensor_shape: List[int]
    tensor_dtype: str
    device: str
    qfhrr_phase_codes: bytes
    sagnac_delta: float
    coherence: float
    transition_loss: float
    selected_action: str
    external_outcome_status: str
    zone_c_checkpoint_id: str
    parent_hash: str = "0" * 64
    event_hash: str = ""

    def compute_hash(self, parent_hash: str) -> str:
        """Computes SHA-256 hash over canonical serialized fields."""
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "environment_id": self.environment_id,
            "source_commit": self.source_commit,
            "state_kind": self.state_kind,
            "tensor_shape": self.tensor_shape,
            "tensor_dtype": self.tensor_dtype,
            "device": self.device,
            "sagnac_delta": round(self.sagnac_delta, 6),
            "coherence": round(self.coherence, 6),
            "transition_loss": round(self.transition_loss, 6),
            "selected_action": self.selected_action,
            "external_outcome_status": self.external_outcome_status,
            "zone_c_checkpoint_id": self.zone_c_checkpoint_id,
            "parent_hash": parent_hash,
            "phase_code_sha256": hashlib.sha256(self.qfhrr_phase_codes).hexdigest(),
        }
        canonical_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class qFHRRAuditLedger:
    """
    Hash-chained audit ledger enforcing deterministic provenance for HENRI state events.
    Appends events to TimescaleDB 'qfhrr_state_events' table and local JSONL mirror.
    """

    def __init__(self, db_dsn: Optional[str] = None, log_file: Optional[str] = None):
        self.db_dsn = db_dsn or "postgres://postgres:postgres@localhost:10100/henri"
        self.log_file = log_file or os.path.join(
            os.path.expanduser("~"), "HENRI_telemetry_exports", "qfhrr_audit_ledger.jsonl"
        )
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.last_hash = "0" * 64

    def record_event(self, packet: ReadoutPacket) -> str:
        """Computes hash-chained event hash and records event to TimescaleDB and JSONL."""
        packet.parent_hash = self.last_hash
        packet.event_hash = packet.compute_hash(self.last_hash)
        self.last_hash = packet.event_hash

        # 1. Mirror to local JSONL
        record_data = asdict(packet)
        record_data["qfhrr_phase_codes"] = packet.qfhrr_phase_codes.hex()
        record_data["created_at"] = time.time()

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_data) + "\n")

        # 2. Persist to TimescaleDB
        if psycopg and self.db_dsn and not self.db_dsn.startswith("offline"):
            try:
                conn = psycopg.connect(self.db_dsn, connect_timeout=2)
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO qfhrr_state_events (
                        parent_hash, event_hash, run_id, step_id, state_kind,
                        phase_codes, codec_version, source_tensor_hash
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        packet.parent_hash,
                        packet.event_hash,
                        packet.run_id,
                        packet.step_id,
                        packet.state_kind,
                        packet.qfhrr_phase_codes,
                        packet.schema_version,
                        hashlib.sha256(packet.qfhrr_phase_codes).hexdigest(),
                    ),
                )
                conn.commit()
                cur.close()
                conn.close()
            except Exception as err:
                print(f"[Ledger Warning] DB commit skipped: {err}", file=sys.stderr)

        return packet.event_hash

    def verify_chain_integrity(self) -> Tuple[bool, int, str]:
        """Verifies hash-chained ledger continuity from JSONL file."""
        if not os.path.exists(self.log_file):
            return True, 0, "Empty ledger log file."

        verified_count = 0
        expected_parent = "0" * 64

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                pkt = ReadoutPacket(
                    schema_version=data["schema_version"],
                    run_id=data["run_id"],
                    step_id=data["step_id"],
                    environment_id=data["environment_id"],
                    source_commit=data["source_commit"],
                    state_kind=data["state_kind"],
                    tensor_shape=data["tensor_shape"],
                    tensor_dtype=data["tensor_dtype"],
                    device=data["device"],
                    qfhrr_phase_codes=bytes.fromhex(data["qfhrr_phase_codes"]),
                    sagnac_delta=data["sagnac_delta"],
                    coherence=data["coherence"],
                    transition_loss=data["transition_loss"],
                    selected_action=data["selected_action"],
                    external_outcome_status=data["external_outcome_status"],
                    zone_c_checkpoint_id=data["zone_c_checkpoint_id"],
                    parent_hash=data["parent_hash"],
                    event_hash=data["event_hash"],
                )

                if pkt.parent_hash != expected_parent:
                    msg = f"Parent hash mismatch at line {line_idx}: expected {expected_parent[:8]}, got {pkt.parent_hash[:8]}"
                    print(f"[Ledger Integrity Failure] {msg}")
                    return False, line_idx, msg

                calc_hash = pkt.compute_hash(expected_parent)
                if calc_hash != pkt.event_hash:
                    msg = f"Event hash mismatch at line {line_idx}: calculated {calc_hash[:8]}, stored {pkt.event_hash[:8]}"
                    print(f"[Ledger Integrity Failure] {msg}")
                    return False, line_idx, msg

                expected_parent = pkt.event_hash
                verified_count += 1

        return True, verified_count, "Chain verified intact."


if __name__ == "__main__":
    ledger = qFHRRAuditLedger(db_dsn="offline://surrogate")
    sample_codes = os.urandom(65536)
    packet = ReadoutPacket(
        schema_version="1.0.0",
        run_id="run_test_01",
        step_id=1,
        environment_id="env_ar25_0c556536",
        source_commit="f802548",
        state_kind="qfhrr_phase_codes",
        tensor_shape=[8192, 8],
        tensor_dtype="uint8",
        device="cuda",
        qfhrr_phase_codes=sample_codes,
        sagnac_delta=0.0431,
        coherence=0.9569,
        transition_loss=0.7258,
        selected_action="ACTION_MOVE_NORTH",
        external_outcome_status="STEP_COMPLETED",
        zone_c_checkpoint_id="chk_001",
    )
    event_hash = ledger.record_event(packet)
    print(f"Recorded ReadoutPacket event hash: {event_hash}")

    valid, count, msg = ledger.verify_chain_integrity()
    print(f"Ledger Chain Integrity Verified: {valid} ({count} events checked, msg: {msg})")
    assert valid, "Ledger integrity check failed"
    print("qFHRRAuditLedger successfully verified.")
