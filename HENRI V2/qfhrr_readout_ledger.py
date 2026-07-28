"""
qFHRR Readout Ledger and Typed Event Packet Contract for Project HENRI V2.

Implements ReadoutPacket (schema 2.0.0) and qFHRRAuditLedger enforcing
cryptographic SHA-256 parent-hash chaining and JSONL logging.
"""

import json
import hashlib
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple
import torch

SCHEMA_VERSION = "2.0.0"
GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class ReadoutPacket:
    """
    Typed readout contract bridging continuous wave mechanics to downstream decoders & ledgers.
    Explicitly separates continuous state, quantized projections, telemetry, and external outcomes.
    """
    schema_version: str
    run_id: str
    step_id: int
    environment_id: str
    source_commit: str
    state_kind: str  # e.g., 'active_inference_step', 'checkpoint', 'task_terminal'

    # Tensor Contract Specs
    tensor_shape: List[int]  # e.g., [8192, 8]
    tensor_dtype: str        # e.g., 'float32'
    device: str              # e.g., 'cuda:0'

    # Payload References & Projections
    continuous_wave_ref: str       # Storage URI / Checkpoint ID
    qfhrr_phase_codes: List[int]   # Quantized phase indices

    # Physical Telemetry
    sagnac_delta: float
    coherence: float
    transition_loss: float

    # Task & External Grounding
    selected_action: int
    external_outcome_status: str   # e.g., 'STEP_SUCCESS', 'RESET_LEGITIMATE', 'NO_PROGRESS'
    zone_c_checkpoint_id: Optional[str] = None

    # Hash Chaining Provenance
    parent_hash: str = GENESIS_HASH
    created_at: float = field(default_factory=time.time)
    packet_hash: str = ""

    def __post_init__(self):
        if not self.packet_hash:
            self.packet_hash = self.compute_canonical_hash()

    def compute_canonical_hash(self) -> str:
        """
        Computes deterministic SHA-256 hash over canonical JSON representation of state fields.
        Excludes the packet_hash field itself to avoid circular dependency.
        """
        canonical_dict = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "environment_id": self.environment_id,
            "source_commit": self.source_commit,
            "state_kind": self.state_kind,
            "tensor_shape": self.tensor_shape,
            "tensor_dtype": self.tensor_dtype,
            "continuous_wave_ref": self.continuous_wave_ref,
            "qfhrr_phase_codes": self.qfhrr_phase_codes,
            "sagnac_delta": round(float(self.sagnac_delta), 6),
            "coherence": round(float(self.coherence), 6),
            "transition_loss": round(float(self.transition_loss), 6),
            "selected_action": int(self.selected_action),
            "external_outcome_status": self.external_outcome_status,
            "parent_hash": self.parent_hash,
        }
        serialized = json.dumps(canonical_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class qFHRRAuditLedger:
    """
    Append-only, hash-chained ledger storing state event projections.
    Provides verified provenance without introducing second-order distributed consensus overhead.
    """

    def __init__(self, log_filepath: str):
        self.log_filepath = log_filepath
        self.last_hash = GENESIS_HASH
        self._initialize_parent_hash()

    def _initialize_parent_hash(self):
        """Reads existing JSONL ledger to recover the head hash."""
        try:
            with open(self.log_filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    last_record = json.loads(lines[-1])
                    self.last_hash = last_record.get("packet_hash", self.last_hash)
        except FileNotFoundError:
            pass

    def record_step(
        self,
        run_id: str,
        step_id: int,
        environment_id: str,
        source_commit: str,
        state_kind: str,
        wave_tensor: torch.Tensor,
        qfhrr_codes: torch.Tensor,
        wave_ref: str,
        sagnac_delta: float,
        coherence: float,
        transition_loss: float,
        selected_action: int,
        external_outcome_status: str,
        zone_c_checkpoint_id: Optional[str] = None,
    ) -> ReadoutPacket:
        """
        Constructs a ReadoutPacket, chains its hash to the previous head, and appends to the ledger.
        """
        assert wave_tensor.ndim == 2, f"Expected 2D wave tensor [num_blocks, 8], got {wave_tensor.shape}"
        assert wave_tensor.shape[1] == 8, f"Clifford wave must have 8 real components per block, got {wave_tensor.shape[1]}"

        packet = ReadoutPacket(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            step_id=step_id,
            environment_id=environment_id,
            source_commit=source_commit,
            state_kind=state_kind,
            tensor_shape=list(wave_tensor.shape),
            tensor_dtype=str(wave_tensor.dtype).replace('torch.', ''),
            device=str(wave_tensor.device),
            continuous_wave_ref=wave_ref,
            qfhrr_phase_codes=qfhrr_codes.detach().cpu().tolist(),
            sagnac_delta=float(sagnac_delta),
            coherence=float(coherence),
            transition_loss=float(transition_loss),
            selected_action=int(selected_action),
            external_outcome_status=external_outcome_status,
            zone_c_checkpoint_id=zone_c_checkpoint_id,
            parent_hash=self.last_hash,
        )

        with open(self.log_filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(packet)) + '\n')

        self.last_hash = packet.packet_hash
        return packet

    @staticmethod
    def verify_chain_integrity(log_filepath: str) -> Tuple[bool, int, str]:
        """
        Verifies cryptographic parent-hash chain integrity across the entire event log.
        Returns: (is_valid, records_verified, failure_reason)
        """
        try:
            with open(log_filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return False, 0, f"Ledger file {log_filepath} not found."

        expected_parent = GENESIS_HASH
        for idx, line in enumerate(lines):
            record = json.loads(line)

            # Check parent hash alignment
            if record["parent_hash"] != expected_parent:
                return False, idx, f"Parent hash break at record {idx}. Expected {expected_parent}, got {record['parent_hash']}"

            # Recompute canonical packet hash
            packet_obj = ReadoutPacket(**record)
            recomputed_hash = packet_obj.compute_canonical_hash()
            if recomputed_hash != record["packet_hash"]:
                return False, idx, f"Hash mismatch at record {idx}. Stored {record['packet_hash']}, recomputed {recomputed_hash}"

            expected_parent = record["packet_hash"]

        return True, len(lines), "Chain intact and verified."
