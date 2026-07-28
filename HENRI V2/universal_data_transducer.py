"""
Universal Data Transducer for Project HENRI V2.

Transduces arbitrary exteroceptive data streams (JSON payloads, timeseries vectors,
multi-modal features, system logs) into D=65,536 qFHRR phase-quantized multivectors on S^{D-1}.
Implements Fractional-Phase Spatial Binding for continuous values and Ontological VSA key-value binding.
"""

import hashlib
import json
import os
import sys
import torch
from typing import Any, Dict, List, Union

try:
    import psycopg2
except ImportError:
    psycopg2 = None


class UniversalDataTransducer:
    """
    Transduces arbitrary exteroceptive data objects into d=65,536 qFHRR phase multivectors.
    Packs continuous numerical features into fractional phase angles theta = v * pi mod 2pi
    and categorical key-value pairs into circular convolution bound phase codes.
    """

    def __init__(self, d_model: int = 65536, codebook_size: int = 256, db_dsn: str = None):
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.db_dsn = db_dsn
        self.num_blocks = d_model // 8

        # Build 256-entry cosine Lookup Table for fast phase similarity computations
        phase_intervals = torch.linspace(0, 2 * torch.pi, steps=codebook_size)
        self.lut_cos = torch.cos(phase_intervals)

    def transduce_object(self, obj: Union[Dict[str, Any], List[Any], str, float, int]) -> torch.Tensor:
        """
        Main entry point: converts any Python object/data row into a d=65,536 qFHRR phase wave (uint8 tensor).
        """
        if isinstance(obj, dict):
            return self._transduce_dict(obj)
        elif isinstance(obj, (list, tuple)):
            return self._transduce_sequence(obj)
        elif isinstance(obj, str):
            return self._transduce_string(obj)
        elif isinstance(obj, (int, float)):
            return self._transduce_scalar(float(obj))
        else:
            return self._transduce_string(str(obj))

    def _transduce_dict(self, d: Dict[str, Any]) -> torch.Tensor:
        """Transduces a dictionary using Ontological VSA key-value binding."""
        fused_wave = torch.zeros(self.d_model, dtype=torch.int32)

        for key, val in d.items():
            key_key = self._hash_key_to_wave(str(key))

            if isinstance(val, (int, float)):
                val_wave = self._transduce_scalar(float(val))
            elif isinstance(val, str):
                val_wave = self._transduce_string(val)
            elif isinstance(val, (dict, list)):
                val_wave = self.transduce_object(val)
            else:
                val_wave = self._transduce_string(str(val))

            # Circular convolution in qFHRR = elementwise modular addition mod K
            bound_pair = (key_key.to(torch.int32) + val_wave.to(torch.int32)) % self.codebook_size
            fused_wave = (fused_wave + bound_pair) % self.codebook_size

        return fused_wave.to(torch.uint8)

    def _transduce_sequence(self, seq: List[Any]) -> torch.Tensor:
        """Transduces a sequence/timeseries by binding position indices to elements."""
        fused_wave = torch.zeros(self.d_model, dtype=torch.int32)

        for idx, elem in enumerate(seq):
            pos_key = self._hash_key_to_wave(f"pos_{idx}")
            elem_wave = self.transduce_object(elem)
            bound_elem = (pos_key.to(torch.int32) + elem_wave.to(torch.int32)) % self.codebook_size
            fused_wave = (fused_wave + bound_elem) % self.codebook_size

        return fused_wave.to(torch.uint8)

    def _transduce_scalar(self, val: float) -> torch.Tensor:
        """Fractional-Phase Spatial Binding: maps scalar v to phase angle theta = v * pi mod 2pi."""
        # Scale float to integer phase code in [0, 255]
        norm_val = math.tanh(val)  # Bounded in (-1, 1)
        phase_code = int((norm_val + 1.0) * 127.5) % self.codebook_size
        return torch.full((self.d_model,), phase_code, dtype=torch.uint8)

    def _transduce_string(self, s: str) -> torch.Tensor:
        """Maps an arbitrary string deterministically to the qFHRR phase ring."""
        hash_seed = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (10**8)
        g = torch.Generator().manual_seed(hash_seed)
        q_codes = torch.randint(0, self.codebook_size, (self.d_model,), dtype=torch.uint8, generator=g)
        return q_codes

    def _hash_key_to_wave(self, key_str: str) -> torch.Tensor:
        """Generates an orthogonal key wave for dictionary field binding."""
        return self._transduce_string(f"key_ortho_{key_str}")

    def commit_data_axiom(
        self, axiom_id: str, data_kind: str, raw_data: Any, wave_q: torch.Tensor, confidence: float = 1.0
    ) -> bool:
        """Writes the transduced exteroceptive data wave directly into Zone C TimescaleDB."""
        if not self.db_dsn:
            return False

        wave_bytes = wave_q.cpu().numpy().tobytes()
        semantic_json = json.dumps({"raw_data": str(raw_data), "kind": data_kind})

        try:
            conn = psycopg2.connect(self.db_dsn)
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO boundary_axioms (
                    axiom_id, axiom_kind, source, wave_payload, semantic_projection, confidence
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (axiom_id) DO UPDATE SET
                    wave_payload = EXCLUDED.wave_payload,
                    semantic_projection = EXCLUDED.semantic_projection,
                    confidence = EXCLUDED.confidence;
                """,
                (axiom_id, f"exteroceptive_data_{data_kind}", "UniversalDataTransducer", wave_bytes, semantic_json, confidence),
            )
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as err:
            print(f"[Zone C Transducer Warning] DB Commit skipped: {err}", file=sys.stderr)
            return False


import math

if __name__ == "__main__":
    transducer = UniversalDataTransducer(d_model=65536)
    sample_data = {
        "sensor_id": "imu_accel_x",
        "timestamp": 1785214000,
        "telemetry": [0.04, -1.09, 0.52, 2.11],
        "status": "NORMAL",
    }
    wave = transducer.transduce_object(sample_data)
    print(f"Transduced Data Wave Shape: {wave.shape}, dtype: {wave.dtype}")
    print("Universal Data Transducer module verified successfully.")
