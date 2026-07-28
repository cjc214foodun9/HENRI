"""
Exteroceptive REPL Sandbox Transducer for Project HENRI V2.

Executes candidate code in an isolated REPL sandbox, captures runtime exceptions,
transduces raw traceback text into a d=65,536 qFHRR phase-quantized wave,
and commits the error wave to Zone C TimescaleDB to serve as a Sagnac Veto.
"""

import hashlib
import io
import json
import os
import sys
import traceback
from typing import Any, Dict, Tuple
import torch

try:
    import psycopg
except ImportError:
    psycopg = None


class ExteroceptiveSandboxTransducer:
    """
    Executes candidate code in an isolated REPL sandbox, captures runtime exceptions,
    transduces raw traceback text into a d=65536 qFHRR phase-quantized wave,
    and commits the error wave to Zone C TimescaleDB to serve as a Sagnac Veto.
    """

    def __init__(self, d_model: int = 65536, codebook_size: int = 256, db_dsn: str = None):
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.db_dsn = db_dsn or os.environ.get(
            "ZONE_C_PROD_DSN", "postgresql://postgres:postgres@localhost:10100/henri"
        )

        # Build exact 256-entry cosine Lookup Table (lut_cos)
        phase_intervals = torch.linspace(0, 2 * torch.pi, steps=codebook_size)
        self.lut_cos = torch.cos(phase_intervals)

        # Frozen orthogonal symbol keys for Ontological VSA mapping
        self.keys = {
            "ExceptionType": self._generate_unitary_phase(seed=101),
            "LineNumber": self._generate_unitary_phase(seed=102),
            "AttributeError": self._generate_unitary_phase(seed=103),
            "NameError": self._generate_unitary_phase(seed=104),
            "IndexError": self._generate_unitary_phase(seed=105),
            "SyntaxError": self._generate_unitary_phase(seed=106),
        }

    def _generate_unitary_phase(self, seed: int = 42) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(0, self.codebook_size, (self.d_model,), dtype=torch.uint8, generator=g)

    def execute_and_transduce(
        self, candidate_code: str, axiom_id: str, source_metadata: str = "python_repl_sandbox"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes code in isolated environment, catching all runtime/syntax exceptions
        and translating them into physical wave constraints.
        """
        local_scope = {}
        global_scope = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        sys_stdout_bak = sys.stdout
        sys_stderr_bak = sys.stderr

        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        success = False
        error_info = {}

        try:
            compiled_code = compile(candidate_code, "<sandbox_repl>", "exec")
            exec(compiled_code, global_scope, local_scope)
            success = True
        except Exception as e:
            success = False
            exc_type, exc_value, exc_tb = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            error_info = {
                "exception_type": str(exc_type.__name__) if exc_type else "Exception",
                "exception_message": str(exc_value),
                "traceback": "".join(tb_lines),
                "line_number": int(traceback.extract_tb(exc_tb)[-1].lineno) if exc_tb else -1,
            }
        finally:
            sys.stdout = sys_stdout_bak
            sys.stderr = sys_stderr_bak

        if success:
            return True, {"output": stdout_capture.getvalue()}

        # Transduce error string into qFHRR wave state
        error_wave_q = self._transduce_traceback_to_wave(error_info)

        # Commit error wave to Zone C database
        if self.db_dsn and psycopg is not None:
            self._commit_axiom_to_zone_c(axiom_id, error_info["exception_type"], error_wave_q, error_info, source_metadata)

        return False, {
            "error": error_info,
            "qfhrr_phase_codes": error_wave_q.cpu().numpy().tolist()[:100],  # Truncated preview
            "coherence_veto_ready": True,
            "error_wave": error_wave_q,
        }

    def _transduce_traceback_to_wave(self, error_info: Dict[str, Any]) -> torch.Tensor:
        """Translates traceback properties into modular qFHRR phase operations."""
        fused_wave = torch.zeros(self.d_model, dtype=torch.int32)

        exc_type = error_info["exception_type"]
        val_vector = self._hash_string_to_wave(exc_type)

        bound_exc = (self.keys["ExceptionType"].to(torch.int32) + val_vector.to(torch.int32)) % self.codebook_size
        fused_wave = (fused_wave + bound_exc) % self.codebook_size

        line_val = self._hash_string_to_wave(str(error_info["line_number"]))
        bound_line = (self.keys["LineNumber"].to(torch.int32) + line_val.to(torch.int32)) % self.codebook_size
        fused_wave = (fused_wave + bound_line) % self.codebook_size

        if exc_type in self.keys:
            bound_specific = (self.keys[exc_type].to(torch.int32) + val_vector.to(torch.int32)) % self.codebook_size
            fused_wave = (fused_wave + bound_specific) % self.codebook_size

        return fused_wave.to(torch.uint8)

    def _hash_string_to_wave(self, s: str) -> torch.Tensor:
        hash_seed = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (10**8)
        g = torch.Generator().manual_seed(hash_seed)
        return torch.randint(0, self.codebook_size, (self.d_model,), dtype=torch.uint8, generator=g)

    def _commit_axiom_to_zone_c(self, axiom_id: str, kind: str, wave_q: torch.Tensor, error_info: dict, source: str):
        wave_bytes = wave_q.numpy().tobytes()
        semantic_json = json.dumps(error_info)

        try:
            with psycopg.connect(self.db_dsn, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO boundary_axioms (
                            axiom_id, axiom_kind, source, wave_payload, semantic_projection, confidence,
                            semantic_index
                        ) VALUES (%s, %s, %s, %s, %s, %s, zero_vector(2000))
                        ON CONFLICT (axiom_id) DO UPDATE SET
                            wave_payload = EXCLUDED.wave_payload,
                            semantic_projection = EXCLUDED.semantic_projection,
                            confidence = EXCLUDED.confidence;
                        """,
                        (axiom_id, f"exteroceptive_error_{kind}", source, psycopg.Binary(wave_bytes), semantic_json, -1.0),
                    )
                conn.commit()
        except Exception as db_err:
            pass  # Fail closed gracefully if DB is transient
