import threading
import queue
import datetime
import uuid
import sys
import os
import torch
import psycopg
from typing import List, Dict, Any, Tuple

from zone_c_env import assert_zone_c_env

class ThermodynamicTelemetryLogger:
    """
    Zone C Interconnect: Asynchronous, lock-free telemetry extraction.
    Bridges the continuous phase-space of Zone B to the discrete Hypertable of Zone C.
    """
    def __init__(self, db_conn_str: str, batch_size: int = 500, flush_interval_sec: float = 1.0):
        self.db_conn_str = db_conn_str
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        
        # The ring buffer absorbs high-frequency phase state logs without locking the main thread.
        self.telemetry_queue = queue.Queue(maxsize=10000)
        self.shutdown_flag = threading.Event()
        
        self._initialize_hypertable()
        
        self.worker_thread = threading.Thread(target=self._async_batch_writer, daemon=True)
        self.worker_thread.start()

    def _initialize_hypertable(self):
        """
        Executes the baseline DDL to enforce the rigid, hardware-aligned memory strides 
        required for the dual-array (Real/Imaginary) Phase Tensor layout.
        """
        schema_sql = """
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

        CREATE TABLE IF NOT EXISTS zone_c_resonant_hypersphere (
            id UUID NOT NULL,
            domain VARCHAR(64) NOT NULL,
            subdomain VARCHAR(64) NOT NULL,
            concept_key VARCHAR(128) NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL,
            
            real_phases REAL[] NOT NULL,
            imag_phases REAL[] NOT NULL,
            
            phase_delta REAL NOT NULL,
            sagnac_clearance BOOLEAN NOT NULL,
            run_id VARCHAR(64),
            arm_id VARCHAR(16),
            commit_sha VARCHAR(40),
            domain_family VARCHAR(16) DEFAULT 'general',
            PRIMARY KEY (id, recorded_at)
        );

        -- Convert to a time-series hypertable if it isn't one already.
        -- This ensures temporal queries scale logarithmically, bounding retrieval costs.
        SELECT create_hypertable('zone_c_resonant_hypersphere', 'recorded_at', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);
        """
        with psycopg.connect(self.db_conn_str) as conn:
            expected = os.environ.get("ZONE_C_ENV", "dev").strip().lower()
            assert_zone_c_env(conn, expected)
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()

    def log_trajectory(self, domain: str, subdomain: str, concept_key: str, predicted_wave: torch.Tensor, phase_delta: float, is_valid: bool,
                       run_id: str = None, arm_id: str = None, commit_sha: str = None, domain_family: str = "general"):
        """
        Exposed method for the main orchestrator to offload phase states.
        Extracts Real and Imaginary components from the complex tensor structure.
        CLASS49: attribution fields (run_id/arm_id/commit_sha) + domain_family.
        """
        if predicted_wave.is_complex():
            real_phases = predicted_wave.real.detach().cpu().numpy().tolist()
            imag_phases = predicted_wave.imag.detach().cpu().numpy().tolist()
        else:
            # Fallback for standard float tensors mapped to polar angles
            real_phases = torch.cos(predicted_wave).detach().cpu().numpy().tolist()
            imag_phases = torch.sin(predicted_wave).detach().cpu().numpy().tolist()

        record = (
            str(uuid.uuid4()),
            domain,
            subdomain,
            concept_key,
            datetime.datetime.now(datetime.timezone.utc),
            real_phases,
            imag_phases,
            phase_delta,
            is_valid,
            run_id or os.environ.get("HENRI_RUN_ID"),
            arm_id or os.environ.get("HENRI_ARM_ID"),
            commit_sha or os.environ.get("HENRI_COMMIT_SHA"),
            domain_family,
        )
        
        try:
            self.telemetry_queue.put_nowait(record)
        except queue.Full:
            print("[Warning] Telemetry reservoir saturated. Epistemic frame dropped to maintain physical velocity.")

    def _async_batch_writer(self):
        """
        Background process that drains the telemetry reservoir and commits it to the TimescaleDB hypertable
        via bulk copy operations, minimizing database transaction lock delays.
        """
        while not self.shutdown_flag.is_set():
            batch = []
            try:
                # Block until an item is available or timeout occurs
                item = self.telemetry_queue.get(timeout=self.flush_interval_sec)
                batch.append(item)
                
                # Drain the queue up to the configured batch limit
                while len(batch) < self.batch_size:
                    try:
                        batch.append(self.telemetry_queue.get_nowait())
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                pass # Timeout reached, proceed to commit loop if batch has items
            
            if batch:
                self._commit_batch(batch)

    def _commit_batch(self, batch: List[Tuple]):
        """
        Executes a rapid PostgreSQL COPY sequence.
        """
        try:
            with psycopg.connect(self.db_conn_str) as conn:
                expected = os.environ.get("ZONE_C_ENV", "dev").strip().lower()
                assert_zone_c_env(conn, expected)
                with conn.cursor() as cur:
                    with cur.copy(
                        "COPY zone_c_resonant_hypersphere (id, domain, subdomain, concept_key, recorded_at, real_phases, imag_phases, phase_delta, sagnac_clearance, run_id, arm_id, commit_sha, domain_family) FROM STDIN"
                    ) as copy:
                        for record in batch:
                            copy.write_row(record)
                conn.commit()
        except Exception as e:
            # Do not claim that a failed database write entered simulation
            # mode.  The logger is a sink, not a persistence surrogate.
            print(f"[Telemetry Sink Error] Database write failed; batch not persisted: {e}")

    def shutdown(self):
        """
        Gracefully collapse the non-blocking thread and flush residual states.
        """
        self.shutdown_flag.set()
        self.worker_thread.join()
        
        # Final flush
        residual_batch = []
        while not self.telemetry_queue.empty():
            residual_batch.append(self.telemetry_queue.get_nowait())
            
        if residual_batch:
            self._commit_batch(residual_batch)