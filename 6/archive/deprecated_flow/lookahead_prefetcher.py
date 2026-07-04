import numpy as np
import threading
import queue
import time
import os
import psycopg
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from henri_contract import db_to_complex, DIMS
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class DatabaseConnectionStub:
    """
    High-fidelity database stub that generates deterministic mock vectors 
    when the local or cloud TimescaleDB instance is unreachable.
    """
    def __init__(self, source_dim=1024):
        self.source_dim = source_dim
        
    def fetch_neighborhood_vectors(self, semantic_hint_id: int, vocab_size: int) -> list:
        # Simulate fetching vectors from the TimescaleDB hypertable
        raw_rows = []
        rng = np.random.default_rng(seed=semantic_hint_id & 0xffffffff)
        for idx in range(vocab_size):
            row_id = (semantic_hint_id * 1000) + idx
            # 1024 complex float row
            real_part = rng.normal(size=self.source_dim)
            imag_part = rng.normal(size=self.source_dim)
            vec_1024 = real_part + 1j * imag_part
            vec_1024 = vec_1024 / (np.linalg.norm(vec_1024) + 1e-8)
            raw_rows.append({
                "id": row_id,
                "vector": vec_1024
            })
        return raw_rows

class ZoneCPostgresEmulatorBridge:
    """
    Asynchronous lookahead prefetcher for the Zone C TimescaleDB/TigerDB hypertable.
    Hides storage bandwidth limits by loading upcoming semantic neighborhoods into SRAM
    and upscaling them dynamically from 1024D to 4096D via isometric projections.
    """
    def __init__(self, db_connection=None, vocab_size: int = 1024, source_dim: int = 1024, target_dim: int = 4096, test_mode: bool = False):
        self.vocab_size = vocab_size
        self.source_dim = source_dim
        self.target_dim = target_dim
        
        # Determine database setup
        db_url = os.environ.get("DATABASE_URL")
        if db_connection is not None:
            self.db = db_connection
            self.is_stub = False
        elif db_url:
            try:
                # Test connection
                with psycopg.connect(db_url, connect_timeout=2) as conn:
                    pass
                self.db = self  # Self handles active connection
                self.db_url = db_url
                self.is_stub = False
                print("[SYSTEM] Prefetcher connected successfully to TigerDB TimescaleDB.")
            except Exception as e:
                print(f"[WARNING] TigerDB unreachable ({e}). Using high-fidelity Database Connection Stub.")
                self.db = DatabaseConnectionStub(source_dim=self.source_dim)
                self.is_stub = True
        elif test_mode:
            print("[SYSTEM] Prefetcher booting in test/mock mode. Disabling network database connections.")
            self.db = DatabaseConnectionStub(source_dim=self.source_dim)
            self.is_stub = True
        else:
            self.db = DatabaseConnectionStub(source_dim=self.source_dim)
            self.is_stub = True
            
        # Local SRAM cache buffer (double-buffered vocabulary)
        self.sram_vocab_buffer = None
        self.cached_ids = []
        
        # Async prefetch queue
        self.prefetch_queue = queue.Queue()
        self.stop_signal = threading.Event()
        
        # Initialize the rigid, invariant Hyperspace Projection Matrix (W_up)
        # Using a fixed seed guarantees the geometric topology remains identical across boots
        # We construct a strictly orthonormal projection matrix without heavy LAPACK QR calls to prevent thread livelocks
        self.W_up = np.zeros((self.target_dim, self.source_dim), dtype=np.complex64)
        rng = np.random.default_rng(seed=42)
        step = self.target_dim // self.source_dim
        for col in range(self.source_dim):
            row = col * step
            phasor = np.exp(1j * rng.uniform(-np.pi, np.pi))
            self.W_up[row, col] = phasor
        
        # Boot background lookahead prefetch worker thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker_loop, daemon=True)
        self.prefetch_thread.start()
        
        # Proactively load a baseline neighborhood (id 0) to avoid initial buffer stalls
        self.trigger_lookahead(0)

    def fetch_neighborhood_vectors(self, semantic_hint_id: int, vocab_size: int) -> list:
        """Standard TimescaleDB hypertable query wrapper when running natively."""
        raw_rows = []
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # 1. Fetch axioms from hrr_canonical_lexicon
                    cur.execute("""
                        SELECT semantic_label, hrr_wavefront 
                        FROM hrr_canonical_lexicon 
                        LIMIT %s;
                    """, (vocab_size,))
                    rows = cur.fetchall()
                    for idx, r in enumerate(rows):
                        label, vec_str = r[0], r[1]
                        vec_complex = db_to_complex(vec_str, DIMS.hrr_dim)
                        
                        # Pad or truncate to source_dim
                        if len(vec_complex) < self.source_dim:
                            vec_complex = np.pad(vec_complex, (0, self.source_dim - len(vec_complex)))
                        else:
                            vec_complex = vec_complex[:self.source_dim]
                            
                        concept_id = hash(label) & 0xffffffff
                        raw_rows.append({
                            "id": concept_id,
                            "vector": vec_complex
                        })
                        
                    # 2. If we need more rows, fetch from gemma_latent_history
                    remaining_slots = vocab_size - len(raw_rows)
                    if remaining_slots > 0:
                        cur.execute("""
                            SELECT step_id, latent_wave_vector 
                            FROM gemma_latent_history 
                            ORDER BY step_id DESC 
                            LIMIT %s;
                        """, (remaining_slots,))
                        rows = cur.fetchall()
                        for r in rows:
                            step_id, vec_str = r[0], r[1]
                            vec_complex = db_to_complex(vec_str, DIMS.hrr_dim)
                            
                            if len(vec_complex) < self.source_dim:
                                vec_complex = np.pad(vec_complex, (0, self.source_dim - len(vec_complex)))
                            else:
                                vec_complex = vec_complex[:self.source_dim]
                                
                            raw_rows.append({
                                "id": step_id,
                                "vector": vec_complex
                            })
        except Exception as e:
            print(f"[Zone C DB Prefetch Error] {e}")
            
        # 3. Fill the rest of the vocabulary using the deterministic stub to reach vocab_size exactly
        needed = vocab_size - len(raw_rows)
        if needed > 0:
            stub = DatabaseConnectionStub(source_dim=self.source_dim)
            stub_rows = stub.fetch_neighborhood_vectors(semantic_hint_id, needed)
            raw_rows.extend(stub_rows)
            
        return raw_rows

    def _prefetch_worker_loop(self):
        """Asynchronously streams upcoming semantic vector blocks into SRAM to bypass IO latency."""
        while not self.stop_signal.is_set():
            try:
                # Wait for lookahead cue from Agentic Router
                semantic_hint_id = self.prefetch_queue.get(timeout=0.1)
                
                # Fetch vectors from storage (SSD / DB Hypertable)
                raw_rows = self.db.fetch_neighborhood_vectors(semantic_hint_id, self.vocab_size)
                if not raw_rows:
                    self.prefetch_queue.task_done()
                    continue
                
                # Extract all vectors and IDs
                vec_list = [row['vector'] for row in raw_rows]
                temp_ids = [row['id'] for row in raw_rows]
                
                # Stack into a single matrix of shape (vocab_size, 1024)
                vec_matrix = np.vstack(vec_list)
                
                # Batch up-project: (vocab_size, 1024) @ (1024, 4096) -> (vocab_size, 4096)
                # W_up is shape (4096, 1024)
                projected_matrix = np.dot(vec_matrix, self.W_up.T)
                
                # Batch normalize row-wise
                norms = np.linalg.norm(projected_matrix, axis=1, keepdims=True) + 1e-8
                temp_vocab = projected_matrix / norms
                
                # Commit staging buffer to active SRAM cache
                self.sram_vocab_buffer = temp_vocab
                self.cached_ids = temp_ids
                
                self.prefetch_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SYSTEM] Prefetch worker error: {e}")

    def trigger_lookahead(self, current_concept_id: int):
        """Non-blocking call to notify the prefetcher where semantic momentum is heading."""
        if self.prefetch_queue.qsize() == 0:
            self.prefetch_queue.put(current_concept_id)

    def close(self):
        """Graceful shutdown of prefetch threads."""
        self.stop_signal.set()
        if self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=2.0)
