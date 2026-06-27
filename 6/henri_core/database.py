import numpy as np
import torch
try:
    import psycopg
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

class HenriTimescaleConnector:
    """
    Manages telemetry connections and weight disaggregation plane (Zone C) in TimescaleDB.
    Logs step-by-step relaxation telemetry, including Free Energy and the continuous 4096-D wave state.
    """
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.is_connected = False
        self.prefetch_buffer = {}
        
        # Hardwire a fixed, non-learned permutation index to break structural symmetries (seed=10177)
        self.permutation = np.random.RandomState(seed=10177).permutation(4096)
        
        if not HAS_PSYCOPG:
            print("[DATABASE] Warning: psycopg package not installed. Logging and Zone C disaggregation are disabled.")
            return

        try:
            with psycopg.connect(self.db_uri, connect_timeout=3) as conn:
                self.is_connected = True
                self._init_schemas(conn)
            print(f"[DATABASE] Connected to TimescaleDB at {self.db_uri} and verified schemas.")
        except Exception as e:
            print(f"[DATABASE] Warning: TimescaleDB unreachable at {self.db_uri} ({e}). Running in offline cache mode.")

    def _init_schemas(self, conn):
        with conn.cursor() as cur:
            conn.autocommit = True
            # Create the Zone C Resonant Hypersphere table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS zone_c_resonant_hypersphere (
                    id UUID NOT NULL,
                    domain VARCHAR(64) NOT NULL,
                    subdomain VARCHAR(64) NOT NULL,
                    concept_key VARCHAR(128) NOT NULL,
                    turn_index INT NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL,
                    real_phases REAL[] NOT NULL,
                    imag_phases REAL[] NOT NULL,
                    epiplexity_floor REAL NOT NULL,
                    lipschitz_bound REAL NOT NULL,
                    metadata JSONB,
                    PRIMARY KEY (id, recorded_at)
                );
            """)
            try:
                cur.execute("SELECT create_hypertable('zone_c_resonant_hypersphere', 'recorded_at', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');")
            except Exception as e:
                if "already a hypertable" not in str(e).lower():
                    print(f"[DATABASE] Warning hypertable creation failed: {e}")

    def generate_holographic_key(self, wave_tensor: torch.Tensor) -> np.ndarray:
        """
        Permutes elements and applies bipolar splitting: H_key = sign(Pi * M_H)
        """
        wave_np = wave_tensor.detach().cpu().numpy().astype(np.float32).flatten()
        if np.iscomplexobj(wave_np):
            wave_np = np.real(wave_np)
        if wave_np.shape[0] < 4096:
            wave_np = np.pad(wave_np, (0, 4096 - wave_np.shape[0]))
        else:
            wave_np = wave_np[:4096]
            
        permuted = wave_np[self.permutation]
        h_key = np.sign(permuted).astype(np.int8)
        return h_key

    def async_prefetch_attractors(self, h_key: np.ndarray, domain: str):
        """
        Simulates background Direct Memory Access (DMA) pre-fetching across the CXL bus.
        Queries the database for Zone C attractors and stores them in self.prefetch_buffer.
        """
        if not self.is_connected:
            return
            
        import threading
        def fetch_worker():
            try:
                # Find matching row inside zone_c_resonant_hypersphere using domain tags
                with psycopg.connect(self.db_uri) as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT id, concept_key, real_phases, imag_phases
                            FROM zone_c_resonant_hypersphere
                            WHERE domain = %s
                            ORDER BY recorded_at DESC
                            LIMIT 5;
                        """, (domain,))
                        rows = cur.fetchall()
                        if rows:
                            results = []
                            for r in rows:
                                real_p = np.array(r[2], dtype=np.float32)
                                imag_p = np.array(r[3], dtype=np.float32)
                                # Re-constitute wave vector (4096-D complex)
                                wave = real_p + 1j * imag_p
                                results.append({
                                    "id": r[0],
                                    "key": r[1],
                                    "wave": wave
                                })
                            self.prefetch_buffer[domain] = results
            except Exception as e:
                print(f"[DATABASE] DMA Prefetch failed: {e}")
                
        threading.Thread(target=fetch_worker, daemon=True).start()

    def log_relaxation_step(self, step_id: int, free_energy: float, alpha: float, wave_tensor: torch.Tensor):
        """
        Logs relaxation telemetry into gemma_latent_history.
        Maps the real-valued wave_tensor to vector(8192) by padding with zero imaginary components.
        """
        if not self.is_connected:
            return
            
        try:
            # Flatten and move to CPU
            wave_np = wave_tensor.detach().cpu().numpy().astype(np.float32).flatten()
            hrr_dim = wave_np.shape[0]
            
            # Lossless encoding to 8192-D (real + imaginary split)
            db_vec = np.zeros(2 * hrr_dim, dtype=np.float32)
            db_vec[:hrr_dim] = wave_np
            # db_vec[hrr_dim:] is 0.0 (imaginary part)
            
            # Convert to pgvector literal format: '[val1,val2,...]'
            vector_str = "[" + ",".join(map(str, db_vec.tolist())) + "]"
            
            with psycopg.connect(self.db_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO gemma_latent_history (
                            timestamp, step_id, token_id, generated_text, 
                            sagnac_delta_value, langevin_heat_value, alpha_value, 
                            latent_wave_vector
                        ) VALUES (
                            NOW(), %s, 0, 'relaxation_step', 
                            %s, 0.0, %s, 
                            %s::vector
                        );
                    """, (step_id, free_energy, alpha, vector_str))
                    conn.commit()
        except Exception as e:
            print(f"[DATABASE] Error logging relaxation step {step_id}: {e}")
