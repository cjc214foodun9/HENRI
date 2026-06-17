import numpy as np
import torch
try:
    import psycopg
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

class HenriTimescaleConnector:
    """
    Manages telemetry connections to TimescaleDB.
    Logs step-by-step relaxation telemetry, including Free Energy and the continuous 4096-D wave state.
    """
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.is_connected = False
        
        if not HAS_PSYCOPG:
            print("[DATABASE] Warning: psycopg package not installed. Logging is disabled.")
            return

        try:
            with psycopg.connect(self.db_uri, connect_timeout=3) as conn:
                self.is_connected = True
            print(f"[DATABASE] Connected to TimescaleDB at {self.db_uri}")
        except Exception as e:
            print(f"[DATABASE] Warning: TimescaleDB unreachable at {self.db_uri} ({e}). Logging in offline mode.")

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
