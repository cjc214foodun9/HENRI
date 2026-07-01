import torch
import torch.nn as nn
import math
import os

class CachedHRRMemoryEngine(nn.Module):
    """
    Implements a Hierarchical Growing Memory Cache using Phase-Only INT8 Quantization
    on the S^1 Manifold and orthonormal Vector Symbolic Architecture (VSA) binding.
    Integrates with pgvector and TimescaleDB for persistent long-term storage of older states.
    """
    def __init__(self, wave_dim=4096, coherence_threshold=0.70, accumulation_limit=15, db_url=None):
        super().__init__()
        self.wave_dim = wave_dim
        self.coherence_threshold = coherence_threshold
        self.accumulation_limit = accumulation_limit
        
        # Working memory: 4096-D complex polar tensor, initialized to transparent window (zeros phase)
        self.register_buffer('active_wave', torch.polar(torch.ones(wave_dim), torch.zeros(wave_dim)))
        
        # Counter for tracking how many states have been superposed into working memory
        self.accumulation_counter = 0
        
        # Historical memory buffers (quantized to int8 phase angles to save DRAM and L3 cache footprint)
        self.cached_waves = [] # List of torch.ByteTensor (int8)
        self.cached_keys = []  # List of torch.Tensor (float32 signatures)
        
        # pgvector database connection setup
        self.db_url = db_url or os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
        self.is_db_connected = False
        
        if self.db_url:
            try:
                import psycopg
                # Connect briefly to check extensions and setup database
                with psycopg.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS henri_conversation_memory (
                                id SERIAL,
                                session_id VARCHAR(255) DEFAULT 'default_session',
                                turn_index INT,
                                embedding vector(4096),
                                signature vector(4096),
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                PRIMARY KEY (id, created_at)
                            );
                        """)
                        
                        # Set up TimescaleDB hypertable if not already configured
                        cur.execute("""
                            SELECT count(*) FROM _timescaledb_catalog.hypertable 
                            WHERE table_name = 'henri_conversation_memory';
                        """)
                        row = cur.fetchone()
                        if row and row[0] == 0:
                            try:
                                cur.execute("SELECT create_hypertable('henri_conversation_memory', 'created_at', if_not_exists => TRUE);")
                            except Exception:
                                pass
                        
                        conn.commit()
                self.is_db_connected = True
                print("[DATABASE] pgvector TimescaleDB hypertable initialized successfully.")
            except Exception as e:
                print(f"[DATABASE] Warning: Failed to connect to pgvector database ({e}). Running in-memory cache only.")

    def _tensor_to_vector_str(self, tensor):
        """Converts a 1D tensor to pgvector format '[v1, v2, ...]'."""
        lst = tensor.flatten().tolist()
        return f"[{','.join(map(str, lst))}]"

    def complex_circular_convolution(self, a, b):
        """
        Performs VSA circular binding using 1D Orthonormal Fast Fourier Transforms.
        Preserves vector magnitude profiles.
        """
        a_fft = torch.fft.fft(a, norm="ortho")
        b_fft = torch.fft.fft(b, norm="ortho")
        bound_fft = a_fft * b_fft
        return torch.fft.ifft(bound_fft, norm="ortho")

    def compute_phase_resonance(self, wave_a, wave_b):
        """
        Calculates the Cosine Similarity of phase angles (phase resonance)
        between two unit-modulus wavefronts. Range: [-1.0, 1.0]
        """
        resonance = torch.real(wave_a * torch.conj(wave_b)).mean()
        return resonance.item()

    def quantize_phase_to_int8(self, complex_wave):
        """
        Topologically maps phase angles from [-pi, pi) to [-128, 127] (INT8).
        Bypasses amplitude quantization noise completely.
        """
        phases = torch.angle(complex_wave)
        scaled_phases = torch.round(phases * (128.0 / math.pi))
        clamped_phases = torch.clamp(scaled_phases, -128, 127)
        return clamped_phases.to(torch.int8)

    def dequantize_int8_to_complex(self, int8_phases):
        """
        Reconstructs a perfect unit-modulus complex wavefront from INT8 phases.
        """
        float_phases = int8_phases.to(torch.float32) * (math.pi / 128.0)
        return torch.polar(torch.ones_like(float_phases), float_phases)

    def push_to_growing_cache(self, signature_key):
        """
        Quantizes the active working memory wave, stores it in database or DRAM cache,
        and resets active memory back to the uniform transparent state.
        """
        if self.is_db_connected:
            try:
                import psycopg
                with psycopg.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        emb_str = self._tensor_to_vector_str(torch.angle(self.active_wave).cpu())
                        sig_str = self._tensor_to_vector_str(signature_key.cpu())
                        cur.execute("""
                            INSERT INTO henri_conversation_memory (session_id, turn_index, embedding, signature)
                            VALUES (%s, %s, %s, %s);
                        """, ('default_session', self.accumulation_counter, emb_str, sig_str))
                        conn.commit()
            except Exception as e:
                print(f"[DATABASE] Error inserting memory: {e}")
        else:
            # Fallback to local DRAM cache
            quantized_wave = self.quantize_phase_to_int8(self.active_wave)
            self.cached_waves.append(quantized_wave)
            self.cached_keys.append(signature_key.detach().clone().cpu())
            
        # Reset active working memory state and counter
        self.active_wave.copy_(torch.polar(torch.ones(self.wave_dim, device=self.active_wave.device), 
                                           torch.zeros(self.wave_dim, device=self.active_wave.device)))
        self.accumulation_counter = 0

    def update_active_memory(self, token_activation, position_key, signature_key):
        """
        Bridges digital activations into the working holographic wavefront via circular
        convolution, tracks coherence degradation, and selectively offloads to database/DRAM.
        """
        device = self.active_wave.device
        token_activation = token_activation.to(device)
        position_key = position_key.to(device)
        if signature_key is not None:
            signature_key = signature_key.to(device)

        # 1. Bind token activation to sequence position key via circular convolution
        bound_state = self.complex_circular_convolution(token_activation, position_key)
        
        # 2. Bundle (superpose) the bound state into our active working memory
        old_active = self.active_wave.clone()
        new_active = old_active + bound_state
        
        # Re-enforce unit-modulus constraint to maintain physical wave compatibility
        active_phases = torch.angle(new_active)
        self.active_wave.copy_(torch.polar(torch.ones_like(active_phases), active_phases))
        
        self.accumulation_counter += 1
        
        # 3. Analyze coherence deflection
        coherence = self.compute_phase_resonance(self.active_wave, old_active)
        
        # 4. Check trigger thresholds (coherence collapse or raw limit)
        if coherence < self.coherence_threshold or self.accumulation_counter >= self.accumulation_limit:
            self.push_to_growing_cache(signature_key)
            return True # Indicates cache offload occurred
            
        return False

    def retrieve_from_cache(self, query_key):
        """
        Performs a soft phase-resonance lookup over pgvector or DRAM cache.
        Returns the dequantized reconstructed memory wavefront.
        """
        if self.is_db_connected:
            try:
                import psycopg
                with psycopg.connect(self.db_url) as conn:
                    with conn.cursor() as cur:
                        q_str = self._tensor_to_vector_str(query_key.cpu())
                        # Order by cosine distance index search and get top 10
                        cur.execute("""
                            SELECT embedding, 1 - (signature <=> %s) AS similarity 
                            FROM henri_conversation_memory 
                            ORDER BY signature <=> %s 
                            LIMIT 10;
                        """, (q_str, q_str))
                        rows = cur.fetchall()
                        
                        if not rows:
                            return torch.polar(torch.ones(self.wave_dim, device=self.active_wave.device), 
                                               torch.zeros(self.wave_dim, device=self.active_wave.device))
                        
                        embeddings = []
                        similarities = []
                        for row in rows:
                            emb_val = [float(x) for x in row[0].strip('[]').split(',')]
                            embeddings.append(torch.tensor(emb_val, dtype=torch.float32, device=self.active_wave.device))
                            similarities.append(row[1])
                            
                        sim_tensor = torch.tensor(similarities, dtype=torch.float32, device=self.active_wave.device)
                        weights = torch.softmax(sim_tensor, dim=0)
                        
                        accumulated_phases = torch.zeros(self.wave_dim, device=self.active_wave.device)
                        for idx, emb_tensor in enumerate(embeddings):
                            accumulated_phases += weights[idx] * emb_tensor
                            
                        return torch.polar(torch.ones_like(accumulated_phases), accumulated_phases)
            except Exception as e:
                print(f"[DATABASE] Error retrieving memory: {e}. Falling back to uniform wave.")
                return torch.polar(torch.ones(self.wave_dim, device=self.active_wave.device), 
                                   torch.zeros(self.wave_dim, device=self.active_wave.device))
        else:
            # Fallback to local DRAM cache lookup
            if not self.cached_waves:
                return torch.polar(torch.ones(self.wave_dim, device=self.active_wave.device), 
                                   torch.zeros(self.wave_dim, device=self.active_wave.device))
                
            stacked_keys = torch.stack(self.cached_keys)
            query_key_cpu = query_key.detach().clone().cpu()
            
            dot_products = torch.mv(stacked_keys, query_key_cpu)
            key_norms = torch.norm(stacked_keys, dim=1)
            query_norm = torch.norm(query_key_cpu)
            similarities = dot_products / (key_norms * query_norm + 1e-8)
            
            weights = torch.softmax(similarities, dim=0).to(self.active_wave.device)
            
            accumulated_phases = torch.zeros(self.wave_dim, device=self.active_wave.device)
            for idx, quantized_wave in enumerate(self.cached_waves):
                reconstructed = self.dequantize_int8_to_complex(quantized_wave).to(self.active_wave.device)
                accumulated_phases += weights[idx] * torch.angle(reconstructed)
                
            return torch.polar(torch.ones_like(accumulated_phases), accumulated_phases)
