"""  
Project HENRI: Holographic Associative DMA Retrieval Engine  
Component: Core Memory Ingestion, Structural Symmetry Breaking, and Phase Retrieval  
Author: Joseph Valentine (Bespoke Silicon Photonic Architecture Core)  
Date: 2026-06-23  
"""

import math  
import uuid  
import sys
import os
import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.fft

# Ensure parent directory is in sys.path to run tests cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  
    import psycopg  
    from psycopg.rows import dict_row  
    HAS_PSYCOPG = True  
except ImportError:  
    HAS_PSYCOPG = False

class HolographicDMALookupCore(nn.Module):  
    """  
    Content-addressable DMA lookup engine operating over high-rank vector symbolic channels.  
    Bypasses text translation layers by executing direct phase-resonance traces.  
    """  
    def __init__(self, dimension: int = 4096, db_uri: str = None):  
        super().__init__()  
        self.dim = dimension  
        self.db_uri = db_uri  
        self.register_buffer("device_sentinel", torch.empty(0))  
          
        # Hardwire a fixed, non-learned permutation matrix index to break structural symmetries  
        # Enforces coordinate isolation without mutating vector L2 norm footprints  
        torch.manual_seed(10177)  # Unitary reference scalar anchor  
        perm_indices = torch.randperm(self.dim)  
        self.register_buffer("symmetry_permutation", perm_indices)  
          
        # Calculate matching phase index scaler for packed phase operations modulo 256  
        self.phase_scale = 256.0 / (2.0 * math.pi)

    def apply_symmetry_breaking(self, wave_tensor: torch.Tensor) -> torch.Tensor:  
        """  
        Permutes coordinate elements across the final vector axis to decouple concurrent tracks.  
        Supports shapes of both (Batch, Dim) and (Batch, SeqLen, Dim).  
        """  
        return torch.index_select(wave_tensor, dim=-1, index=self.symmetry_permutation)

    def circular_correlation(self, active_wave: torch.Tensor, memory_wave: torch.Tensor) -> torch.Tensor:  
        """  
        Computes pure wave-domain unbinding traces using Real Fast Fourier Transforms (RFFT) on CPU.  
        Correlation: A * B = IRFFT( RFFT(A) * conj(RFFT(B)) )  
        """  
        # Force computation on CPU to prevent VRAM page-swapping on accelerator
        dev = active_wave.device
        active_cpu = active_wave.cpu()  
        mem_wave_clean = memory_wave.cpu()  
          
        # Move into the complex frequency spectrum  
        fft_active = torch.fft.rfft(active_cpu, dim=-1)  
        fft_memory = torch.fft.rfft(mem_wave_clean, dim=-1)  
          
        # Apply phase conjugation to execute the unbinding alignment check  
        correlation_spectrum = fft_active * torch.conj(fft_memory)  
          
        # Return to continuous spatial bounds, forcing exact coordinate sizing boundary clamps  
        unbind_result = torch.fft.irfft(correlation_spectrum, n=self.dim, dim=-1)
        
        # Project back to the original accelerator device track
        return unbind_result.to(device=dev)

    def reconstitute_db_vector_to_wave(self, db_vector_list: list) -> torch.Tensor:  
        """  
        Converts raw 8192-float unrolled rows from pgvector back into standard complex wave structures.  
        Matches the database.py layout format: Real components (0:4096), Imaginary components (4096:8192).  
        """  
        vec_np = np.array(db_vector_list, dtype=np.float32)  
        real_part = torch.from_numpy(vec_np[:self.dim])  
        imag_part = torch.from_numpy(vec_np[self.dim:])  
          
        # Forge the unified complex plane tensor  
        return torch.complex(real_part, imag_part).to(self.device_sentinel.device)

    def execute_dma_lookup(self, active_context_wave: torch.Tensor, top_k: int = 5, domain: str = "programming_logic") -> list:  
        """  
        Ingests a 4096-D continuous wave key, formats it, and runs a parallel pgvector inner-product query.  
        Returns a list of highly resonant sub-axioms, complete with pre-calculated similarity metrics.  
        """  
        if not HAS_PSYCOPG or not self.db_uri:  
            print("[DMA CORE] Warning: Lookup skipped. System running in disconnected, local-only mode.")  
            return []

        # Flatten wave tensor, isolate batch footprints, and move matrix references to host memory  
        wave_flattened = active_context_wave.detach().clone()  
        if wave_flattened.ndim > 1:  
            wave_flattened = wave_flattened[0] # Isolate principal streaming frame  
        if wave_flattened.is_complex():  
            real_np = wave_flattened.real.cpu().numpy().astype(np.float32).flatten()  
            imag_np = wave_flattened.imag.cpu().numpy().astype(np.float32).flatten()  
        else:  
            real_np = wave_flattened.cpu().numpy().astype(np.float32).flatten()  
            imag_np = np.zeros(self.dim, dtype=np.float32)

        # Build the exact unrolled 8192-D real-imaginary data payload  
        db_payload = np.zeros(2 * self.dim, dtype=np.float32)  
        db_payload[:self.dim] = real_np  
        db_payload[self.dim:] = imag_np

        # Convert to exact pgvector literal array formatting string  
        vector_literal = "[" + ",".join(map(str, db_payload.tolist())) + "]"  
        retrieved_artifacts = []

        try:  
            # Open a direct connection and fetch parameters via inner-product sorting (negative similarity)  
            with psycopg.connect(self.db_uri, connect_timeout=3) as conn:  
                with conn.cursor(row_factory=dict_row) as cur:  
                    query = """  
                        SELECT concept_hash, semantic_label, domain_tag, epiplexity_weight,  
                               hrr_wavefront,  
                               (hrr_wavefront <#> %s::vector) AS inner_product_distance  
                        FROM hrr_canonical_lexicon  
                        WHERE domain_tag = %s  
                        ORDER BY hrr_wavefront <#> %s::vector ASC  
                        LIMIT %s;  
                    """  
                    cur.execute(query, (vector_literal, domain, vector_literal, top_k))  
                    rows = cur.fetchall()  
                      
                    for row in rows:  
                        # Extract the string representation of pgvector and unpack floats safely  
                        vec_str = row["hrr_wavefront"]  
                        if isinstance(vec_str, str):  
                            vec_str = vec_str.strip("[]")  
                            float_array = [float(x) for x in vec_str.split(",") if x.strip()]  
                        else:  
                            float_array = list(vec_str)  
                              
                        # Reconstitute structural wave vectors back into hardware memory channels  
                        complex_wave = self.reconstitute_db_vector_to_wave(float_array)  
                          
                        # Invert index metrics to compute normal, positive cosine phase alignment resonance  
                        cosine_resonance = -float(row["inner_product_distance"])  
                          
                        retrieved_artifacts.append({  
                            "uuid": row["concept_hash"],  
                            "label": row["semantic_label"],  
                            "resonance": cosine_resonance,  
                            "wave": complex_wave  
                        })  
        except Exception as e:  
            print(f"[DMA CORE] Error executing content-addressable database lookup: {e}")  
              
        return retrieved_artifacts

def run_clean_room_validation():  
    print("=== INITIALIZING HENRI HOLOGRAPHIC DMA RETRIEVAL SUITE ===")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"[BOOT] Target accelerator environment initialized: {device}")

    # 1. Instantiate Core Substrates  
    from henri_core.hrr import HRRInputLayer  
    input_layer = HRRInputLayer(dim=4096).to(device)  
    db_uri = os.getenv("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/henri")  
    dma_core = HolographicDMALookupCore(dimension=4096, db_uri=db_uri).to(device)  
    print("[SUCCESS] Vector symbolic engines and hardware buffers bound.")

    # 2. Verify Symmetry-Breaking Permutation Invariance  
    torch.manual_seed(42)  
    mock_active_ctx = torch.randn(1, 4096, device=device)  
    permuted_wave = dma_core.apply_symmetry_breaking(mock_active_ctx)  
      
    # Calculate geometric dot-product distance metrics  
    base_norm = torch.norm(mock_active_ctx, p=2, dim=-1)  
    perm_norm = torch.norm(permuted_wave, p=2, dim=-1)  
    norm_delta = torch.abs(base_norm - perm_norm).item()  
    cosine_similarity = torch.clamp(F.cosine_similarity(mock_active_ctx, permuted_wave, dim=-1), -1.0, 1.0).item()

    print(f"[MANIFOLD] Wave norm baseline: {base_norm.item():.4f} | Permuted norm footprint: {perm_norm.item():.4f}")  
    assert norm_delta < 1e-5, "Fatal: Orthogonal phase permutation altered wave energy invariants!"  
    print(f"[SUCCESS] Hypersphere energy conservation verified (Norm Delta: {norm_delta:.2e}).")  
    print(f"[MANIFOLD] Symmetry breaking trace cross-talk index (Cosine Similarity): {cosine_similarity:.4f}")  
    assert abs(cosine_similarity) < 0.05, "Warning: Permutation matrix failed to break spatial coordinate patterns!"  
    print("[SUCCESS] Structural symmetry breaking verified. Phase spaces isolated.")

    # 3. Shape Invariant Testing  
    mock_db_vector = [0.1] * 8192  
    reconstituted_tensor = dma_core.reconstitute_db_vector_to_wave(mock_db_vector)  
    print(f"[DATA INTEGRITY] Reconstituted wave dimension profile: {reconstituted_tensor.shape} | Type: {reconstituted_tensor.dtype}")  
    assert reconstituted_tensor.shape == torch.Size([4096]), "Fatal: Database vector unrolling caused dimensional shift!"  
    assert reconstituted_tensor.is_complex(), "Fatal: Reconstituted tensor dropped complex phase properties!"  
    print("[SUCCESS] Database unrolling checks passed. Structural formatting is stable.")  
    print("=== PHASE 1 COGNITIVE RETRIEVAL INFRASTRUCTURE SECURED ===")

if __name__ == "__main__":  
    run_clean_room_validation()
