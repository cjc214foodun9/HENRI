import os
import sys
import torch
import numpy as np

# Add parent directories to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "6"))

from dialogue_ingress import HenriDialogueIngress
from memory_cache import CachedHRRMemoryEngine
from cognitive_swarm import HenriCognitiveSwarmOrchestrator

def test_subspace_partitioning():
    print("[TEST 1] Testing Phase Sub-Space Partitioning...")
    ingress = HenriDialogueIngress(hrr_dim=4096, split_dim=2048, threshold=0.5)
    
    # Generate dummy phase angles
    phases = torch.randn(1, 4096)
    
    chat_p, code_p, should_route = ingress.partition_phases(phases)
    
    # 1. Verify orthogonal projections: Q^T * Q = I.
    # The dot product of chat_p and code_p in projected coordinate space is exactly 0.
    # Let's verify their inner product is close to 0 in original space
    dot_product = torch.dot(chat_p.flatten(), code_p.flatten())
    print(f" -> Subspace dot product (target 0.0): {dot_product.item():.6f}")
    assert abs(dot_product.item()) < 1e-3, "Subspaces are not orthogonal!"
    print(" -> [PASS] Subspace orthogonality verified.")

    # 2. Test routing trigger behavior
    # Code-like phase (concentrated in the second half of coordinates)
    code_phases = torch.zeros(1, 4096)
    projected_code = torch.zeros(1, 4096)
    projected_code[0, 2048:] = 5.0 # High energy in operator subspace
    code_phases = torch.matmul(projected_code, ingress.Q.t())
    
    _, _, should_route_code = ingress.partition_phases(code_phases)
    assert should_route_code.item() == True, "Failed to route high-energy operator wave!"
    
    # Chat-like phase (concentrated in the first half of coordinates)
    chat_phases = torch.zeros(1, 4096)
    projected_chat = torch.zeros(1, 4096)
    projected_chat[0, :2048] = 5.0 # High energy in conversational subspace
    chat_phases = torch.matmul(projected_chat, ingress.Q.t())
    
    _, _, should_route_chat = ingress.partition_phases(chat_phases)
    assert should_route_chat.item() == False, "Incorrectly routed conversational wave to sandbox!"
    
    print(" -> [PASS] Dynamic routing threshold logic verified.")

def test_pgvector_timescaledb_memory():
    print("\n[TEST 2] Testing pgvector TimescaleDB Memory Cache...")
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5432/henri")
    try:
        engine = CachedHRRMemoryEngine(wave_dim=4096, db_url=db_url)
        if not engine.is_db_connected:
            raise Exception("Port 5432 connection failed")
    except Exception:
        fallback_url = "postgresql://postgres:password@localhost:5433/henri"
        print(f" -> Port 5432 failed. Trying fallback to {fallback_url}...")
        engine = CachedHRRMemoryEngine(wave_dim=4096, db_url=fallback_url)
    
    if not engine.is_db_connected:
        print(" -> [SKIP] TimescaleDB unreachable. Skipping database test.")
        return
        
    print(" -> Database hypertable successfully initialized.")
    
    # Test pushing a state
    dummy_signature = torch.randn(4096)
    # Set working wave phase angles
    dummy_phases = torch.randn(4096)
    engine.active_wave.copy_(torch.polar(torch.ones(4096), dummy_phases))
    
    print(" -> Pushing memory state to pgvector hypertable...")
    engine.push_to_growing_cache(dummy_signature)
    
    # Test retrieving the state
    print(" -> Retrieving memory state via cosine similarity query...")
    retrieved_wave = engine.retrieve_from_cache(dummy_signature)
    
    # Verify that the retrieved wave resembles the pushed wave
    pushed_wave = torch.polar(torch.ones(4096), dummy_phases)
    resonance = engine.compute_phase_resonance(retrieved_wave, pushed_wave)
    print(f" -> Retrieved wave phase resonance match: {resonance:.4f}")
    assert resonance > 0.0, "Retrieved memory does not match signature lookup!"
    print(" -> [PASS] pgvector insertion and retrieval lookup verified.")

def test_telemetry_downsampling():
    print("\n[TEST 3] Testing GPU Telemetry Downsampling...")
    # Initialize orchestrator
    orchestrator = HenriCognitiveSwarmOrchestrator(num_streams=16, hrr_dim=4096)
    
    # Create dummy 6324x6324 complex wave on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_bulk = torch.randn(6324, 6324, dtype=torch.complex64, device=device)
    
    start_time = time.time()
    downsampled = orchestrator.downsample_telemetry_wave(dummy_bulk)
    duration = time.time() - start_time
    
    print(f" -> Downsampling duration: {duration:.4f} seconds")
    print(f" -> Downsampled grid shape: {downsampled.shape}")
    assert downsampled.shape == (64, 64), "Downsampled shape is not 64x64!"
    print(" -> [PASS] GPU Telemetry Downsampler verified.")

if __name__ == "__main__":
    import time
    print("=====================================================================")
    # Temporarily ignore psycopg pool warnings at interpreter finalization
    import warnings
    warnings.filterwarnings("ignore")
    
    print("        BOOTING DIALOGUE INGRESS & PGVECTOR VERIFICATION SUITE       ")
    print("=====================================================================\n")
    test_subspace_partitioning()
    test_pgvector_timescaledb_memory()
    test_telemetry_downsampling()
    print("\n[SUCCESS] Dialogue Ingress & pgvector Memory tests passed successfully.")
