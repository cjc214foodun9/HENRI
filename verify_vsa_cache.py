import torch
import sys
import os

# Ensure import paths are correct
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vsa_cache_stream import SwarmTemporalCacheManager

def run_vsa_substrate_test():
    print("[*] Launching VSA Holographic Caching test harness...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")
      
    # Instantiate manager for your 16 active streams
    manager = SwarmTemporalCacheManager(num_streams=16, hidden_dim=4096)
      
    # Replicate a synthetic mock wave coming out of your 30 offloaded GPU layers
    mock_hidden_wave = torch.randn(16, 4096, device=device)
    mock_hidden_wave = torch.nn.functional.normalize(mock_hidden_wave, p=2, dim=-1)
      
    # Process across 5 sequential micro-epoch iterations
    for micro_step in range(5):
        print(f"\n--- STEP {micro_step} ---")
        manager.process_and_cache_segment(mock_hidden_wave, segment_step_idx=micro_step)
          
    # Verify hash addressing generation passes cleanly
    hash_signatures = manager.generate_holographic_hash()
    print(f"\n[+] Generated Bipolar Hash Table Address Shape: {list(hash_signatures.shape)}")
    assert hash_signatures.shape == (16, 4096), "Memory alignment verification mismatch."
    
    # Ensure there are no 0.0 values (must be strictly 1.0 or -1.0)
    has_zeros = torch.any(hash_signatures == 0.0).item()
    assert not has_zeros, "Found dead 0.0 states in the bipolar hash space!"
    
    # Ensure all values are 1.0 or -1.0
    valid_states = torch.all((hash_signatures == 1.0) | (hash_signatures == -1.0)).item()
    assert valid_states, "Bipolar hash has invalid states (not 1.0 or -1.0)!"
    
    print("[SUCCESS] All VSA cache loop checks passed. Manifold perspective is perfectly contiguous.")

if __name__ == "__main__":
    run_vsa_substrate_test()
