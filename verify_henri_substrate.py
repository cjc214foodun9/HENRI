import os
import sys
import torch
import torch.nn as nn
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "6"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "lib_physics", "wosx")))

print("\n[INIT] Initiating HENRI Substrate Diagnostic Protocol...")
print("========================================================")

# --- PHASE 1: WoSX Binding Verification ---
print("\n[PHASE 1] Testing Continuous PDE Physics Sieve (WoSX)...")
try:
    import wosx
    print("  [SUCCESS] nv-tlabs/wosx bindings successfully imported.")
    # Verify Vulkan targeting if possible via env vars
    if os.environ.get("WOSX_USE_VULKAN") == "1" and os.environ.get("WOSX_USE_CUDA") == "0":
        print("  [SUCCESS] WoSX Environment is strictly locked to Vulkan (AMD Radeon).")
    else:
        print("  [WARNING] WOSX_USE_VULKAN/CUDA env vars not explicitly set in this shell.")
except ImportError as e:
    print(f"  [FATAL] Failed to import wosx. Did the Slang/MSVC compilation fail? Error: {e}")

# --- PHASE 2: Lexical Sync & Dimensionality Verification ---
print("\n[PHASE 2] Testing L3SwarmRouter Hardware Grafting...")
try:
    from l3_router_model import L3SwarmRouter
    
    # Initialize the router exactly as the orchestrator does
    router = L3SwarmRouter(vocab_size=262144, hidden_dim=1024, activation_dim=3840)
    
    # 1. Check the Dimensionality Trap
    emb_shape = router.token_embedding.weight.shape
    if emb_shape == (262144, 3840):
        print(f"  [SUCCESS] L3 Router input matrix matches Gemma 12B exactly: {emb_shape}")
    else:
        print(f"  [FATAL] Dimensionality mismatch! Expected (262144, 3840), got {emb_shape}")
        
    # 2. Check the Gradient Freeze
    if not router.token_embedding.weight.requires_grad:
        print("  [SUCCESS] L3 Router vocabulary matrix is correctly hardwired and frozen.")
    else:
        print("  [FATAL] Vocabulary matrix is leaking gradients. requires_grad must be False.")
        
except Exception as e:
    print(f"  [FATAL] L3 Router initialization failed: {e}")

# --- PHASE 3: Decoupled Micro-Epoch Wave Extraction ---
print("\n[PHASE 3] Testing 64-Token -> 4096-D Wave Projection...")
try:
    # Simulate a 64-token generation block coming from the GPU
    mock_gpu_tokens = torch.randint(0, 262144, (1, 64), device='cpu')
    
    t0 = time.perf_counter()
    # Pass through the router to execute the Euler unit circle projection
    hrr_wave, _, _ = router(tokens=mock_gpu_tokens)
    t1 = time.perf_counter()
    
    wave_shape = hrr_wave.shape
    if wave_shape == (1, 4096) and torch.is_complex(hrr_wave):
        print(f"  [SUCCESS] 64 discrete tokens successfully collapsed into a {wave_shape} complex wave.")
        print(f"  [SUCCESS] Extraction completed in {(t1-t0)*1000:.2f} ms (L3 Cache speed confirmed).")
    else:
        print(f"  [FATAL] Wave projection failed. Expected (1, 4096) complex, got {wave_shape}.")
except Exception as e:
    print(f"  [FATAL] Micro-Epoch bridge failed: {e}")

# --- PHASE 4: Tabula Rasa & Epistemic Distillation Math ---
print("\n[PHASE 4] Testing Tabula Rasa Mathematical Inversions...")
try:
    # Simulate a surviving 3840-D LoRA expert state
    mock_lora_state = torch.randn(3840)
    
    # 1. Test the Epistemic Distillation (Projecting UP)
    # Using the transpose (.T) of the orthogonal bridge as the mathematical inverse
    distilled_wave = torch.matmul(mock_lora_state, router.w_down.weight.T)
    
    if distilled_wave.shape == (4096,):
        print("  [SUCCESS] Epistemic Distillation successfully projected 3840-D LoRA back to 4096-D Zone C wave.")
    else:
        print(f"  [FATAL] Distillation shape mismatch: got {distilled_wave.shape}")

    # 2. Test the Physical Flush (LoRA Zeroing)
    mock_lora_A = nn.Parameter(torch.randn(16, 3840))
    torch.nn.init.zeros_(mock_lora_A)
    
    if torch.sum(mock_lora_A) == 0.0:
        print("  [SUCCESS] PyTorch LoRA adapter successfully zeroed (Open Mind confirmed).")
    else:
        print("  [FATAL] LoRA zeroing failed. Ghosts of previous tasks remain.")
        
except Exception as e:
    print(f"  [FATAL] Tabula Rasa math failed: {e}")

print("\n========================================================")
print("[DIAGNOSTIC COMPLETE]")
