import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure path is set to import modules from HENRI and HENRI/6
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "6"))

from henri_core.core import ProprietaryHENRICore
from diffusion_canvas import NonAutoregressiveCanvasSampler, BirkhoffTopologicalLoss

def run_diffusion_canvas_verification():
    print("=====================================================================")
    # Set seed for deterministic test outputs
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    # Use a small test dimension
    dim = 256
    depth = 2
    num_experts = 2
    vocab_size = 1000
    seq_len = 64
    num_steps = 10

    print(f"[*] Initializing test components (dim={dim}, vocab_size={vocab_size})...")
    core = ProprietaryHENRICore(dim=dim, depth=depth, num_fluid_states=num_experts).to(device)
    translation_head = nn.Linear(dim, vocab_size).to(device)

    # ========================================================
    # [PART 1] Verifying BirkhoffTopologicalLoss
    # ========================================================
    print("\n========================================================")
    print("[TEST 1] Testing BirkhoffTopologicalLoss...")
    print("========================================================")
    
    criterion = BirkhoffTopologicalLoss(translation_head, alpha=1.0, beta=0.05, eta=0.1)
    
    # Mock tensors
    pred_score = torch.randn(2, seq_len, dim, device=device, requires_grad=True)
    target_score = torch.randn(2, seq_len, dim, device=device)
    canvas_state = torch.randn(2, seq_len, dim, device=device, requires_grad=True)
    
    # Forward pass
    loss, metrics = criterion(pred_score, target_score, canvas_state)
    
    print(f"Computed loss: {loss.item():.6f}")
    print("Metrics telemetry:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.6f}")
        
    assert loss.ndim == 0, "Loss must be a scalar tensor"
    assert "loss_score_mse" in metrics, "Missing score loss metric"
    assert "complexity_entropy_C" in metrics, "Missing entropy C metric"
    assert "roughness_TV_O" in metrics, "Missing TV order metric"
    assert "birkhoff_measure_estimate" in metrics, "Missing Birkhoff measure estimate"
    
    # Backward pass check to ensure gradients propagate safely
    loss.backward()
    assert pred_score.grad is not None, "Gradients failed to propagate to pred_score"
    assert canvas_state.grad is not None, "Gradients failed to propagate to canvas_state"
    print("[+] BirkhoffTopologicalLoss backpropagation verification: PASS")

    # ========================================================
    # [PART 2] Verifying NonAutoregressiveCanvasSampler
    # ========================================================
    print("\n========================================================")
    print("[TEST 2] Testing NonAutoregressiveCanvasSampler...")
    print("========================================================")
    
    sampler = NonAutoregressiveCanvasSampler(core, translation_head, num_diffusion_steps=num_steps)
    
    # Mock swarm trajectory vector
    swarm_trajectory = torch.randn(1, dim, device=device)
    swarm_trajectory = F.normalize(swarm_trajectory, p=2, dim=-1)
    
    # Crystallize motif
    target_tokens = sampler.crystallize_motif(
        swarm_trajectory=swarm_trajectory,
        sequence_length=seq_len,
        guidance_scale=4.5
    )
    
    print(f"Output tokens shape: {target_tokens.shape}")
    assert target_tokens.shape == (1, seq_len), f"Unexpected target token shape: {target_tokens.shape}"
    print(f"Output tokens sample: {target_tokens[0][:10].tolist()}...")
    print("[+] NonAutoregressiveCanvasSampler trajectory crystallization: PASS")

    print("\n=====================================================================")
    print("      [SUCCESS] ALL DIFFUSION CANVAS PIPELINE TESTS PASSED           ")
    print("=====================================================================")

if __name__ == "__main__":
    run_diffusion_canvas_verification()
