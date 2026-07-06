import torch
import traceback
from closed_loop_engine import ClosedLoopThermodynamicEngine
from test_time_inference_engine import DeploymentPipeline
from multi_expert_swarm_pre_training_engine import FreshHENRIOrchestrator

def run_real_loop():
    print("=========================================================================")
    print("      PROJECT HENRI: FULL CYBERNETIC CLOSED-LOOP ENGINE                  ")
    print("=========================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Native hardware acceleration target: {device}")
    
    # 1. Initialize the Core Swarm with the trained physics weights
    print("[*] Initializing 16-expert Stiefel Manifold Core...")
    orchestrator = FreshHENRIOrchestrator(vocab_size=32000, dim=4096, num_experts=16)
    
    try:
        # Note: henri_fresh_core.pt is not fully downloaded locally due to vast.ai ssh timeout, 
        # so we will initialize with raw physics weights if it fails.
        checkpoint = torch.load("henri_fresh_core.pt", map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'core' in checkpoint:
            orchestrator.core.load_state_dict(checkpoint['core'])
        else:
            orchestrator.core.load_state_dict(checkpoint)
        print(" -> Successfully loaded pure physics weights (henri_fresh_core.pt).")
    except Exception as e:
        print(f" -> [WARNING] Weights not found, initializing base entropy state: {e}")
        
    orchestrator.to(device)
    
    # 2. Construct the Deployment Pipeline (wrapping the Logit Sieve)
    print("[*] Engaging WCAG Logit Sieve and MimicryMasterOrchestrator...")
    pipeline = DeploymentPipeline(
        core_swarm=orchestrator.core, 
        vocab_map={"<|python_begin|>": 1, "<|python_end|>": 2}, # Stub vocab for local test
        wcag_regex=r".*", # Bypassing the lookahead crash in google-re2
        dim=4096
    ).to(device)
    
    # Try loading the aligned phase transducer weights
    try:
        pipeline.mimicry_orchestrator.phase_transducer.load_state_dict(
            torch.load("aligned_phase_transducer.pt", map_location=device, weights_only=True)
        )
        print(" -> Successfully locked HolographicPhaseTransducer to trained projection matrix.")
    except Exception as e:
        print(f" -> [WARNING] Could not load aligned_phase_transducer.pt, using default orthogonal projection: {e}")

    # 3. Construct the Closed-Loop Engine
    print("[*] Constructing Closed-Loop Engine with 16 Viscoelastic Creep cycles...")
    engine = ClosedLoopThermodynamicEngine(vocab_map={"a": 1}, max_thermal_cycles=16)
    engine.pipeline = pipeline
    engine.transducer.to(device)
    
    # 4. Generate the Environmental Boundary State (target_grid)
    # The absolute Dirichlet boundary: A highly structured 10x10 geometric grid
    target_grid = torch.ones(10, 10, dtype=torch.float32, device=device)
    target_grid[2:8, 2:8] = 0.5
    
    # Initial arbitrary wavefront representing the task prompt (must be complex64 for the holographic core)
    initial_wavefront = torch.randn(4096, dtype=torch.float32, device=device) + 1j * torch.randn(4096, dtype=torch.float32, device=device)
    
    # 5. EXECUTE THE LOOP
    print("\n>>> INITIATING CYBERNETIC LOOP <<<\n")
    try:
        success, best_code, cycles_used = engine.execute_viscoelastic_creep(
            initial_wavefront=initial_wavefront,
            target_grid=target_grid,
            task_context={"strict_mode": True}
        )
        
        print("\n=========================================================================")
        print(f"[*] CYBERNETIC LOOP TERMINATED.")
        print(f"[*] Resonance Achieved: {success}")
        print(f"[*] Cycles Consumed: {cycles_used}/16")
        print("[*] Extracted Syntactic Payload:")
        print("-------------------------------------------------------------------------")
        print(best_code)
        print("-------------------------------------------------------------------------")
    except Exception as e:
        print("\n[CATASTROPHIC TEAR] The loop failed to contain the tensor graph:")
        traceback.print_exc()

if __name__ == "__main__":
    run_real_loop()
