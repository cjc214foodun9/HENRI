import torch
import traceback
from multi_expert_swarm_pre_training_engine import FreshHENRIOrchestrator
from test_time_inference_engine import DeploymentPipeline
from holographic_vector_lifter import HolographicVectorLifter
import string

def run_comprehensive_inference():
    print("=========================================================================")
    print("      PROJECT HENRI: COMPREHENSIVE VALIDATION PIPELINE                   ")
    print("=========================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Native hardware acceleration target: {device}")
    
    # Simple character-level vocab for validation
    chars = list(string.ascii_letters + string.digits + " \n\t.,;:!?()[]{}<>=")
    vocab_map = {c: i+1 for i, c in enumerate(chars)}
    vocab_map["<EOS>"] = 0
    vocab_size = len(vocab_map)
    
    # 1. Initialize Lifter
    print("[*] Initializing Holographic Vector Lifter (Epistemic Transducer)...")
    lifter = HolographicVectorLifter(vocab_size=vocab_size, dim=4096).to(device)
    
    # 2. Initialize the Core Swarm with the trained physics weights
    print("[*] Initializing 16-expert Stiefel Manifold Core...")
    orchestrator = FreshHENRIOrchestrator(vocab_size=vocab_size, dim=4096, num_experts=16)
    
    try:
        checkpoint = torch.load("henri_fresh_core.pt", map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'core' in checkpoint:
            orchestrator.core.load_state_dict(checkpoint['core'])
        else:
            orchestrator.core.load_state_dict(checkpoint)
        print(" -> Successfully loaded pure physics weights (henri_fresh_core.pt).")
    except Exception as e:
        print(f" -> [WARNING] Weights not found, initializing base entropy state: {e}")
        
    orchestrator.to(device)
    
    # 3. Construct the Deployment Pipeline (wrapping the Logit Sieve)
    print("[*] Engaging Non-Autoregressive Deployment Pipeline...")
    pipeline = DeploymentPipeline(
        core_swarm=orchestrator.core, 
        vocab_map=vocab_map,
        wcag_regex=None, # Bypass RE2 crash for testing
        dim=4096
    ).to(device)
    
    # Initialize the ADMA canonical lexicon with uniform topology for testing
    dummy_lexicon = torch.randn(vocab_size, 4096, device=device)
    dummy_lexicon = torch.nn.functional.normalize(dummy_lexicon, p=2, dim=-1)
    pipeline.canvas_sampler.egress_assembler.adma_fetch.load_zone_c_attractors(dummy_lexicon)
    
    # 4. Generate the Prompt Wave
    prompt_text = "def solve_entropy():"
    print(f"\n[INPUT] Raw Prompt: '{prompt_text}'")
    
    # Tokenize (character level)
    input_ids = [vocab_map.get(c, 0) for c in prompt_text]
    input_tensor = torch.tensor([input_ids], device=device)
    
    # Lift to continuous wave
    print("[*] Lifting tokens to continuous 4096D holographic wave...")
    with torch.no_grad():
        prompt_wave = lifter(input_tensor)
        
    # The sampler and RightKanPullback expect [Batch, Seq, Dim]. 
    # If lifter outputs [Batch, Dim], we unsqueeze.
    if prompt_wave.dim() == 2:
        prompt_wave = prompt_wave.unsqueeze(1)
        
    print(f" -> Prompt Wave Shape: {prompt_wave.shape}")
    print(f" -> Wave Modulus Intact: {torch.allclose(torch.norm(prompt_wave, dim=-1).float(), torch.ones_like(prompt_wave[..., 0]).float())}")

    # 5. Execute 25-step Euler-Maruyama Cosinespace Relaxation
    print("\n[*] Propagating Wave through Cognitive Swarm Core...")
    with torch.no_grad():
        output_string = pipeline.generate_compliant_sequence(
            initial_state=prompt_wave, 
            target_axiom=None, 
            max_len=64
        )
        
    print(f"\n[OUTPUT] Crystallized Syntax:")
    print("-------------------------------------------------------------------------")
    print(output_string)
    print("-------------------------------------------------------------------------")
    print("[*] VALIDATION PIPELINE COMPLETE.")

if __name__ == "__main__":
    run_comprehensive_inference()
