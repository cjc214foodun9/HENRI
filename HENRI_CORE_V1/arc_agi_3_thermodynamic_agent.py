import os
import sys
import torch
import arc_agi
from arcengine import GameAction
from unified_cognitive_pipeline import UnifiedCognitivePipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=========================================================================")
    print("      PROJECT HENRI: ARC AGI 3 LIVE AGENT (CONTINUOUS WAVE)              ")
    print("=========================================================================")
    
    # Using CUDA (GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Target Architecture: {device}")
    
    vocab_size = 32000
    dim = 4096
    
    # Mocking Tokenizer for pure coordinate/action translations
    class MockTokenizer:
        def encode(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
            if return_tensors == 'pt':
                return torch.tensor([[1] + [ord(c) % vocab_size for c in text]])
            return [1] + [ord(c) % vocab_size for c in text]
    tokenizer = MockTokenizer()
    
    print("[*] Booting Unified Continuous Wave Execution Engine...")
    engine = UnifiedCognitivePipeline(vocab_size=vocab_size, dim=dim, spatial_resolution=64).to(device)
    
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "henri_fresh_core.pt")
    
    if not os.path.exists(weights_path):
        print(f"[*] Production weights not found at {weights_path}. Downloading 7GB core from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="Chandler/HENRI8.6Bswarm", 
                filename="henri_fresh_core.pt", 
                local_dir=os.path.dirname(os.path.abspath(__file__)), 
                token="HFAKUz2BkSyewRWxJX4KZaLy9rs5J5w"
            )
        except Exception as e:
            print(f"[!] Warning: Failed to download weights. Error: {e}")
            
    if os.path.exists(weights_path):
        print(f"[*] Loading Production Core Weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            # If the file contains only the core or specific modules, we load strictly
            engine.load_state_dict(state_dict, strict=False)
            print("[*] Weights loaded successfully.")
        except Exception as e:
            print(f"[!] Warning: Failed to load weights. Error: {e}")
    else:
        print(f"[!] WARNING: Production weights still not found. Running with uninitialized physics core.")
    
    
    print("[*] Initializing Canonical Target Axioms (Phase Zero)...")
    target_axioms_complex = torch.polar(
        torch.ones(1, dim, device=device),
        torch.empty(1, dim, device=device).uniform_(-3.14159, 3.14159)
    )

    print("\n>>> INITIATING ARC-AGI-3 CYBERNETIC ARCADE <<<")
    try:
        arc = arc_agi.Arcade()
        print("[*] ARC-AGI Toolkit successfully connected.")
    except Exception as e:
        print(f"[!] Critical Failure: Could not initialize arc_agi.Arcade(). Is arc-agi installed? {e}")
        return

    # Using the ls20 game as the primary testing environment
    env_name = "ls20"
    print(f"[*] Constructing Environment: {env_name}")
    env = arc.make(env_name, render_mode="terminal")
    
    obs, info = env.reset()
    
    print(f"[*] Environment Initialized. Entering Thermodynamic Action Loop.")
    
    max_steps = 100
    for step_idx in range(max_steps):
        # 1. State Perception
        # Flatten observation into a string buffer
        obs_str = str(obs)
        input_str = f"<|arc_state|>\n{obs_str}\n<|arc_end|>"
        
        # 2. Vector Lifting
        tokens = tokenizer.encode(input_str, return_tensors='pt').to(device)
        
        # 3. Continuous Wave Processing
        with torch.no_grad():
            clean_logits, telemetry = engine(tokens, target_axioms_complex)
            
        # 4. Wave-to-Action Collapse
        # The logits correspond to vocab_size. We map the strongest resonance to the GameAction space.
        action_space = list(GameAction)
        predicted_token_id = torch.argmax(clean_logits[0, -1, :]).item()
        
        action_idx = predicted_token_id % len(action_space)
        action = action_space[action_idx]
        
        # 5. Execute Action
        print(f"[STEP {step_idx}] Heat: {telemetry.get('Langevin_Heat_Integral', 0.0):.2f} | Reflection: {telemetry.get('Sagnac_Reflection_Energy', 1.0):.3f} | Action Triggered: {action.name}")
        
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            print(f"\n[*] Episode Terminated. Final Scorecard:")
            print(arc.get_scorecard())
            break

    print("\n=========================================================================")
    print("[*] ARC-AGI-3 LIVE EVALUATION COMPLETE")
    print("=========================================================================")

if __name__ == "__main__":
    main()
