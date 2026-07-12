import os
import sys
import torch
import arc_agi
from arcengine import GameAction
from unified_cognitive_pipeline import UnifiedCognitivePipeline
from arc_action_harness import ArcActionHarness
from viscoelastic_jepa_core import ViscoelasticOneShotLearner
from holographic_cache_manager import HolographicCacheManager

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
    
    print("[*] Initializing Ontological Action Manifold (O-AM)...")
    action_harness = ArcActionHarness(dim=dim).to(device)
    
    print("[*] Booting Viscoelastic JEPA Core...")
    jepa_planner = ViscoelasticOneShotLearner(hidden_dim=dim, num_experts=16).to(device)
    
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
    
    print("[*] Instantiating Predictive Associative DMA Bridge to Zone C...")
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/henri")
    cache_manager = HolographicCacheManager(db_url=db_url, dim=dim, device=device)

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
    
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        obs = reset_output[0]
    else:
        obs = reset_output
    
    print(f"[*] Environment Initialized. Entering Thermodynamic Action Loop.")
    
    max_steps = 10
    for step_idx in range(max_steps):
        # 1. State Perception
        # Flatten observation into a string buffer
        obs_str = str(obs)
        input_str = f"<|arc_state|>\n{obs_str}\n<|arc_end|>"
        
        # 2. Vector Lifting
        tokens = tokenizer.encode(input_str, return_tensors='pt').to(device)
        
        # 3. Continuous Wave Processing
        with torch.no_grad():
            clean_logits, clean_wave, telemetry = engine(tokens, target_axioms_complex)
            
        # Query Zone C to retrieve the absolute best-matching structural grid rules
        target_axioms_complex = cache_manager.retrieve_resonant_boundary(
            active_thought_wave=clean_wave[0].unsqueeze(0), 
            limit=1
        )
            
        # 4. Wave-to-Action Collapse via ArcActionHarness
        # We pass the pure continuous wave and the Langevin heat (T_t) to the Sagnac Selector
        langevin_heat = torch.tensor([telemetry.get('Langevin_Heat_Integral', 0.01)], device=device)
        chosen_actions, action_probs, resonance, chosen_indices = action_harness.select_action(clean_wave, langevin_heat)
        
        action_name = chosen_actions[0]
        action_idx = chosen_indices[0].item()
        action_wavefront = action_harness.action_manifold[action_idx]
        
        # 5. Viscoelastic Planning (NextLat JEPA)
        # Simulate the future state by binding the active wave with the chosen action
        predicted_future_state = action_harness._hrr_circular_convolution(clean_wave[0], action_wavefront)
        
        # Check against target axioms via Sagnac Veto
        transmission, sagnac_delta = jepa_planner.sagnac_veto(predicted_future_state.unsqueeze(0), target_axioms_complex)
        
        print(f"[STEP {step_idx}] Heat (T_t): {langevin_heat.item():.4f} | Action Proposed: {action_name}")
        
        if sagnac_delta.mean().item() > jepa_planner.sagnac_veto.energy_threshold:
            print(f"   [JEPA] Internal violation detected. Predicted state hit boundary constraint. Action rejected.")
            # Inject localized heat and continue without physically executing
            telemetry['Langevin_Heat_Integral'] += sagnac_delta.mean().item()
            continue
            
        print(f"   [RESONANCE] Lookahead validated. Transmission Truth: {transmission.mean().item():.4f}")
        
        # 6. Execute Validated Action Physically
        action = getattr(GameAction, action_name, GameAction.ACTION1)
        step_output = env.step(action)
        
        # Robustly parse the step output depending on the custom ARC Engine version
        if isinstance(step_output, tuple):
            obs = step_output[0]
            done = step_output[2] if len(step_output) > 2 and isinstance(step_output[2], bool) else False
            truncated = step_output[3] if len(step_output) > 3 and isinstance(step_output[3], bool) else False
        else:
            obs = step_output
            done, truncated = False, False
        
        if done or truncated:
            print(f"\n[*] Episode Terminated. Final Scorecard:")
            print(arc.get_scorecard())
            break

    print(f"\n[*] Episode Terminated. Final Scorecard:")
    print(arc.get_scorecard())
    
    print("\n=========================================================================")
    print("[*] ARC-AGI-3 LIVE EVALUATION COMPLETE")
    print("=========================================================================")

if __name__ == "__main__":
    main()
