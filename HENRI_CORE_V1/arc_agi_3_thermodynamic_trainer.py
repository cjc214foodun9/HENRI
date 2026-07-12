import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import arc_agi
import json
from arcengine import GameAction
from unified_cognitive_pipeline import UnifiedCognitivePipeline
from arc_action_harness import ArcActionHarness
from viscoelastic_jepa_core import ViscoelasticOneShotLearner
from thermodynamic_melting_controller import ThermodynamicMeltingController

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=========================================================================")
    print("      PROJECT HENRI: ARC AGI 3 THERMODYNAMIC RL TRAINER                  ")
    print("=========================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Target Architecture: {device}")
    
    vocab_size = 32000
    dim = 4096
    
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
    if os.path.exists(weights_path):
        print(f"[*] Loading Production Core Weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            engine.load_state_dict(state_dict, strict=False)
            print("[*] Weights loaded successfully.")
        except Exception as e:
            print(f"[!] Warning: Failed to load weights. Error: {e}")
            
    print("[*] Initializing Canonical Target Axioms (Phase Zero)...")
    target_axioms_complex = torch.polar(
        torch.ones(1, dim, device=device),
        torch.empty(1, dim, device=device).uniform_(-3.14159, 3.14159)
    )
    target_axioms_complex = F.normalize(target_axioms_complex, p=2, dim=-1)

    # Wrap the pure physics core in AdamW
    print("[*] Initializing Thermodynamic Optimizer (AdamW)...")
    optimizer = optim.AdamW(engine.parameters(), lr=1e-4, weight_decay=1e-2)

    try:
        arc = arc_agi.Arcade()
        print("[*] ARC-AGI Toolkit successfully connected.")
    except Exception as e:
        print(f"[!] Critical Failure: Could not initialize arc_agi.Arcade(): {e}")
        return

    env_name = "ls20"
    print(f"[*] Constructing Environment: {env_name}")
    # Run without terminal rendering to speed up training
    env = arc.make(env_name)
    
    num_episodes = 5000
    max_steps_per_episode = 100
    
    print(f"\n[*] Initiating Episodic REINFORCE Training Loop ({num_episodes} Episodes)")
    
    thermostat = ThermodynamicMeltingController(
        base_threshold=0.35, 
        cooling_rate=0.08, 
        thermal_mass=1.5,
        alpha=0.4
    ).to(device)
    
    for episode in range(num_episodes):
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            obs = reset_output[0]
        else:
            obs = reset_output
            
        episode_reward = 0.0
        trajectory_deltas = []
        
        for step_idx in range(max_steps_per_episode):
            obs_str = str(obs)
            input_str = f"<|arc_state|>\n{obs_str}\n<|arc_end|>"
            tokens = tokenizer.encode(input_str, return_tensors='pt').to(device)
            
            # Forward pass WITH gradients
            clean_logits, clean_wave, telemetry = engine(tokens, target_axioms_complex)
            
            langevin_heat = torch.tensor([telemetry.get('Langevin_Heat_Integral', 0.01)], device=device)
            chosen_actions, action_probs, resonance, chosen_indices = action_harness.select_action(clean_wave, langevin_heat)
            
            action_name = chosen_actions[0]
            action_idx = chosen_indices[0].item()
            action_wavefront = action_harness.action_manifold[action_idx]
            
            # Predict future state (computational graph maintained)
            predicted_future_state = action_harness._hrr_circular_convolution(clean_wave[0], action_wavefront)
            predicted_future_state = F.normalize(predicted_future_state, p=2, dim=-1)
            
            # Compute Sagnac Delta against target axioms
            transmission, sagnac_delta = jepa_planner.sagnac_veto(predicted_future_state.unsqueeze(0), target_axioms_complex)
            
            current_sagnac = sagnac_delta.mean().item()
            dynamic_threshold, active_temp = thermostat.evaluate_and_update(
                sagnac_delta=sagnac_delta, 
                was_vetoed=True
            )
            
            if current_sagnac > dynamic_threshold:
                # Internal violation: action rejected
                telemetry['Langevin_Heat_Integral'] += current_sagnac
                # We don't append to trajectory because the physical action wasn't taken
                continue
                
            # Action passed! Cool the thermostat for the next step.
            _, _ = thermostat.evaluate_and_update(sagnac_delta=sagnac_delta, was_vetoed=False)
                
            # Physically execute validated action
            action = getattr(GameAction, action_name, GameAction.ACTION1)
            step_output = env.step(action)
            
            if isinstance(step_output, tuple):
                obs = step_output[0]
                reward = step_output[1] if len(step_output) > 1 and isinstance(step_output[1], (int, float)) else 0.0
                done = step_output[2] if len(step_output) > 2 and isinstance(step_output[2], bool) else False
                truncated = step_output[3] if len(step_output) > 3 and isinstance(step_output[3], bool) else False
            else:
                obs = step_output
                reward, done, truncated = 0.0, False, False
                
            if reward > 0.0:
                thermostat.reset_thermostat()
                
            episode_reward += reward
            trajectory_deltas.append(sagnac_delta.mean())
            
            if done or truncated:
                break
                
        # Episodic Update
        if len(trajectory_deltas) > 0:
            # Thermodynamic REINFORCE Loss
            total_delta = torch.stack(trajectory_deltas).sum()
            
            if episode_reward > 0:
                # Winning path: Burn the strategy into the core (Minimize Sagnac Delta)
                loss = total_delta * episode_reward
            else:
                # Losing path: Push the phases away from the failing trajectory (Maximize Sagnac Delta)
                loss = -total_delta * 0.1
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
            optimizer.step()
            
            print(f"[Episode {episode+1}/{num_episodes}] Steps: {len(trajectory_deltas)} | Reward: {episode_reward:.2f} | Loss: {loss.item():.4f} | Temp: {active_temp:.2f}")
        else:
            print(f"[Episode {episode+1}/{num_episodes}] Vetoed all actions. Skipping optimization. Temp: {active_temp:.2f}")
            
        # Checkpoint every 100 episodes
        if (episode + 1) % 100 == 0:
            torch.save(engine.state_dict(), weights_path)
            print(f"[*] Checkpoint saved to {weights_path}")

    print("\n=========================================================================")
    print("[*] THERMODYNAMIC RL TRAINING COMPLETE")
    print("=========================================================================")

if __name__ == "__main__":
    main()
