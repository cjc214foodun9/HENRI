import arc_agi
from arcengine import GameAction
from logic_puzzle_active_inference_harness import LogicPuzzleActiveInference, ThermodynamicTelemetry, ARCEgress
from arc_agi_axiom_crystallizer import ARCAxiomCrystallizer
import time

def run_benchmark():
    print("[ALETHEIA] Initializing Full ARC-AGI-3 Production Swarm...")
    
    arc = arc_agi.Arcade()
    available_envs = arc.available_environments
    print(f"[ALETHEIA] Retrieved {len(available_envs)} novel environments.")
    
    master_telemetry = ThermodynamicTelemetry(session_name="full_production_run")
    harness = LogicPuzzleActiveInference(master_telemetry)
    crystallizer = ARCAxiomCrystallizer()
    
    for idx, env_obj in enumerate(available_envs):
        game_id = env_obj.game_id if hasattr(env_obj, 'game_id') else 'unknown'
        print(f"\n[ALETHEIA] ========================================================")
        print(f"[ALETHEIA] Instantiating Environment {idx+1}/{len(available_envs)}: {game_id}")
        
        try:
            env = arc.make(game_id)
        except Exception as e:
            print(f"[ALETHEIA ERROR] Failed to instantiate {game_id}: {e}")
            continue
            
        # Epistemic Rolling Axiom: We must extract the invariant law dynamically.
        # We perturb the environment once to observe the physics.
        obs = env.reset()
        if obs is None or not hasattr(obs, 'frame') or len(obs.frame) == 0:
            print(f"[ALETHEIA] Skipping {game_id}: No valid frame data.")
            continue
            
        state_0 = obs.frame[0].tolist()
        
        # We probe the environment to extract the transformation geometry
        probe_obs = env.step(GameAction.ACTION1)
        state_1 = probe_obs.frame[0].tolist()
        
        # Isolate the boundary_axiom from the physical feedback
        try:
            boundary_axiom = crystallizer.crystallize_boundary_axiom([{"input": state_0, "output": state_1}])
        except Exception as e:
            print(f"[ALETHEIA ERROR] Crystallization Failed for {game_id}: {e}")
            continue
            
        env.reset()
        obs = env.step(GameAction.ACTION2) # Break symmetry
        
        print(f"[ALETHEIA] Axiom Crystallized. Entering Thermodynamic Inference Loop...")
        
        steps = 0
        while steps < 10: # Limit steps per environment to prevent infinite loops in terminal states
            if obs is None or not hasattr(obs, 'state'):
                print(f"[ALETHEIA WARNING] Environment crashed or returned Null observation. Abandoning attractor.")
                break
            
            if obs.state.name in ["WIN", "GAME_OVER"]:
                print(f"[ALETHEIA] Attractor Exhausted: {obs.state.name}")
                break
                
            grid_state = obs.frame[0].tolist()
            task_wave = crystallizer.encode_grid_to_wave(grid_state)
            
            resolved_policy_wave = harness.solve_puzzle(
                task_id=f"ARC_{game_id}_STEP_{steps}",
                task_wave=task_wave,
                boundary_axiom=boundary_axiom,
                max_epochs=150
            )
            
            optimal_action = ARCEgress.wave_to_action(resolved_policy_wave, crystallizer)
            print(f"[ALETHEIA Egress] {game_id} | Step {steps} | Crystallized Action: {optimal_action.name}")
            
            obs = env.step(optimal_action)
            steps += 1
            
    print("\n[ALETHEIA] ========================================================")
    print("[ALETHEIA] Full Production Run Completed.")
    print("[ALETHEIA] Master Scorecard:")
    print(arc.get_scorecard())
    
    master_telemetry.close()

if __name__ == "__main__":
    run_benchmark()
