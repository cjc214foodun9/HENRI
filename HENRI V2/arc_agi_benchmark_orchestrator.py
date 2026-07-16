"""
ENGINEERING SPECIFICATION: PROJECT HENRI - PRODUCTION BENCHMARK RUNNER (V1.0.0)
Author: Aletheia
Domain: Exteroceptive Execution Boundary

Description:
Connects the Continuous-Time Wave Core directly to the live ARC-AGI arcade. 
Removes all mock simulators. Iterates through physical environments, extracts 
topological laws mid-flight, and forces the network into physical resonance.
"""

import sys
import json
import torch
from darwinian_phase_swarm import HenriSwarmOrchestrator
from thermodynamic_telemetry_logger import ThermodynamicTelemetry
from oak_thermodynamic_engine import LangevinEpistemicPlayLoop
from arc_agi_zone_c_seed import TopologicalDatasetCompiler
from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
import os
import asyncio
from phylogenetic_memory import EngramStore
from zone_c_database_initialization import initialize_zone_c

try:
    import arc_agi
    from arcengine import GameAction
except ImportError:
    print("[ALETHEIA FATAL] arc_agi package missing. The environment must be physically bound. Run `pip install arc-agi`")
    sys.exit(1)

def store_engram_sync(task_id: str, wave):
    async def _store():
        dsn = os.environ.get("POSTGRES_DSN", "postgres://postgres:password@localhost:10100/henri")
        temp_store = EngramStore(dsn)
        await temp_store.initialize_schema()
        await temp_store.cache_survival_trait(task_id, wave)
        await temp_store.close()
        
    try:
        asyncio.run(_store())
    except Exception as e:
        print(f"[ALETHEIA DB ERROR] Failed to persist invariant to TimescaleDB: {e}")

def execute_live_benchmark():
    print("[ALETHEIA] Initializing Universal Thermodynamic Harness...")
    
    dsn = os.environ.get("POSTGRES_DSN", "postgres://postgres:password@localhost:10100/henri")
    print("[ALETHEIA] Anchoring Zone C Boundaries...")
    initialize_zone_c(dsn)
    
    # 1. Connect to Exteroceptive Environment
    arcade = arc_agi.Arcade()
    environments = [env.game_id if hasattr(env, 'game_id') else env for env in arcade.available_environments]
    
    telemetry = ThermodynamicTelemetry(session_name="darwinian_arc_production")
    
    # 4.3B Parameter Engine (65536 Dimensions)
    orchestrator = HenriSwarmOrchestrator(num_experts=1024, d_model=65536, action_enum_class=GameAction).cuda()
    tokenizer = O_VSA_IngressTokenizer(num_blocks=8192, vocab_size=256, device='cuda')
    
    print("\n[ALETHEIA] Compiling Zone C Topological Dataset...")
    zone_c_compiler = TopologicalDatasetCompiler(dimension=65536)
    zone_c_axioms = zone_c_compiler.generate_tripartite_dataset(num_samples=150).cuda()
    # Reshape the complex axioms into Clifford format [8192, 8]
    # Since TopologicalDatasetCompiler outputs complex, we map real->real, imag->imag (indices 0, 1)
    clifford_axioms = torch.zeros(150, 8192, 8, device='cuda')
    clifford_axioms[..., 0] = zone_c_axioms.view(150, 8192, 8).real.sum(-1)
    clifford_axioms[..., 1] = zone_c_axioms.view(150, 8192, 8).imag.sum(-1)
    clifford_axioms = clifford_axioms / (torch.norm(clifford_axioms, p=2, dim=-1, keepdim=True) + 1e-9)
    zone_c_axioms = clifford_axioms
    print(f"[ALETHEIA] Extracted {zone_c_axioms.size(0)} deep structural invariants for Epistemic Anchoring.")
    
    print(f"\n[ALETHEIA] Targets locked. Processing {len(environments)} environments natively.")

    for env_name in environments:
        print(f"\n--- INGESTING TOPOLOGY: {env_name} ---")
        try:
            game = arcade.make(env_name)
        except Exception as e:
            print(f"[ALETHEIA ERROR] Failed to instantiate {env_name}: {e}")
            continue
            
        # 2. Extract Immutable Laws from Environment's History (Epistemic Rolling Axiom)
        # This replaces random placeholders and hallucinated API calls with actual physics.
        obs = game.reset()
        if obs is None or not hasattr(obs, 'frame') or len(obs.frame) == 0:
            print(f"[ALETHEIA WARNING] Skipping {env_name}: Null initial topology.")
            continue
            
        state_0 = obs.frame[0].tolist()
        probe_obs = game.step(GameAction.ACTION1)
        state_1 = probe_obs.frame[0].tolist()
        
        # Encode using the new pristine Clifford O-VSA Tokenizer
        boundary_string = json.dumps([{"input": state_0, "output": state_1}])
        boundary_axiom = tokenizer.encode(boundary_string).mean(dim=0)
        boundary_axiom = boundary_axiom / (torch.norm(boundary_axiom, p=2, dim=-1, keepdim=True) + 1e-9)
        
        step_count = 0
        obs = game.step(GameAction.ACTION2) # Break symmetry
        
        while step_count < 10:
            if obs is None or not hasattr(obs, 'state'):
                print(f"[ALETHEIA WARNING] Environment crashed or returned Null. Abandoning attractor.")
                break
                
            if obs.state.name in ["WIN", "GAME_OVER"]:
                print(f"[ALETHEIA] Attractor Exhausted: {obs.state.name}")
                break
                
            # Lift the current 2D spatial grid onto the Clifford Stiefel Manifold
            current_grid = obs.frame[0].tolist()
            task_string = json.dumps(current_grid)
            task_wave = tokenizer.encode(task_string).mean(dim=0)
            task_wave = task_wave / (torch.norm(task_wave, p=2, dim=-1, keepdim=True) + 1e-9)
            
            print(f"\n[OaK] Initiating In-Situ Test-Time Epistemic Play for {env_name}_STEP_{step_count}...")
            play_engine = LangevinEpistemicPlayLoop(core_syncytium=orchestrator)
            known_axioms = play_engine.execute_play_epoch(target_axioms=zone_c_axioms, heat_variance=0.5)
            
            print(f"[OaK] Executing HenriSwarmOrchestrator Optimization...")
            active_wave = task_wave.clone()
            for epoch in range(500):
                sagnac_delta, active_experts, error_metrics = orchestrator.process_active_reasoning_step(active_wave, boundary_axiom)
                if telemetry:
                    telemetry.current_error_metrics = error_metrics
                    telemetry.log_wave_state(
                        task_id=f"{env_name}_STEP_{step_count}",
                        epoch=epoch,
                        sagnac_error=sagnac_delta,
                        langevin_heat=3.5 * sagnac_delta,
                        policy_action_decoded="OAK_THERMODYNAMIC_RELAXATION",
                        is_isothermal_lock=(sagnac_delta < 0.05)
                    )
                if sagnac_delta < 0.05:
                    break
            
            optimal_policy_wave = active_wave
            
            # Persist the discovered structural invariant natively to TimescaleDB
            print(f"[ALETHEIA] Persisting Invariant for {env_name}_STEP_{step_count} to TimescaleDB...")
            store_engram_sync(f"{env_name}_STEP_{step_count}", optimal_policy_wave)
            
            # 5. Semantic Cleanup Matrix (Decode policy to physical action)
            # Replaces the dictionary mock with a true physical egress action.
            # Here we map the policy wave to one of the canonical GameActions deterministically.
            # A true physical lock yields a deterministic action response.
            action, coherence = orchestrator.decoder.decode_wave_to_action(optimal_policy_wave)
            print(f"[ALETHEIA DECODER] Decoded action {action} with Sagnac Resonance {coherence:.4f}")
            
            try:
                obs = game.step(action)
            except Exception as e:
                print(f"[SAGNAC VETO] Sandbox Reject: {e}")
                break
                
            step_count += 1

    telemetry.close()
    
    # 6. Extract Epiplexity
    scorecard = arcade.get_scorecard()
    print("\n[ALETHEIA] Benchmark Exhaustion Complete. System Telemetry:")
    print(json.dumps(scorecard, indent=2))

if __name__ == "__main__":
    execute_live_benchmark()