import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from holographic_vector_lifter import HolographicVectorLifter
from cognitive_swarm_orchestrator import FreshHENRIOrchestrator
from test_time_inference_engine import DeploymentPipeline
from closed_loop_engine import ClosedLoopThermodynamicEngine

def grid_to_string(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])

def main():
    print("=========================================================================")
    print("      PROJECT HENRI: ARC AGI 2 LIVE BENCHMARK (CONTINUOUS WAVE)          ")
    print("=========================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Target Architecture: {device}")
    
    vocab_size = 32000
    dim = 4096
    
    print("[*] Initializing LLaMA Tokenizer and Holographic Vector Lifter...")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    lifter = HolographicVectorLifter(vocab_size=vocab_size, dim=dim).to(device)
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_data_dir = os.path.join(base_dir, "ARC-AGI-2-main", "data", "evaluation")
    
    if not os.path.exists(eval_data_dir):
        print(f"[!] Error: Cannot find {eval_data_dir}")
        return
        
    print("[*] Booting Continuous Wave Execution Engine (536M Physics Core)...")
    orchestrator = FreshHENRIOrchestrator(dim=dim, num_experts=16)
    
    pipeline = DeploymentPipeline(core_swarm=orchestrator.core, vocab_map=tokenizer.get_vocab(), dim=dim).to(device)
    
    print("[*] Initializing Holographic ADMA Zone C Attractors...")
    zone_c_path = os.path.join(base_dir, "zone_c_timescaledb.pt")
    try:
        pipeline.canvas_sampler.egress_assembler.adma_fetch.load_zone_c_attractors(zone_c_path)
        print("[*] ADMA Master Context Loaded successfully (including Epsilon/ARC quadrant).")
    except Exception as e:
        print(f"[!] Warning: Could not load real zone_c_timescaledb: {e}. Using simulated attractors.")
        dummy = torch.randn(1024, 4096, device=device)
        pipeline.canvas_sampler.egress_assembler.adma_fetch.canonical_lexicon = F.normalize(dummy, p=2, dim=-1)
    
    engine = ClosedLoopThermodynamicEngine(vocab_map=tokenizer.get_vocab(), max_thermal_cycles=16)
    engine.pipeline = pipeline
    engine.pipeline.tot.core.bjorck_newton_orthonormalize() # Enforce Stiefel stability before run
    
    print("\n>>> INITIATING CYBERNETIC LIVE EVALUATION LOOP <<<\n")
    
    total_tasks = 0
    successes = 0
    total_heat_consumed = 0.0
    
    eval_files = [f for f in os.listdir(eval_data_dir) if f.endswith('.json')]
    print(f"[*] Found {len(eval_files)} evaluation tasks.")
    
    for idx, fname in enumerate(eval_files):
        with open(os.path.join(eval_data_dir, fname), 'r') as f:
            task_data = json.load(f)
            
        if 'test' not in task_data or len(task_data['test']) == 0:
            continue
            
        test_input_grid = task_data['test'][0]['input']
        test_output_grid = task_data['test'][0].get('output', []) # Might not exist in hidden eval
        
        # Build prompt semantic wave
        input_str = f"<|arc_input|>\n{grid_to_string(test_input_grid)}\n<|arc_end|>"
        tokens = tokenizer.encode(input_str, add_special_tokens=False)
        tokens = [min(t, vocab_size - 1) for t in tokens]
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            phasors = lifter(token_tensor)
            bound_wavefront = torch.prod(phasors, dim=1).squeeze(0)
            initial_wavefront = F.normalize(bound_wavefront, p=2, dim=-1).unsqueeze(0).to(torch.complex64)
            
            # Since this is an evaluation, we do NOT pass a target_axiom or dynamic_target_complex.
            # The system must rely strictly on Epiplexity and the Zone C attractor routing.
            success, best_code, cycles_used, task_heat = engine.execute_viscoelastic_creep(
                initial_wavefront=initial_wavefront,
                target_grid=torch.tensor(test_output_grid) if test_output_grid else torch.zeros(1, 1),
                task_context={"strict_mode": True}
            )
            
        total_tasks += 1
        total_heat_consumed += task_heat
        if success:
            successes += 1
            
        print(f"[TASK {fname}] Status: {'RESONANCE' if success else 'COLLAPSE'} | Cycles: {cycles_used}/16 | Heat: {task_heat:.2f}")
        
        # Cap the test run to avoid waiting forever during testing
        if total_tasks >= 10:
            break
            
    print("\n=========================================================================")
    print("[*] LIVE EVALUATION COMPLETE")
    print(f"[*] Total Tasks Evaluated: {total_tasks}")
    print(f"[*] Topological Resonance Achieved: {successes} ({successes/total_tasks*100:.2f}%)")
    print(f"[*] Cumulative Langevin Heat Exerted: {total_heat_consumed:.2f}")
    print("=========================================================================")

if __name__ == "__main__":
    main()
