import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add paths to sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CRITPT_SRC_DIR = os.path.join(PROJECT_DIR, "archive", "CritPt-main", "CritPt-main", "src")
sys.path.append(PROJECT_DIR)
sys.path.append(CRITPT_SRC_DIR)

from cognitive_swarm import HenriCognitiveSwarmOrchestrator, AletheiaAgent
from critpt.data_loader import NotebookDataLoader

def main():
    print("=====================================================================")
    print("             HENRI COGNITIVE SWARM CRITPT BENCHMARK ENGINE          ")
    print("=====================================================================")
    
    # 1. Load the CritPt dataset notebook
    notebook_path = Path(PROJECT_DIR) / "archive" / "CritPt-main" / "CritPt-main" / "data" / "example_challenges" / "quantum_error_correction_main.ipynb"
    print(f"[BENCHMARK] Loading CritPt notebook: {notebook_path.name}")
    
    if not notebook_path.exists():
        print(f"[ERROR] CritPt notebook not found at: {notebook_path}")
        sys.exit(1)
        
    loader = NotebookDataLoader(notebook_path)
    print(f"[BENCHMARK] Dataset Identifier: {loader.dataset_name}")
    
    try:
        problems = loader.load_all_problems()
        print(f"[BENCHMARK] Successfully loaded {len(problems)} CritPt problems.")
    except Exception as e:
        print(f"[ERROR] Failed to load CritPt problems: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Select the main problem to run tests and inference on
    target_problem = None
    for p in problems:
        if p.problem_type == "main":
            target_problem = p
            break
            
    if not target_problem:
        target_problem = problems[0]
        
    print(f"\n--- [TARGET PROBLEM: {target_problem.problem_id}] ---")
    desc_lines = target_problem.problem_description.split("\n")
    print("\n".join(desc_lines[:10]))
    if len(desc_lines) > 10:
        print("... [TRUNCATED] ...")
        
    # 2. Initialize the Orchestrator with memory optimizations
    print("\n=====================================================================")
    print("             INITIALIZING OPTIMIZED NEURAL NETWORK SWARM             ")
    print("=====================================================================")
    
    start_init = time.perf_counter()
    orchestrator = HenriCognitiveSwarmOrchestrator(
        model_path="Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf",
        num_streams=16
    )
    init_duration = time.perf_counter() - start_init
    print(f"[BENCHMARK] Swarm orchestrator initialized in {init_duration:.2f}s.")
    
    # Report hardware configuration status
    print("\n--- Hardware Cache & RAM Optimization Report ---")
    import psutil
    process = psutil.Process(os.getpid())
    try:
        affinity = process.cpu_affinity()
        print(f"  - CPU Core Affinity: Pinned to Cores {affinity}")
    except Exception as aff_err:
        print(f"  - CPU Core Affinity Check Failed: {aff_err}")
        
    print(f"  - PyTorch Active CPU Threads: {torch.get_num_threads()}")
    
    # 3. Instantiate the Aletheia math research agent
    agent = AletheiaAgent(orchestrator)
    
    # 4. Measure latency on isolated neural network steps
    print("\n=====================================================================")
    print("            INFERENCE STEP PERFORMANCE PROFILE BENCHMARK            ")
    print("=====================================================================")
    
    test_prompt = target_problem.problem_description[:200]
    
    # Time embedding creation
    t0 = time.perf_counter()
    emb_res = orchestrator.base_model.create_embedding(test_prompt)
    emb_latency = (time.perf_counter() - t0) * 1000.0
    h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
    if len(h_7b_raw.shape) == 2:
        h_7b_raw = torch.mean(h_7b_raw, dim=0)
    
    # Time Swarm routing (L3 Router transformer pass)
    t0 = time.perf_counter()
    with torch.no_grad():
        hrr_wave = orchestrator.l3_router.activation_to_wave(h_7b_raw)
    router_latency = (time.perf_counter() - t0) * 1000.0
    
    # Time Zone B D2NN propagation
    target_vector = orchestrator.get_stream_address(0)
    target_np = target_vector.detach().numpy().astype(np.complex64)
    hrr_flat = hrr_wave.flatten()
    hrr_np = hrr_flat.detach().numpy().astype(np.complex64)
    
    t0 = time.perf_counter()
    truth_np, _, alignment = orchestrator.optical_core.forward(
        hr_wavefront=hrr_np,
        target_manifold=target_np,
        langevin_heat=0.0
    )
    d2nn_latency = (time.perf_counter() - t0) * 1000.0
    
    # Time Boundary Validation
    truth_tensor = torch.tensor(truth_np, dtype=torch.complex64)
    t0 = time.perf_counter()
    is_valid, _, error_energy, _ = orchestrator.boundary_validator.validate_boundary(truth_tensor)
    val_latency = (time.perf_counter() - t0) * 1000.0
    
    print(f"  - Gemma E4B Embedding Generation: {emb_latency:.2f} ms")
    print(f"  - L3 Cache Swarm Router Projection:  {router_latency:.2f} ms")
    print(f"  - Zone B Physical D2NN Core Feed:   {d2nn_latency:.2f} ms")
    print(f"  - Boundary Axiom Validator Veto:     {val_latency:.2f} ms")
    print(f"  - Cache Status: L3 preheat completed. Windows working set limits applied.")
    
    # 5. Run the actual agent solver reasoning loop
    print("\n=====================================================================")
    print("             EXECUTING Aletheia AGENT CLOSED COGNITIVE LOOP          ")
    print("=====================================================================")
    
    problem_prompt = (
        f"Problem: {target_problem.problem_description}\n\n"
        f"Code Template:\n{target_problem.code_template}\n\n"
        "Solve the challenge and write a python script conforming to the code template. "
        "Return the coefficients or analytical solution in the answer function."
    )
    
    print("[BENCHMARK] Initiating multi-turn generator-verifier-reviser solver loop...")
    loop_start = time.perf_counter()
    
    solution, revision_turns, status = agent.execute_reasoning_loop(
        prompt=problem_prompt,
        target_label="Optics_Resonance",
        max_revisions=2
    )
    
    loop_duration = time.perf_counter() - loop_start
    print("\n=====================================================================")
    print("                      BENCHMARK RUN SUMMARY                          ")
    print("=====================================================================")
    print(f"  - Problem Solved:     {target_problem.problem_id}")
    print(f"  - Convergence Status: {status}")
    print(f"  - Revision Turns:     {revision_turns}")
    print(f"  - Execution Latency:  {loop_duration:.2f} seconds")
    print("\nFinal Solution Generated:")
    print("---------------------------------------------------------------------")
    print(solution)
    print("---------------------------------------------------------------------")
    print("=====================================================================")

if __name__ == "__main__":
    main()
