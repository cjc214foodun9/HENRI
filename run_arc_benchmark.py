import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

# Ensure Vulkan environment variables do not disable the backend
os.environ.pop("GGML_DISABLE_VULKAN", None)
os.environ.pop("GGML_VULKAN_DISABLE", None)
os.environ["GGML_OPENCL_DISABLE"] = "1"

# Add paths to sys.path: prioritizing the 6/ folder for unified cognitive engine imports
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, "6"))

from cognitive_swarm import HenriCognitiveSwarmOrchestrator

def serialize_grid(grid):
    """Translates a 2D grid of integers into a compact text layout."""
    if not isinstance(grid, list):
        return str(grid)
    if not grid or not isinstance(grid[0], list):
        return str(grid)
    return "\n".join("".join(str(cell) for cell in row) for row in grid)

def build_arc_prompt(task_dict):
    """Builds a reasoning prompt from demonstration pairs and test inputs."""
    prompt = (
        "You are an expert AI programmer and puzzle solver. Your task is to write a Python function "
        "to perform the transformation shown in the demonstration pairs. You must output exactly TWO blocks:\n\n"
        "BLOCK 1: Reasoning\n"
        "Enclose this block in <|reasoning_begin|> and <|reasoning_end|> tags. Inside, you must:\n"
        "1. Identify the background color.\n"
        "2. Briefly describe the objects and pattern (MAXIMUM 80 words, DO NOT write out row-by-row or coordinate-by-coordinate lists of grids, keep it extremely concise).\n"
        "3. Deduce the topological/geometric transformation rule.\n"
        "4. Determine the Hybrid Execution Policy.\n\n"
        "BLOCK 2: Python Code\n"
        "Enclose this block in <|python_begin|> and <|python_end|> tags. Inside, write the executable `def transform(grid: list[list[int]]) -> list[list[int]]` function.\n"
        "CRITICAL RULES:\n"
        "- KEEP THE REASONING BLOCK VERY BRIEF (MAX 80 WORDS) to leave token budget for the Python code block.\n"
        "- NumPy is strictly forbidden. You must use PyTorch (torch).\n"
        "- Absolutely NO explanations or comments outside the tags. The output must start with <|reasoning_begin|>.\n"
        "- Your program MUST print(json.dumps(answer)) as its final line.\n\n"
        "Here are the training demonstration inputs and outputs:\n\n"
    )
    
    for idx, pair in enumerate(task_dict["train"]):
        prompt += f"--- DEMONSTRATION PAIR {idx+1} ---\n"
        prompt += "Input Grid:\n"
        prompt += serialize_grid(pair["input"]) + "\n\n"
        prompt += "Output Grid:\n"
        prompt += serialize_grid(pair["output"]) + "\n\n"
        
    prompt += "--- TEST INPUT GRID ---\n"
    prompt += "Please transform this grid following the same rule:\n"
    prompt += serialize_grid(task_dict["test"][0]["input"]) + "\n\n"
    
    guidelines = ""
    return prompt, guidelines

class ARCSolverAgent:
    """
    ARC solver agent delegating execution loops to the ActiveExperimentationEngine.
    """
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def solve_task(self, task_dict, time_limit=1200) -> tuple:
        """Hand off control to ActiveExperimentationEngine for closed-loop execution."""
        print(f"[SYSTEM] Task ingested. Initializing ActiveExperimentationEngine...")
        from active_experimentation_engine import ActiveExperimentationEngine
        
        engine = ActiveExperimentationEngine(
            orchestrator=self.orchestrator
        )
        
        final_solution, revisions, status = engine.execute_task_manifold(task_dict, time_limit=time_limit)
        return final_solution, revisions, status

def main():
    print("=====================================================================")
    print("            HENRI COGNITIVE SWARM ARC-AGI-2 BENCHMARK ENGINE         ")
    print("=====================================================================")
    
    # 1. Setup Database Connection URL fallback
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        default_port = 5432 if torch.cuda.is_available() else 5433
        db_url = f"postgresql://postgres:password@127.0.0.1:{default_port}/henri"
        os.environ["DATABASE_URL"] = db_url
        print(f"[SYSTEM] DATABASE_URL not set. Defaulting to: {db_url}")
    else:
        print(f"[SYSTEM] Using configured DATABASE_URL: {db_url}")

    # 2. Load ARC Evaluation Tasks with fallback resolution
    eval_dir = Path(PROJECT_DIR) / "ARC-AGI-2" / "data" / "evaluation"
    if not eval_dir.exists():
        eval_dir = Path(PROJECT_DIR) / "archive" / "ARC-AGI-2-main" / "data" / "evaluation"
        
    if not eval_dir.exists():
        print(f"[ERROR] ARC Evaluation folder not found at: {eval_dir}")
        sys.exit(1)
        
    task_files = sorted(list(eval_dir.glob("*.json")))
    print(f"[SYSTEM] Found {len(task_files)} public evaluation tasks.")
    
    max_test_tasks = len(task_files)
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--max-tasks="):
            max_test_tasks = int(arg.split("=")[1])
        elif arg == "--max-tasks" and i + 1 < len(sys.argv):
            max_test_tasks = int(sys.argv[i + 1])
            
    tasks_to_test = task_files[:max_test_tasks]
    print(f"[SYSTEM] Selecting first {len(tasks_to_test)} tasks for benchmarking.")
    
    # 3. Initialize the Swarm Orchestrator with memory and GPU optimizations
    print("\n[SYSTEM] Booting optimized Swarm Orchestrator...")
    orchestrator = HenriCognitiveSwarmOrchestrator(
        num_streams=16
    )
    
    # Enforce explicit CUDA device validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE ACCELERATION] Binding execution substrate natively to: {device.type.upper()}")
    
    # Lift the primary orchestration components to the accelerator card
    if device.type == "cuda":
        print(f"  - Active GPU Target detected: {torch.cuda.get_device_name(0)}")
        # Ensure the core model graph is explicitly cast to high-efficiency bfloat16
        orchestrator.to(device=device, dtype=torch.bfloat16)
        # Enable PyTorch's native cuDNN autotuner to optimize the relaxation convolutions
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARNING] No CUDA engine initialized. Operating under legacy CPU thread restrictions.")
        
    agent = ARCSolverAgent(orchestrator)
    
    solved_count = 0
    total_time = 0.0
    results_summary = []
    
    dispersion_history = []
    ma_history = []
    
    # 4. Loop through tasks
    for idx, t_file in enumerate(tasks_to_test):
        print(f"\n--- [TASK {idx+1}/{len(tasks_to_test)}: {t_file.name}] ---")
        with open(t_file, "r") as f:
            task_dict = json.load(f)
            
        start_time = time.perf_counter()
        prediction, revisions, status = agent.solve_task(task_dict, time_limit=1200)
        elapsed = time.perf_counter() - start_time
        total_time += elapsed
        
        # Verify prediction on test set ground truth
        expected_test_output = task_dict["test"][0]["output"]
        is_correct = (prediction == expected_test_output)
        
        if is_correct:
             solved_count += 1
             print(f"[+] Task {t_file.name} SOLVED successfully! (Time: {elapsed:.2f}s)")
        else:
             print(f"[-] Task {t_file.name} FAILED. (Time: {elapsed:.2f}s)")
             
        results_summary.append({
            "task": t_file.name,
            "revisions": revisions,
            "status": status,
            "solved": is_correct,
            "duration": elapsed
        })
        
        # Hard Manifold Reset between tasks to prevent VRAM/RAM leakage and context pollution
        orchestrator.flush_cognitive_manifold()
        
        # Log dispersion metric
        dispersion = orchestrator.l3_router.measure_centroid_dispersion()
        print(f"[METRIC] Mean expert centroid dispersion: {dispersion:.6f}")
        dispersion_history.append(dispersion)
        
        current_ma = sum(dispersion_history[-5:]) / len(dispersion_history[-5:])
        ma_history.append(current_ma)
        
        if len(ma_history) >= 5:
            delta = ma_history[-1] - ma_history[-5]
            print(f"[METRIC] 5-task moving average of dispersion: {current_ma:.6f} (delta: {delta:+.6f})")
            if current_ma < 0.1:
                raise SystemExit("FATAL: MoE Specialization Failure. Centroids have collapsed.")
                
    # 5. Print Summary
    print("\n=====================================================================")
    print("                     ARC-AGI-2 RUN SUMMARY                           ")
    print("=====================================================================")
    print(f"  - Tasks Attempted: {len(tasks_to_test)}")
    print(f"  - Tasks Solved:    {solved_count}")
    print(f"  - Accuracy:        {(solved_count / len(tasks_to_test)) * 100:.2f}%")
    print(f"  - Total Duration:  {total_time:.2f} seconds")
    print("\nDetails:")
    for res in results_summary:
        status_str = "PASS" if res["solved"] else "FAIL"
        print(f"  [{status_str}] Task {res['task']}: Solved={res['solved']} | Revisions={res['revisions']} | Status={res['status']} | {res['duration']:.2f}s")
    print("=====================================================================")

if __name__ == "__main__":
    main()
