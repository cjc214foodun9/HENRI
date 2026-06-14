import os
import sys
import json
import time
import argparse
import traceback

# Ensure project and subdirectory paths are resolved
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, "6"))

# Force Vulkan environment variables compatibility
os.environ.pop("GGML_DISABLE_VULKAN", None)
os.environ.pop("GGML_VULKAN_DISABLE", None)
os.environ["GGML_OPENCL_DISABLE"] = "1"
os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"

from cognitive_swarm import HenriCognitiveSwarmOrchestrator
from active_experimentation_engine import ActiveExperimentationEngine

def parse_args():
    parser = argparse.ArgumentParser(description="HENRI Multi-Epoch ARC Distillation Sprint")
    parser.add_argument(
        "--arc-folder",
        type=str,
        default=os.path.join("ARC-AGI-2", "data", "training"),
        help="Path to the ARC AGI training tasks folder"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to load in Epoch 1 (useful for dry runs)"
    )
    parser.add_argument(
        "--epoch1-timeout",
        type=int,
        default=300,
        help="Timeout in seconds per task for Epoch 1 (5 minutes default)"
    )
    parser.add_argument(
        "--epoch2-timeout",
        type=int,
        default=600,
        help="Timeout in seconds per task for Epoch 2 (10 minutes default)"
    )
    parser.add_argument(
        "--epoch3-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds per task for Epoch 3 (30 minutes default)"
    )
    parser.add_argument(
        "--min-resonance-epoch2",
        type=float,
        default=0.01,
        help="Minimum training accuracy/resonance required in Epoch 1 to proceed to Epoch 2"
    )
    parser.add_argument(
        "--min-resonance-epoch3",
        type=float,
        default=0.5,
        help="Minimum training accuracy/resonance required in Epoch 2 to proceed to Epoch 3"
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=os.path.join("results", "distillation_summary.json"),
        help="Path to save the distillation run summary"
    )
    return parser.parse_args()

def run_task_safely(engine, task_dict, timeout, domain_tag):
    """Executes a single task, trapping exceptions to prevent batch crash."""
    try:
        prediction, revisions, status = engine.execute_task_manifold(
            task_dict=task_dict,
            time_limit=timeout,
            domain_tag=domain_tag
        )
        # Check resonance achieved (best sandbox training accuracy)
        resonance = getattr(engine, "best_sandbox_accuracy", 0.0)
        return status, resonance, revisions
    except Exception as e:
        print(f"[ERROR] Task failed with exception: {e}")
        traceback.print_exc()
        return "CRASHED", 0.0, 0

def ingest_arc_training_folder(args):
    print(f"\n=====================================================================")
    print(f"        HENRI CASCADING MULTI-EPOCH ARC DISTILLATION SPRINT         ")
    print(f"=====================================================================")
    print(f"  - Target Folder:       {args.arc_folder}")
    print(f"  - Max Tasks Limit:     {args.max_tasks}")
    print(f"  - Epoch 1 Timeout:     {args.epoch1-timeout if hasattr(args, 'epoch1-timeout') else args.epoch1_timeout}s")
    print(f"  - Epoch 2 Timeout:     {args.epoch2_timeout}s")
    print(f"  - Epoch 3 Timeout:     {args.epoch3_timeout}s")
    print(f"=====================================================================")

    # Initialize the cognitive swarm orchestrator
    print("[INIT] Loading Optimized Swarm Orchestrator (Gemma 4 12B in RAM)...")
    orchestrator = HenriCognitiveSwarmOrchestrator(
        model_path="Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf",
        num_streams=16
    )
    engine = ActiveExperimentationEngine(orchestrator)

    # Ingest the task files
    if not os.path.exists(args.arc_folder):
        print(f"[ERROR] Training folder not found at: {args.arc_folder}")
        sys.exit(1)

    all_task_files = sorted([f for f in os.listdir(args.arc_folder) if f.endswith(".json")])
    print(f"[INIT] Found {len(all_task_files)} total tasks in folder.")
    
    if args.max_tasks is not None:
        all_task_files = all_task_files[:args.max_tasks]
        print(f"[INIT] Limiting run to first {len(all_task_files)} tasks.")

    # Storage for historical run statistics
    # Structure: { task_id: { "epoch1": {...}, "epoch2": {...}, "epoch3": {...}, "final_status": str } }
    run_stats = {}
    
    # -------------------------------------------------------------
    # EPOCH 1: THE QUICK SPRINT
    # -------------------------------------------------------------
    epoch1_name = "Epoch 1: The Quick Sprint"
    timeout_e1 = args.epoch1_timeout
    print(f"\n>>> BOOTING {epoch1_name.upper()} (Timeout: {timeout_e1}s | Tasks: {len(all_task_files)})")
    
    epoch2_candidates = []
    
    for idx, filename in enumerate(all_task_files):
        task_id = filename.replace(".json", "")
        task_path = os.path.join(args.arc_folder, filename)
        print(f"\n--- [EPOCH 1] Task {idx+1}/{len(all_task_files)}: {filename} ---")
        
        with open(task_path, "r") as f:
            task_dict = json.load(f)
            
        start_t = time.perf_counter()
        domain_tag = f"ARC_Task_{task_id}"
        
        status, resonance, revisions = run_task_safely(engine, task_dict, timeout_e1, domain_tag)
        elapsed = time.perf_counter() - start_t
        
        print(f"[RESULT] Task {task_id}: Status={status} | Resonance={resonance:.4f} | Revisions={revisions} | Duration={elapsed:.2f}s")
        
        run_stats[task_id] = {
            "epoch1": {
                "status": status,
                "resonance": resonance,
                "revisions": revisions,
                "duration": elapsed
            },
            "epoch2": None,
            "epoch3": None,
            "final_status": status
        }
        
        # Hard memory reset and manifold flush between tasks to prevent leakage
        orchestrator.flush_cognitive_manifold()
        
        # Filter for Epoch 2: Skip solved tasks, and skip least coherent tasks (resonance == 0)
        if status == "SUCCESS":
            print(f"[+] Task {task_id} SOLVED in Epoch 1! Skipping subsequent epochs.")
        elif resonance >= args.min_resonance_epoch2:
            print(f"[+] Task {task_id} qualifies for Epoch 2 (Resonance {resonance:.4f} >= {args.min_resonance_epoch2}).")
            epoch2_candidates.append(filename)
        else:
            print(f"[-] Task {task_id} SKIPPED (Resonance {resonance:.4f} is too low/problematic).")

    # -------------------------------------------------------------
    # EPOCH 2: THE SYNTHESIS
    # -------------------------------------------------------------
    epoch2_name = "Epoch 2: The Synthesis"
    timeout_e2 = args.epoch2_timeout
    print(f"\n>>> BOOTING {epoch2_name.upper()} (Timeout: {timeout_e2}s | Target Tasks: {len(epoch2_candidates)})")
    
    epoch3_candidates = []
    
    for idx, filename in enumerate(epoch2_candidates):
        task_id = filename.replace(".json", "")
        task_path = os.path.join(args.arc_folder, filename)
        print(f"\n--- [EPOCH 2] Task {idx+1}/{len(epoch2_candidates)}: {filename} ---")
        
        with open(task_path, "r") as f:
            task_dict = json.load(f)
            
        start_t = time.perf_counter()
        domain_tag = f"ARC_Task_{task_id}"
        
        status, resonance, revisions = run_task_safely(engine, task_dict, timeout_e2, domain_tag)
        elapsed = time.perf_counter() - start_t
        
        print(f"[RESULT] Task {task_id}: Status={status} | Resonance={resonance:.4f} | Revisions={revisions} | Duration={elapsed:.2f}s")
        
        run_stats[task_id]["epoch2"] = {
            "status": status,
            "resonance": resonance,
            "revisions": revisions,
            "duration": elapsed
        }
        run_stats[task_id]["final_status"] = status
        
        orchestrator.flush_cognitive_manifold()
        
        if status == "SUCCESS":
            print(f"[+] Task {task_id} SOLVED in Epoch 2! Skipping Epoch 3.")
        elif resonance >= args.min_resonance_epoch3:
            print(f"[+] Task {task_id} qualifies for Epoch 3 (Resonance {resonance:.4f} >= {args.min_resonance_epoch3}).")
            epoch3_candidates.append(filename)
        else:
            print(f"[-] Task {task_id} SKIPPED for Epoch 3 (Resonance {resonance:.4f} is too low).")

    # -------------------------------------------------------------
    # EPOCH 3: DEEP LITERACY (UNLEASH UNLIMITED REASONING)
    # -------------------------------------------------------------
    epoch3_name = "Epoch 3: Deep Literacy (Extended Reasoning)"
    timeout_e3 = args.epoch3_timeout
    print(f"\n>>> BOOTING {epoch3_name.upper()} (Timeout: {timeout_e3}s | Target Tasks: {len(epoch3_candidates)})")
    
    for idx, filename in enumerate(epoch3_candidates):
        task_id = filename.replace(".json", "")
        task_path = os.path.join(args.arc_folder, filename)
        print(f"\n--- [EPOCH 3] Task {idx+1}/{len(epoch3_candidates)}: {filename} ---")
        
        with open(task_path, "r") as f:
            task_dict = json.load(f)
            
        start_t = time.perf_counter()
        domain_tag = f"ARC_Task_{task_id}"
        
        status, resonance, revisions = run_task_safely(engine, task_dict, timeout_e3, domain_tag)
        elapsed = time.perf_counter() - start_t
        
        print(f"[RESULT] Task {task_id}: Status={status} | Resonance={resonance:.4f} | Revisions={revisions} | Duration={elapsed:.2f}s")
        
        run_stats[task_id]["epoch3"] = {
            "status": status,
            "resonance": resonance,
            "revisions": revisions,
            "duration": elapsed
        }
        run_stats[task_id]["final_status"] = status
        
        orchestrator.flush_cognitive_manifold()

    # -------------------------------------------------------------
    # TELEMETRY & REPORT SERIALIZATION
    # -------------------------------------------------------------
    print(f"\n[TELEMETRY] Saving distillation sprint stats to {args.output_summary}...")
    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)
    with open(args.output_summary, "w") as f:
        json.dump(run_stats, f, indent=4)
        
    # Summarize solved
    solved_epoch1 = sum(1 for k, v in run_stats.items() if v["epoch1"] and v["epoch1"]["status"] == "SUCCESS")
    solved_epoch2 = sum(1 for k, v in run_stats.items() if v["epoch2"] and v["epoch2"]["status"] == "SUCCESS")
    solved_epoch3 = sum(1 for k, v in run_stats.items() if v["epoch3"] and v["epoch3"]["status"] == "SUCCESS")
    total_solved = solved_epoch1 + solved_epoch2 + solved_epoch3
    
    print("\n=====================================================================")
    print("                     DISTILLATION SPRINT SUMMARY                     ")
    print("=====================================================================")
    print(f"  - Total Tasks Loaded:  {len(all_task_files)}")
    print(f"  - Solved in Epoch 1:   {solved_epoch1}")
    print(f"  - Solved in Epoch 2:   {solved_epoch2}")
    print(f"  - Solved in Epoch 3:   {solved_epoch3}")
    print(f"  - Total Solved:        {total_solved} ({(total_solved / len(all_task_files)) * 100:.2f}%)")
    print("=====================================================================")

if __name__ == "__main__":
    args = parse_args()
    ingest_arc_training_folder(args)
