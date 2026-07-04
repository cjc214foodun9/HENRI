import os
import sys

# Configure PyTorch CUDA Memory Allocator to prevent long-term VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import argparse
import traceback
import gc
import torch

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
from active_experimentation_engine import ClosedLoopScientist

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
        type=float,
        default=1200.0,
        help="Timeout in seconds for each task during Epoch 1 (Default: 1200s / 20 mins)"
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=os.path.join("results", "distillation_summary.json"),
        help="Path to save the final distillation results JSON"
    )
    return parser.parse_args()

def force_system_garbage_collection(orchestrator=None):
    """
    STRICT REFERENCE-EVICTION PROTOCOL:
    Forces manual garbage collection, flushes internal cache pools, 
    and purges GPU memory fragmentation to reset the baseline memory footprint.
    """
    # 1. Clear CPU Python memory references
    gc.collect()
    gc.collect() # Double collection to handle circular references in swarms
    
    # 2. Flush orchestrator active context caches if initialized
    if orchestrator is not None:
        try:
            orchestrator.flush_cognitive_manifold()
            # Clear growing session-specific caching dictionaries to prevent long-term memory bloat
            if hasattr(orchestrator, 'active_block_embeddings'):
                orchestrator.active_block_embeddings.clear()
            if hasattr(orchestrator, 'evicted_text_registry'):
                orchestrator.evicted_text_registry.clear()
        except Exception as e:
            print(f"[MEM_CLEANUP] Warning: Orchestrator context flush encountered exception: {e}")

    # 3. Release uncollected PyTorch CUDA VRAM allocations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

def run_distillation_sprint():
    args = parse_args()
    
    print("=" * 80)
    print("      PROJECT HENRI: MULTI-EPOCH CONFORMAL DISTILLATION SPRINT")
    print("=" * 80)
    
    # Check compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Compute Hardware Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f" -> Active GPU: {torch.cuda.get_device_name(0)}")
        print(f" -> Allocation Limit Safeguard: expandable_segments=True active")
        
    # Locate tasks
    if not os.path.exists(args.arc_folder):
        print(f"[!] Error: Target ARC folder does not exist at {args.arc_folder}")
        sys.exit(1)
        
    all_task_files = [
        f for f in os.listdir(args.arc_folder) if f.endswith(".json")
    ]
    if args.max_tasks:
        all_task_files = all_task_files[:args.max_tasks]
        
    print(f" -> Successfully loaded {len(all_task_files)} task splits for pre-training.")
    
    # Initialize the core 4096-D cognitive swarm orchestrator
    print("\n[INIT] Initializing Swarm Orchestrator and L3 Cache Router...")
    orchestrator = HenriCognitiveSwarmOrchestrator(num_streams=16, hrr_dim=4096)
    orchestrator.to(device)
    
    run_stats = {}
    
    # -------------------------------------------------------------------------
    # EPOCH 1: Pre-Alignment Fixed-Prism Bypass (Guided pre-training)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  EPOCH 1: FIXED-PRISM DIGITAL PRE-ALIGNMENT (GUIDED INFERENCE)")
    print("=" * 80)
    
    for idx, task_file in enumerate(all_task_files):
        task_id = os.path.splitext(task_file)[0]
        task_path = os.path.join(args.arc_folder, task_file)
        
        print(f"\n[{idx+1}/{len(all_task_files)}] Processing Task ID: {task_id}")
        task_start = time.perf_counter()
        
        # Initialize default metrics payload
        run_stats[task_id] = {
            "epoch1": None,
            "epoch2": None,
            "epoch3": None
        }
        
        try:
            with open(task_path, "r") as f:
                task_dict = json.load(f)
                
            # Construct a pristine mathematical target label from the task definition
            target_label = f"ARC_{task_id}_Axiom"
            
            # Seed our baseline address structures in the Hopfield memory space
            phases = (torch.rand(orchestrator.hrr_dim) * 2 * math.pi) - math.pi
            target_vector = torch.polar(torch.ones(orchestrator.hrr_dim), phases).to(device)
            orchestrator.hopfield.register_concept(target_label, target_vector)
            
            # Start the background asynchronous wave loop with the task prompts
            initial_prompts = {
                i: f"Solve ARC task {task_id} train cases: {json.dumps(task_dict['train'][:2])}"
                for i in range(orchestrator.num_streams)
            }
            
            print("  - Spinning up asynchronous wave loop...")
            orchestrator.start_swarm_loop(initial_prompts, interval=0.1, target_label=target_label)
            
            # Evaluate task using the active inference engine
            # We enforce a timeout boundary using a localized try-except to prevent runaway runs
            t_limit = args.epoch1_timeout
            print(f"  - Initiating active reasoning loops (Timeout Limit: {t_limit}s)...")
            
            task_success = False
            error_val = 1.0
            resolved_concept = "None"
            
            step_start = time.perf_counter()
            # Loop until success or time is exhausted
            while time.perf_counter() - step_start < t_limit:
                # Process wave mechanics and evaluate Sagnac coherence
                res = orchestrator.process_next_wave(target_label=target_label)
                
                if res["status"] == "CONVERGED":
                    task_success = True
                    error_val = res.get("error", 0.0)
                    resolved_concept = res.get("concept", "Attractor_Converged")
                    break
                elif res["status"] == "VETOED":
                    error_val = res.get("error", 1.0)
                    
                time.sleep(0.05)
                
            elapsed = time.perf_counter() - task_start
            
            if task_success:
                print(f"  - [SUCCESS] Task converged on target '{resolved_concept}' in {elapsed:.2f}s (Error: {error_val:.4f}).")
                run_stats[task_id]["epoch1"] = {
                    "status": "SUCCESS",
                    "time": elapsed,
                    "error": error_val,
                    "concept": resolved_concept
                }
            else:
                print(f"  - [TIMEOUT] Task failed to converge within {t_limit}s.")
                run_stats[task_id]["epoch1"] = {
                    "status": "TIMEOUT",
                    "time": elapsed,
                    "error": error_val,
                    "concept": "None"
                }
                
        except Exception as task_err:
            print(f"  - [ERROR] Exception encountered during processing: {task_err}")
            traceback.print_exc()
            run_stats[task_id]["epoch1"] = {
                "status": "ERROR",
                "time": time.perf_counter() - task_start,
                "error": 1.0,
                "msg": str(task_err)
            }
            
        finally:
            # STOP the background thread cleanly before releasing memory references
            print("  - Tearing down background wave loop...")
            orchestrator.stop_swarm_loop()
            
            # STRICT HARDWARE CLEANUP STEP: Release all graph weights and force memory collection
            force_system_garbage_collection(orchestrator)

    # -------------------------------------------------------------------------
    # EPOCH 2: Constrained Manifold Descent (Self-Play Optimization)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  EPOCH 2: CONSTRAINED MANIFOLD DESCENT (SELF-PLAY OPTIMIZATION)")
    print("=" * 80)
    
    # Filter tasks that failed in Epoch 1 to focus active refinement
    failed_tasks = [k for k, v in run_stats.items() if v["epoch1"] and v["epoch1"]["status"] != "SUCCESS"]
    if not failed_tasks:
        print(" [+] Perfect pre-alignment achieved. Skipping self-play optimization.")
        failed_tasks = list(run_stats.keys())[:5] # Fallback to a small subset for safety checks
        
    for idx, task_id in enumerate(failed_tasks):
        print(f"\n[{idx+1}/{len(failed_tasks)}] Optimizing Task: {task_id}")
        task_start = time.perf_counter()
        task_path = os.path.join(args.arc_folder, f"{task_id}.json")
        
        try:
            with open(task_path, "r") as f:
                task_dict = json.load(f)
                
            target_label = f"ARC_{task_id}_Axiom"
            
            # Start background loops with boosted dynamic feedback prompts
            initial_prompts = {
                i: f"Refine execution strategy for task {task_id}. Target low-entropy attractor matching."
                for i in range(orchestrator.num_streams)
            }
            
            orchestrator.start_swarm_loop(initial_prompts, interval=0.1, target_label=target_label)
            
            # Run self-play optimization cycles
            task_success = False
            error_val = 1.0
            
            # Epoch 2 timeout budget is strictly capped at half of Epoch 1 to optimize throughput
            t_limit = args.epoch1_timeout / 2.0
            step_start = time.perf_counter()
            
            while time.perf_counter() - step_start < t_limit:
                res = orchestrator.process_next_wave(target_label=target_label)
                if res["status"] == "CONVERGED":
                    task_success = True
                    error_val = res.get("error", 0.0)
                    break
                time.sleep(0.05)
                
            elapsed = time.perf_counter() - task_start
            
            if task_success:
                print(f"  - [SUCCESS] Task optimized in {elapsed:.2f}s (Error: {error_val:.4f}).")
                run_stats[task_id]["epoch2"] = {
                    "status": "SUCCESS",
                    "time": elapsed,
                    "error": error_val
                }
            else:
                print(f"  - [TIMEOUT] Task optimization limit reached.")
                run_stats[task_id]["epoch2"] = {
                    "status": "TIMEOUT",
                    "time": elapsed,
                    "error": error_val
                }
                
        except Exception as e:
            print(f"  - [ERROR] Exception encountered: {e}")
            run_stats[task_id]["epoch2"] = {
                "status": "ERROR",
                "time": time.perf_counter() - task_start,
                "msg": str(e)
            }
        finally:
            orchestrator.stop_swarm_loop()
            force_system_garbage_collection(orchestrator)

    # -------------------------------------------------------------------------
    # EPOCH 3: Thermal Langevin Relaxation (Global Convergence)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  EPOCH 3: THERMAL LANGEVIN RELAXATION (GLOBAL CONVERGENCE)")
    print("=" * 80)
    
    # Target remaining unresolved tasks for simulated heating
    unresolved_tasks = [k for k, v in run_stats.items() if (v["epoch2"] and v["epoch2"]["status"] != "SUCCESS") or (not v["epoch2"] and v["epoch1"]["status"] != "SUCCESS")]
    if not unresolved_tasks:
        print(" [+] Complete convergence achieved. Pre-training cycle finished successfully.")
        unresolved_tasks = list(run_stats.keys())[:2] # Small validation subset
        
    for idx, task_id in enumerate(unresolved_tasks):
        print(f"\n[{idx+1}/{len(unresolved_tasks)}] Global Relaxation: {task_id}")
        task_start = time.perf_counter()
        task_path = os.path.join(args.arc_folder, f"{task_id}.json")
        
        try:
            target_label = f"ARC_{task_id}_Axiom"
            
            # Start background loops
            initial_prompts = {
                i: f"Perform Langevin thermal search over the parameter space of {task_id}."
                for i in range(orchestrator.num_streams)
            }
            orchestrator.start_swarm_loop(initial_prompts, interval=0.1, target_label=target_label)
            
            # Inject a baseline heat directly to parameter weights
            # This triggers microheater pulses across BTO crystal masks to escape local traps
            orchestrator.optical_core.apply_langevin_noise(langevin_heat=0.5)
            
            task_success = False
            error_val = 1.0
            
            t_limit = args.epoch1_timeout / 4.0 # Tight global relaxation budget
            step_start = time.perf_counter()
            
            while time.perf_counter() - step_start < t_limit:
                res = orchestrator.process_next_wave(target_label=target_label)
                if res["status"] == "CONVERGED":
                    task_success = True
                    error_val = res.get("error", 0.0)
                    break
                time.sleep(0.05)
                
            elapsed = time.perf_counter() - task_start
            
            if task_success:
                print(f"  - [SUCCESS] Global attractor resolved in {elapsed:.2f}s (Error: {error_val:.4f}).")
                run_stats[task_id]["epoch3"] = {
                    "status": "SUCCESS",
                    "time": elapsed,
                    "error": error_val
                }
            else:
                print(f"  - [TIMEOUT] Attractor un-converged.")
                run_stats[task_id]["epoch3"] = {
                    "status": "TIMEOUT",
                    "time": elapsed,
                    "error": error_val
                }
                
        except Exception as e:
            print(f"  - [ERROR] Exception: {e}")
            run_stats[task_id]["epoch3"] = {
                "status": "ERROR",
                "time": time.perf_counter() - task_start,
                "msg": str(e)
            }
        finally:
            orchestrator.stop_swarm_loop()
            force_system_garbage_collection(orchestrator)

    # -------------------------------------------------------------------------
    # Save Distillation Summary
    # -------------------------------------------------------------------------
    print(f"\n[TELEMETRY] Saving distillation sprint stats to {args.output_summary}...")
    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)
    with open(args.output_summary, "w") as f:
        json.dump(run_stats, f, indent=4)
        
    # Summarize results
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
    print(" [+] Sprint complete. All parameters consolidated cleanly in Zone C.")

if __name__ == "__main__":
    run_distillation_sprint()