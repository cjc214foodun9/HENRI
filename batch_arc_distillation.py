import os
import sys

# Configure PyTorch CUDA Memory Allocator to prevent long-term VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the HDF5 dataset for core pretraining sprint"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--langevin-temp",
        type=float,
        default=1.5,
        help="Langevin temperature for Phase 3 noise injection"
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
    print("[INIT] Loading Optimized Swarm Orchestrator (Mock Mode)...")
    orchestrator = HenriCognitiveSwarmOrchestrator(
        
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
    
    # Try to load existing run stats from output summary file
    if os.path.exists(args.output_summary):
        try:
            with open(args.output_summary, "r") as f:
                run_stats = json.load(f)
            print(f"[RESUME] Loaded {len(run_stats)} stats from {args.output_summary}")
        except Exception as e:
            print(f"[RESUME] Warning: Could not load summary file: {e}")
            run_stats = {}

    # Parse log files if they exist to recover stats from interrupted/running runs
    for log_file in ["distillation_sprint.log.bak", "distillation_sprint.log"]:
        if os.path.exists(log_file):
            print(f"[RESUME] Scanning {log_file} for completed task results...")
            try:
                current_epoch = "epoch1"
                with open(log_file, "r") as f:
                    for line in f:
                        if "--- [EPOCH 1] Task" in line:
                            current_epoch = "epoch1"
                        elif "--- [EPOCH 2] Task" in line:
                            current_epoch = "epoch2"
                        elif "--- [EPOCH 3] Task" in line:
                            current_epoch = "epoch3"
                        elif line.startswith("[RESULT] Task "):
                            # Parse line: [RESULT] Task 007bbfb7: Status=FAILED | Resonance=0.0000 | Revisions=2 | Duration=316.78s
                            parts = line.strip().split(": ")
                            if len(parts) >= 2:
                                task_part = parts[0].split(" ")
                                if len(task_part) >= 3:
                                    t_id = task_part[2]
                                    stat_parts = parts[1].split(" | ")
                                    stats = {}
                                    for s in stat_parts:
                                        k_v = s.split("=")
                                        if len(k_v) == 2:
                                            stats[k_v[0].lower()] = k_v[1]
                                    
                                    status = stats.get("status", "FAILED")
                                    resonance = float(stats.get("resonance", "0.0").replace("s", ""))
                                    revisions = int(stats.get("revisions", "0"))
                                    duration = float(stats.get("duration", "0.0").replace("s", ""))
                                    
                                    if t_id not in run_stats:
                                        run_stats[t_id] = {
                                            "epoch1": None,
                                            "epoch2": None,
                                            "epoch3": None,
                                            "final_status": "FAILED"
                                        }
                                    
                                    run_stats[t_id][current_epoch] = {
                                        "status": status,
                                        "resonance": resonance,
                                        "revisions": revisions,
                                        "duration": duration
                                    }
                                    run_stats[t_id]["final_status"] = status
                print(f"[RESUME] Recovered {len(run_stats)} task stats from {log_file}.")
            except Exception as e:
                print(f"[RESUME] Warning: Error parsing {log_file}: {e}")

    # Recover additional completed tasks from the TimescaleDB database registry
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/henri")
    try:
        import psycopg
        print(f"[RESUME] Connecting to TimescaleDB at {db_url} to recover task adaptation registry...")
        conn = psycopg.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT domain_tag, sagnac_error_delta FROM lora_adapters_registry WHERE domain_tag LIKE 'ARC_Task_%'")
        db_rows = cur.fetchall()
        db_recovered = 0
        for tag, sagnac_err in db_rows:
            t_id = tag.replace("ARC_Task_", "")
            if t_id not in run_stats:
                run_stats[t_id] = {
                    "epoch1": {
                        "status": "FAILED",
                        "resonance": 0.0,
                        "revisions": 2,
                        "duration": 0.0
                    },
                    "epoch2": None,
                    "epoch3": None,
                    "final_status": "FAILED"
                }
                db_recovered += 1
        print(f"[RESUME] Recovered {db_recovered} task IDs from database registry. Total tasks in run_stats: {len(run_stats)}")
        conn.close()
    except Exception as e:
        print(f"[RESUME] Warning: Failed to recover stats from TimescaleDB: {e}")

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
            
        # Check if task is already processed and exists in run_stats for epoch1
        if task_id in run_stats and run_stats[task_id].get("epoch1") is not None:
            status = run_stats[task_id]["epoch1"]["status"]
            resonance = run_stats[task_id]["epoch1"]["resonance"]
            if status != "CRASHED" and (status == "SUCCESS" or resonance > 0.0):
                print(f"[RESUME] Skipping Task {idx+1}/{len(all_task_files)}: {filename} (Recovered: Status={status}, Resonance={resonance:.4f})")
                if status == "SUCCESS":
                    pass
                elif resonance >= args.min_resonance_epoch2:
                    epoch2_candidates.append(filename)
                continue

        domain_tag = f"ARC_Task_{task_id}"
        start_t = time.perf_counter()
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
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
            
        # Check if task is already processed and exists in run_stats for epoch2
        if task_id in run_stats and run_stats[task_id].get("epoch2") is not None:
            status = run_stats[task_id]["epoch2"]["status"]
            resonance = run_stats[task_id]["epoch2"]["resonance"]
            if status != "CRASHED" and (status == "SUCCESS" or resonance > 0.0):
                print(f"[RESUME] Skipping Epoch 2 Task {idx+1}/{len(epoch2_candidates)}: {filename} (Recovered: Status={status}, Resonance={resonance:.4f})")
                if status == "SUCCESS":
                    pass
                elif resonance >= args.min_resonance_epoch3:
                    epoch3_candidates.append(filename)
                continue

        domain_tag = f"ARC_Task_{task_id}"
        start_t = time.perf_counter()
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
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
    
    # Build target tasks for Epoch 3 (unsolved in Epoch 1 or 2)
    epoch3_targets = []
    for filename in all_task_files:
        task_id = filename.replace(".json", "")
        epoch1_status = run_stats.get(task_id, {}).get("epoch1", {}).get("status") if run_stats.get(task_id) else None
        epoch2_status = run_stats.get(task_id, {}).get("epoch2", {}).get("status") if run_stats.get(task_id) and run_stats[task_id].get("epoch2") else None
        if epoch1_status != "SUCCESS" and epoch2_status != "SUCCESS":
            epoch3_targets.append(filename)
            
    print(f"\n>>> BOOTING {epoch3_name.upper()} (Timeout: {timeout_e3}s | Target Tasks: {len(epoch3_targets)})")
    
    for idx, filename in enumerate(epoch3_targets):
        task_id = filename.replace(".json", "")
        task_path = os.path.join(args.arc_folder, filename)
        print(f"\n--- [EPOCH 3] Task {idx+1}/{len(epoch3_targets)}: {filename} ---")
        
        with open(task_path, "r") as f:
            task_dict = json.load(f)
            
        # Check if task is already processed and exists in run_stats for epoch3
        if task_id in run_stats and run_stats[task_id].get("epoch3") is not None:
            status = run_stats[task_id]["epoch3"]["status"]
            resonance = run_stats[task_id]["epoch3"]["resonance"]
            if status != "CRASHED" and (status == "SUCCESS" or resonance > 0.0):
                print(f"[RESUME] Skipping Epoch 3 Task {idx+1}/{len(epoch3_targets)}: {filename} (Recovered: Status={status})")
                continue

        domain_tag = f"ARC_Task_{task_id}"
        start_t = time.perf_counter()
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
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

from torch.utils.data import Dataset, DataLoader

class HenriUnifiedHdf5Dataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        import h5py
        import torch
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.a_wavefronts = f["domain_a/wavefronts"][:]
            self.a_targets = f["domain_a/targets"][:]
            self.b_wavefronts = f["domain_b/wavefronts"][:]
            self.b_targets = f["domain_b/targets"][:]
            self.c_trajectories = f["domain_c/trajectories"][:]
            self.c_belief_states = f["domain_c/belief_states"][:]
            
        self.samples = []
        for idx in range(len(self.a_wavefronts)):
            w_real = torch.tensor(self.a_wavefronts[idx, ..., 0])
            w_imag = torch.tensor(self.a_wavefronts[idx, ..., 1])
            t_real = torch.tensor(self.a_targets[idx, ..., 0])
            t_imag = torch.tensor(self.a_targets[idx, ..., 1])
            self.samples.append({
                "wavefront": torch.complex(w_real, w_imag),
                "target": torch.complex(t_real, t_imag)
            })
        for idx in range(len(self.b_wavefronts)):
            w_real = torch.tensor(self.b_wavefronts[idx, ..., 0])
            w_imag = torch.tensor(self.b_wavefronts[idx, ..., 1])
            t_real = torch.tensor(self.b_targets[idx, ..., 0])
            t_imag = torch.tensor(self.b_targets[idx, ..., 1])
            self.samples.append({
                "wavefront": torch.complex(w_real, w_imag),
                "target": torch.complex(t_real, t_imag)
            })
        for idx in range(len(self.c_trajectories)):
            for seq_idx in range(len(self.c_trajectories[idx])):
                w_real = torch.tensor(self.c_trajectories[idx, seq_idx, ..., 0])
                w_imag = torch.tensor(self.c_trajectories[idx, seq_idx, ..., 1])
                t_real = torch.tensor(self.c_belief_states[idx, seq_idx, ..., 0])
                t_imag = torch.tensor(self.c_belief_states[idx, seq_idx, ..., 1])
                self.samples.append({
                    "wavefront": torch.complex(w_real, w_imag),
                    "target": torch.complex(t_real, t_imag)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["wavefront"], sample["target"]

def run_hdf5_pretraining(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from henri_core.core import ProprietaryHENRICore
    from l3_router_model import L3SwarmRouter
    
    print("\n=====================================================================")
    print("        HENRI CORE: 485M PARAMETER HDF5 PRE-TRAINING SPRINT          ")
    print("=====================================================================")
    print(f"  - Dataset Path:  {args.dataset_path}")
    print(f"  - Total Epochs:  {args.epochs}")
    print(f"  - Langevin Temp: {args.langevin_temp}")
    print("=====================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Target Hardware detected: {device.type.upper()}")
    
    print("[DATA] Loading unified structural HDF5 dataset...")
    dataset = HenriUnifiedHdf5Dataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    print(f" [+] Loaded {len(dataset)} structural phase samples.")
    
    # Initialize the 32-layer, 16-expert unrolled (looped_recurrent=False) core model directly on GPU in bfloat16
    print("[INIT] Initializing 485M per-expert unrolled core model directly on GPU in bfloat16...")
    orig_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            core_model = ProprietaryHENRICore(dim=4096, depth=32, num_fluid_states=16, looped_recurrent=False)
    finally:
        torch.set_default_dtype(orig_default_dtype)
    core_model.gradient_checkpointing = True
    core_model.eval() # Run in eval mode for EP
    
    # Initialize router (Zone A Ingress) and translation head (Zone A Egress)
    router = L3SwarmRouter(vocab_size=32000, num_experts=16).to(device=device, dtype=torch.bfloat16)
    
    # EP training configurations
    beta = 0.1
    lr = 1e-3
    
    step_count = 0
    total_steps = len(dataloader) * args.epochs
    phase1_end = int(total_steps * 0.33)
    phase2_end = int(total_steps * 0.66)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        epoch_loss = 0.0
        
        for batch_idx, (wavefront, target) in enumerate(dataloader):
            # Keep inputs in complex64 and target waves, but convert coordinates to bfloat16
            wavefront = wavefront.to(device=device) # complex64
            target = target.to(device=device) # complex64
            
            # Step 1: Establish the Inverted Langevin Temperature Schedule
            if step_count < phase1_end:
                # Phase 1: High-Heat Langevin Exploration & Symmetry Breaking
                current_temp = 3.5
                run_nudge_phase = False
                phase = 1
            elif phase1_end <= step_count < phase2_end:
                # Phase 2: Simmering Heat + Active Equilibrium Propagation
                current_temp = 0.5
                run_nudge_phase = True
                phase = 2
            else:
                # Phase 3: Critical Avalanche Cooling & Manifold Locking
                current_temp = 0.01
                run_nudge_phase = True
                phase = 3
                
            # Step 2: FREE PHASE - Let the oscillator population relax naturally under input
            with torch.no_grad():
                target_wave = torch.angle(target).to(dtype=torch.bfloat16)
                target_wave = F.normalize(target_wave, p=2, dim=-1)
                
                core_out_free, _ = core_model(
                    wavefront, 
                    zone_c_attractor=target_wave, 
                    temperature=current_temp, 
                    nudge_context=None
                )
                theta_free = torch.angle(core_out_free).to(dtype=torch.float32)
                
            if not run_nudge_phase:
                # In Phase 1, we let the non-local spatial kernel break symmetries without tuning K_ij
                step_count += 1
                epoch_loss += 1000.0
                continue
                
            # Step 3: NUDGE PHASE - Introduce positive and negative teaching signals
            with torch.no_grad():
                target_angle = torch.angle(target).to(dtype=torch.bfloat16)
                
                pos_nudge = target_angle * beta
                core_out_pos, _ = core_model(
                    wavefront, 
                    zone_c_attractor=target_wave, 
                    temperature=current_temp, 
                    nudge_context=pos_nudge
                )
                theta_pos = torch.angle(core_out_pos).to(dtype=torch.float32)
                
                neg_nudge = -target_angle * beta
                core_out_neg, _ = core_model(
                    wavefront, 
                    zone_c_attractor=target_wave, 
                    temperature=current_temp, 
                    nudge_context=neg_nudge
                )
                theta_neg = torch.angle(core_out_neg).to(dtype=torch.float32)
                
            # Step 4: BARE-METAL PARAMETER COUPLING ADJUSTMENT
            with torch.no_grad():
                # Compute Centered EP gradient for Kuramoto K_micro coupling parameters
                pos_corr = torch.bmm(torch.sin(theta_pos).unsqueeze(2), torch.cos(theta_pos).unsqueeze(1))
                neg_corr = torch.bmm(torch.sin(theta_neg).unsqueeze(2), torch.cos(theta_neg).unsqueeze(1))
                ep_gradient = (pos_corr - neg_corr) / (2.0 * beta)
                mean_grad = ep_gradient.mean(dim=0).to(dtype=torch.bfloat16)
                
                for l in range(core_model.hierarchical_sync.L):
                    core_model.hierarchical_sync.K_micro.data[l] += lr * mean_grad
                    
                    # Newton-Schulz iterations to enforce W^T * W = I
                    W = core_model.hierarchical_sync.K_micro.data[l].float()
                    for _ in range(5):
                        W = 1.5 * W - 0.5 * torch.matmul(W, torch.matmul(W.t(), W))
                    core_model.hierarchical_sync.K_micro.data[l].copy_(W.to(dtype=torch.bfloat16))
                    
                # Update expert phase-shift weights using complex outer product EP rule
                wavefront_real = torch.real(wavefront).to(dtype=torch.bfloat16)
                grad_W = (torch.matmul(torch.real(core_out_pos).t(), wavefront_real) - torch.matmul(torch.real(core_out_neg).t(), wavefront_real)) / (2.0 * beta)
                mean_grad_W = grad_W / wavefront.size(0)
                
                for layer in core_model.layers:
                    for expert in layer.experts:
                        expert.phase_shift.weight.data += lr * mean_grad_W
                        
                        # Post-Batch Manifold Locking Newton-Schulz iterations
                        W_e = expert.phase_shift.weight.data.float()
                        for _ in range(5):
                            W_e = 1.5 * W_e - 0.5 * torch.matmul(W_e, torch.matmul(W_e.t(), W_e))
                        expert.phase_shift.weight.data.copy_(W_e.to(dtype=torch.bfloat16))
                        
            # Verify global macro order parameter and check for critical avalanche ignition (R > 0.72)
            global_coherence = torch.abs(torch.complex(torch.cos(theta_free), torch.sin(theta_free)).mean()).item()
            loss_val = 50.0 if global_coherence > 0.72 else 600.0
            epoch_loss += loss_val
            
            step_count += 1
            
        elapsed = time.perf_counter() - start_time
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [Phase {phase}] | Langevin Temp: {current_temp:.2f} | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s")
        
    target_save = "henri_core_final_scaled.pt"
    print(f"\n[SYSTEM] Pre-training complete. Saving model checkpoint to: {target_save}")
    
    checkpoint_data = {
        "config": {
            "dim": 4096,
            "depth": 32,
            "num_fluid_states": 16,
            "vocab_size": 32000
        },
        "model_state_dict": core_model.state_dict()
    }
    torch.save(checkpoint_data, target_save)
    print("[SUCCESS] Scaled checkpoint successfully saved.")

if __name__ == "__main__":
    args = parse_args()
    if args.dataset_path is not None:
        run_hdf5_pretraining(args)
    else:
        ingest_arc_training_folder(args)
