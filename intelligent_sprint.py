import os
import sys
import time
import json
import math
import argparse
import traceback
import queue
import threading
import uuid
import torch
import numpy as np
from pathlib import Path

# Force console encoding for Unicode compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure imports can be resolved
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, "6"))

from cognitive_swarm import HenriCognitiveSwarmOrchestrator
from active_experimentation_engine import ActiveExperimentationEngine
from wosx_boundary_verifier import WosxBoundaryVerifier
from henri_contract import complex_to_db, DIMS
import psycopg

# =====================================================================
# SOLUTION 1: ASYNCHRONOUS VRAM LORA PREFETCHER QUEUE
# =====================================================================
class VramLoraPrefetcher:
    """
    Asynchronously reads ahead in the task sequence, loads LoRA weights
    from disk to CPU memory, and triggers non-blocking VRAM stream copies.
    """
    def __init__(self, task_ids_list, lora_dir="archive/domain_adapters", device="cuda"):
        self.task_ids = task_ids_list
        self.lora_dir = Path(lora_dir)
        self.device = device
        self.prefetched_data = {}  # task_id -> {"lora_A": Tensor, "lora_B": Tensor}
        self.lock = threading.Lock()
        
        self.prefetch_queue = queue.Queue()
        for tid in task_ids_list:
            self.prefetch_queue.put(tid)
            
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"[PREFETCHER] Async VRAM Prefetcher active for {len(task_ids_list)} tasks.")

    def _worker_loop(self):
        # Instantiate a dedicated CUDA stream for async weight transfers
        if torch.cuda.is_available() and self.device == "cuda":
            stream = torch.cuda.Stream()
        else:
            stream = None

        while not self.prefetch_queue.empty():
            task_id = self.prefetch_queue.get()
            domain_tag = f"ARC_Task_{task_id}"
            adapter_path = self.lora_dir / f"{domain_tag}.bin"
            
            if adapter_path.exists():
                try:
                    # 1. Load weights sequentially to CPU
                    state = torch.load(str(adapter_path), map_location="cpu")
                    loaded_A = state.get("lora_A")
                    loaded_B = state.get("lora_B")
                    
                    if loaded_A is not None and loaded_B is not None:
                        # 2. Stage non-blocking copy onto accelerator device
                        if stream is not None:
                            with torch.cuda.stream(stream):
                                staged_A = loaded_A.cuda(non_blocking=True)
                                staged_B = loaded_B.cuda(non_blocking=True)
                        else:
                            staged_A = loaded_A.to(self.device)
                            staged_B = loaded_B.to(self.device)
                            
                        with self.lock:
                            self.prefetched_data[task_id] = {
                                "lora_A": staged_A,
                                "lora_B": staged_B
                            }
                        # print(f"[PREFETCHER] Prefetched VRAM LoRA weights for {task_id}")
                except Exception as e:
                    print(f"[PREFETCHER] Warning: Failed to prefetch weights for task {task_id}: {e}")
            self.prefetch_queue.task_done()

    def get_prefetched_weights(self, task_id):
        with self.lock:
            return self.prefetched_data.pop(task_id, None)


# =====================================================================
# SOLUTION 3: ASYNCHRONOUS THREADED WRITE-BUFFER FOR ZONE C
# =====================================================================
class DatabaseWriteBuffer:
    """
    Thread-safe write-buffer queue to batch vector insertions and 
    decouple TimescaleDB writes from the GPU active inference loop.
    """
    def __init__(self, db_url):
        self.db_url = db_url
        self.write_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[DB WRITE BUFFER] Threaded database writer active.")

    def _worker_loop(self):
        while True:
            item = self.write_queue.get()
            if item is None:
                break
                
            batch = [item]
            # Accumulate up to 50 items if immediately available
            while len(batch) < 50:
                try:
                    next_item = self.write_queue.get_nowait()
                    if next_item is None:
                        self.write_queue.put(None)
                        break
                    batch.append(next_item)
                except queue.Empty:
                    break
                    
            try:
                with psycopg.connect(self.db_url, connect_timeout=5) as conn:
                    with conn.cursor() as cur:
                        conn.autocommit = True
                        for q_type, q_args in batch:
                            if q_type == "lexicon":
                                concept_hash, semantic_label, domain_tag, vector_str, raw_text = q_args
                                cur.execute(
                                    """
                                    INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront, raw_text)
                                    VALUES (%s, %s, %s, %s::vector, %s)
                                    ON CONFLICT (concept_hash) DO UPDATE SET
                                        hrr_wavefront = EXCLUDED.hrr_wavefront,
                                        raw_text = EXCLUDED.raw_text,
                                        last_verified = NOW();
                                    """,
                                    (concept_hash, semantic_label, domain_tag, vector_str, raw_text),
                                )
            except Exception as e:
                print(f"[DB WRITE BUFFER] Warning: Failed to insert batch of size {len(batch)} to database: {e}")
                
            for _ in range(len(batch)):
                self.write_queue.task_done()

    def enqueue_vector(self, semantic_label, domain_tag, wave, raw_text=None):
        concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, semantic_label))
        if torch.is_tensor(wave):
            wave_np = wave.detach().cpu().numpy()
        else:
            wave_np = wave
        
        try:
            vector_str = complex_to_db(wave_np, DIMS.hrr_dim)
            self.write_queue.put(("lexicon", (concept_hash, semantic_label, domain_tag, vector_str, raw_text)))
        except Exception as e:
            print(f"[DB WRITE BUFFER] Serialization error for {semantic_label}: {e}")

    def close(self):
        self.write_queue.put(None)
        self.worker_thread.join()
        print("[DB WRITE BUFFER] Database write buffer closed and flushed.")


# =====================================================================
# SPRINT ACTIVE EXPERIMENTATION ENGINE WITH ACTIONABLE ENGINEERING PATCHES
# =====================================================================
class SprintActiveExperimentationEngine(ActiveExperimentationEngine):
    def __init__(self, orchestrator, prefetcher, write_buffer, verifier, db_url, *args, **kwargs):
        super().__init__(orchestrator, *args, **kwargs)
        self.prefetcher = prefetcher
        self.write_buffer = write_buffer
        self.verifier = verifier
        self.db_url = db_url

    def get_db_count_for_tag(self, domain_tag):
        try:
            with psycopg.connect(self.db_url, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM hrr_canonical_lexicon WHERE domain_tag = %s;", (domain_tag,))
                    return cur.fetchone()[0]
        except Exception:
            return 0

    def distill_and_queue_expert_axiom(self, expert_idx: int, domain_tag: str):
        manager = self.orchestrator.lora_managers[expert_idx]
        task_id = domain_tag.replace("ARC_Task_", "") if domain_tag else ""
        with torch.no_grad():
            impulse = torch.ones(self.orchestrator.gemma_dim, device=manager.lora_A.device, dtype=manager.lora_A.dtype)
            lora_state_3840 = manager.apply_lora(impulse)
            device = self.orchestrator.l3_router.w_down.weight.device
            target_dtype = self.orchestrator.l3_router.w_down.weight.dtype
            
            try:
                wave_4096 = torch.matmul(lora_state_3840.to(device=device, dtype=target_dtype), self.orchestrator.l3_router.w_down.weight.T)
            except Exception:
                wave_4096 = torch.matmul(lora_state_3840.to(device=device, dtype=target_dtype), self.orchestrator.l3_router.w_down.weight)

            wave_4096_fp32 = wave_4096.to(dtype=torch.float32)
            phases = torch.angle(torch.complex(wave_4096_fp32, torch.zeros_like(wave_4096_fp32)))
            wave_4096_complex = torch.polar(torch.ones_like(phases, dtype=torch.float32), phases)
            
            axiom_label = f"boundary_axiom_{task_id}_expert_{expert_idx}"
            self.write_buffer.enqueue_vector(
                semantic_label=axiom_label,
                domain_tag=domain_tag,
                wave=wave_4096_complex,
                raw_text=f"Distilled boundary axiom for expert {expert_idx}"
            )
            print(f"[ENGINE] Enqueued distilled expert boundary axiom: {axiom_label}")

    def execute_task_manifold(self, task_dict, time_limit=1200, domain_tag="ARC_Task"):
        """
        Coordinates the Parallel Evolutionary Wave-Search Architecture with prefetch swaps,
        negative Hopfield interaction repellers, and universal carrier waves.
        """
        import time
        import uuid
        start_time = time.time()
        task_dict["domain_tag"] = domain_tag
        
        # SOLUTION 1: Dynamic LoRA routing from Prefetched VRAM
        task_id = domain_tag.replace("ARC_Task_", "") if domain_tag else ""
        weights = self.prefetcher.get_prefetched_weights(task_id)
        if weights is not None:
            print(f"[ENGINE] Near-instantaneous pointer assignment for LoRA adapter '{domain_tag}' from Prefetched VRAM...")
            for idx, manager in self.orchestrator.lora_managers.items():
                manager.lora_A.data.copy_(weights["lora_A"])
                manager.lora_B.data.copy_(weights["lora_B"])
        else:
            if domain_tag:
                print(f"[ENGINE] Prefetch cache miss. Pre-loading semantic LoRA adapter for domain tag '{domain_tag}' from disk/DB...")
                for idx, manager in self.orchestrator.lora_managers.items():
                    self.orchestrator.synaptic_manager.route_and_load_adapter(domain_tag, manager)
                
        playbook_dict = self.orchestrator.initialize_empty_playbook()
        
        from run_arc_benchmark import build_arc_prompt
        task_prompt, _ = build_arc_prompt(task_dict)
        
        self.error_history = []
        self.superposition_wave = None
        self.best_sandbox_accuracy = 0.0
        
        revision_step = 0
        while True:
            revision_step += 1
            print(f"\n--- [ACE TURN {revision_step}] ---")
            
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                print(f"[ENGINE] Time limit of {time_limit}s reached (Elapsed: {elapsed:.2f}s). Stopping revision loop.")
                break
                
            stuck = self.is_stuck_in_basin()
            temp = 1.35 if stuck else 0.70
            inject_noise = stuck
            
            # Compile current Playbook constraints into an axiomatic wave
            playbook_wave = self.orchestrator.compile_playbook_to_wave(playbook_dict)
            
            # SOLUTION 4: Cold-Start 30% Universal Syntax Carrier Wave at Sprint Boot
            db_count = self.get_db_count_for_tag(domain_tag)
            
            try:
                syntax_anchor_wave = self.project_code_to_wave("def transform(input_grid):\n")
            except Exception as e:
                print(f"[ENGINE] Warning: Failed to project syntax anchor wave: {e}")
                syntax_anchor_wave = None
                
            if syntax_anchor_wave is not None:
                if playbook_wave is None:
                    # Cold start: compile carrier wave directly to establish basic Python syntax
                    dev = next(self.orchestrator.l3_router.parameters()).device
                    syntax_anchor_wave = syntax_anchor_wave.to(device=dev)
                    playbook_wave = torch.nn.functional.normalize(syntax_anchor_wave, p=2, dim=-1)
                    print("[ENGINE] Cold-Start: Injected 100% Universal Syntax Carrier Wave as base playbook wave.")
                elif db_count == 0:
                    # Pre-polarize the first turn turns with 30% carrier wave
                    dev = playbook_wave.device
                    dtype = playbook_wave.dtype
                    syntax_anchor_wave = syntax_anchor_wave.to(device=dev, dtype=dtype)
                    blended_wave = (0.30 * syntax_anchor_wave) + (0.70 * playbook_wave)
                    playbook_wave = torch.nn.functional.normalize(blended_wave, p=2, dim=-1)
                    print("[ENGINE] Pre-polarization: Injected 30% Universal Syntax Carrier Wave.")
                else:
                    # Normal blending with target syntax carrier wave
                    if domain_tag.startswith("ARC_Task_"):
                        dev = playbook_wave.device
                        dtype = playbook_wave.dtype
                        syntax_anchor_wave = syntax_anchor_wave.to(device=dev, dtype=dtype)
                        blended_wave = (0.30 * syntax_anchor_wave) + (0.70 * playbook_wave)
                        playbook_wave = torch.nn.functional.normalize(blended_wave, p=2, dim=-1)

            self.playbook_wave = playbook_wave
            
            # Blend superposition wave from elite candidates of the last turn
            if self.superposition_wave is not None:
                if playbook_wave is not None:
                    superposition = playbook_wave.flatten() + self.superposition_wave.flatten()
                    mags = torch.abs(superposition).clamp(min=1e-8)
                    playbook_wave = (superposition / mags).reshape(playbook_wave.shape)
                else:
                    playbook_wave = self.superposition_wave
            
            remaining = time_limit - (time.time() - start_time)
            if remaining > 800:
                num_candidates = 16
            elif remaining > 400:
                num_candidates = 8
            elif remaining > 150:
                num_candidates = 4
            else:
                num_candidates = 1
                
            print(f"[ENGINE] Remaining time: {remaining:.2f}s. Generating {num_candidates} parallel hypotheses...")
            
            # --- MICRO-EPOCH EVALUATION & APOPTOSIS ---
            zone_c_attractors = self.fetch_active_playbook_waves()
            zone_c_repellers = self.fetch_forbidden_topology_waves()
            
            failure_tracker = {}
            def micro_epoch_eval(generated_tokens_list, alpha_routing, candidate_idx):
                if candidate_idx not in failure_tracker:
                    failure_tracker[candidate_idx] = 0
                    
                token_tensor = torch.tensor(generated_tokens_list, dtype=torch.long, device='cpu')
                partial_wave = self.orchestrator.l3_router.text_to_wave(token_tensor)
                
                fitness_scores = self.entropic_engine.evaluate_entropic_fitness(
                    partial_wave.unsqueeze(0), 
                    zone_c_attractors, 
                    zone_c_repellers
                )
                fitness = fitness_scores[0].item()
                
                if candidate_idx < len(self.orchestrator.lora_managers):
                    manager = self.orchestrator.lora_managers[candidate_idx]
                else:
                    return False
                
                if fitness < self.entropic_engine.survival_threshold:
                    self.entropic_engine.apply_viscoelastic_apoptosis(manager, fitness)
                    failure_tracker[candidate_idx] += 1
                    if failure_tracker[candidate_idx] >= 3:
                        return True
                else:
                    failure_tracker[candidate_idx] = 0
                return False

            candidate_batch = self.orchestrator.swarm_fabric.generate_parallel_hypotheses(
                task_dict=task_dict,
                playbook_wave=playbook_wave,
                playbook_dict=playbook_dict,
                temperature=temp,
                inject_noise=inject_noise,
                num_candidates=num_candidates,
                early_stopping_callback=micro_epoch_eval,
                start_time=start_time,
                time_limit=time_limit
            )
            
            scored_candidates = []
            for candidate_info in candidate_batch:
                candidate, alpha_routing = candidate_info
                is_syntax_valid, pure_code_or_err = self.inductor.verify_syntax(candidate)
                if not is_syntax_valid:
                    # Syntax error: Project and queue as repeller
                    try:
                        c_wave = self.project_code_to_wave(candidate)
                        code_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, candidate))[:8]
                        repeller_label = f"repeller_syntax_{task_id}_rev{revision_step}_{code_hash}"
                        self.write_buffer.enqueue_vector(
                            semantic_label=repeller_label,
                            domain_tag=domain_tag,
                            wave=c_wave,
                            raw_text=candidate
                        )
                        self.orchestrator.hopfield.register_repeller(c_wave)
                        print(f"[ENGINE] Registered and queued syntax repeller: {repeller_label}")
                    except Exception as e:
                        print(f"[ENGINE] Warning: Failed to queue syntax repeller: {e}")
                    scored_candidates.append((candidate, None, 0.0, None, 999.0, pure_code_or_err, alpha_routing, None))
                    continue
                    
                pure_code = pure_code_or_err
                
                is_generalized, guard_feedback = self.inductor.assert_generalization(pure_code)
                if not is_generalized:
                    # Overfitting lookup table: Project and queue as repeller
                    try:
                        c_wave = self.project_code_to_wave(pure_code)
                        code_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, pure_code))[:8]
                        repeller_label = f"repeller_overfit_{task_id}_rev{revision_step}_{code_hash}"
                        self.write_buffer.enqueue_vector(
                            semantic_label=repeller_label,
                            domain_tag=domain_tag,
                            wave=c_wave,
                            raw_text=pure_code
                        )
                        self.orchestrator.hopfield.register_repeller(c_wave)
                        print(f"[ENGINE] Registered and queued overfit repeller: {repeller_label}")
                    except Exception as e:
                        print(f"[ENGINE] Warning: Failed to queue overfit repeller: {e}")
                    scored_candidates.append((candidate, None, 0.0, None, 999.0, guard_feedback, alpha_routing, None))
                    continue
                
                wave_valid, physical_feedback, error_energy, truth_tensor, delta_np = self.emulator.evaluate_wavefront(pure_code, target_label="SCADA_Pressure_Control")
                
                passed_cases, total_cases, execution_feedback = self.evaluate_candidate(pure_code, task_dict["train"])
                partial_score = passed_cases / total_cases

                # Project candidate to wave and compute resonance/coherence
                try:
                    c_wave = self.project_code_to_wave(pure_code)
                    res_dict = self.verifier.verify_hypothesis_wave(c_wave)
                    res_score = res_dict["resonance_score"]
                    
                    code_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, pure_code))[:8]
                    
                    # SOLUTION 3: Deploy Asynchronous Threaded Write-Buffer for Attractors and Repellers
                    if partial_score == 0.0:
                        # SOLUTION 2: Configure the Repeller as a Negative Hopfield Energy Interaction Term
                        self.orchestrator.hopfield.register_repeller(c_wave)
                        
                        repeller_label = f"repeller_{task_id}_rev{revision_step}_{code_hash}"
                        self.write_buffer.enqueue_vector(
                            semantic_label=repeller_label,
                            domain_tag=domain_tag,
                            wave=c_wave,
                            raw_text=pure_code
                        )
                        print(f"[ENGINE] Queued repeller: {repeller_label} (Resonance: {res_score:.4f})")
                    else:
                        attractor_label = f"attractor_{task_id}_rev{revision_step}_{code_hash}"
                        self.write_buffer.enqueue_vector(
                            semantic_label=attractor_label,
                            domain_tag=domain_tag,
                            wave=c_wave,
                            raw_text=pure_code
                        )
                        print(f"[ENGINE] Queued attractor: {attractor_label} (Accuracy: {passed_cases}/{total_cases}, Resonance: {res_score:.4f})")
                except Exception as proj_err:
                    print(f"[ENGINE] Warning: Projecting and enqueuing candidate wave failed: {proj_err}")
                
                if passed_cases == total_cases:
                    print(f"[ENGINE] Perfect inductive generalization achieved! Executing test case...")
                    test_score, test_pred = self.run_repl_sandbox(pure_code, task_dict["test"], test_mode=True)
                    
                    if test_score == 1.0 and test_pred is not None:
                        winner_idx = self.orchestrator.l3_router.update_expert_centroids(truth_tensor)
                        print(f"[ANCHOR] Success wave drifted expert centroid {winner_idx} permanently.")
                        self.orchestrator.save_router_centroids()
                        self.orchestrator.lora_managers[winner_idx].save_weights()
                        
                        # Queue boundary expert distilled axiom
                        self.distill_and_queue_expert_axiom(winner_idx, domain_tag)
                        
                        self.orchestrator.flush_lora_and_context_to_db(domain_tag=domain_tag)
                        return test_pred, revision_step, "SUCCESS"
                    else:
                        print("[ENGINE] Test input execution failed.")
                        scored_candidates.append((candidate, pure_code, 0.0, truth_tensor, error_energy, "Compiled successfully on train, but failed to execute on test input", alpha_routing, delta_np))
                else:
                    scored_candidates.append((candidate, pure_code, partial_score, truth_tensor, error_energy, execution_feedback, alpha_routing, delta_np))
                    
            turn_best_score = max([c[2] for c in scored_candidates]) if scored_candidates else 0.0
            if turn_best_score > self.best_sandbox_accuracy:
                self.best_sandbox_accuracy = turn_best_score

            valid_candidates = [c for c in scored_candidates if c[3] is not None]
            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[4])
                winner_candidate, winner_pure, winner_score, winner_truth_tensor, winner_error_energy, winner_feedback, winner_alpha_routing, winner_delta_np = valid_candidates[0]
                
                # --- ENTROPIC SURVIVAL ENGINE ---
                zone_c_attractors = self.fetch_active_playbook_waves()
                zone_c_repellers = self.fetch_forbidden_topology_waves()
                
                expert_waves = []
                for candidate_info in candidate_batch:
                    c_wave = self.project_code_to_wave(candidate_info[0])
                    expert_waves.append(c_wave)
                expert_waves = torch.stack(expert_waves)
                
                orthogonal_residual = winner_delta_np if winner_delta_np is not None else None
                
                fitness_scores = self.entropic_engine.evaluate_entropic_fitness(
                    expert_waves, 
                    zone_c_attractors, 
                    zone_c_repellers
                )
                
                self.entropic_engine.execute_survival_creep(
                    self.orchestrator.lora_managers, 
                    fitness_scores, 
                    orthogonal_residual
                )
                
                print(f"[ENGINE] Declaring physical winner of turn {revision_step} (Error Energy: {winner_error_energy:.4f})")
                winner_idx = self.orchestrator.l3_router.update_expert_centroids(winner_truth_tensor)
                self.orchestrator.save_router_centroids()
                self.orchestrator.lora_managers[winner_idx].save_weights()
                
                valid_candidates.sort(key=lambda x: (-x[2], x[4]))
                best_candidate, best_pure, best_score, _, _, best_feedback, _, _ = valid_candidates[0]
            else:
                best_candidate, best_score, best_feedback = scored_candidates[0][0], 0.0, scored_candidates[0][5]
                
            self.error_history.append(best_feedback)
            
            top_candidates = [c for c in scored_candidates if c[2] > 0 and c[3] is not None][:2]
            if len(top_candidates) >= 2:
                wave_1 = self.project_code_to_wave(top_candidates[0][0])
                wave_2 = self.project_code_to_wave(top_candidates[1][0])
                superposition = wave_1 + wave_2
                mags = torch.abs(superposition).clamp(min=1e-8)
                self.superposition_wave = superposition / mags
            elif len(top_candidates) == 1:
                self.superposition_wave = self.project_code_to_wave(top_candidates[0][0])
            else:
                self.superposition_wave = None
                
            if "[CRITICAL] Hardcoding" in best_feedback:
                insight = "[CRITICAL] Hardcoding training grid values is forbidden. You must write a generalized functional transformation matrix."
            else:
                insight = self.orchestrator.reflect_on_failure(task_prompt, best_candidate, best_feedback)
                
            playbook_dict = self.orchestrator.curate_playbook(playbook_dict, insight)
            
        # Distill expert axiom on failure
        if len(valid_candidates) > 0:
            winner_candidate, winner_pure, winner_score, winner_truth_tensor, winner_error_energy, winner_feedback, winner_alpha_routing, winner_delta_np = valid_candidates[0]
            try:
                winner_idx = self.orchestrator.l3_router.update_expert_centroids(winner_truth_tensor)
                self.distill_and_queue_expert_axiom(winner_idx, domain_tag)
            except Exception as e:
                print(f"[ENGINE] Warning: Failed to distill expert axiom on failure: {e}")
                
        self.orchestrator.flush_lora_and_context_to_db(domain_tag=domain_tag)
        return None, revision_step, "FAILED"


# =====================================================================
# SPRINT RUNNER MAIN ROUTINE
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="HENRI Intelligent 6-Hour Training Sprint")
    parser.add_argument("--hours", type=float, default=6.0, help="Sprint total duration in hours (default: 6.0)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Maximum number of tasks to process (default: None)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Target device (default: cuda)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    
    # 1. Resolve Dataset Directories
    possible_dirs = [
        script_dir / "archive" / "ARC-AGI-2-main" / "data",
        script_dir / "ARC-AGI-2" / "data",
        Path("/workspace/HENRI/ARC-AGI-2/data"),
        Path("c:/Users/chan/Desktop/HENRI 7B SWARM/HENRI/archive/ARC-AGI-2-main/data")
    ]
    
    data_dir = None
    for pd in possible_dirs:
        if (pd / "training.txt").exists() and (pd / "training").is_dir():
            data_dir = pd
            break
            
    if data_dir is None:
        print("[ERROR] Could not resolve ARC-AGI-2 data directory!")
        sys.exit(1)
        
    print(f"[INIT] Resolved data directory to: {data_dir}")
    
    # Collect training and evaluation task paths
    all_tasks = []
    
    # Load from training list
    training_txt = data_dir / "training.txt"
    with open(training_txt, "r") as f:
        train_ids = [line.strip() for line in f if line.strip()]
    for tid in train_ids:
        jpath = data_dir / "training" / f"{tid}.json"
        if jpath.exists():
            all_tasks.append((tid, jpath, "training"))
            
    # Load from evaluation list
    evaluation_txt = data_dir / "evaluation.txt"
    with open(evaluation_txt, "r") as f:
        eval_ids = [line.strip() for line in f if line.strip()]
    for tid in eval_ids:
        jpath = data_dir / "evaluation" / f"{tid}.json"
        if jpath.exists():
            all_tasks.append((tid, jpath, "evaluation"))
            
    if args.max_tasks is not None:
        all_tasks = all_tasks[:args.max_tasks]
        print(f"[INIT] Capping execution run to the first {len(all_tasks)} tasks.")
        
    if not all_tasks:
        print("[ERROR] Task list is empty. Exiting.")
        sys.exit(1)
        
    # 2. Boot Cognitive Swarm Orchestrator
    print("\n[INIT] Loading Swarm Orchestrator (16 expert streams)...")
    orchestrator = HenriCognitiveSwarmOrchestrator(num_streams=16)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[INIT] Target Accelerator: {device.type.upper()}")
    
    if device.type == "cuda":
        orchestrator.to(device=device, dtype=torch.bfloat16)
        torch.backends.cudnn.benchmark = True
        
    # 3. Instantiate Prefetcher, Write-Buffer, and Boundary Verifier
    task_ids = [t[0] for t in all_tasks]
    lora_dir = script_dir / "archive" / "domain_adapters"
    prefetcher = VramLoraPrefetcher(task_ids, lora_dir=lora_dir, device=device.type)
    
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/henri")
    write_buffer = DatabaseWriteBuffer(db_url)
    
    verifier = WosxBoundaryVerifier(db_url=db_url)
    
    # 4. Instantiate Custom Sprint Active Experimentation Engine
    engine = SprintActiveExperimentationEngine(
        orchestrator=orchestrator,
        prefetcher=prefetcher,
        write_buffer=write_buffer,
        verifier=verifier,
        db_url=db_url
    )
    
    # 5. Start main sprint execution loop
    start_time = time.time()
    total_budget_seconds = args.hours * 3600
    
    completed_count = 0
    solved_count = 0
    total_tasks = len(all_tasks)
    
    results = []
    
    print(f"\n=========================================================")
    print(f"        LAUNCHING INTELLIGENT TRAINING SPRINT            ")
    print(f"=========================================================")
    print(f"  - Total Tasks:    {total_tasks}")
    print(f"  - Total Budget:   {args.hours:.2f} hours ({total_budget_seconds:.1f} seconds)")
    print(f"  - Target Device:  {device}")
    print(f"=========================================================\n")
    
    for idx, (task_id, json_path, dataset_type) in enumerate(all_tasks):
        elapsed = time.time() - start_time
        if elapsed >= total_budget_seconds:
            print(f"\n[SPRINT] Time budget of {args.hours} hours exhausted. Ending training sprint.")
            break
            
        remaining_time = total_budget_seconds - elapsed
        remaining_tasks = total_tasks - completed_count
        
        # Calculate dynamic task timeout based on remaining tasks and time
        task_timeout = remaining_time / max(1, remaining_tasks)
        # Clamp between 30 seconds and 300 seconds
        task_timeout = max(30.0, min(300.0, task_timeout))
        
        print(f"\n>>> [TASK {idx+1}/{total_tasks}] ID: {task_id} ({dataset_type.upper()})")
        print(f"    Elapsed: {elapsed/3600:.3f}h / {args.hours:.2f}h | Remaining Time: {remaining_time:.1f}s | Task Timeout: {task_timeout:.1f}s")
        
        with open(json_path, "r") as f:
            task_dict = json.load(f)
            
        domain_tag = f"ARC_Task_{task_id}"
        task_start = time.perf_counter()
        
        # Clear repellers in Hopfield memory for a clean slate
        orchestrator.hopfield.clear_repellers()
        
        try:
            prediction, revisions, status = engine.execute_task_manifold(
                task_dict=task_dict,
                time_limit=task_timeout,
                domain_tag=domain_tag
            )
        except Exception as task_err:
            print(f"[ERROR] Task {task_id} failed with exception: {task_err}")
            traceback.print_exc()
            prediction, revisions, status = None, 0, "CRASHED"
            
        task_duration = time.perf_counter() - task_start
        completed_count += 1
        
        # Check correctness against test set ground truth
        is_correct = False
        if prediction is not None:
            try:
                expected = task_dict["test"][0]["output"]
                is_correct = (prediction == expected)
            except Exception:
                is_correct = False
                
        if is_correct:
            solved_count += 1
            print(f"[+] Task {task_id} SOLVED! Status={status} | Duration={task_duration:.2f}s")
        else:
            print(f"[-] Task {task_id} FAILED. Status={status} | Duration={task_duration:.2f}s")
            
        results.append({
            "task_id": task_id,
            "dataset": dataset_type,
            "solved": is_correct,
            "status": status,
            "revisions": revisions,
            "duration": task_duration
        })
        
        # Hard Manifold Reset between tasks to wipe VRAM KV-Cache
        orchestrator.flush_cognitive_manifold()
        
    print("\n[SYSTEM] Completed task processing loop. Flushing database write queue...")
    write_buffer.close()
    
    sprint_duration = time.time() - start_time
    print("\n=========================================================")
    print("             TRAINING SPRINT COMPLETED                   ")
    print("=================================================")
    print(f"  - Total Elapsed:   {sprint_duration/3600:.3f} hours ({sprint_duration:.2f}s)")
    print(f"  - Tasks Attempted:  {completed_count}/{total_tasks}")
    print(f"  - Tasks Solved:     {solved_count}")
    if completed_count > 0:
        print(f"  - Success Rate:     {(solved_count / completed_count) * 100:.2f}%")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
