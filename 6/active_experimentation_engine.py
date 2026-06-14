import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import collections
from typing import List, Callable, Dict, Any

class ExperimentDesigner(torch_nn.Module):
    """
    Identifies parameter regions where competing mathematical theories disagree.
    It uses gradient ascent on the input space to maximize the prediction variance
    across an ensemble of hypotheses, effectively targeting the topological 
    boundaries of the network's current knowledge.
    """
    def __init__(self, state_dim: int, lr: float = 0.05, optimization_steps: int = 50):
        super().__init__()
        self.state_dim = state_dim
        self.lr = lr
        self.optimization_steps = optimization_steps

    def design_optimal_experiment(self, hypothesis_ensemble: List[Callable], seed_state: torch.Tensor) -> torch.Tensor:
        """
        Takes a seed state and perturbs it until the hypotheses disagree the most.
        Returns the optimal 'x' (experimental parameters) to test in reality.
        """
        # We optimize the input state, NOT the network weights.
        with torch.enable_grad():
            experimental_x = seed_state.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([experimental_x], lr=self.lr)

            for _ in range(self.optimization_steps):
                optimizer.zero_grad()
                
                predictions = []
                for hypothesis in hypothesis_ensemble:
                    predictions.append(hypothesis(experimental_x))
                stacked_preds = torch.stack(predictions) # (Num_Theories, Batch, State_Dim)
                
                # 1. Physics-Aware Epistemic Uncertainty (Phase Disagreement)
                # Calculate how much the theories disagree on the wave vector directions
                mean_pred = stacked_preds.mean(dim=0, keepdim=True)
                # Cosine distance between each theory and the mean ensemble prediction
                cos_sim = F.cosine_similarity(stacked_preds, mean_pred.expand_as(stacked_preds), dim=-1)
                # We want to MAXIMIZE disagreement, which means MINIMIZING cosine similarity
                disagreement = -cos_sim.mean() 
                
                # 2. Hardware Boundary Regularization (Soft Clamp)
                # Penalize the optimizer if it tries to push the experimental parameters 
                # beyond the safe normalized bounds [-1, 1].
                boundary_penalty = torch.relu(torch.abs(experimental_x) - 1.0).sum() * 10.0
                
                # Loss to minimize: -disagreement + penalty
                loss = disagreement + boundary_penalty 
                loss.backward()
                optimizer.step()
                
            # Apply a hard physical clamp at the very end just to be perfectly safe
            return torch.clamp(experimental_x.detach(), min=-1.0, max=1.0)


class PhysicalSubstrateInterface:
    """
    Mock interface representing the physical boundary. 
    In deployment, this sends telemetry out to the Barium Titanate waveguides, 
    triggers the physical wave mechanics, and records the emergent topological state.
    """
    def execute(self, experimental_params: torch.Tensor) -> torch.Tensor:
        # Simulate an unknown physical ground truth (e.g., a non-linear wave interaction)
        # The agent does not have access to this equation.
        noise = torch.randn_like(experimental_params) * 0.05
        return torch.sin(experimental_params * 2.0) + torch.tanh(experimental_params) + noise


class ClosedLoopScientist(torch_nn.Module):
    """
    The orchestrator of Phase 4. 
    Couples hypothesis generation (Phase 3) with hypothesis-conditioned 
    experiment selection, actively probing the environment to resolve uncertainty.
    """
    def __init__(self, program_inductor, state_dim: int, ensemble_size: int = 5, memory_size: int = 1000):
        super().__init__()
        self.inductor = program_inductor # Instance of ProgramInductor from Phase 3
        self.designer = ExperimentDesigner(state_dim)
        self.physical_world = PhysicalSubstrateInterface()
        self.ensemble_size = ensemble_size
        
        self.active_theories = [] 
        
        # Use a bounded deque to prevent OOM and forget outdated thermodynamic states
        self.empirical_observations = collections.deque(maxlen=memory_size)
        
        # ModuleList to register active theory modules for serialization & device casting
        self.active_theory_modules = torch_nn.ModuleList()

    def _sync_theory_modules(self):
        """Registers all local primitive modules from active ASTs to a ModuleList for serialization/device casts."""
        modules = []
        for theory in self.active_theories:
            modules.extend(self._collect_ast_modules(theory['ast']))
        self.active_theory_modules = torch_nn.ModuleList(modules)

    def _collect_ast_modules(self, ast: Dict[str, Any]) -> List[torch_nn.Module]:
        modules = []
        if ast["type"] == "branch":
            modules.append(ast["branch_logic"])
            modules.append(ast["condition_primitive"])
            modules.extend(self._collect_ast_modules(ast["true_tree"]))
            modules.extend(self._collect_ast_modules(ast["false_tree"]))
        elif ast["type"] == "primitive":
            modules.append(ast["op"])
            for inp in ast["inputs"]:
                modules.extend(self._collect_ast_modules(inp))
        return modules

    def bootstrap_hypotheses(self, x_init: torch.Tensor, y_init: torch.Tensor):
        """Generates the initial competing theories based on early data."""
        self.active_theories = []
        for _ in range(self.ensemble_size):
            ast, loss = self.inductor.fit_program_to_data(x_init, y_init, num_architectures=3)
            # Create a callable wrapper for the AST
            callable_theory = lambda x, ast=ast: self.inductor.execute_program(ast, x)
            self.active_theories.append({'ast': ast, 'callable': callable_theory, 'score': loss})
        self._sync_theory_modules()

    def run_discovery_cycle(self, seed_state: torch.Tensor):
        """
        The core active inference loop.
        """
        # 1. DESIGN: Find where theories disagree most
        callables = [t['callable'] for t in self.active_theories]
        optimal_x = self.designer.design_optimal_experiment(callables, seed_state)
        
        # 2. EXECUTE: Query the physical substrate
        true_y = self.physical_world.execute(optimal_x)
        self.empirical_observations.append((optimal_x, true_y))
        
        # 3. VERIFY & UPDATE: Cull weak theories, mutate strong ones
        # Re-evaluate all theories on the new, highly-informative data point
        for theory in self.active_theories:
            pred_y = theory['callable'](optimal_x)
            error = F.mse_loss(pred_y, true_y).item()
            # Exponential moving average to update the theory's trust score
            theory['score'] = (0.7 * theory['score']) + (0.3 * error)
            
        # Sort theories by how well they survived the physical experiment
        self.active_theories.sort(key=lambda t: t['score'])
        
        # 4. ITERATE: Discard the worst theory and synthesize a new one using RECENT data
        # Grab up to the last 50 highly-informative experiments to train the new theory
        recent_obs = list(self.empirical_observations)[-50:]
        best_x_history = torch.cat([obs[0] for obs in recent_obs])
        best_y_history = torch.cat([obs[1] for obs in recent_obs])
        
        # Induce a new program to replace the weakest link
        new_ast, new_loss = self.inductor.fit_program_to_data(best_x_history, best_y_history, num_architectures=5)
        new_callable = lambda x, ast=new_ast: self.inductor.execute_program(ast, x)
        
        # Replace the worst performing theory (Darwinian selection on the logic space)
        self.active_theories[-1] = {'ast': new_ast, 'callable': new_callable, 'score': new_loss}
        self._sync_theory_modules()
        
        return optimal_x, true_y, self.active_theories[0] # Return the resulting best theory


class ActiveExperimentationEngine:
    def __init__(self, orchestrator, max_revisions=3):
        self.orchestrator = orchestrator  # Points to the base model wrapper
        self.max_revisions = max_revisions
        
        from neurosymbolic_program_induction import ProgramInductor
        from zone_b_emulator import ZoneBPhysicalEmulator
        from dynamic_lora import DynamicLoRAEngine
        from entropic_survival_engine import EntropicSurvivalEngine
        
        self.inductor = ProgramInductor(state_dim=128)
        self.emulator = ZoneBPhysicalEmulator(self.orchestrator)
        self.lora_engine = DynamicLoRAEngine()
        self.entropic_engine = EntropicSurvivalEngine(num_experts=self.orchestrator.num_streams)
        
        # Track errors in history for stagnation/basin escape
        self.error_history = []
        self.superposition_wave = None
        self.playbook_wave = None

    def fetch_active_playbook_waves(self):
        if hasattr(self, 'playbook_wave') and self.playbook_wave is not None:
            return self.playbook_wave.unsqueeze(0).to(torch.complex64)
        return torch.zeros((1, 4096), dtype=torch.complex64)

    def fetch_forbidden_topology_waves(self):
        if not hasattr(self, 'error_history') or not self.error_history:
            return None
        repellers = []
        for err in self.error_history[-5:]:
            repellers.append(self.project_code_to_wave(err))
        if len(repellers) > 0:
            return torch.stack(repellers).to(torch.complex64)
        return None

    def is_stuck_in_basin(self) -> bool:
        """Checks if consecutive turns produce the exact same error footprint."""
        if len(self.error_history) < 2:
            return False
        return self.error_history[-1] == self.error_history[-2]

    def sync_vulkan_kv_cache_slot(self, src_idx: int, dest_idx: int):
        """
        Overwrite the Vulkan KV-cache for this specific sequence to match the leader.
        """
        if not hasattr(self.orchestrator, "gen_model") or self.orchestrator.gen_model is None or self.orchestrator.is_mock:
            return
        try:
            import llama_cpp
            import llama_cpp.llama_cpp as lcpp
            ctx = self.orchestrator.gen_model.llama.ctx
            # Remove dest sequence
            lcpp.llama_memory_seq_rm(ctx, dest_idx, 0, -1)
            # Copy src sequence to dest sequence
            lcpp.llama_memory_seq_cp(ctx, src_idx, dest_idx, 0, -1)
            print(f"[KV CACHE SWAP] Successfully synced Vulkan KV-cache sequence slot {src_idx} -> {dest_idx}")
        except Exception as e:
            print(f"[KV CACHE SWAP] Warning: KV-cache sync failed: {e}")

    def project_code_to_wave(self, candidate_code: str) -> torch.Tensor:
        """Projects candidate code to 4096-D complex wave state."""
        emb_res = self.orchestrator.base_model.create_embedding(candidate_code)
        h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
        h_7b_lora = self.orchestrator.lora_managers[0].apply_lora(h_7b_raw)
        if len(h_7b_lora.shape) == 2:
            h_7b_lora = torch.mean(h_7b_lora, dim=0)
            
        psi_candidate_focused = self.orchestrator.l3_router.activation_to_wave(h_7b_lora)
        if len(psi_candidate_focused.shape) == 2:
            psi_candidate_focused = torch.mean(psi_candidate_focused, dim=0)
            
        return psi_candidate_focused.flatten()

    def run_repl_sandbox(self, code_payload, training_cases, test_mode=False):
        import copy
        sandbox_globals = {}
        
        common_imports = (
            "import math\n"
            "import collections\n"
            "from collections import defaultdict, deque, Counter\n"
            "import itertools\n"
            "import copy\n"
            "import numpy as np\n"
            "import torch\n"
            "import torch.nn as torch_nn\n"
            "import wosx\n"
        )
        full_code = common_imports + code_payload
        
        try:
            exec(full_code, sandbox_globals)
            if 'transform' not in sandbox_globals:
                return 0.0, "Execution Error: Function 'transform' not defined."
            transform_func = sandbox_globals['transform']
            
            if test_mode:
                # CRITICAL FIX: Deep-copy the input grid to prevent in-place memory corruption
                safe_input = copy.deepcopy(training_cases[0]['input'])
                prediction = transform_func(safe_input)
                return 1.0, prediction
            
            passed = 0
            for case in training_cases:
                # CRITICAL FIX: Deep-copy the input grid to prevent in-place memory corruption
                safe_input = copy.deepcopy(case['input'])
                expected_output = case['output']
                
                # Execute transformation on the isolated memory block
                if transform_func(safe_input) == expected_output:
                    passed += 1
                    
            if passed == len(training_cases):
                return 1.0, "All training cases passed."
            return (passed / len(training_cases)), f"Failed. Passed {passed}/{len(training_cases)}."
            
        except Exception as e:
            return 0.0, f"Execution Error: {str(e)}"

    def evaluate_candidate(self, pure_code: str, train_cases: list) -> tuple:
        """
        Executes candidate program in REPL sandbox against all training cases.
        Returns: (passed_cases, total_cases, execution_feedback)
        """
        score, feedback = self.run_repl_sandbox(pure_code, train_cases)
        total_cases = len(train_cases)
        passed_cases = int(score * total_cases)
        return passed_cases, total_cases, feedback

    def execute_task_manifold(self, task_dict, time_limit=1200, domain_tag="ARC_Task"):
        """
        Coordinates the Parallel Evolutionary Wave-Search Architecture.
        """
        import time
        start_time = time.time()
        
        # Pre-load specialized LoRA weights from the registry if they exist
        if domain_tag:
            print(f"[ENGINE] Pre-loading semantic LoRA adapter for domain tag '{domain_tag}'...")
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
            self.playbook_wave = playbook_wave
            
            # Blend superposition wave from elite candidates of the last turn
            if self.superposition_wave is not None:
                if playbook_wave is not None:
                    # Sum wave states into active interference pattern
                    superposition = playbook_wave.flatten() + self.superposition_wave.flatten()
                    mags = torch.abs(superposition).clamp(min=1e-8)
                    playbook_wave = (superposition / mags).reshape(playbook_wave.shape)
                else:
                    playbook_wave = self.superposition_wave
            
            # 2. Parallel Monte Carlo Generation: fire swarm to generate parallel hypotheses
            remaining = time_limit - (time.time() - start_time)
            if remaining > 800:
                num_candidates = 16  # Fire all 16 parallel agents as requested
            elif remaining > 400:
                num_candidates = 8
            elif remaining > 150:
                num_candidates = 4
            else:
                num_candidates = 1
                
            print(f"[ENGINE] Remaining time: {remaining:.2f}s. Generating {num_candidates} parallel hypotheses...")
            
            # --- MICRO-EPOCH APOPTOSIS CALLBACK ---
            zone_c_attractors = self.fetch_active_playbook_waves()
            zone_c_repellers = self.fetch_forbidden_topology_waves()
            
            failure_tracker = {}
            def micro_epoch_eval(generated_tokens_list, alpha_routing, candidate_idx):
                if candidate_idx not in failure_tracker:
                    failure_tracker[candidate_idx] = 0
                    
                # 1. Format the raw GPU token IDs for the CPU Router
                token_tensor = torch.tensor(generated_tokens_list, dtype=torch.long, device='cpu')
                
                # 2. Extract the continuous 4096-D wave (Executes in CPU L3 Cache)
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
            # --------------------------------------

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
                # 3. Inductive Generalization Guard (AST Sieve)
                is_generalized, guard_feedback = self.inductor.assert_generalization(candidate)
                if not is_generalized:
                    print(f"[ENGINE] Generalization check failed: {guard_feedback}")
                    # Store as invalid (None truth_tensor)
                    scored_candidates.append((candidate, None, 0.0, None, 999.0, guard_feedback, alpha_routing, None))
                    continue
                    
                pure_code = guard_feedback
                
                # 4. The Physical Sagnac Veto: evaluate wavefront in Zone B emulator
                wave_valid, physical_feedback, error_energy, truth_tensor, delta_np = self.emulator.evaluate_wavefront(pure_code, target_label="SCADA_Pressure_Control")
                if not wave_valid:
                    print(f"[ENGINE] Physical wave check failed: {physical_feedback}")
                    # Store as invalid but keep the truth_tensor/error_energy for drift
                    scored_candidates.append((candidate, pure_code, 0.0, truth_tensor, error_energy, physical_feedback, alpha_routing, delta_np))
                    continue
                    
                # 5. Measure sandbox execution accuracy against training set
                passed_cases, total_cases, execution_feedback = self.evaluate_candidate(pure_code, task_dict["train"])
                partial_score = passed_cases / total_cases
                
                # REINFORCE Update: Compute reward signal relative to the best accuracy seen so far
                if partial_score > self.best_sandbox_accuracy:
                    reward_signal = 1.0
                elif partial_score < self.best_sandbox_accuracy:
                    reward_signal = -1.0
                else:
                    reward_signal = 0.0
                    
                # LoRA reinforced updates removed in favor of thermodynamic Entropic Survival Engine
                if passed_cases == total_cases:
                    print(f"[ENGINE] Perfect inductive generalization achieved! Executing test case...")
                    test_score, test_pred = self.run_repl_sandbox(pure_code, task_dict["test"], test_mode=True)
                    
                    if test_score == 1.0 and test_pred is not None:
                        # Success: perform final Drift crystallization
                        winner_idx = self.orchestrator.l3_router.update_expert_centroids(truth_tensor)
                        print(f"[ANCHOR] Success wave drifted expert centroid {winner_idx} permanently.")
                        self.orchestrator.save_router_centroids()
                        self.orchestrator.lora_managers[winner_idx].save_weights()
                        
                        self.orchestrator.flush_lora_and_context_to_db(domain_tag=domain_tag)
                        return test_pred, revision_step, "SUCCESS"
                    else:
                        print("[ENGINE] Test input execution failed.")
                        scored_candidates.append((candidate, pure_code, 0.0, truth_tensor, error_energy, "Compiled successfully on train, but failed to execute on test input", alpha_routing, delta_np))
                else:
                    print(f"[ENGINE] Candidate score: {passed_cases}/{total_cases}")
                    scored_candidates.append((candidate, pure_code, partial_score, truth_tensor, error_energy, execution_feedback, alpha_routing, delta_np))
                    
            # Update best accuracy seen so far
            turn_best_score = max([c[2] for c in scored_candidates]) if scored_candidates else 0.0
            if turn_best_score > self.best_sandbox_accuracy:
                self.best_sandbox_accuracy = turn_best_score

            # 6. The Drift: Select winning candidate with lowest physical error energy
            valid_candidates = [c for c in scored_candidates if c[3] is not None] # c[3] is truth_tensor
            if valid_candidates:
                # Sort by physical error energy ascending
                valid_candidates.sort(key=lambda x: x[4]) # x[4] is error_energy
                winner_candidate, winner_pure, winner_score, winner_truth_tensor, winner_error_energy, winner_feedback, winner_alpha_routing, winner_delta_np = valid_candidates[0]
                
                # --- NEW ENTROPIC SURVIVAL ENGINE LOGIC ---
                zone_c_attractors = self.fetch_active_playbook_waves()
                zone_c_repellers = self.fetch_forbidden_topology_waves()
                
                # construct expert waves from the 16 candidates
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
                # ------------------------------------------
                
                print(f"[ENGINE] Declaring physical winner of turn {revision_step} (Error Energy: {winner_error_energy:.4f})")
                winner_idx = self.orchestrator.l3_router.update_expert_centroids(winner_truth_tensor)
                print(f"[ANCHOR] Expert centroid {winner_idx} drifted toward winner concept space.")
                self.orchestrator.save_router_centroids()
                
                # Save specialized expert weights
                self.orchestrator.lora_managers[winner_idx].save_weights()
                
                # Find the best candidate for playbook curation and error tracking (highest train score)
                valid_candidates.sort(key=lambda x: (-x[2], x[4]))
                best_candidate, best_pure, best_score, _, _, best_feedback, _, _ = valid_candidates[0]
            else:
                best_candidate, best_score, best_feedback = scored_candidates[0][0], 0.0, scored_candidates[0][5]
                
            self.error_history.append(best_feedback)
            
            # Selection & Superposition of top 2 candidates for next turn steering
            top_candidates = [c for c in scored_candidates if c[2] > 0 and c[3] is not None][:2]
            if len(top_candidates) >= 2:
                print("[ENGINE] Superposing top 2 candidate trajectories...")
                wave_1 = self.project_code_to_wave(top_candidates[0][0])
                wave_2 = self.project_code_to_wave(top_candidates[1][0])
                superposition = wave_1 + wave_2
                mags = torch.abs(superposition).clamp(min=1e-8)
                self.superposition_wave = superposition / mags
            elif len(top_candidates) == 1:
                self.superposition_wave = self.project_code_to_wave(top_candidates[0][0])
            else:
                self.superposition_wave = None
                
            # 7. Curate the playbook based on the best candidate's failure
            if "[CRITICAL] Hardcoding" in best_feedback:
                insight = "[CRITICAL] Hardcoding training grid values is forbidden. You must write a generalized functional transformation matrix."
            else:
                insight = self.orchestrator.reflect_on_failure(task_prompt, best_candidate, best_feedback)
                
            playbook_dict = self.orchestrator.curate_playbook(playbook_dict, insight)
            print("[ENGINE] Playbook curated.")
            
        self.orchestrator.flush_lora_and_context_to_db(domain_tag=domain_tag)
        return None, revision_step, "FAILED"