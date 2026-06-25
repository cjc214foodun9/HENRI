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
        
        # Initialize the agential thermostat Modulator Head
        from henri_core.thermodynamics import AgentialLangevinThermostat
        device = next(self.orchestrator.l3_router.parameters()).device if list(self.orchestrator.l3_router.parameters()) else torch.device("cpu")
        self.thermostat = AgentialLangevinThermostat(dim=4096).to(device)

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

    @torch.no_grad()
    def project_code_to_wave(self, candidate_code: str) -> torch.Tensor:
        """Projects candidate code to 4096-D complex wave state."""
        device = next(self.orchestrator.l3_router.parameters()).device
        
        # Intercept the incoming embedding payload array and migrate its memory page to the card
        raw_embedding = self.orchestrator.base_model.create_embedding(candidate_code)
        # Push the tensor straight into the active GPU memory channel
        h_7b_raw = torch.tensor(raw_embedding["data"][0]["embedding"], device=device, dtype=torch.bfloat16)
        
        h_7b_lora = self.orchestrator.lora_managers[0].apply_lora(h_7b_raw)
        if len(h_7b_lora.shape) == 2:
            h_7b_lora = torch.mean(h_7b_lora, dim=0)
            
        psi_candidate_focused = self.orchestrator.l3_router.activation_to_wave(h_7b_lora)
        if len(psi_candidate_focused.shape) == 2:
            psi_candidate_focused = torch.mean(psi_candidate_focused, dim=0)
            
        return psi_candidate_focused.flatten()

    def run_repl_sandbox(self, code_payload, training_cases, test_mode=False):  
        import copy  
        import numpy as np  
        import torch  
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
              
            def safe_dynamic_execute(input_grid):  
                """Safely tries multi-type ingestion pathways to prevent slicing and attribute crashes."""  
                # Pathway A: Try execution as a pure standard Python list  
                try:  
                    res = transform_func(copy.deepcopy(input_grid))  
                    if hasattr(res, "tolist"): return res.tolist()  
                    if isinstance(res, np.ndarray): return res.tolist()  
                    return res  
                except Exception:  
                    # Pathway B: Fallback to an optimized NumPy array wrapper  
                    try:  
                        res = transform_func(np.array(input_grid))  
                        if hasattr(res, "tolist"): return res.tolist()  
                        if isinstance(res, np.ndarray): return res.tolist()  
                        return res  
                    except Exception:  
                        # Pathway C: Fallback to an unflattened PyTorch tensor wrapper  
                        try:  
                            res = transform_func(torch.tensor(input_grid))  
                            if hasattr(res, "tolist"): return res.tolist()  
                            return res  
                        except Exception as final_backend_exception:  
                            raise final_backend_exception

            if test_mode:  
                safe_input = training_cases[0]['input']  
                prediction = safe_dynamic_execute(safe_input)  
                return 1.0, prediction  
              
            passed = 0  
            for case in training_cases:  
                safe_input = case['input']  
                expected_output = case['output']  
                  
                # Execute the candidate across the type-casting sieve  
                actual_output = safe_dynamic_execute(safe_input)  
                  
                # Normalize expected output type arrays if necessary  
                if hasattr(expected_output, "tolist"):  
                    expected_output = expected_output.tolist()  
                  
                # Perform an ironclad structural value comparison  
                if actual_output == expected_output:  
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

    @torch.no_grad()
    def execute_task_manifold(self, task_dict, time_limit=1200, domain_tag="ARC_Task"):
        """
        Coordinates the Parallel Evolutionary Wave-Search Architecture with 
        an added Thermodynamic Best-Effort Fallback Egress gate.
        """
        import time
        import copy
        start_time = time.time()
        task_dict["domain_tag"] = domain_tag
        
        # Pre-load specialized LoRA weights from the registry if they exist
        if domain_tag:
            print(f"[ENGINE] Pre-loading semantic LoRA adapter for domain tag '{domain_tag}'...")
            for idx, manager in self.orchestrator.lora_managers.items():
                self.orchestrator.synaptic_manager.route_and_load_adapter(domain_tag, manager)
                
        playbook_dict = self.orchestrator.initialize_empty_playbook()
        
        # Pre-fetch mastered sub-axiom visual primitives from TimescaleDB
        if hasattr(self.orchestrator, "prefetch_mastered_sub_axioms"):
            try:
                # Compile initial playbook to wave to act as the query vector
                initial_playbook_wave = self.orchestrator.compile_playbook_to_wave(playbook_dict)
                self.orchestrator.prefetch_mastered_sub_axioms(query_wave=initial_playbook_wave)
            except Exception as e:
                print(f"[ENGINE WARNING] Dynamic pre-fetch failed: {e}. Falling back to default pre-fetch.")
                self.orchestrator.prefetch_mastered_sub_axioms()
        
        from run_arc_benchmark import build_arc_prompt
        task_prompt, _ = build_arc_prompt(task_dict)
        
        self.error_history = []
        self.superposition_wave = None
        self.best_sandbox_accuracy = 0.0
        
        # --- NEW ACCUMULATION REGISTER: Best-effort trajectory fallback cache ---
        global_best_candidate_tracker = {
            "score": -1.0,
            "error_energy": float('inf'),
            "pure_code": None,
            "provisional_prediction": None
        }
        # ------------------------------------------------------------------------
        best_candidate = "def transform(input_grid):\n    return input_grid"
        
        stuck_high_energy_turns = 0
        revision_step = 0
        while True:
            revision_step += 1
            print(f"\n--- [ACE TURN {revision_step}] ---")
            
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                print(f"[ENGINE] Time limit of {time_limit}s reached. Engaging Fallback Egress...")
                break
                
            # Step 1: Extract the current raw continuous thought wave state
            current_thought_wave = self.project_code_to_wave(best_candidate)
            if current_thought_wave.dim() == 1:
                current_thought_wave = current_thought_wave.unsqueeze(0)

            # Step 2: Query HENRI's internal thermostat to decide on noise injection
            # Query the public.zone_c_resonant_hypersphere using complex_hypersphere_resonance FFI
            import psycopg
            import os
            db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/henri")
            
            # Map domain tag to DB domain
            domain_map = {
                "ARC_Task": "Agential Retraining",
                "ARC": "Agential Retraining",
                "Agential Retraining": "Agential Retraining",
                "Condensed Matter Physics": "Condensed Matter Physics",
                "Topology": "Topology",
                "Wave Mechanics": "Wave Mechanics",
                "Basal Biocomputation": "Basal Biocomputation",
                "Information Theory": "Information Theory",
                "Esoteric Philosophy": "Esoteric Philosophy",
                "Quantum Optics": "Quantum Optics",
                "Fluid Dynamics": "Fluid Dynamics",
                "Mixed-Signal Silicon": "Mixed-Signal Silicon",
                "Category Theory": "Category Theory"
            }
            db_domain = domain_map.get(domain_tag, "Agential Retraining")
            
            # Extract real and imaginary lists for SQL FFI
            wave_flat = current_thought_wave.flatten()
            r_list = torch.real(wave_flat).tolist()
            i_list = torch.imag(wave_flat).tolist()
            
            lexicon_list = []
            max_resonance_score = 0.0
            
            try:
                with psycopg.connect(db_url, connect_timeout=3) as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT real_phases, imag_phases, complex_hypersphere_resonance(real_phases, imag_phases, %s::real[], %s::real[]) AS resonance_score
                            FROM public.zone_c_resonant_hypersphere
                            WHERE domain = %s
                            ORDER BY resonance_score DESC
                            LIMIT 1024;
                        """, (r_list, i_list, db_domain))
                        rows = cur.fetchall()
                        
                        for row in rows:
                            real_p, imag_p, score = row[0], row[1], row[2]
                            max_resonance_score = max(max_resonance_score, score)
                            
                            # Rehydrate vector to torch tensor
                            r_tensor = torch.tensor(real_p, dtype=torch.float32)
                            i_tensor = torch.tensor(imag_p, dtype=torch.float32)
                            c_vec = torch.complex(r_tensor, i_tensor)
                            lexicon_list.append(torch.real(c_vec).detach().cpu())
            except Exception as e:
                print(f"[ENGINE DB RETRIEVAL WARNING] Failed to fetch from resonant hypersphere: {e}")
            
            if lexicon_list:
                zone_c_lexicon = torch.stack(lexicon_list)
                print(f"[ENGINE DB RETRIEVAL] Loaded {len(lexicon_list)} attractors from public.zone_c_resonant_hypersphere. Max resonance score: {max_resonance_score:.4f}")
            else:
                # Fallback to Hopfield vocabulary if DB query fails or is empty
                if hasattr(self.orchestrator, 'hopfield') and self.orchestrator.hopfield.vocabulary:
                    vocab_vals = list(self.orchestrator.hopfield.vocabulary.values())
                    for v in vocab_vals:
                        v_real = torch.real(v) if torch.is_complex(v) else v
                        lexicon_list.append(v_real.detach().cpu())
                    zone_c_lexicon = torch.stack(lexicon_list)
                else:
                    zone_c_lexicon = torch.randn(10, 4096)
            
            zone_c_lexicon = zone_c_lexicon.to(device=current_thought_wave.device, dtype=current_thought_wave.dtype)
            
            # Prevent Catastrophic Topological Domain Lock
            # 1. Track linewidth drift delta_phi
            if not hasattr(self, 'prev_wave'):
                self.prev_wave = None
            if self.prev_wave is not None:
                # Phase angle difference between consecutive turns
                phase_diff = torch.angle(current_thought_wave * torch.conj(self.prev_wave))
                delta_phi = torch.mean(torch.abs(phase_diff)).item()
                print(f"[LINEWIDTH AUDIT] Linewidth drift delta_phi: {delta_phi:.6f}")
            self.prev_wave = current_thought_wave.clone()
            
            # 2. Avoid cosine similarity > 0.15 in Zone C:
            # If maximum resonance score (cosine similarity) exceeds 0.15, trigger preventive thermal fluctuation
            force_noise_injection = False
            if max_resonance_score > 0.15:
                print(f"[DOMAIN LOCK WARNING] Cosine similarity in Zone C is {max_resonance_score:.4f} > 0.15. Triggering preventive thermal fluctuation.")
                force_noise_injection = True
                
            agential_noise, applied_voltage = self.thermostat.calculate_agential_perturbation(
                active_wave_state=current_thought_wave,
                zone_c_lexicon=zone_c_lexicon
            )
            
            temp = 0.70 + (applied_voltage / 3.5) * 0.65
            inject_noise = (applied_voltage > 0.5) or force_noise_injection
            
            # Compile current Playbook constraints into an axiomatic wave
            playbook_wave = self.orchestrator.compile_playbook_to_wave(playbook_dict)
            
            # Blend syntax anchor attractor wave to enforce clean python syntax
            if playbook_wave is not None and domain_tag == "ARC_Task":
                try:
                    syntax_anchor_wave = self.project_code_to_wave("def transform(input_grid):\n")
                    dev = playbook_wave.device
                    dtype = playbook_wave.dtype
                    
                    syntax_anchor_wave = syntax_anchor_wave.to(device=dev, dtype=dtype)
                    blended_wave = (0.30 * syntax_anchor_wave) + (0.70 * playbook_wave)
                    playbook_wave = torch.nn.functional.normalize(blended_wave, p=2, dim=-1)
                except Exception as e:
                    print(f"[ENGINE] Warning: Failed to blend syntax anchor: {e}")
                    
            self.playbook_wave = playbook_wave
            
            # Blend superposition wave from elite candidates of the last turn
            if self.superposition_wave is not None:
                # Select the superposition wave directly to prevent destructive linear averaging cancellation
                playbook_wave = self.superposition_wave.reshape(playbook_wave.shape) if playbook_wave is not None else self.superposition_wave
            
            # Step 3: Superimpose the predicted noise burst straight onto the active playbook trajectory
            # This physically shakes the latent space to unlock previously unverified vectors mid-flight
            if applied_voltage > 0.0 and playbook_wave is not None:
                agential_noise = agential_noise.to(device=playbook_wave.device, dtype=playbook_wave.dtype)
                playbook_wave = torch.nn.functional.normalize(
                    playbook_wave + agential_noise.reshape(playbook_wave.shape), 
                    p=2, dim=-1
                )
            
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
            
            # --- STIRRUP ROBOTIC HARNESS GROUNDING CHECK ---
            # NOTE: "Robotic" refers metaphorically to the software tool-use agent. 
            # This check uses the Stirrup harness to evaluate the sequence of software actions 
            # (such as file reads/writes and REPL execution) in a simulated, sandboxed dry-run 
            # before selecting the candidate. Successful trajectories are logged to the 
            # TimescaleDB hypertable registry (Shared Knowledge Attractor Grid).
            if hasattr(self.orchestrator, "stirrup") and self.orchestrator.stirrup is not None and len(candidate_batch) > 0:
                print(f"[STIRRUP] Grounding candidate batch of size {len(candidate_batch)} via sensory-motor harness...")
                try:
                    translation = task_dict.get("translation", (0.745, -1.204, 2.891))
                    rotation = task_dict.get("rotation", (0.05, -0.12, 1.41))
                    target_setpoint = task_dict.get("target_setpoint", (0.750, -1.200, 2.900))
                    
                    # Project candidates to 4096-D waves
                    horizon = getattr(self.orchestrator, "h_mpc_horizon", 4)
                    candidate_waves = []
                    for cand_text, _ in candidate_batch:
                        try:
                            c_wave = self.project_code_to_wave(cand_text)
                            c_wave_real = torch.real(c_wave)
                        except Exception:
                            # Fallback if code projection fails
                            c_wave_real = torch.randn(4096)
                        # Expand across lookahead horizon
                        seq = torch.stack([c_wave_real] * horizon, dim=0)
                        candidate_waves.append(seq)
                    candidate_motor_waves = torch.stack(candidate_waves, dim=0) # [num_candidates, horizon, 4096]
                    
                    device = self.orchestrator.stirrup.registry.bulk_lifter.weight.device
                    control_telemetry = self.orchestrator.stirrup.execute_grounded_control_tick(
                        translation=translation,
                        rotation=rotation,
                        target_setpoint=target_setpoint,
                        candidate_motor_waves=candidate_motor_waves.to(device=device, dtype=self.orchestrator.stirrup.registry.bulk_lifter.weight.dtype),
                        horizon=horizon
                    )
                    
                    print("\n=== STIRRUP GROUNDED ACTUATION REPORT ===")
                    print(f"  Selected Candidate Index: {control_telemetry['selected_plan_index']}")
                    print(f"  Thermodynamic Stress:     {control_telemetry['thermodynamic_stress_cost']:.6f}")
                    print(f"  SIGReg Separation Score:  {control_telemetry['sigreg_disentanglement_score']:.6f}")
                    print(f"  Transduced Token ID:      {control_telemetry['transduced_motor_token_id']}")
                    print(f"  Actuated Command:         \"{control_telemetry['actuated_environment_command']}\"")
                    print("=========================================\n")
                    
                    # Log telemetry to TimescaleDB
                    self.orchestrator.stirrup.log_telemetry_to_db(control_telemetry)

                    # Trigger dynamic gear-shifter transmission execute step
                    if hasattr(self.orchestrator, "gear_bridge") and self.orchestrator.gear_bridge is not None:
                        self.orchestrator.gear_bridge.execute_synchronized_gear_shift({
                            "thermodynamic_stress_cost": control_telemetry.get("thermodynamic_stress_cost", 0.5),
                            "sigreg_disentanglement_score": control_telemetry.get("sigreg_disentanglement_score", 3.0)
                        })
                    
                except Exception as stirrup_err:
                    print(f"[STIRRUP] Grounding check encountered an error: {stirrup_err}")
            # -----------------------------------------------
            
            scored_candidates = []
            for candidate_info in candidate_batch:
                candidate, alpha_routing = candidate_info
                # 3. Inductive Generalization Guard (AST Sieve)
                is_syntax_valid, pure_code_or_err = self.inductor.verify_syntax(candidate)
                if not is_syntax_valid:
                    print(f"[ENGINE] Syntax check failed: {pure_code_or_err}")
                    scored_candidates.append((candidate, None, 0.0, None, 999.0, pure_code_or_err, alpha_routing, None))
                    continue
                    
                pure_code = pure_code_or_err
                
                is_generalized, guard_feedback = self.inductor.assert_generalization(pure_code)
                if not is_generalized:
                    print(f"[ENGINE] Generalization check failed: {guard_feedback}")
                    # Store as invalid (None truth_tensor)
                    scored_candidates.append((candidate, None, 0.0, None, 999.0, guard_feedback, alpha_routing, None))
                    continue
                
                # 4. The Physical Sagnac Veto: evaluate wavefront in Zone B emulator
                wave_valid, physical_feedback, error_energy, truth_tensor, delta_np = self.emulator.evaluate_wavefront(pure_code, target_label="SCADA_Pressure_Control")
                if not wave_valid and domain_tag != "ARC_Task":
                    print(f"[ENGINE] Physical wave check failed: {physical_feedback}")
                    # Store as invalid but keep the truth_tensor/error_energy for drift
                    scored_candidates.append((candidate, pure_code, 0.0, truth_tensor, error_energy, physical_feedback, alpha_routing, delta_np))
                    continue
                    
                # 5. Measure sandbox execution accuracy against training set
                passed_cases, total_cases, execution_feedback = self.evaluate_candidate(pure_code, task_dict["train"])
                partial_score = passed_cases / total_cases
                
                # Mid-Flight Sub-Axiom Harvesting Loop (HARVEST_THRESHOLD = 0.5)
                if partial_score >= 0.5:
                    try:
                        code_lower = pure_code.lower()
                        if "rotate" in code_lower or "rotation" in code_lower:
                            label = "rotation_primitive"
                        elif "border" in code_lower or "boundary" in code_lower or "contain" in code_lower:
                            label = "border_containment_primitive"
                        elif "color" in code_lower or "palette" in code_lower or "scale" in code_lower:
                            label = "color_scaling_primitive"
                        else:
                            label = "general_spatial_primitive"
                        
                        c_wave = self.project_code_to_wave(candidate)
                        if hasattr(self.orchestrator, "harvest_and_persist_sub_axiom"):
                            self.orchestrator.harvest_and_persist_sub_axiom(label, pure_code, c_wave)
                    except Exception as harvest_err:
                        print(f"[HARVEST ERROR] Failed to harvest sub-axiom: {harvest_err}")
                
                # REINFORCE Update: Compute reward signal relative to the best accuracy seen so far
                if partial_score > self.best_sandbox_accuracy:
                    reward_signal = 1.0
                elif partial_score < self.best_sandbox_accuracy:
                    reward_signal = -1.0
                else:
                    reward_signal = 0.0
                    
                # Execute evaluation of the exact sandbox predictions for the best-effort ledger cache
                try:
                    _, provisional_pred = self.run_repl_sandbox(pure_code, task_dict["test"], test_mode=True)
                except Exception:
                    provisional_pred = None

                # --- TRACK EXCELLENCE: Cache the highest-scoring candidate seen globally ---
                if (partial_score > global_best_candidate_tracker["score"]) or \
                   (partial_score == global_best_candidate_tracker["score"] and error_energy < global_best_candidate_tracker["error_energy"]):
                    if provisional_pred is not None and (not isinstance(provisional_pred, str) or "Error" not in str(provisional_pred)):
                        global_best_candidate_tracker["score"] = partial_score
                        global_best_candidate_tracker["error_energy"] = error_energy
                        global_best_candidate_tracker["pure_code"] = pure_code
                        global_best_candidate_tracker["provisional_prediction"] = copy.deepcopy(provisional_pred)
                # ----------------------------------------------------------------------------
                    
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
            
            # Selection of top candidate trajectory (preventing linear phase cancellation)
            top_candidates = [c for c in scored_candidates if c[2] > 0 and c[3] is not None][:2]
            if len(top_candidates) >= 1:
                print("[ENGINE] Selecting top-1 candidate trajectory (preventing linear phase cancellation)...")
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

            # Track high-energy error reflection loop anomaly (Topological Logic Lock)
            current_error_val = 0.0
            if valid_candidates:
                current_error_val = winner_error_energy
            else:
                # If no valid candidate, treat as maximum error
                current_error_val = 999.0
            
            # Error Energy > 0.80 corresponds to E_reflect > 80.0 a.u.
            if current_error_val > 0.80:
                stuck_high_energy_turns += 1
                print(f"[ANOMALY AUDIT] High-energy reflection detected ({current_error_val:.4f} > 0.80). Stuck count: {stuck_high_energy_turns}/50")
            else:
                stuck_high_energy_turns = 0

            if stuck_high_energy_turns > 50:
                print(f"\n[ANOMALY TRIGGER] Topological Logic Lock detected for task {domain_tag} (>50 turns stuck in high-energy reflection loop). Truncating context thread.")
                self.orchestrator.flush_cognitive_manifold()
                # Reset search state to escape the lock
                stuck_high_energy_turns = 0
                best_candidate = "def transform(input_grid):\n    return input_grid"
                self.superposition_wave = None
                playbook_dict = self.orchestrator.initialize_empty_playbook()

            
        # --- TIMEOUT OVERDRIVE RELEASE: Submit the single best-scoring provisional answer ---
        if global_best_candidate_tracker["provisional_prediction"] is not None:
            print(f"\n[THERMODYNAMIC EVACUATION] Hard cap reached. Releasing best-effort trajectory candidate.")
            print(f" -> Retained Accuracy Tier: {global_best_candidate_tracker['score'] * 100:.2f}% validation coverage.")
            self.orchestrator.flush_lora_and_context_to_db(domain_tag=domain_tag)
            return global_best_candidate_tracker["provisional_prediction"], revision_step, "SUCCESS_BEST_EFFORT"
        # ------------------------------------------------------------------------------------
        
        self.orchestrator.flush_lora_and_context_to_db(domain_tag=domain_tag)
        return None, revision_step, "FAILED"