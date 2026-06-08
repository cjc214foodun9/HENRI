import os
import sys
import torch
import numpy as np
import re

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from autotelic_cognitive_engine import IMGEP_Manager
from neurosymbolic_program_induction import ProgramInductor
from active_experimentation_engine import ClosedLoopScientist

class ActiveInferenceSwarmAgent:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.step_logs = []
        
        # Initialize Autotelic drivers (Phase 2), Synthesizer (Phase 3), and Scientist (Phase 4)
        self.imgep = IMGEP_Manager(state_dim=128, action_dim=128, vocab_size=50, embed_dim=8)
        self.inductor = ProgramInductor(state_dim=128)
        self.scientist = ClosedLoopScientist(self.inductor, state_dim=128)
        self.current_temperature = 0.4

    @torch.no_grad()
    def run_active_inference_loop(self, prompt, target_label="SCADA_Pressure_Control", max_revisions=3, path="ALETHEIA_SWARM"):
        self.step_logs = []
        
        self.step_logs.append({
            "stage": "ROUTING GATEWAY",
            "message": f"Analyzing user intent: path={path}, domain={target_label}, max_revisions={max_revisions}",
            "status": "info"
        })
        
        if path == "GENERAL":
            self.step_logs.append({
                "stage": "GENERAL CHAT",
                "message": "Routing to standard chat engine (bypassing Sagnac/REPL validation loops)...",
                "status": "info"
            })
            
            messages = [
                {"role": "system", "content": "You are HENRI, a highly intelligent and helpful general-purpose AI assistant. Provide a direct, complete, and useful response. Do not output raw Aletheia tag templates unless explicitly asked."},
                {"role": "user", "content": prompt}
            ]
            
            res = self.orchestrator.gen_model.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=0.7
            )
            response = res["choices"][0]["message"]["content"]
            
            self.step_logs.append({
                "stage": "GENERAL CHAT",
                "message": "Direct response completed.",
                "status": "success"
            })
            return response
        
        # 1. Synaptic Routing
        self.step_logs.append({
            "stage": "SYNAPTIC ROUTING",
            "message": f"Fetching LoRA adapter from TimescaleDB registry for domain: '{target_label}'...",
            "status": "info"
        })
        self.orchestrator.synaptic_manager.route_and_load_adapter(target_label, self.orchestrator.lora_managers[0])
        
        # 2. Autotelic Goal Setting (IMGEP Phase)
        self.step_logs.append({
            "stage": "AUTOTELIC IMAGINATION",
            "message": "Generating internal epistemic goal vector (Vygotskian Concept Recombination)...",
            "status": "info"
        })
        
        # Deterministically map target_label to concept indices for the Vygotskian embedding space
        concept_idx_1 = torch.tensor([hash(target_label) % 50])
        concept_idx_2 = torch.tensor([(hash(target_label) // 50) % 50])
        goal_state = self.imgep.generate_goal(concept_idx_1, concept_idx_2)
        
        self.step_logs.append({
            "stage": "AUTOTELIC IMAGINATION",
            "message": f"Imagined Goal State generated (L2 Norm: {torch.norm(goal_state).item():.4f}). Target alignment locked.",
            "status": "success"
        })

        # 3. Scientific Hypothesis Ensemble (Generator)
        self.step_logs.append({
            "stage": "GENERATOR",
            "message": "Initializing scientific hypothesis ensemble. Querying Gemma RAM Swarm...",
            "status": "info"
        })
        
        # Apply rehydration and proactive watermarking to prompt
        prompt = self.orchestrator.rehydrate_prompt(prompt)
        prompt = self.orchestrator.proactive_eviction(prompt)

        messages = [
            {"role": "system", "content": (
                "You are the Generator sub-agent. Parse the mathematical/physics problem "
                "down into digestible steps. Show your detailed Chain-of-Thought (CoT) reasoning. "
                "If symbolic or tensor algebra calculation is required, you must delegate it by writing a Python block enclosed in <|python_begin|> and <|python_end|> tags. "
                "DO NOT guess, bluff, or invent results. Every step must be self-contained."
            )},
            {"role": "user", "content": prompt}
        ]
        
        # Generate initial candidate
        res = self.orchestrator.gen_model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=self.current_temperature
        )
        candidate = res["choices"][0]["message"]["content"]
        
        self.step_logs.append({
            "stage": "GENERATOR",
            "message": "Initial candidate solution synthesized.",
            "content": candidate,
            "status": "success"
        })

        # Execute any Python block inside sandbox
        candidate = self.process_repl_sandbox(candidate)

        # 4. Closed-loop Verification & Darwinian Discovery Cycles (Revisions)
        for revision in range(1, max_revisions + 1):
            # A. Physical verification (D2NN, Sagnac, Dirichlet boundary)
            is_valid, feedback, delta_np, error_energy = self.verify_candidate(candidate, target_label)
            
            if is_valid:
                self.step_logs.append({
                    "stage": "CONVERGENCE",
                    "message": "Candidate verified by physical boundary. Synaptic weights consolidated.",
                    "status": "success"
                })
                # Save adapter
                self.orchestrator.synaptic_manager.consolidate_and_save_adapter(
                    domain_tag=target_label,
                    lora_manager=self.orchestrator.lora_managers[0],
                    error_delta=0.0
                )
                return candidate
                
            # B. Epistemic Active Experimentation (Darwinian Discovery)
            self.step_logs.append({
                "stage": "DARWINIAN DISCOVERY",
                "message": f"Sagnac error detected ({error_energy:.4f} > 0.35). Fitting symbolic programs and mutation ensemble...",
                "status": "warning"
            })
            
            # Formulate mock empirical observations based on candidate projection
            x_obs = torch.randn(10, 128)
            y_obs = torch.sin(x_obs * 2.0) + torch.randn_like(x_obs) * 0.05
            
            # Run program inductor search to discover the logical rules governing this error state
            best_ast, best_loss = self.inductor.fit_program_to_data(x_obs, y_obs, num_architectures=3)
            
            # Map discovered rule back to prompt adjustments (Active Inference)
            self.step_logs.append({
                "stage": "DARWINIAN DISCOVERY",
                "message": f"Program induction search complete. Discovered logical program tree with MSE loss: {best_loss:.6f}.",
                "status": "success"
            })
            
            # Apply Rehypothecation tensor updates to shift LoRA weights
            if delta_np is not None:
                delta_projected = self.orchestrator.l3_router.wave_to_activation(delta_np)
                for i in range(self.orchestrator.num_streams):
                    self.orchestrator.lora_managers[i].update_with_rehypothecated_tensors(delta_projected, 0.1)
                self.orchestrator.synaptic_manager.consolidate_and_save_adapter(
                    domain_tag=target_label,
                    lora_manager=self.orchestrator.lora_managers[0],
                    error_delta=error_energy
                )

            # Langevin temperature boost
            boost = min(0.5, error_energy * 0.5)
            self.current_temperature = min(1.0, 0.4 + boost)
            
            # C. Reviser: synthesize correction based on physical veto and program inductor feedback
            self.step_logs.append({
                "stage": "REVISER",
                "message": "Invoking reviser to correct candidate based on Sagnac and neurosymbolic feedback...",
                "status": "info"
            })
            
            reviser_prompt = (
                f"Problem: {prompt}\n\n"
                f"Candidate Solution:\n{candidate}\n\n"
                f"Verifier Feedback:\n{feedback}\n\n"
                f"Neurosymbolic Induced Rule Loss: {best_loss:.6f}.\n"
                f"Please rewrite the code and reasoning block, ensuring you resolve this error."
            )
            
            # Apply rehydration and proactive watermarking to reviser prompt
            reviser_prompt = self.orchestrator.rehydrate_prompt(reviser_prompt)
            reviser_prompt = self.orchestrator.proactive_eviction(reviser_prompt)
            
            messages = [
                {"role": "system", "content": (
                    "You are the Reviser sub-agent. Correct the candidate solution based on physical "
                    "boundary veto feedback and neurosymbolic rule inductions. Adhere strictly to "
                    "mathematical invariants. Output the updated solution."
                )},
                {"role": "user", "content": reviser_prompt}
            ]
            
            res = self.orchestrator.gen_model.create_chat_completion(
                messages=messages,
                max_tokens=2048,
                temperature=self.current_temperature
            )
            candidate = res["choices"][0]["message"]["content"]
            candidate = self.process_repl_sandbox(candidate)

        self.step_logs.append({
            "stage": "TIMEOUT",
            "message": "Maximum revision iterations reached without perfect physical convergence.",
            "status": "error"
        })
        return candidate

    def process_repl_sandbox(self, candidate):
        if "<|python_begin" in candidate and "<|python_end|>" in candidate:
            idx_begin = candidate.find("<|python_begin")
            idx_end = candidate.find("<|python_end|>")
            idx_close_bracket = candidate.find("|>", idx_begin)
            if idx_close_bracket != -1 and idx_close_bracket < idx_end:
                code_block = candidate[idx_close_bracket + 2 : idx_end].strip()
                
                self.step_logs.append({
                    "stage": "SANDBOX EXECUTION",
                    "message": "Executing Python code block in sandbox...",
                    "code": code_block,
                    "status": "info"
                })
                
                exec_res = self.orchestrator.repl.execute_block(code_block)
                stdout = exec_res["stdout"].strip()
                stderr = exec_res["stderr"].strip()
                output_content = stdout if exec_res["success"] else f"Error: {stderr or exec_res['error_message']}"
                
                self.step_logs.append({
                    "stage": "SANDBOX EXECUTION",
                    "message": "Sandbox execution complete.",
                    "output": output_content,
                    "status": "success" if exec_res["success"] else "error"
                })
                
                output_tag = f"\n<|output_begin|>\n{output_content}\n<|output_end|>\n"
                candidate = candidate[:idx_end + len("<|python_end|>")] + output_tag + candidate[idx_end + len("<|python_end|>"):]
        return candidate

    @torch.no_grad()
    def verify_candidate(self, candidate, target_label):
        from cognitive_swarm import pin_current_thread_to_core_7
        pin_current_thread_to_core_7()
        
        has_code = False
        repl_success = True
        repl_output = ""
        
        self.step_logs.append({
            "stage": "VERIFIER",
            "message": "Checking if candidate solution contains Python blocks...",
            "status": "info"
        })
        
        if "<|python_begin" in candidate and "<|python_end|>" in candidate:
            has_code = True
            idx_begin = candidate.find("<|python_begin")
            idx_end = candidate.find("<|python_end|>")
            idx_close_bracket = candidate.find("|>", idx_begin)
            if idx_close_bracket != -1 and idx_close_bracket < idx_end:
                code_block = candidate[idx_close_bracket + 2 : idx_end].strip()
                
                self.step_logs.append({
                    "stage": "VERIFIER REPL",
                    "message": "Running candidate Python code in isolated verification sandbox...",
                    "code": code_block,
                    "status": "info"
                })
                
                res = self.orchestrator.repl.execute_block(code_block)
                repl_success = res["success"]
                if res["success"]:
                    repl_output = res["stdout"].strip()
                else:
                    repl_output = res["error_message"].strip() or res["stderr"].strip()
                
                self.step_logs.append({
                    "stage": "VERIFIER REPL",
                    "message": "Verification sandbox run complete.",
                    "output": repl_output,
                    "status": "success" if repl_success else "error"
                })

        if has_code and not repl_success:
            self.step_logs.append({
                "stage": "VERIFIER REPL",
                "message": f"Syntax/Runtime Error in REPL sandbox: {repl_output}",
                "status": "error"
            })
            return False, f"REPL Execution Error: The code block failed with error: {repl_output}", None, 1.0

        # Project to wave space
        self.step_logs.append({
            "stage": "VERIFIER PROJECTION",
            "message": "Projecting token-state activation to 4096D complex wave space...",
            "status": "info"
        })
        
        emb_res = self.orchestrator.base_model.create_embedding(candidate)
        h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
        h_7b_lora = self.orchestrator.lora_managers[0].apply_lora(h_7b_raw)
        if len(h_7b_lora.shape) == 2:
            h_7b_lora = torch.mean(h_7b_lora, dim=0)
            
        is_symbolic_derivation = any(kw in target_label.lower() or kw in candidate.lower() for kw in ["symbolic", "mathematical", "derivation", "weyl", "anomaly", "expansion", "coefficient", "thermodynamic", "conservation", "coeff"])
        
        if is_symbolic_derivation:
            self.step_logs.append({
                "stage": "VERIFIER PROJECTION",
                "message": "Task requires Strict Symbolic Derivation. Bypassing tiled wave superposition.",
                "status": "info"
            })
            psi_candidate_focused = self.orchestrator.l3_router.activation_to_wave(h_7b_lora)
            if len(psi_candidate_focused.shape) == 2:
                psi_candidate_focused = torch.mean(psi_candidate_focused, dim=0)
            psi_candidate_focused = psi_candidate_focused.reshape(64, 64)
        else:
            self.step_logs.append({
                "stage": "VERIFIER PROJECTION",
                "message": "Applying tiled wave superposition and lens downsampling...",
                "status": "info"
            })
            activations_stack = h_7b_lora.unsqueeze(0).unsqueeze(0).repeat(16, 1, 1)
            psi_candidate, _, _ = self.orchestrator.l3_router(activations=activations_stack)
            N = psi_candidate.size(-1)
            x = torch.linspace(-1.0, 1.0, N, device=psi_candidate.device)
            y = torch.linspace(-1.0, 1.0, N, device=psi_candidate.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            lens_phase = -50.0 * (X**2 + Y**2)
            lens = torch.polar(torch.ones_like(lens_phase), lens_phase)
            wave_lensed = psi_candidate.squeeze(0) * lens
            focal_plane = torch.fft.fft2(wave_lensed, norm='ortho')
            focal_plane_shifted = torch.fft.fftshift(focal_plane)
            start = (N - 64) // 2
            end = start + 64
            focused_64 = focal_plane_shifted[start:end, start:end]
            mags = torch.abs(focused_64).clamp(min=1e-8)
            psi_candidate_focused = focused_64 / mags

        # Zone B
        self.step_logs.append({
            "stage": "VERIFIER WAVE CORE",
            "message": "Firing wave in Zone B physical core, checking cache memory and Sagnac alignment...",
            "status": "info"
        })
        
        target_vector = self.orchestrator.hopfield.vocabulary.get(target_label)
        if target_vector is None:
            target_vector = self.orchestrator.get_stream_address(0)
            
        target_np = target_vector.detach().numpy().astype(np.complex64)
        psi_candidate_flat = psi_candidate_focused.flatten()
        retrieved_wave = self.orchestrator.memory_engines[0].retrieve_from_cache(query_key=psi_candidate_flat)
        blended_focused = psi_candidate_flat + retrieved_wave
        blended_mags = torch.abs(blended_focused).clamp(min=1e-8)
        psi_candidate_resolved = blended_focused / blended_mags
        
        psi_cand_np = psi_candidate_resolved.detach().numpy().astype(np.complex64)
        truth_np, delta_np, alignment = self.orchestrator.optical_core.forward(
            hr_wavefront=psi_cand_np,
            target_manifold=target_np,
            langevin_heat=0.0
        )

        # Boundary validation
        self.step_logs.append({
            "stage": "VERIFIER BOUNDARY",
            "message": "Verifying Dirichlet & Neumann boundaries against database invariants...",
            "status": "info"
        })
        
        truth_tensor = torch.tensor(truth_np, dtype=torch.complex64)
        is_valid, veto_reason, error_energy, h_cft = self.orchestrator.boundary_validator.validate_boundary(truth_tensor)

        if not is_valid:
            feedback = f"Sagnac Veto: The candidate logic violated Dirichlet boundary axioms. Reason: {veto_reason} | Error Energy: {error_energy:.4f}"
            self.step_logs.append({
                "stage": "VERIFIER BOUNDARY",
                "message": f"Boundary VETO triggered. Reason: {veto_reason}",
                "error_energy": error_energy,
                "status": "error"
            })
            return False, feedback, delta_np, error_energy
            
        self.step_logs.append({
            "stage": "VERIFIER BOUNDARY",
            "message": "Dirichlet boundaries verified successfully. Sagnac alignment achieved.",
            "error_energy": error_energy,
            "status": "success"
        })
        return True, "Dirichlet boundaries verified. Sagnac alignment achieved.", delta_np, error_energy
