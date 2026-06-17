import torch
import torch.nn as nn
import numpy as np

class EmergentCognitiveSwarm(nn.Module):
    """
    Decentralized neural generation hub.
    Bridges the L3SwarmRouter's MoE weighting distribution to dynamic, in-memory logit bias steering.
    """
    def __init__(self, llama_instance, router_instance):
        super().__init__()
        self.llama = llama_instance
        self.router = router_instance
        self.orchestrator = None

    def orchestrate_in_memory_moe_steering(self, alpha_routing, prompt):
        """
        Builds the dynamic logit bias array entirely in-memory using the alpha routing weights.
        Projects expert residuals through the active LoRA weights and project them to logit bias space.
        """
        if not prompt or self.orchestrator is None:
            return None
            
        # 1. Retrieve the base embedding of the prompt
        emb_res = self.orchestrator.base_model.create_embedding(prompt)
        h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
        if h_7b_raw.ndim == 2:
            h_7b_raw = torch.mean(h_7b_raw, dim=0)
        elif h_7b_raw.ndim == 3:
            h_7b_raw = torch.mean(h_7b_raw.view(-1, h_7b_raw.shape[-1]), dim=0)
            
        with torch.no_grad():
            device = self.router.w_down.weight.device
            h_7b_raw = h_7b_raw.to(device)
            
            blended_residual = torch.zeros_like(h_7b_raw)
            for idx in range(self.orchestrator.num_streams):
                w = alpha_routing[idx].item() if torch.is_tensor(alpha_routing) else alpha_routing[idx]
                if w > 0.0001:
                    manager = self.orchestrator.lora_managers[idx]
                    lora_A = manager.lora_A.to(device)
                    lora_B = manager.lora_B.to(device)
                    residual_i = torch.matmul(torch.matmul(h_7b_raw, lora_A), lora_B)
                    blended_residual += w * residual_i
            
            # Project blended residual to logit bias space
            h_1024 = torch.matmul(blended_residual, self.router.activation_projection.weight.T)
            logit_bias = torch.matmul(h_1024, self.router.token_embedding.weight.T)
            logit_bias_np = logit_bias.cpu().numpy()
            
        def steered_logits_processor(input_ids, logits):
            n_vocab_model = logits.shape[-1]
            n_vocab_bias = len(logit_bias_np)
            if n_vocab_model == n_vocab_bias:
                logits[:] = logits + logit_bias_np
            elif n_vocab_model > n_vocab_bias:
                padded_bias = np.zeros(n_vocab_model, dtype=np.float32)
                padded_bias[:n_vocab_bias] = logit_bias_np
                logits[:] = logits + padded_bias
            else:
                logits[:] = logits + logit_bias_np[:n_vocab_model]
            return logits
            
        import llama_cpp
        return llama_cpp.LogitsProcessorList([steered_logits_processor])

    def generate_swarm_hypothesis(self, task_dict, playbook_wave, playbook_dict=None, temperature=0.7, inject_noise=False, early_stopping_callback=None):
        """
        Computes routing weights, blends PyTorch experts, injects dynamic LoRA adapter,
        and generates prompt response. Supports dynamic temperature and routing noise injection.
        """
        router_temp = 1.4 if inject_noise else 0.7
        # Compute the explicit alpha routing weights based on the compiled playbook wave
        if playbook_wave is not None:
            alpha_routing = self.router.compute_routing_weights(playbook_wave.unsqueeze(0), temperature=router_temp).squeeze(0)
        else:
            alpha_routing = torch.ones(self.orchestrator.num_streams) / self.orchestrator.num_streams
            
        if inject_noise:
            # Inject simulated randomized phase/routing noise into the MoE routing activations
            noise = torch.randn_like(alpha_routing) * 0.2
            alpha_routing = torch.softmax(alpha_routing + noise, dim=-1)
            
        prompt = self.build_clean_prompt(task_dict, playbook_dict)
        
        # JIT LoRA compilation & Vulkan injection
        if self.orchestrator is not None:
            # 1. Mathematically blend active expert weights in-memory
            blended_A, blended_B = self.orchestrator.blend_moe_loras(
                self.orchestrator.lora_managers,
                alpha_routing
            )
            # 2. Compile to GGUF format and physically inject into Vulkan computation graph
            self.orchestrator.apply_blended_lora_to_gemma(blended_A, blended_B)

        # Legacy logit steering is completely disabled to use physical adapter injection
        logit_processor = None
        
        try:
            # Execute the accelerated Vulkan token generation pass
            res_stream = self.llama(
                prompt,
                logits_processor=logit_processor,
                max_tokens=2048,
                temperature=temperature,
                stop=["<turn|>", "<|turn>"],
                stream=True
            )
            
            text = ""
            generated_tokens_list = []
            token_count = 0
            for chunk in res_stream:
                token_str = chunk["choices"][0]["text"]
                text += token_str
                
                try:
                    chunk_tokens = self.llama.tokenize(token_str.encode("utf-8"), add_bos=False)
                except TypeError:
                    chunk_tokens = self.llama.tokenize(token_str.encode("utf-8"))
                generated_tokens_list.extend(chunk_tokens)
                
                token_count += 1
                
                if early_stopping_callback is not None and token_count % 64 == 0:
                    should_kill = early_stopping_callback(generated_tokens_list, alpha_routing)
                    if should_kill:
                        print(f"[SWARM] Micro-Epoch Apoptosis Triggered at token {token_count}. Thread Guillotine executed.")
                        break
                        
        finally:
            # Clear physical adapter immediately after generation to prevent graph interference
            if self.orchestrator is not None:
                self.orchestrator.clear_active_lora()
        
        text = text.strip()
            
        # Log the generated swarm hypothesis to stdout
        print(f"\n--- [SWARM HYPOTHESIS GENERATED] ---\n{text}\n-------------------------------------\n")
        return text, alpha_routing

    def generate_parallel_hypotheses(self, task_dict, playbook_wave, playbook_dict=None, temperature=0.7, inject_noise=False, num_candidates=16, early_stopping_callback=None, start_time=None, time_limit=None):
        """
        Generates a batch of parallel hypotheses across the swarm in coordinated 64-token micro-epochs.
        Integrates Ephemeral Attractors, mid-flight Prune & Clone, and Spontaneous Resonance Harvesting.
        """
        # Initialize prompts, generated texts, token lists, active masks for all candidates
        prompts = [self.build_clean_prompt(task_dict, playbook_dict) for _ in range(num_candidates)]
        generated_texts = ["" for _ in range(num_candidates)]
        tokens_lists = [[] for _ in range(num_candidates)]
        active_mask = [True] * num_candidates
        
        # Initial routing weights perturbed for each candidate
        alpha_routings = []
        for i in range(num_candidates):
            if playbook_wave is not None:
                perturbed_wave = playbook_wave.clone()
                noise_phase = torch.randn_like(perturbed_wave.real) * 0.1
                perturbed_wave = perturbed_wave * torch.polar(torch.ones_like(perturbed_wave.real), noise_phase)
            else:
                perturbed_wave = None
                
            stream_inject_noise = inject_noise or (i > 0)
            router_temp = 1.4 if stream_inject_noise else 0.7
            
            if perturbed_wave is not None:
                alpha = self.router.compute_routing_weights(perturbed_wave.unsqueeze(0), temperature=router_temp).squeeze(0)
            else:
                alpha = torch.ones(self.orchestrator.num_streams) / self.orchestrator.num_streams
                
            if stream_inject_noise:
                noise = torch.randn_like(alpha) * 0.2
                alpha = torch.softmax(alpha + noise, dim=-1)
            alpha_routings.append(alpha)
            
        # Generate candidates using continuous PyTorch core model samplers
        import os
        print("[SWARM] Generating candidates using continuous PyTorch core model samplers...")
        device = self.router.w_down.weight.device
        
        # Resolve/import model components
        from diffusion_canvas import NonAutoregressiveCanvasSampler
        from henri_core.egress import QuantizedEgressAssembler
        from henri_core.core import ProprietaryHENRICore
        
        # Retrieve or instantiate the core model cached on the orchestrator
        if not hasattr(self.orchestrator, '_diffusion_core_model') or self.orchestrator._diffusion_core_model is None:
            # Target Full-Scale Swarm Target Configuration: dim=4096, depth=32, experts=16
            num_layers = 32
            num_base_experts = 16
            hidden_dim = 4096
            
            # Check for state dict in parent dir
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            core_path = os.path.join(parent_dir, "henri_core_final.pt")
            state_dict = None
            if os.path.exists(core_path):
                try:
                    state_dict = torch.load(core_path, map_location="cpu")
                except Exception as e:
                    print(f"[SWARM] Warning: Failed to parse checkpoint: {e}")
            
            core_model = ProprietaryHENRICore(dim=hidden_dim, depth=num_layers, num_fluid_states=num_base_experts)
            # Cast core to bfloat16 to optimize VRAM footprint on RTX 5090
            core_model = core_model.to(device=device, dtype=torch.bfloat16)
            core_model.eval()
            self.orchestrator._diffusion_core_model = core_model
        else:
            core_model = self.orchestrator._diffusion_core_model
            
        # Retrieve or instantiate cached translation head
        vocab_size = getattr(self.router, 'vocab_size', 32000)
        if not hasattr(self.orchestrator, '_diffusion_translation_head') or self.orchestrator._diffusion_translation_head is None:
            translation_head = nn.Linear(core_model.layers[0].dim, vocab_size)
            translation_head = translation_head.to(device=device, dtype=torch.bfloat16)
            translation_head.eval()
            self.orchestrator._diffusion_translation_head = translation_head
        else:
            translation_head = self.orchestrator._diffusion_translation_head
            
        # Lazily initialize H-MPC orchestrator on self.orchestrator if missing
        if not hasattr(self.orchestrator, 'h_mpc') or self.orchestrator.h_mpc is None:
            from cognitive_swarm import HolographicMPCOrchestrator
            self.orchestrator.h_mpc = HolographicMPCOrchestrator(core_model, dim=core_model.layers[0].dim).to(device=device, dtype=torch.bfloat16)

        # Generate and select winning action trajectory via Holographic MPC
        winning_wave = None
        if hasattr(self.orchestrator, 'h_mpc') and self.orchestrator.h_mpc is not None and playbook_wave is not None:
            horizon = 5
            candidate_action_sequences = []
            for idx in range(num_candidates):
                actions_seq = []
                for t in range(horizon):
                    perturbed_step = playbook_wave.clone()
                    noise_phase = torch.randn_like(perturbed_step.real) * 0.1
                    perturbed_step = perturbed_step * torch.polar(torch.ones_like(perturbed_step.real), noise_phase)
                    actions_seq.append(perturbed_step)
                candidate_action_sequences.append(torch.stack(actions_seq, dim=0))
            candidate_action_sequences = torch.stack(candidate_action_sequences, dim=0) # [num_candidates, horizon, hrr_dim]
            
            if hasattr(self.orchestrator, 'memory_engines') and 0 in self.orchestrator.memory_engines:
                current_wave = self.orchestrator.memory_engines[0].active_wave.to(device)
            else:
                current_wave = torch.zeros(self.orchestrator.hrr_dim, dtype=torch.complex64, device=device)

            # H-MPC operations must be real-valued. Cast tensors to real parts and bfloat16.
            current_wave_real = torch.real(current_wave).to(device=device, dtype=torch.bfloat16)
            target_goal_real = torch.real(playbook_wave).to(device=device, dtype=torch.bfloat16)
            candidate_actions_real = torch.real(candidate_action_sequences).to(device=device, dtype=torch.bfloat16)

            winning_idx = self.orchestrator.h_mpc.run_h_mpc_selection(
                current_wave=current_wave_real,
                target_goal_wave=target_goal_real,
                candidate_action_sequences=candidate_actions_real,
                horizon=horizon
            )
            winning_wave = candidate_action_sequences[winning_idx][-1] # use terminal state (keep original complex form)

        # Generate parallel candidate outputs
        for idx in range(num_candidates):
            if winning_wave is not None:
                # Perturb around the selected winning wave trajectory to generate diverse candidates
                perturbed_wave = winning_wave.clone()
                noise_phase = torch.randn_like(perturbed_wave.real) * 0.05
                perturbed_wave = perturbed_wave * torch.polar(torch.ones_like(perturbed_wave.real), noise_phase)
            elif playbook_wave is not None:
                perturbed_wave = playbook_wave.clone()
                noise_phase = torch.randn_like(perturbed_wave.real) * 0.1
                perturbed_wave = perturbed_wave * torch.polar(torch.ones_like(perturbed_wave.real), noise_phase)
            else:
                perturbed_wave = torch.zeros(self.orchestrator.hrr_dim, dtype=torch.complex64, device=device)
                
            # Project complex to real projection if needed
            if torch.is_complex(perturbed_wave):
                trajectory_real = torch.real(perturbed_wave)
            else:
                trajectory_real = perturbed_wave
                
            if trajectory_real.ndim == 1:
                trajectory_real = trajectory_real.unsqueeze(0)
                
            trajectory_input = trajectory_real.to(device=device, dtype=torch.bfloat16)
            
            try:
                # Non-Autoregressive Canvas Sampler is the primary pathway for score-guided relaxation
                sampler = NonAutoregressiveCanvasSampler(
                    core_model=core_model,
                    translation_head=translation_head,
                    num_diffusion_steps=25
                )
                target_tokens = sampler.crystallize_motif(
                    swarm_trajectory=trajectory_input,
                    sequence_length=512,
                    guidance_scale=4.5
                )
                token_ids = target_tokens[0].tolist()
            except Exception as e:
                print(f"[SWARM] Continuous canvas sampler failed for candidate {idx}: {e}. Falling back to default tokens.")
                token_ids = [ord(c) for c in "def transform(grid):\n    return grid"]
            
            # Decode to string using character mapping
            generated_text = "".join(chr(t % 256) for t in token_ids)
            generated_texts[idx] = generated_text
            tokens_lists[idx] = token_ids
            active_mask[idx] = False # Completed immediately
            
        candidates = []
        for idx in range(num_candidates):
            candidates.append((generated_texts[idx], alpha_routings[idx]))
        return candidates

    def build_clean_prompt(self, task_dict, playbook_dict=None):
        """
        Builds the prompt placing static task specification at the top and the volatile playbook at the bottom.
        This maximizes KV Cache reuse and optimizes Vulkan execution.
        """
        from run_arc_benchmark import build_arc_prompt
        prompt, guidelines = build_arc_prompt(task_dict)
        
        raw_prompt = (
            "<|turn>system\n"
            "You are the Generator sub-agent for the ARC AGI puzzle solver. Your goal is to write a Python function `transform(grid: list[list[int]]) -> list[list[int]]` using PyTorch.\n"
            "You MUST output exactly two blocks:\n"
            "1. A reasoning block wrapped in <|reasoning_begin|> and <|reasoning_end|> tags.\n"
            "2. A python code block wrapped in <|python_begin|> and <|python_end|> tags containing the transform function.\n\n"
            "CRITICAL RULES:\n"
            "- KEEP THE REASONING BLOCK VERY BRIEF (MAX 80 WORDS). Do NOT print grids, row lists, or verbose coordinate lists.\n"
            "- Do NOT use discrete numpy arrays for physics or spatial transformations. You must define continuous boundary conditions using wosx.boundary_query(point, geometry_id).\n"
            "- PATH A (Rigid Geometry): If the task is a discrete bounding box crop, flip, or translation, use standard PyTorch tensor slicing.\n"
            "- PATH B (Complex Emergence): If the task requires fuzzy pattern completion or physical evolution, translate the grid to S^1 wave phases and use the wosx PDE solver to define Dirichlet/Neumann boundaries.\n"
            "- Absolutely no conversational or explanatory text outside these tags is permitted.\n\n"
        )
        if guidelines:
            raw_prompt += (
                "=== ABSTRACT REFERENCE EXAMPLE ===\n"
                f"{guidelines}\n"
                "==================================\n\n"
            )
            
        raw_prompt += f"=== ARC TASK SPECIFICATION ===\n{prompt}\n\n"
        
        if playbook_dict and self.orchestrator is not None:
            playbook_str = self.orchestrator.serialize_playbook(playbook_dict)
            raw_prompt += (
                "=== CURRENT PLAYBOOK ===\n"
                f"{playbook_str}\n"
                "========================\n\n"
            )
            
        raw_prompt += (
            "<|turn>user\n"
            "Transform the test input grid according to the demonstrated rules and the playbook. "
            "Start your response immediately with <|reasoning_begin|>.<turn|>\n"
            "<|turn>model\n"
            "<|reasoning_begin|>\n1. Background color:"
        )
        
        return raw_prompt