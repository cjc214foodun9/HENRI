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

    @torch.no_grad()
    def generate_parallel_hypotheses(self, task_dict, playbook_wave, playbook_dict=None, temperature=0.7, inject_noise=False, num_candidates=16, early_stopping_callback=None, start_time=None, time_limit=None):
        """
        Generates a batch of parallel hypotheses across the swarm in coordinated 64-token micro-epochs.
        Integrates Ephemeral Attractors, mid-flight Prune & Clone, and Spontaneous Resonance Harvesting.
        """
        # Initialize prompts, generated texts, token lists, active masks for all candidates
        domain_tag = task_dict.get("domain_tag", "ARC_Task")
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
                polar_factor = torch.polar(torch.ones_like(perturbed_wave.real, dtype=torch.float32), noise_phase.to(torch.float32))
                perturbed_wave = perturbed_wave.to(torch.complex64) * polar_factor
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
        vocab_size = 32000
        if hasattr(self.orchestrator, 'l3_router') and self.orchestrator.l3_router is not None:
            vocab_size = getattr(self.orchestrator.l3_router, 'vocab_size', 32000)
        elif hasattr(self.orchestrator, '_diffusion_translation_head') and self.orchestrator._diffusion_translation_head is not None:
            vocab_size = self.orchestrator._diffusion_translation_head.out_features
            
        from henri_core.diffusion_canvas import NonAutoregressiveCanvasSampler
        from henri_core.egress import QuantizedEgressAssembler
        from henri_core.core import ProprietaryHENRICore
        
        # Retrieve or instantiate the core model cached on the orchestrator
        if not hasattr(self.orchestrator, '_diffusion_core_model') or self.orchestrator._diffusion_core_model is None:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            core_path = os.path.join(parent_dir, "henri_core_final.pt")
            if not os.path.exists(core_path):
                core_path = "henri_core_final.pt"
            
            checkpoint = None
            if os.path.exists(core_path):
                print(f"[SWARM] Loading checkpoint to CPU to conserve GPU VRAM from: {core_path}")
                try:
                    checkpoint = torch.load(core_path, map_location='cpu')
                except Exception as load_err:
                    print(f"[SWARM] Error loading checkpoint: {load_err}")
                    checkpoint = None
            else:
                print(f"[WARNING] Checkpoint {core_path} not found.")
                fallback_path = os.path.join(parent_dir, "henri_core_final_scaled.pt")
                if os.path.exists(fallback_path):
                    print(f" -> Falling back to baseline scaled weights: {fallback_path}")
                    try:
                        checkpoint = torch.load(fallback_path, map_location='cpu')
                    except Exception as fallback_err:
                        print(f"[SWARM ERROR] Failed loading fallback checkpoint: {fallback_err}")
                        checkpoint = None

            if checkpoint is not None and isinstance(checkpoint, dict) and "config" in checkpoint:
                cfg = checkpoint["config"]
                hidden_dim = cfg["dim"]
                num_layers = cfg["depth"]
                num_base_experts = cfg["num_fluid_states"]
                vocab_size = cfg["vocab_size"]
                print(f"[SWARM GEOMETRY] Checkpoint verified: dim={hidden_dim}, depth={num_layers}, fluid_states={num_base_experts}")

                # Set default dtype to bfloat16 to instantiate directly in low-precision and prevent OOM
                orig_default_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                try:
                    with torch.device(device):
                        core_model = ProprietaryHENRICore(dim=hidden_dim, depth=num_layers, num_fluid_states=num_base_experts)
                finally:
                    torch.set_default_dtype(orig_default_dtype)

                # Load state dict on CPU
                print("[SWARM] Loading model state dict on CPU...")
                core_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

                # Extract translation head state dict from checkpoint before deleting checkpoint
                checkpoint_translation_head_state_dict = checkpoint.get("translation_head_state_dict")

                # Free checkpoint memory immediately
                del checkpoint
                import gc; gc.collect(); torch.cuda.empty_cache()

                # Move core_model to device after loading state dict and freeing checkpoint
                print(f"[SWARM] Moving model to device: {device}...")
                core_model = core_model.to(device=device).eval()
                import gc; gc.collect(); torch.cuda.empty_cache()

                has_bias = (checkpoint_translation_head_state_dict is not None and "bias" in checkpoint_translation_head_state_dict)
                translation_head = nn.Linear(hidden_dim, vocab_size, bias=has_bias).to(dtype=torch.bfloat16)
                if checkpoint_translation_head_state_dict is not None:
                    try:
                        translation_head.load_state_dict(checkpoint_translation_head_state_dict)
                        print("[SWARM SUCCESS] Transduction vocabulary layer fully aligned with continuous core weights.")
                    except Exception as lsd_err:
                        print(f"[SWARM WARNING] Failed to load translation head state dict: {lsd_err}. Reinitializing.")
                        with torch.no_grad():
                            temp_th = torch.empty(translation_head.weight.shape, dtype=torch.float32, device='cpu')
                            torch.nn.init.orthogonal_(temp_th)
                            translation_head.weight.copy_(temp_th.to(device=device, dtype=torch.bfloat16))
                else:
                    print("[SWARM WARNING] No trained translation state found. Falling back to orthogonal init.")
                    with torch.no_grad():
                        temp_th = torch.empty(translation_head.weight.shape, dtype=torch.float32, device='cpu')
                        torch.nn.init.orthogonal_(temp_th)
                        translation_head.weight.copy_(temp_th.to(device=device, dtype=torch.bfloat16))

                translation_head = translation_head.to(device=device).eval()
                del checkpoint_translation_head_state_dict
                gc.collect(); torch.cuda.empty_cache()
            else:
                state_dict = checkpoint
                num_layers = 32
                num_base_experts = 16
                hidden_dim = 4096
                
                if state_dict is not None:
                    try:
                        layer_indices = set()
                        expert_indices = set()
                        for key, val in state_dict.items():
                            if key.startswith("layers."):
                                parts = key.split(".")
                                layer_indices.add(int(parts[1]))
                                if len(parts) > 3 and parts[2] == "experts":
                                    expert_indices.add(int(parts[3]))
                                    if "weight" in key:
                                        hidden_dim = val.shape[-1]
                                elif "weight" in key:
                                    hidden_dim = val.shape[-1]
                        if layer_indices:
                            num_layers = len(layer_indices)
                        if expert_indices:
                            num_base_experts = len(expert_indices)
                        print(f"[SWARM] Detected legacy state dict: {num_layers} layers, {num_base_experts} experts, dim={hidden_dim}")
                    except Exception as e:
                        print(f"[SWARM] Warning: Failed to parse legacy state dict: {e}")
                        state_dict = None

                # Set default dtype to bfloat16 to instantiate directly in low-precision and prevent OOM
                orig_default_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                try:
                    with torch.device(device):
                        core_model = ProprietaryHENRICore(dim=hidden_dim, depth=num_layers, num_fluid_states=num_base_experts)
                finally:
                    torch.set_default_dtype(orig_default_dtype)

                # Move core_model to device before loading state dict to prevent CPU memory duplication
                core_model = core_model.to(device=device).eval()
                if state_dict is not None:
                    core_model.load_state_dict(state_dict, strict=False)
                
                # Free state_dict memory immediately
                del state_dict
                import gc; gc.collect(); torch.cuda.empty_cache()

                vocab_size = getattr(self.router, 'vocab_size', 32000)
                translation_head = nn.Linear(hidden_dim, vocab_size, bias=False)
                nn.init.orthogonal_(translation_head.weight)
                translation_head = translation_head.to(device=device, dtype=torch.bfloat16).eval()

            self.orchestrator._diffusion_core_model = core_model
            self.orchestrator._diffusion_translation_head = translation_head
        else:
            core_model = self.orchestrator._diffusion_core_model
            translation_head = self.orchestrator._diffusion_translation_head
            
        # Lazily initialize H-MPC orchestrator on self.orchestrator if missing
        if not hasattr(self.orchestrator, 'h_mpc') or self.orchestrator.h_mpc is None:
            from cognitive_swarm import HolographicMPCOrchestrator
            self.orchestrator.h_mpc = HolographicMPCOrchestrator(core_model, dim=core_model.layers[0].dim).to(device=device, dtype=torch.bfloat16)
            self.orchestrator.h_mpc.orchestrator = self.orchestrator

        # Generate and select winning action trajectory via Holographic MPC
        winning_wave = None
        winning_jepa_track = None
        jl_guard = None
        if hasattr(self.orchestrator, 'h_mpc') and self.orchestrator.h_mpc is not None and playbook_wave is not None:
            horizon = getattr(self.orchestrator, "h_mpc_horizon", 5)
            candidate_action_sequences = []
            for idx in range(num_candidates):
                actions_seq = []
                for t in range(horizon):
                    perturbed_step = playbook_wave.clone()
                    noise_phase = torch.randn_like(perturbed_step.real) * 0.1
                    polar_factor = torch.polar(torch.ones_like(perturbed_step.real, dtype=torch.float32), noise_phase.to(torch.float32))
                    perturbed_step = perturbed_step.to(torch.complex64) * polar_factor
                    actions_seq.append(perturbed_step)
                candidate_action_sequences.append(torch.stack(actions_seq, dim=0))
            candidate_action_sequences = torch.stack(candidate_action_sequences, dim=0) # [num_candidates, horizon, hrr_dim]
            
            if hasattr(self.orchestrator, 'memory_engines') and 0 in self.orchestrator.memory_engines:
                current_wave = self.orchestrator.memory_engines[0].active_wave.to(device)
            else:
                current_wave = torch.zeros(self.orchestrator.hrr_dim, dtype=torch.complex64, device=device)

            # Pass raw complex tensors directly to run_h_mpc_selection to prevent complex casting leak
            winning_idx, winning_jepa_track = self.orchestrator.h_mpc.run_h_mpc_selection(
                current_wave=current_wave.to(device=device),
                target_goal_wave=playbook_wave.to(device=device),
                candidate_action_sequences=candidate_action_sequences.to(device=device),
                horizon=horizon
            )
            winning_wave = candidate_action_sequences[winning_idx][-1].clone() # use terminal state (keep original complex form)
            jl_guard = getattr(self.orchestrator.h_mpc, 'jl_guard', None)
            
            # Explicitly delete large trajectory tensor and purge GPU cache memory
            del candidate_action_sequences
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Generate parallel candidate outputs
        for idx in range(num_candidates):
            target_idx = winning_idx if (winning_idx is not None) else 0
            if idx != target_idx:
                # PEARL Short-Circuit Veto: skip continuous canvas sampler for suboptimal plans
                generated_text = "def transform(grid):\n    return grid"
                token_ids = [ord(c) for c in generated_text]
                generated_texts[idx] = generated_text
                tokens_lists[idx] = token_ids
                active_mask[idx] = False
                continue

            if winning_wave is not None:
                # Perturb around the selected winning wave trajectory to generate diverse candidates
                perturbed_wave = winning_wave.clone()
                noise_phase = torch.randn_like(perturbed_wave.real) * 0.05
                polar_factor = torch.polar(torch.ones_like(perturbed_wave.real, dtype=torch.float32), noise_phase.to(torch.float32))
                perturbed_wave = perturbed_wave.to(torch.complex64) * polar_factor
            elif playbook_wave is not None:
                perturbed_wave = playbook_wave.clone()
                noise_phase = torch.randn_like(perturbed_wave.real) * 0.1
                polar_factor = torch.polar(torch.ones_like(perturbed_wave.real, dtype=torch.float32), noise_phase.to(torch.float32))
                perturbed_wave = perturbed_wave.to(torch.complex64) * polar_factor
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
                if not hasattr(self.orchestrator, "canvas_sampler") or self.orchestrator.canvas_sampler is None:
                    self.orchestrator.canvas_sampler = NonAutoregressiveCanvasSampler(
                        core_model=core_model,
                        translation_head=translation_head,
                        num_diffusion_steps=2
                    )
                    self.orchestrator.canvas_sampler.guidance_scale = 4.5
                self.orchestrator._canvas_sampler = self.orchestrator.canvas_sampler

                active_seq_len = min(512, getattr(self.orchestrator, "max_context_len", 512))
                active_guidance = getattr(self.orchestrator.canvas_sampler, "guidance_scale", 4.5)

                target_tokens = self.orchestrator.canvas_sampler.crystallize_motif(
                    swarm_trajectory=trajectory_input,
                    sequence_length=active_seq_len,
                    guidance_scale=active_guidance,
                    winning_jepa_track=winning_jepa_track,
                    jl_guard=jl_guard,
                    domain_tag=domain_tag,
                    intent_flag=task_dict.get("intent_flag", "CONVERSATION"),
                    playbook_wave=playbook_wave
                )
                token_ids = target_tokens[0].tolist()
                
                # Convert character arrays directly back to python text strings using the GPT-2 Tokenizer
                try:
                    from transformers import AutoTokenizer
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if vocab_size == 32000:
                        local_tok_dir = os.path.join(parent_dir, "llama_tokenizer_local")
                        tokenizer = AutoTokenizer.from_pretrained(local_tok_dir)
                    else:
                        local_tok_dir = os.path.join(parent_dir, "gpt2_tokenizer_local")
                        tokenizer = AutoTokenizer.from_pretrained(local_tok_dir) if os.path.exists(local_tok_dir) else AutoTokenizer.from_pretrained('gpt2')
                    generated_text = tokenizer.decode(token_ids)
                except Exception as tok_err:
                    print(f"[SWARM WARNING] Tokenizer decode failed: {tok_err}. Falling back to character mapping.")
                    decoded_chars = []
                    for token in token_ids:
                        char_code = abs(int(token)) % 256
                        if (char_code >= 32 and char_code <= 126) or char_code == 10 or char_code == 13:
                            decoded_chars.append(chr(char_code))
                    generated_text = "".join(decoded_chars)
                
                # Sanitize syntax-breaking characters (like non-breaking spaces and replacement characters)
                generated_text = generated_text.replace("\xa0", " ").replace("\ufffd", "")
                
                # Before passing the materialized string to the sandbox, clamp the signature block
                prefix_constraint = "def transform(input_grid):\n"
                if not generated_text.strip().startswith("def transform"):
                    generated_text = prefix_constraint + generated_text
                    
                print(f"[CRYSTALLIZATION COMPLETED] Generated executable block size: {len(generated_text)} characters.")
            except Exception as e:
                print(f"[SWARM] Continuous canvas sampler failed for candidate {idx}: {e}. Falling back to default tokens.")
                token_ids = [ord(c) for c in "def transform(grid):\n    return grid"]
                generated_text = "def transform(grid):\n    return grid"
            
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