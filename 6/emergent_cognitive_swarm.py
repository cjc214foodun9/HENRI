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
            
        # Volatile attraction CPU buffer
        ephemeral_attractors = []
        
        # Coordinated micro-epoch generation loop
        max_epochs = 32 # 32 epochs * 64 tokens = 2048 max tokens
        APOPTOSIS_THRESHOLD = 0.35
        DYNAMIC_STEERING_THRESHOLD = 0.85
        HARVEST_THRESHOLD = 0.95
        
        device = self.router.w_down.weight.device
        
        for epoch in range(max_epochs):
            any_active = False
            for idx in range(num_candidates):
                if not active_mask[idx]:
                    continue
                any_active = True
                
                # Generate next 64 tokens for this candidate
                current_prompt = prompts[idx] + generated_texts[idx]
                
                # Blend LoRAs and apply
                if self.orchestrator is not None:
                    blended_A, blended_B = self.orchestrator.blend_moe_loras(
                        self.orchestrator.lora_managers,
                        alpha_routings[idx]
                    )
                    self.orchestrator.apply_blended_lora_to_gemma(blended_A, blended_B)
                    
                new_text = ""
                try:
                    res = self.llama(
                        current_prompt,
                        max_tokens=64,
                        temperature=temperature,
                        stop=["<turn|>", "<|turn>"],
                        stream=False
                    )
                    new_text = res["choices"][0]["text"]
                except Exception as gen_err:
                    print(f"[SWARM] Error during generation for candidate {idx}: {gen_err}")
                finally:
                    if self.orchestrator is not None:
                        self.orchestrator.clear_active_lora()
                        
                generated_texts[idx] += new_text
                
                # Tokenize and append
                try:
                    chunk_tokens = self.llama.tokenize(new_text.encode("utf-8"), add_bos=False)
                except TypeError:
                    chunk_tokens = self.llama.tokenize(new_text.encode("utf-8"))
                tokens_lists[idx].extend(chunk_tokens)
                
                # Check for stop conditions
                if not new_text or any(stop_word in new_text for stop_word in ["<turn|>", "<|turn>"]):
                    active_mask[idx] = False
                    
                # Call early stopping/apoptosis callback if active
                if active_mask[idx] and early_stopping_callback is not None:
                    should_kill = early_stopping_callback(tokens_lists[idx], alpha_routings[idx], idx)
                    if should_kill:
                        print(f"[SWARM] Micro-Epoch Apoptosis Triggered at epoch {epoch} for candidate {idx}. Thread execution aborted.")
                        active_mask[idx] = False
                    
            if not any_active:
                break
                
            # Check for overall timeout limit
            if start_time is not None and time_limit is not None:
                import time
                if time.time() - start_time >= time_limit:
                    print(f"[SWARM] Swarm generation timeout reached (Elapsed: {time.time() - start_time:.2f}s >= {time_limit}s). Stopping batch generation.")
                    break
                
            # --- EVALUATE RESONANCE FOR ACTIVE SWARM ---
            expert_waves = []
            for idx in range(num_candidates):
                t_list = tokens_lists[idx] if tokens_lists[idx] else [0]
                token_tensor = torch.tensor(t_list, dtype=torch.long, device='cpu')
                wave = self.router.text_to_wave(token_tensor)
                # Ensure wave has shape [4096]
                if wave.ndim == 2:
                    wave = torch.mean(wave, dim=0)
                expert_waves.append(wave)
            expert_waves_tensor = torch.stack(expert_waves)
            
            # Retrieve stable Zone C axioms
            zone_c_attractors = playbook_wave.unsqueeze(0) if playbook_wave is not None else torch.zeros((1, 4096), dtype=torch.complex64)
            
            # Combine with ephemeral attractors
            if ephemeral_attractors:
                combined_attractors = torch.cat([zone_c_attractors, torch.stack(ephemeral_attractors)], dim=0)
            else:
                combined_attractors = zone_c_attractors
                
            # Entropic survival calculation
            resonances = self.orchestrator.entropic_engine.evaluate_entropic_fitness(
                expert_waves_tensor,
                combined_attractors,
                zone_c_repellers=None
            )
            
            # Rank the experts
            ranked_experts = torch.argsort(resonances, descending=True)
            top_expert_idx = ranked_experts[0].item()
            # Bottom 25% or at least the bottom candidate if num_candidates is small
            k_bottom = max(1, num_candidates // 4)
            bottom_experts = ranked_experts[-k_bottom:]
            
            # --- THE EPHEMERAL HARVEST ---
            top_resonance = resonances[top_expert_idx].item()
            if top_resonance > DYNAMIC_STEERING_THRESHOLD:
                # Extract the winning LoRA state
                winning_lora = self.orchestrator.lora_managers[top_expert_idx].lora_A.data.mean(dim=1)
                top_wave = torch.matmul(winning_lora.to(device), self.router.w_down.weight.T)
                top_wave = torch.nn.functional.normalize(top_wave, p=2, dim=0)
                
                # Convert to unit-modulus complex representation
                phases = torch.angle(torch.complex(top_wave, torch.zeros_like(top_wave)))
                top_wave_complex = torch.polar(torch.ones_like(phases), phases)
                
                # Add to the volatile CPU buffer
                ephemeral_attractors.append(top_wave_complex)
                print(f"[EPHEMERAL] Expert {top_expert_idx} established new local attractor. Resonance: {top_resonance:.4f}")
                
            # --- SPONTANEOUS RESONANCE HARVEST ---
            for idx in range(num_candidates):
                res_val = resonances[idx].item()
                if res_val > HARVEST_THRESHOLD:
                    print(f"[ZONE B] SPONTANEOUS RESONANCE DETECTED ({res_val:.4f}) on Expert {idx}. Harvesting fragment...")
                    winning_lora = self.orchestrator.lora_managers[idx].lora_A.data.mean(dim=1)
                    top_wave = torch.matmul(winning_lora.to(device), self.router.w_down.weight.T)
                    top_wave = torch.nn.functional.normalize(top_wave, p=2, dim=0)
                    phases = torch.angle(torch.complex(top_wave, torch.zeros_like(top_wave)))
                    top_wave_complex = torch.polar(torch.ones_like(phases), phases)
                    
                    # Save directly to TimescaleDB via orchestrator's save_wave_to_db method
                    fragment_name = f"fragment_{task_dict.get('id', 'task')}_step_{len(tokens_lists[idx])}"
                    self.orchestrator.save_wave_to_db(fragment_name, top_wave_complex, "wosx_spontaneous_sub_axiom")
                    
            # --- THE PRUNE AND CLONE ---
            for dead_idx in bottom_experts:
                dead_idx = dead_idx.item()
                if dead_idx >= num_candidates:
                    continue
                if dead_idx == top_expert_idx:
                    continue
                if resonances[dead_idx].item() < APOPTOSIS_THRESHOLD and active_mask[dead_idx]:
                    print(f"[REALLOCATION] Pruning lagging Expert {dead_idx} (Resonance: {resonances[dead_idx].item():.4f}). Cloning Expert {top_expert_idx}...")
                    
                    # 1. Overwrite the dead expert's LoRA weights with the leader's weights
                    self.orchestrator.lora_managers[dead_idx].lora_A.data.copy_(
                        self.orchestrator.lora_managers[top_expert_idx].lora_A.data
                    )
                    self.orchestrator.lora_managers[dead_idx].lora_B.data.copy_(
                        self.orchestrator.lora_managers[top_expert_idx].lora_B.data
                    )
                    
                    # 2. Inject Gaussian noise
                    noise_A = torch.randn_like(self.orchestrator.lora_managers[dead_idx].lora_A.data) * 0.01
                    self.orchestrator.lora_managers[dead_idx].lora_A.data.add_(noise_A)
                    
                    # 3. Synchronize text and tokens
                    generated_texts[dead_idx] = generated_texts[top_expert_idx]
                    tokens_lists[dead_idx] = list(tokens_lists[top_expert_idx])
                    
                    # 4. Overwrite Vulkan KV-cache slot
                    if not self.orchestrator.is_mock and hasattr(self.orchestrator.gen_model, "llama"):
                        try:
                            import llama_cpp.llama_cpp as lcpp
                            ctx = self.orchestrator.gen_model.llama.ctx
                            lcpp.llama_memory_seq_rm(ctx, dead_idx, 0, -1)
                            lcpp.llama_memory_seq_cp(ctx, top_expert_idx, dead_idx, 0, -1)
                            print(f"[KV CACHE SWAP] Synced Vulkan KV-cache sequence slot {top_expert_idx} -> {dead_idx}")
                        except Exception as kv_err:
                            print(f"[KV CACHE SWAP] Warning: KV-cache sync failed: {kv_err}")
                            
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