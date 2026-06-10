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

    def generate_parallel_hypotheses(self, task_dict, playbook_wave, playbook_dict=None, temperature=0.7, inject_noise=False, num_candidates=16, early_stopping_callback=None):
        """
        Generates a batch of parallel hypotheses across the swarm sequentially.
        We perturb the playbook wave slightly for each stream to explore alternative trajectories.
        """
        candidates = []
        for i in range(num_candidates):
            if playbook_wave is not None:
                # Perturb the playbook wave slightly for each candidate stream
                perturbed_wave = playbook_wave.clone()
                noise_phase = torch.randn_like(perturbed_wave.real) * 0.1
                # Multiply by e^(i * noise_phase) to perturb the phase wavefront
                perturbed_wave = perturbed_wave * torch.polar(torch.ones_like(perturbed_wave.real), noise_phase)
            else:
                perturbed_wave = None
                
            stream_inject_noise = inject_noise or (i > 0)
            
            if early_stopping_callback is not None:
                stream_cb = lambda t, a, idx=i: early_stopping_callback(t, a, idx)
            else:
                stream_cb = None
            
            print(f"[SWARM] Generating candidate {i+1}/{num_candidates} (temperature={temperature:.2f}, noise={stream_inject_noise})...")
            try:
                candidate, alpha_routing = self.generate_swarm_hypothesis(
                    task_dict=task_dict,
                    playbook_wave=perturbed_wave,
                    playbook_dict=playbook_dict,
                    temperature=temperature,
                    inject_noise=stream_inject_noise,
                    early_stopping_callback=stream_cb
                )
                candidates.append((candidate, alpha_routing))
            except Exception as e:
                print(f"[SWARM] Exception during generation: {e}")
            
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