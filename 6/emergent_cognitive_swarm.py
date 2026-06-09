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

    def generate_swarm_hypothesis(self, task_dict, playbook_wave, playbook_dict=None):
        """
        Computes routing weights, initializes in-memory logit bias, and generates prompt response.
        """
        # Compute the explicit alpha routing weights based on the compiled playbook wave
        if playbook_wave is not None:
            alpha_routing = self.router.compute_routing_weights(playbook_wave.unsqueeze(0), temperature=0.7).squeeze(0)
        else:
            alpha_routing = torch.ones(self.orchestrator.num_streams) / self.orchestrator.num_streams
            
        prompt = self.build_clean_prompt(task_dict, playbook_dict)
        
        # Build the dynamic logit bias array entirely in-memory using the alpha weights
        # This completely avoids writing blended LoRA adapters to disk mid-generation
        logit_processor = self.orchestrate_in_memory_moe_steering(alpha_routing, prompt)
        
        # Execute the accelerated Vulkan token generation pass
        res = self.llama(
            prompt,
            logits_processor=logit_processor,
            max_tokens=2048,
            stop=["<turn|>", "<|turn>"]
        )
        
        if isinstance(res, dict):
            return res["choices"][0]["text"].strip()
        return str(res).strip()

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
            "- NumPy is strictly forbidden. You must use PyTorch (torch).\n"
            "- PATH A (Rigid Geometry): If the task is a discrete bounding box crop, flip, or translation, use standard PyTorch tensor slicing.\n"
            "- PATH B (Complex Emergence): If the task requires fuzzy pattern completion or non-rigid emergence, translate the grid to S^1 wave phases and use the EmergentManifold.\n"
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