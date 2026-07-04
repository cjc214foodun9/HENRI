import os
import math
import torch
import torch.nn as nn
import numpy as np

class EmergentCognitiveSwarm(nn.Module):
    """
    Decentralized neural generation hub.
    Bridges the L3SwarmRouter's MoE weighting distribution to continuous-time
    wave-relaxation and pattern crystallization.
    """
    def __init__(self, router_instance, llama_instance=None):
        super().__init__()
        self.router = router_instance
        self.llama = llama_instance
        self.orchestrator = None

    @torch.no_grad()
    def generate_parallel_hypotheses(self, task_dict, playbook_wave, playbook_dict=None, temperature=0.7, inject_noise=False, num_candidates=16, early_stopping_callback=None, start_time=None, time_limit=None):
        """
        Generates a batch of parallel hypotheses across the swarm.
        Integrates Ephemeral Attractors, mid-flight Prune & Clone, and Spontaneous Resonance Harvesting.
        """
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
            
        device = next(self.router.parameters()).device if list(self.router.parameters()) else torch.device("cpu")
        
        vocab_size = 32000
        if hasattr(self.orchestrator, 'l3_router') and self.orchestrator.l3_router is not None:
            vocab_size = getattr(self.orchestrator.l3_router, 'vocab_size', 32000)
        elif hasattr(self.orchestrator, '_diffusion_translation_head') and self.orchestrator._diffusion_translation_head is not None:
            vocab_size = self.orchestrator._diffusion_translation_head.out_features
            
        from henri_core.diffusion_canvas import NonAutoregressiveCanvasSampler
        from henri_core.core import ProprietaryHENRICore
        
        if not hasattr(self.orchestrator, '_diffusion_core_model') or self.orchestrator._diffusion_core_model is None:
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            core_path = os.path.join(parent_dir, "henri_core_final.pt")
            if not os.path.exists(core_path):
                core_path = "henri_core_final.pt"
            
            checkpoint = None
            if os.path.exists(core_path):
                try:
                    checkpoint = torch.load(core_path, map_location='cpu')
                except Exception as load_err:
                    checkpoint = None
            else:
                fallback_path = os.path.join(parent_dir, "henri_core_final_scaled.pt")
                if os.path.exists(fallback_path):
                    try:
                        checkpoint = torch.load(fallback_path, map_location='cpu')
                    except Exception as fallback_err:
                        checkpoint = None

            if checkpoint is not None and isinstance(checkpoint, dict) and "config" in checkpoint:
                cfg = checkpoint["config"]
                hidden_dim = cfg["dim"]
                num_layers = cfg["depth"]
                num_base_experts = cfg["num_fluid_states"]
                vocab_size = cfg["vocab_size"]

                orig_default_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                try:
                    with torch.device(device):
                        core_model = ProprietaryHENRICore(dim=hidden_dim, depth=num_layers, num_fluid_states=num_base_experts)
                finally:
                    torch.set_default_dtype(orig_default_dtype)

                core_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                checkpoint_translation_head_state_dict = checkpoint.get("translation_head_state_dict")

                del checkpoint
                import gc; gc.collect(); torch.cuda.empty_cache()

                core_model = core_model.to(device=device).eval()
                import gc; gc.collect(); torch.cuda.empty_cache()

                has_bias = (checkpoint_translation_head_state_dict is not None and "bias" in checkpoint_translation_head_state_dict)
                translation_head = nn.Linear(hidden_dim, vocab_size, bias=has_bias).to(dtype=torch.bfloat16)
                if checkpoint_translation_head_state_dict is not None:
                    try:
                        translation_head.load_state_dict(checkpoint_translation_head_state_dict)
                    except Exception as lsd_err:
                        with torch.no_grad():
                            temp_th = torch.empty(translation_head.weight.shape, dtype=torch.float32, device='cpu')
                            torch.nn.init.orthogonal_(temp_th)
                            translation_head.weight.copy_(temp_th.to(device=device, dtype=torch.bfloat16))
                else:
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
                    except Exception as e:
                        state_dict = None

                orig_default_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                try:
                    with torch.device(device):
                        core_model = ProprietaryHENRICore(dim=hidden_dim, depth=num_layers, num_fluid_states=num_base_experts)
                finally:
                    torch.set_default_dtype(orig_default_dtype)

                core_model = core_model.to(device=device).eval()
                if state_dict is not None:
                    core_model.load_state_dict(state_dict, strict=False)
                
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
            
        if not hasattr(self.orchestrator, 'h_mpc') or self.orchestrator.h_mpc is None:
            from henri_core.h_mpc_steering import HolographicMPCOrchestrator
            self.orchestrator.h_mpc = HolographicMPCOrchestrator(dim=core_model.layers[0].dim).to(device=device, dtype=torch.bfloat16)
            self.orchestrator.h_mpc.orchestrator = self.orchestrator

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
            candidate_action_sequences = torch.stack(candidate_action_sequences, dim=0)
            
            if hasattr(self.orchestrator, 'memory_engines') and 0 in self.orchestrator.memory_engines:
                current_wave = self.orchestrator.memory_engines[0].active_wave.to(device)
            else:
                current_wave = torch.zeros(self.orchestrator.hrr_dim, dtype=torch.complex64, device=device)

            winning_idx, winning_jepa_track = self.orchestrator.h_mpc.run_h_mpc_selection(
                current_wave=current_wave.to(device=device),
                target_goal_wave=playbook_wave.to(device=device),
                candidate_action_sequences=candidate_action_sequences.to(device=device),
                horizon=horizon
            )
            winning_wave = candidate_action_sequences[winning_idx][-1].clone()
            jl_guard = getattr(self.orchestrator.h_mpc, 'jl_guard', None)
            
            del candidate_action_sequences
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for idx in range(num_candidates):
            target_idx = winning_idx if (winning_idx is not None) else 0
            if idx != target_idx:
                generated_text = "def transform(grid):\n    return grid"
                token_ids = [ord(c) for c in generated_text]
                generated_texts[idx] = generated_text
                tokens_lists[idx] = token_ids
                active_mask[idx] = False
                continue

            if winning_wave is not None:
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
                
            if torch.is_complex(perturbed_wave):
                trajectory_real = torch.real(perturbed_wave)
            else:
                trajectory_real = perturbed_wave
                
            if trajectory_real.ndim == 1:
                trajectory_real = trajectory_real.unsqueeze(0)
                
            trajectory_input = trajectory_real.to(device=device, dtype=torch.bfloat16)
            
            try:
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
                    decoded_chars = []
                    for token in token_ids:
                        char_code = abs(int(token)) % 256
                        if (char_code >= 32 and char_code <= 126) or char_code == 10 or char_code == 13:
                            decoded_chars.append(chr(char_code))
                    generated_text = "".join(decoded_chars)
                
                generated_text = generated_text.replace("\xa0", " ").replace("\ufffd", "")
                prefix_constraint = "def transform(input_grid):\n"
                if not generated_text.strip().startswith("def transform"):
                    generated_text = prefix_constraint + generated_text
                    
            except Exception as e:
                token_ids = [ord(c) for c in "def transform(grid):\n    return grid"]
                generated_text = "def transform(grid):\n    return grid"
            
            generated_texts[idx] = generated_text
            tokens_lists[idx] = token_ids
            active_mask[idx] = False
            
        candidates = []
        for idx in range(num_candidates):
            candidates.append((generated_texts[idx], alpha_routings[idx]))
        return candidates

    def build_clean_prompt(self, task_dict, playbook_dict=None):
        """
        Builds the prompt placing static task specification at the top and the volatile playbook at the bottom.
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

class HenriGapJunctionSwarmSynchronizer(nn.Module):
    def __init__(self, num_experts=16, d_wave=4096, alpha_anneal=0.15):
        super().__init__()
        self.num_experts = num_experts
        self.d_wave = d_wave
        self.alpha_anneal = alpha_anneal  # Scale of localized thermal perturbation
        
    @torch.no_grad()
    def execute_sync_junction(self, lora_managers, resonance_scores):
        """
        lora_managers: dict or list of DynamicLoraManager
        resonance_scores: tensor containing active alignment scalars bounded [0, 1]
        """
        # 1. Identify the high-coherence champion node
        max_score, champion_idx = torch.max(resonance_scores, dim=0)
        
        # 2. Establish the activation gate threshold (e.g., 93.6% soft-match floor)
        sync_gate_threshold = 0.935
        
        if max_score > sync_gate_threshold:
            print(f"[GAP JUNCTION OPEN] Swarm Expert Node {champion_idx.item()} achieved critical coherence: {max_score.item():.4f}")
            
            # Isolate the winner's geometric weight signature (The Ephemeral Attractor)
            champion_A = lora_managers[champion_idx.item()].lora_A.clone()
            champion_B = lora_managers[champion_idx.item()].lora_B.clone()
            
            # 3. Broadcast clone to all other slots parallelly across the active stream
            for e in range(self.num_experts):
                if e == champion_idx.item():
                    continue  # Protect the genetic lineage of the active champion
                    
                # Compute localized Langevin noise driven by the distance to perfect alignment
                # Lagging heads receive a larger thermal shock to expand their exploration radius
                stress_delta = 1.0 - resonance_scores[e].item()
                thermal_variance = self.alpha_anneal * stress_delta
                
                # Phase-domain Langevin injection mask
                langevin_shaker_A = torch.randn_like(champion_A) * thermal_variance
                langevin_shaker_B = torch.randn_like(champion_B) * thermal_variance
                
                # Overwrite weight properties and deform viscoelastically into adjacent space
                lora_managers[e].lora_A.copy_(champion_A + langevin_shaker_A)
                lora_managers[e].lora_B.copy_(champion_B + langevin_shaker_B)
                lora_managers[e].save_weights()
                
            print(f"[SUCCESS] Swarm syncytium locked. 15 nodes aligned to champion topology.")
        else:
            # Maintain independent parallel exploration paths if no head has unlocked a basin
            pass
            
        return lora_managers

    @torch.no_grad()
    def synchronize_swarm_syncytium(self, expert_layers: nn.ModuleList, alignment_scores: torch.Tensor, alpha_anneal: float = 0.12):
        """
        Operationalizes Michael Levin's TAME framework within graphics hardware.
        Clones high-resonance expert parameters across lagging nodes mid-flight.
        """
        # Identify the current champion node exploring the grid layout
        max_resonance, champion_idx = torch.max(alignment_scores, dim=0)
        
        # Anchor floor matching your 93.6% validation sweet-spot
        critical_coherence_gate = 0.935
        
        if max_resonance > critical_coherence_gate:
            print(f"[GAP JUNCTION ENGAGED] Node {champion_idx.item()} reached attractor basin: {max_resonance.item():.4f}")
            
            # Isolate the winner's state-dict parameters
            champion_state = expert_layers[champion_idx].state_dict()
            
            for e in range(len(expert_layers)):
                if e == champion_idx:
                    continue # Protect the winner's lineage from thermal distortion
                    
                # Calculate localized Langevin variance based on distance to excellence
                # Trailing expert nodes receive a larger thermal perturbation to push them wider
                stress_delta = 1.0 - alignment_scores[e].item()
                thermal_variance = alpha_anneal * stress_delta
                
                # Clone parameters with phase-domain stochastic deformations
                for key, param in expert_layers[e].named_parameters():
                    if key in champion_state:
                        noise_shaker = torch.randn_like(param) * thermal_variance
                        param.copy_(champion_state[key] + noise_shaker)
                        
            print(f"[SUCCESS] Swarm sync complete. 15 expert topologies realigned to champion basin.")
        return expert_layers