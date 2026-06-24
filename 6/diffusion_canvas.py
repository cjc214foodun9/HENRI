import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _safe_core_forward(core_model, canvas, target_goal_wave, t):
    t_val = float(t.mean().item()) if torch.is_tensor(t) else float(t)
    import inspect
    sig = inspect.signature(core_model.forward)
    params = list(sig.parameters.keys())
    num_params = len(params)
    
    if num_params >= 3:
        if target_goal_wave is None:
            target_goal_wave = torch.zeros(canvas.size(0), canvas.size(-1), device=canvas.device, dtype=canvas.dtype)
        else:
            target_goal_wave = target_goal_wave.to(device=canvas.device, dtype=canvas.dtype)
        if target_goal_wave.ndim == 1:
            target_goal_wave = target_goal_wave.unsqueeze(0)
        output = core_model(canvas, target_goal_wave, t_val)
        if isinstance(output, tuple):
            return output[0]
        return output
    else:
        t_tensor = torch.full((canvas.size(0), canvas.size(1), 1), t_val, device=canvas.device, dtype=canvas.dtype)
        try:
            output = core_model(canvas, t_tensor)
        except Exception:
            output = core_model(canvas, t_val)
        if isinstance(output, tuple):
            return output[0]
        return output

class ConsistencyCanvasCrystallizer(nn.Module):
    """
    Collapses the 25-step Euler-Maruyama loop into a 1- or 2-step single-shot pass
    using Continuous-Time Consistency Distillation tailored for the HENRI core.
    """
    def __init__(self, dim=4096, sequence_length=512, epsilon=0.002, T=80.0):
        super().__init__()
        self.dim = dim
        self.seq_len = sequence_length
        self.epsilon = epsilon
        self.T = T
        
        # Hard boundary scaling parameters matching the continuous phase manifolds
        self.sigma_data = 0.5

    def get_boundary_coefficients(self, t):
        """
        Computes c_skip and c_out to guarantee the identity mapping f_theta(x_0, 0) = x_0.
        """
        denom = (t - self.epsilon) ** 2 + self.sigma_data ** 2
        c_skip = self.sigma_data ** 2 / denom
        c_out = (t - self.epsilon) * self.sigma_data / torch.sqrt(denom)
        return c_skip.unsqueeze(-1).unsqueeze(-1), c_out.unsqueeze(-1).unsqueeze(-1)

    def single_shot_crystallize(self, core_model, noisy_canvas, target_goal_wave):
        """
        Executes a 1-step direct mapping from maximum entropy noise (t = T)
        straight back to the low-entropy canonical AST attractor basin.
        """
        device = noisy_canvas.device
        batch_size = noisy_canvas.size(0)
        
        t_max = torch.full((batch_size,), self.T, device=device, dtype=noisy_canvas.dtype)
        c_skip, c_out = self.get_boundary_coefficients(t_max)
        
        with torch.no_grad():
            score_network_output = _safe_core_forward(core_model, noisy_canvas, target_goal_wave, t_max)
            crystallized_manifold = c_skip * noisy_canvas + c_out * score_network_output
            
        return crystallized_manifold

    def two_step_chording_crystallize(self, core_model, noisy_canvas, target_goal_wave, t_mid=20.0):
        """
        Executes a 2-step chording pass. Step 1 leaps from T to t_mid, 
        allowing the out-of-band Sagnac Veto to inject fine-grained localized adjustments
        before Step 2 snaps the manifold directly to the final clean code tokens.
        """
        device = noisy_canvas.device
        batch_size = noisy_canvas.size(0)
        
        # --- STEP 1: Leap from maximum noise (T) to intermediate horizon (t_mid) ---
        t_max = torch.full((batch_size,), self.T, device=device, dtype=noisy_canvas.dtype)
        c_skip_1, c_out_1 = self.get_boundary_coefficients(t_max)
        
        with torch.no_grad():
            score_out_1 = _safe_core_forward(core_model, noisy_canvas, target_goal_wave, t_max)
            x_mid = c_skip_1 * noisy_canvas + c_out_1 * score_out_1
            
            # --- STEP 2: Refinement leap from t_mid down to absolute zero ---
            t_low = torch.full((batch_size,), t_mid, device=device, dtype=noisy_canvas.dtype)
            c_skip_2, c_out_2 = self.get_boundary_coefficients(t_low)
            
            score_out_2 = _safe_core_forward(core_model, x_mid, target_goal_wave, t_low)
            final_crystallized_manifold = c_skip_2 * x_mid + c_out_2 * score_out_2
            
        return final_crystallized_manifold
class TokenLevelFSMDecoderGate(nn.Module):
    """
    Intercepts the language decoding head at the logit layer during crystallization.
    Forces absolute character compliance with the Python AST grammar schema.
    """
    def __init__(self, vocab_size=256):
        super().__init__()
        self.vocab_size = vocab_size

    def enforce_ast_rigidity(self, raw_logits, active_fsm_mask):
        """
        Applies a hard mathematical constraint to the model's prediction distribution.
        
        Args:
            raw_logits: [Batch, Vocab_Size] (Raw tensor outputs from the decoder head)
            active_fsm_mask: [Batch, Vocab_Size] (Binary byte mask from python.gbnf)
        """
        # Convert the active character mask to a strict logit penalty vector
        # Valid tokens (1) receive 0 penalty; Invalid tokens (0) are pushed to negative infinity
        penalty = (1.0 - active_fsm_mask.to(raw_logits.dtype)) * -1e9
        
        # Superimpose the syntax constraint straight onto the hardware register
        bounded_logits = raw_logits + penalty
        return bounded_logits

class NonAutoregressiveCanvasSampler(nn.Module):
    def __init__(self, core_model, translation_head, num_diffusion_steps=2):
        """
        Executes parallel, score-guided relaxation to materialize text blocks all at once.
        
        Args:
            core_model (nn.Module): Your pre-trained parameter unitary base core.
            translation_head (nn.Module): The linear projection matrix mapping 4096-D to vocabulary logits.
            num_diffusion_steps (int): Total number of parallel denoising relaxation steps (O(1) scaling).
        """
        super().__init__()
        self.core = core_model
        self.translation_head = translation_head
        self.N = num_diffusion_steps
        self.hidden_dim = 4096
        
        # Compile blocked indices for vocabulary domain mask exactly once
        vocab_size = self.translation_head.out_features
        mask_tensor = torch.zeros(vocab_size, dtype=torch.float32)
        try:
            import os
            from transformers import GPT2Tokenizer
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_tok_dir = os.path.join(parent_dir, "gpt2_tokenizer_local")
            if os.path.exists(local_tok_dir):
                tokenizer = GPT2Tokenizer.from_pretrained(local_tok_dir)
            else:
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            forbidden_keywords = ["scada", "actuator", "gripper", "torque", "vulkan", "valve", 
                                  "firmware", "reflash", "motor", "fluid", "mixer", "conjugation", 
                                  "pressure", "axis", "hardware", "alleviate"]
            blocked_indices = []
            for idx in range(vocab_size):
                token_str = tokenizer.decode([idx]).lower()
                if any(kw in token_str for kw in forbidden_keywords):
                    blocked_indices.append(idx)
            mask_tensor[blocked_indices] = float('inf')
            print(f"[CANVAS MASK] Compiled {len(blocked_indices)} blocked SCADA/robotics tokens.")
        except Exception as e:
            print(f"[CANVAS MASK] Warning: Failed to compile tokenizer-based mask: {e}")
        
        device = next(self.core.parameters()).device if list(self.core.parameters()) else torch.device("cpu")
        self.register_buffer("blocked_token_mask", mask_tensor.to(device))

    @torch.no_grad()
    def crystallize_motif(self, swarm_trajectory, sequence_length=512, guidance_scale=4.5, winning_jepa_track=None, jl_guard=None, domain_tag=None):
        """
        De-noises a raw high-entropy canvas into a structured, low-entropy English response matrix.
        
        Args:
            swarm_trajectory (Tensor): The optimal geometric path discovered by the 16 swarms [1, 4096].
            sequence_length (int): Fixed spatial token layout limit.
            guidance_scale (float): Strength of the Zone C attractor field injection.
            winning_jepa_track (Tensor, optional): The chronological track of predicted latent states from PEARL MPC.
            jl_guard (nn.Module, optional): The Johnson-Lindenstrauss Guard.
        """
        self.core.eval()
        device = next(self.core.parameters()).device if list(self.core.parameters()) else torch.device("cpu")
        batch_size = 1

        # Adjust dimensions and datatype based on the core model parameters
        model_dtype = torch.float32
        for p in self.core.parameters():
            self.hidden_dim = p.shape[-1]
            model_dtype = p.dtype
            break

        # 1. Instantiate the High-Entropy Staging Canvas (Pure Gaussian Noise)
        # Shape: [1, Sequence_Length, hidden_dim]
        canvas = torch.randn(batch_size, sequence_length, self.hidden_dim, device=device, dtype=model_dtype)
        canvas = F.normalize(canvas, p=2, dim=-1) # Project cleanly onto the hypersphere

        if winning_jepa_track is not None:
            jepa_squeezed = winning_jepa_track.squeeze(0) if winning_jepa_track.ndim == 3 else winning_jepa_track
            if jepa_squeezed.shape[-1] == self.hidden_dim:
                target_goal_wave = jepa_squeezed[-1].unsqueeze(0)
            elif jl_guard is not None:
                W_aligned = jl_guard.W_JL.to(device=device, dtype=model_dtype)
                steering_field_4096 = torch.matmul(jepa_squeezed.to(dtype=model_dtype), W_aligned)
                target_goal_wave = steering_field_4096[-1].unsqueeze(0)
            else:
                target_goal_wave = jepa_squeezed[-1].unsqueeze(0)
        else:
            target_goal_wave = swarm_trajectory.to(device=device, dtype=model_dtype).view(1, self.hidden_dim)

        if target_goal_wave.shape[-1] != self.hidden_dim:
            if target_goal_wave.shape[-1] < self.hidden_dim:
                padded = torch.zeros(target_goal_wave.shape[0], self.hidden_dim, device=device, dtype=model_dtype)
                padded[:, :target_goal_wave.shape[-1]] = target_goal_wave
                target_goal_wave = padded
            else:
                target_goal_wave = target_goal_wave[:, :self.hidden_dim]

        consistency_engine = ConsistencyCanvasCrystallizer(
            dim=self.hidden_dim,
            sequence_length=sequence_length
        ).to(device=device, dtype=model_dtype)
        
        canvas = consistency_engine.two_step_chording_crystallize(
            core_model=self.core,
            noisy_canvas=canvas,
            target_goal_wave=target_goal_wave
        )
        canvas = F.normalize(canvas, p=2, dim=-1)

        print("[+] Canvas relaxation complete. Executing out-of-band Holographic Dictionary Lookup...")

        # 5. Holographic Dictionary Lookup (Platonic Search Paradigm)
        # Location keys generation for unbinding
        generator = torch.Generator(device=device).manual_seed(101)
        phases_keys = (torch.rand(sequence_length, self.hidden_dim, generator=generator, device=device) * 2.0 * math.pi) - math.pi
        location_keys = torch.polar(torch.ones(sequence_length, self.hidden_dim, device=device), phases_keys)

        # Generate deterministic vocabulary waves on CPU to conserve GPU VRAM
        vocab_size = self.translation_head.out_features
        vocab_generator = torch.Generator(device="cpu").manual_seed(202)
        phases_vocab = (torch.rand(vocab_size, self.hidden_dim, generator=vocab_generator, device="cpu") * 2.0 * math.pi) - math.pi
        vocab_waves = torch.polar(torch.ones(vocab_size, self.hidden_dim, device="cpu"), phases_vocab)

        # Map real canvas to complex phase vectors
        canvas_phases = (canvas[0] * 2.0 * math.pi).to(dtype=torch.float32)
        canvas_complex = torch.polar(torch.ones_like(canvas_phases), canvas_phases) # [Sequence_Length, hidden_dim]

        # Bind canvas waves to location keys via circular convolution (element-wise in phase domain)
        bound_waves = canvas_complex * location_keys
        # Superpose into a single joint thought wave
        M_thought = torch.sum(bound_waves, dim=0) # [hidden_dim]
        # Normalize joint wave to unit magnitude
        M_thought = M_thought / (torch.abs(M_thought) + 1e-8)

        # Unbind and clean up all sequence positions in parallel on the CPU
        # Force complex64 (float32 real/imag) representation to ensure CPU matmul compatibility
        Phi_retrieved_all = (M_thought.unsqueeze(0) * torch.conj(location_keys)).to(device="cpu", dtype=torch.complex64)
        vocab_waves_cpu = vocab_waves.to(torch.complex64)
        
        # Parallel inner-product similarity (resonance) over all vocabulary elements
        similarity_all = torch.matmul(Phi_retrieved_all, torch.conj(vocab_waves_cpu).t()) # [Sequence_Length, vocab_size]
        resonance_all = torch.abs(similarity_all)
        
        # Modern Hopfield Network energy interaction
        energy_all = -torch.exp(resonance_all / math.sqrt(self.hidden_dim))
        
        # Apply strict semantic projection mask to block SCADA/robotics subwords
        energy_all = energy_all.to(device)
        if domain_tag == "ARC_Task":
            energy_all = energy_all + self.blocked_token_mask.to(device)

        # Token-by-token crystallization with FSM grammar masking
        winning_tokens = []
        open_quote = None  # None, or character code
        escape_active = False
        brace_stack = []  # List of open brace char codes
        
        # Mapping close braces to open braces
        brace_map = {41: 40, 93: 91, 125: 123}
        open_braces = {40, 91, 123}
        close_braces = {41, 93, 125}
        
        # Strict Python AST/GBNF character whitelist outside strings
        python_fsm_whitelist = set(range(97, 123))  # a-z
        python_fsm_whitelist.update(range(65, 91))   # A-Z
        python_fsm_whitelist.update(range(48, 58))   # 0-9
        python_fsm_whitelist.update([
            95,  # _
            32,  # space
            10,  # \n
            13,  # \r
            40, 41,  # ( )
            91, 93,  # [ ]
            123, 125, # { }
            58,  # :
            44,  # ,
            46,  # .
            61,  # =
            43,  # +
            45,  # -
            42,  # *
            47,  # /
            37,  # %
            60, 62,  # < >
            33,  # !
            34, 39  # " '
        ])
        
        valid_chars_base = torch.zeros(256, dtype=torch.bool, device=device)
        for c in python_fsm_whitelist:
            valid_chars_base[c] = True
            
        decoder_gate = TokenLevelFSMDecoderGate(vocab_size=vocab_size).to(device)
                
        for i in range(sequence_length):
            if open_quote is not None:
                # Inside string literal: allow all printable ASCII, disallow newlines unless escaped
                valid_chars = torch.ones(256, dtype=torch.bool, device=device)
                for c in range(256):
                    if not (32 <= c <= 126 or c in (9, 10, 13)):
                        valid_chars[c] = False
                if not escape_active:
                    valid_chars[10] = False
                    valid_chars[13] = False
                
                # If we are at the last token, force close the quote
                remaining_tokens = sequence_length - 1 - i
                if remaining_tokens == 0:
                    valid_chars[:] = False
                    valid_chars[open_quote] = True
            else:
                # Outside string literal: use strict whitelist
                valid_chars = valid_chars_base.clone()
                
                # Cannot close a brace that doesn't match the top of the stack
                for close_c, open_c in brace_map.items():
                    if not brace_stack or brace_stack[-1] != open_c:
                        valid_chars[close_c] = False
                
                # If running out of tokens, force close open braces
                remaining_tokens = sequence_length - 1 - i
                if remaining_tokens < len(brace_stack):
                    target_close = None
                    for close_c, open_c in brace_map.items():
                        if open_c == brace_stack[-1]:
                            target_close = close_c
                            break
                    if target_close is not None:
                        valid_chars[:] = False
                        valid_chars[target_close] = True

            # Project 256-D mask to vocab_size via repeating/indexing
            repeats = (vocab_size + 255) // 256
            mask_vocab = valid_chars.repeat(repeats)[:vocab_size]
            
            # Format active_fsm_mask for the gate (1.0 for valid, 0.0 for invalid)
            active_fsm_mask = mask_vocab.to(dtype=torch.float32).unsqueeze(0)
            
            # Retrieve logits (negative of energy) at position i
            # raw_logits shape: [1, vocab_size]
            raw_logits = -energy_all[i].unsqueeze(0)
            
            # Apply the FSM gate constraint
            bounded_logits = decoder_gate.enforce_ast_rigidity(raw_logits, active_fsm_mask)
            
            # Select token with the highest bounded logit (argmax)
            selected_token = torch.argmax(bounded_logits).item()
            winning_tokens.append(selected_token)
            
            # Update FSM state based on the chosen character
            chosen_char = selected_token % 256
            if open_quote is not None:
                if escape_active:
                    escape_active = False
                else:
                    if chosen_char == 92:  # Backslash '\'
                        escape_active = True
                    elif chosen_char == open_quote:
                        open_quote = None  # Closed
            else:
                if chosen_char in (34, 39):  # " or '
                    open_quote = chosen_char
                elif chosen_char in open_braces:
                    brace_stack.append(chosen_char)
                elif chosen_char in close_braces:
                    if brace_stack and brace_stack[-1] == brace_map[chosen_char]:
                        brace_stack.pop()
                        
        winning_tokens_tensor = torch.tensor(winning_tokens, dtype=torch.long, device=device)
        return winning_tokens_tensor.unsqueeze(0)

class BirkhoffTopologicalLoss(nn.Module):  
    def __init__(self, translation_head, alpha=1.0, beta=0.05, eta=0.1):  
        """  
        Orchestrates non-autoregressive score matching with aesthetic constraints.  
          
        Args:  
            translation_head (nn.Module): Linear projection mapping [B, L, hidden_dim] to [B, L, Vocabulary_Size].  
            alpha (float): Scaling coefficient for denoising score matching.  
            beta (float): Scaling coefficient for Entropy Minimization (Complexity 'C').  
            eta (float): Scaling coefficient for Trajectory Smoothness (Order 'O').  
        """  
        super().__init__()  
        self.translation_head = translation_head  
        self.alpha = alpha  
        self.beta = beta  
        self.eta = eta

    def forward(self, pred_score, target_score, canvas_state):  
        """  
        Evaluates the thermodynamic structural constraints of the active thought-wave.  
          
        Args:  
            pred_score (Tensor): Predicted score vector field from the unitary layers [B, L, hidden_dim].  
            target_score (Tensor): Target score vector field (the true injected noise gradient) [B, L, hidden_dim].  
            canvas_state (Tensor): The active denoising canvas matrix [B, L, hidden_dim].  
              
        Returns:  
            total_loss (Tensor): Differentiable scalar combining score alignment and Birkhoff constraints.  
            metrics (dict): Un-detached telemetry floats for logging.  
        """  
        # 1. Denoising Score Matching Loss (Base Alignment)  
        loss_score = F.mse_loss(pred_score, target_score, reduction='mean')

        # Project the continuous latent canvas to the discrete token vocabulary space  
        logits = self.translation_head(canvas_state)  
        probs = F.softmax(logits, dim=-1)

        # 2. Entropy Minimization (Complexity Control 'C')  
        # Shannon Entropy across the vocabulary axis  
        epsilon = 1e-9  
        entropy_per_token = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)  
        loss_entropy_C = torch.mean(entropy_per_token)

        # 3. Structural Geodesic Order (Order Control 'O')  
        # We compute the first-order discrete spatial derivative along the sequence axis (L).  
        trajectory_delta = canvas_state[:, 1:, :] - canvas_state[:, :-1, :]  
        loss_roughness_TV = torch.mean(torch.abs(trajectory_delta))

        # 4. Synthesizing the Birkhoff Objective Matrix  
        total_loss = (self.alpha * loss_score) + (self.beta * loss_entropy_C) + (self.eta * loss_roughness_TV)

        metrics = {  
            "loss_score_mse": loss_score.item(),  
            "complexity_entropy_C": loss_entropy_C.item(),  
            "roughness_TV_O": loss_roughness_TV.item(),  
            "birkhoff_measure_estimate": (1.0 / (loss_entropy_C.item() + loss_roughness_TV.item() + epsilon))  
        }

        return total_loss, metrics
