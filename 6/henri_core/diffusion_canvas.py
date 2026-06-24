"""  
Project HENRI: Dynamic Intent-Driven Cosinespace Diffusion Canvas  
Component: Phase 4 Scaled Non-Autoregressive Global Latent Relaxation Core  
Author: Joseph Valentine (Bespoke Silicon Photonic Architecture Core)  
Date: 2026-06-23  
"""

import os  
import math  
import torch  
import torch.nn as nn  
import torch.nn.functional as F

class NonAutoregressiveCanvasSampler(nn.Module):  
    """  
    Executes parallel, score-guided reverse SDE relaxation loops over unrolled sequences.  
    Dynamically swaps vocabulary mask matrices based on active runtime intent flags.  
    """  
    def __init__(self, core_model: nn.Module, translation_head: nn.Module, num_diffusion_steps: int = 25):  
        super().__init__()  
        self.core = core_model  
        self.translation_head = translation_head  
        self.N = num_diffusion_steps  
        self.hidden_dim = 4096  
          
        vocab_size = self.translation_head.out_features  
          
        # 1. Compile Invariant Domain Masks  
        mask_scada = torch.zeros(vocab_size, dtype=torch.float32)  
        mask_code = torch.zeros(vocab_size, dtype=torch.float32)  
          
        try:  
            from transformers import GPT2Tokenizer  
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
            local_tok_dir = os.path.join(parent_dir, "gpt2_tokenizer_local")  
            tokenizer = GPT2Tokenizer.from_pretrained(local_tok_dir) if os.path.exists(local_tok_dir) else GPT2Tokenizer.from_pretrained('gpt2')  
              
            # Scada / Robotics Crosstalk Terms  
            forbidden_keywords = ["scada", "actuator", "gripper", "torque", "vulkan", "valve",   
                                  "firmware", "reflash", "motor", "fluid", "mixer", "conjugation",   
                                  "pressure", "axis", "hardware", "alleviate"]  
              
            # Non-Python Structural Character Noise Filter  
            python_keywords = ["def", "return", "import", "from", "for", "if", "else", "elif", "while", "try", "except", "with", "as", "pass", "in", "not", "and", "or", "is", "lambda", "class"]  
              
            for idx in range(vocab_size):  
                token_str = tokenizer.decode([idx]).lower()  
                # Populate SCADA block walls  
                if any(kw in token_str for kw in forbidden_keywords):  
                    mask_scada[idx] = float('inf')  
                  
                # Populate Non-Python constraints for strict code-generation blocks  
                # Blocks noisy punctuation fragments that break the abstract syntax tree  
                if not any(token_str.strip().startswith(kw) or token_str.strip() in ["", "(", ")", "[", "]", "{", "}", ":", ",", "=", "+", "-", "*", "/", "_", ".", "\n"] for kw in python_keywords):  
                    if len(token_str.strip()) > 3 and not token_str.strip().isidentifier():  
                        mask_code[idx] = 1000.0  # Apply heavy penalization energy barrier  
                          
            print(f"[CANVAS MASK] Dynamic Matrix Compiled. SCADA Bounds: {torch.isinf(mask_scada).sum().item()} | Code Constraints: {(mask_code > 0).sum().item()}")  
        except Exception as e:  
            print(f"[CANVAS MASK] Warning: Tokenizer compilation fallback: {e}")  
              
        device = next(self.core.parameters()).device if list(self.core.parameters()) else torch.device("cpu")  
        self.register_buffer("scada_robotics_mask", mask_scada.to(device))  
        self.register_buffer("strict_python_mask", mask_code.to(device))  
        self.register_buffer("open_conversation_mask", torch.zeros(vocab_size, device=device))

    @torch.no_grad()  
    def crystallize_motif(self, swarm_trajectory: torch.Tensor, sequence_length: int = 512,   
                           guidance_scale: float = 4.5, winning_jepa_track: torch.Tensor = None,   
                           jl_guard: nn.Module = None, domain_tag: str = None, intent_flag: str = "CONVERSATION") -> torch.Tensor:  
        """  
        Denoises a raw random phase canvas globally into structured token configurations.  
        Utilizes intent_flag ("CONVERSATION", "CODE", "RESEARCH") to select vocabulary masks.  
        """  
        self.core.eval()  
        device = next(self.core.parameters()).device if list(self.core.parameters()) else torch.device("cpu")  
          
        model_dtype = torch.float32  
        for p in self.core.parameters():  
            self.hidden_dim = p.shape[-1]  
            model_dtype = p.dtype  
            break

        # Inwardly resolve backward-compatible parameters to intent keys
        if domain_tag == "ARC_Task":
            intent_flag = "CODE"
        elif domain_tag is not None and "research" in domain_tag.lower():
            intent_flag = "RESEARCH"

        # Instantiate Staging Canvas on the complex unit hypersphere  
        canvas = torch.randn(1, sequence_length, self.hidden_dim, device=device, dtype=model_dtype)  
        canvas = F.normalize(canvas, p=2, dim=-1)

        timesteps = torch.linspace(1.0, 0.001, self.N, device=device, dtype=model_dtype)  
        dt = 1.0 / self.N

        if winning_jepa_track is not None and jl_guard is not None:  
            W_aligned = jl_guard.W_JL.to(device=device, dtype=model_dtype)  
            steering_field = torch.matmul(winning_jepa_track.squeeze(0).to(dtype=model_dtype), W_aligned)  
            horizon_steps = steering_field.size(0)  
        else:  
            steering_field = None

        clean_trajectory_fallback = swarm_trajectory.to(device=device, dtype=model_dtype).view(1, self.hidden_dim)

        # Reverse SDE Relaxation Loop  
        for step_idx, t in enumerate(timesteps):  
            t_tensor = torch.full((1, sequence_length, 1), t, device=device, dtype=model_dtype)  
              
            if steering_field is not None:  
                track_idx = min(int((step_idx / self.N) * horizon_steps), horizon_steps - 1)  
                active_steer = steering_field[track_idx]  
                if active_steer.shape[-1] != self.hidden_dim:  
                    active_steer = F.pad(active_steer, (0, self.hidden_dim - active_steer.shape[-1]))[:self.hidden_dim]  
            else:  
                active_steer = clean_trajectory_fallback.squeeze(0)

            if hasattr(self.core, 'layers'):  
                predicted_noise, _ = self.core(canvas, active_steer.unsqueeze(0), float(t))  
            else:  
                predicted_noise = self.core(canvas, t_tensor)

            trajectory_guidance = active_steer.view(1, 1, self.hidden_dim).expand_as(canvas)  
            total_score_direction = predicted_noise + (guidance_scale * trajectory_guidance)

            canvas = canvas - (total_score_direction * dt)  
              
            if t > 0.1:  
                canvas += torch.randn_like(canvas) * (t * 0.001)  
              
            canvas = F.normalize(canvas, p=2, dim=-1)

        # Out-of-Band Holographic Spatial-Spectral Key Unbinding  
        generator = torch.Generator(device=device).manual_seed(101)  
        phases_keys = (torch.rand(sequence_length, self.hidden_dim, generator=generator, device=device) * 2.0 * math.pi) - math.pi  
        location_keys = torch.polar(torch.ones(sequence_length, self.hidden_dim, device=device), phases_keys)

        vocab_size = self.translation_head.out_features  
        vocab_generator = torch.Generator(device="cpu").manual_seed(202)  
        phases_vocab = (torch.rand(vocab_size, self.hidden_dim, generator=vocab_generator, device="cpu") * 2.0 * math.pi) - math.pi  
        vocab_waves = torch.polar(torch.ones(vocab_size, self.hidden_dim, device="cpu"), phases_vocab).to(device=device, dtype=torch.complex64)

        canvas_phases = (canvas[0] * 2.0 * math.pi).to(dtype=torch.float32)  
        canvas_complex = torch.polar(torch.ones_like(canvas_phases), canvas_phases)  
          
        bound_waves = canvas_complex * location_keys  
        M_thought = torch.sum(bound_waves, dim=0)  
        M_thought = M_thought / (torch.abs(M_thought) + 1e-8)

        Phi_retrieved_all = (M_thought.unsqueeze(0) * torch.conj(location_keys)).to(dtype=torch.complex64)  
        similarity_all = torch.matmul(Phi_retrieved_all, torch.conj(vocab_waves).t())  
        resonance_all = torch.abs(similarity_all)  
          
        energy_all = -torch.exp(resonance_all / math.sqrt(self.hidden_dim))  
          
        # 2. Dynamic Intent-Driven Energy Mask Selection Gate  
        if intent_flag == "CODE":  
            energy_all = energy_all + self.strict_python_mask.view(1, vocab_size)  
        elif intent_flag == "RESEARCH":  
            energy_all = energy_all + self.scada_robotics_mask.view(1, vocab_size)  
        else:  # CONVERSATION  
            energy_all = energy_all + self.open_conversation_mask.view(1, vocab_size)

        # Token-by-token crystallization with FSM grammar masking
        winning_tokens = []
        open_quote = None  # None, or character code
        escape_active = False
        brace_stack = []  # List of open brace char codes
        
        # Mapping close braces to open braces
        brace_map = {41: 40, 93: 91, 125: 123}
        open_braces = {40, 91, 123}
        close_braces = {41, 93, 125}
        
        # Precompute constants to avoid device transfers in the loop
        valid_chars_base = torch.ones(256, dtype=torch.bool, device=device)
        for c in range(256):
            if not (32 <= c <= 126 or c in (9, 10, 13)):
                valid_chars_base[c] = False
                
        for i in range(sequence_length):
            valid_chars = valid_chars_base.clone()
            
            # Apply FSM constraints to the 256 possible ASCII byte values
            if open_quote is not None:
                # Inside string literal: disallow newlines unless escaped
                if not escape_active:
                    valid_chars[10] = False
                    valid_chars[13] = False
                
                # If we are at the last token, we must force-close the open quote
                remaining_tokens = sequence_length - 1 - i
                if remaining_tokens == 0:
                    valid_chars[:] = False
                    valid_chars[open_quote] = True
            else:
                # Outside a string literal
                # Cannot close a brace that doesn't match the top of the stack
                for close_c, open_c in brace_map.items():
                    if not brace_stack or brace_stack[-1] != open_c:
                        valid_chars[close_c] = False
                
                # If running out of tokens, force close the open braces in order
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
            
            # Retrieve energy at position i
            energy_i = energy_all[i].clone()
            energy_i[~mask_vocab] = float('inf')
            
            selected_token = torch.argmin(energy_i).item()
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
    def __init__(self, translation_head: nn.Module, alpha: float = 1.0, beta: float = 0.05, eta: float = 0.1):    
        super().__init__()    
        self.translation_head = translation_head    
        self.alpha = alpha    
        self.beta = beta    
        self.eta = eta

    def forward(self, pred_score: torch.Tensor, target_score: torch.Tensor, canvas_state: torch.Tensor) -> tuple:    
        loss_score = F.mse_loss(pred_score, target_score, reduction='mean')  
        logits = self.translation_head(canvas_state)    
        probs = F.softmax(logits, dim=-1)

        epsilon = 1e-9    
        entropy_per_token = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)    
        loss_entropy_C = torch.mean(entropy_per_token)

        trajectory_delta = canvas_state[:, 1:, :] - canvas_state[:, :-1, :]    
        loss_roughness_TV = torch.mean(torch.abs(trajectory_delta))

        total_loss = (self.alpha * loss_score) + (self.beta * loss_entropy_C) + (self.eta * loss_roughness_TV)  
        return total_loss, {"loss_score_mse": loss_score.item(), "complexity_entropy_C": loss_entropy_C.item(), "roughness_TV_O": loss_roughness_TV.item(), "birkhoff_measure_estimate": (1.0 / (loss_entropy_C.item() + loss_roughness_TV.item() + epsilon))}

def run_phase_4_validation():  
    print("=== INITIALIZING HENRI PHASE 4: DIFFUSION CANVAS VALIDATION ===")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"[BOOT] Target accelerator environment initialized: {device}")

    # 1. Instantiate Mock Modules and Heads  
    class MockCore(nn.Module):  
        def __init__(self, dim=4096):  
            super().__init__()  
            self.param = nn.Parameter(torch.randn(1, dim))  
        def forward(self, canvas, t_tensor):  
            return torch.zeros_like(canvas)

    class MockJLGuard(nn.Module):  
        def __init__(self, latent=128, global_dim=4096):  
            super().__init__()  
            self.W_JL = nn.Parameter(torch.randn(latent, global_dim))

    core_model = MockCore(dim=4096).to(device)  
    translation_head = nn.Linear(4096, 50257, bias=False).to(device) # GPT-2 size  
    jl_guard = MockJLGuard(latent=128, global_dim=4096).to(device)

    sampler = NonAutoregressiveCanvasSampler(core_model, translation_head, num_diffusion_steps=5).to(device)  
    criterion = BirkhoffTopologicalLoss(translation_head, alpha=1.0, beta=0.05).to(device)  
    print("[SUCCESS] Production modules initialized and mapped to GPU registers.")

    # 2. Test Single-Pass Canvas Relaxation using a 16-step Lookahead Matrix  
    mock_trajectory = torch.randn(1, 4096, device=device)  
    mock_jepa_track = torch.randn(1, 16, 128, device=device) # 16 planning steps  
      
    print("[DATA INFRASTRUCTURE] Launching global cosinespace relaxation check over 128 tokens...")  
    winning_tokens = sampler.crystallize_motif(  
        swarm_trajectory=mock_trajectory,  
        sequence_length=128,  
        winning_jepa_track=mock_jepa_track,  
        jl_guard=jl_guard,  
        domain_tag="ARC_Task"  
    )

    print(f"[MANIFOLD] Crystallized token matrix shape footprint: {winning_tokens.shape}")  
    assert winning_tokens.shape == torch.Size([1, 128]), "Fatal: Canvas relaxation altered sequence dimensions!"  
    print("[SUCCESS] Parallel relaxation completed without structural layout drift.")

    # 3. Evaluate Birkhoff Loss Matrix Gradient Flow  
    mock_pred = torch.randn(2, 64, 4096, device=device, requires_grad=True)  
    mock_target = torch.randn(2, 64, 4096, device=device)  
    mock_canvas = torch.randn(2, 64, 4096, device=device)

    total_loss, metrics = criterion(mock_pred, mock_target, mock_canvas)  
    total_loss.backward()  
      
    print(f"[METRIC MONITOR] Denoising Score MSE: {metrics['loss_score_mse']:.4f}")  
    print(f"[METRIC MONITOR] Complexity Entropy 'C': {metrics['complexity_entropy_C']:.4f}")  
    print(f"[METRIC MONITOR] Roughness TV 'O': {metrics['roughness_TV_O']:.4f}")  
    print(f"[METRIC MONITOR] Birkhoff Quality Estimate: {metrics['birkhoff_measure_estimate']:.4f}")

    assert mock_pred.grad is not None, "Fatal: Birkhoff loss calculation severed the backpropagation chain!"  
    print("[SUCCESS] Gradient pathways verified. Loss constraints are fully differentiable.")  
    print("=== PHASE 4 NON-AUTOREGRESSIVE DIFFUSION CANVAS ENGINE SECURED ===")

if __name__ == "__main__":  
    run_phase_4_validation()
