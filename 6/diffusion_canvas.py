import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NonAutoregressiveCanvasSampler:
    def __init__(self, core_model, translation_head, num_diffusion_steps=25):
        """
        Executes parallel, score-guided relaxation to materialize text blocks all at once.
        
        Args:
            core_model (nn.Module): Your pre-trained parameter unitary base core.
            translation_head (nn.Module): The linear projection matrix mapping 4096-D to vocabulary logits.
            num_diffusion_steps (int): Total number of parallel denoising relaxation steps (O(1) scaling).
        """
        self.core = core_model
        self.translation_head = translation_head
        self.N = num_diffusion_steps
        self.hidden_dim = 4096
        
        # Compile blocked indices for vocabulary domain mask exactly once
        self.blocked_indices = []
        try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            vocab_size = self.translation_head.out_features
            forbidden_keywords = ["scada", "actuator", "gripper", "torque", "vulkan", "valve", 
                                  "firmware", "reflash", "motor", "fluid", "mixer", "conjugation", 
                                  "pressure", "axis", "hardware", "alleviate"]
            for idx in range(vocab_size):
                token_str = tokenizer.decode([idx]).lower()
                if any(kw in token_str for kw in forbidden_keywords):
                    self.blocked_indices.append(idx)
            print(f"[CANVAS MASK] Compiled {len(self.blocked_indices)} blocked SCADA/robotics tokens.")
        except Exception as e:
            print(f"[CANVAS MASK] Warning: Failed to compile tokenizer-based mask: {e}")

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

        # Construct a discrete variance schedule (cosinespace time-stepping)
        timesteps = torch.linspace(1.0, 0.001, self.N, device=device, dtype=model_dtype)
        dt = 1.0 / self.N

        print(f"[*] Initializing parallel thermodynamic relaxation phase over {self.N} steps...")

        # Ensure swarm_trajectory matches the hidden_dim and model_dtype
        swarm_trajectory = swarm_trajectory.to(device=device, dtype=model_dtype)
        if swarm_trajectory.shape[-1] != self.hidden_dim:
            if swarm_trajectory.shape[-1] < self.hidden_dim:
                padded = torch.zeros(swarm_trajectory.shape[0], self.hidden_dim, device=device, dtype=model_dtype)
                padded[:, :swarm_trajectory.shape[-1]] = swarm_trajectory
                swarm_trajectory = padded
            else:
                swarm_trajectory = swarm_trajectory[:, :self.hidden_dim]

        # Project PEARL trajectory track if provided
        if winning_jepa_track is not None and jl_guard is not None:
            W_aligned = jl_guard.W_JL.to(device=device, dtype=model_dtype)
            # Lift [Horizon, core_dim] to [Horizon, global_dim (4096)] using W_JL. Since W_JL is [core_dim, global_dim],
            # [Horizon, core_dim] @ [core_dim, global_dim] -> [Horizon, global_dim]
            steering_field_4096 = torch.matmul(winning_jepa_track.to(device=device, dtype=model_dtype), W_aligned)
            horizon_steps = steering_field_4096.size(0)
        else:
            steering_field_4096 = None

        # 2. Reverse-Diffusion Denoising Loop
        for step_idx, t in enumerate(timesteps):
            # Broadcast scalar time position to match matrix shape requirements
            t_tensor = torch.full((batch_size, sequence_length, 1), t, device=device)
            
            # Determine active steering/attractor vector for this step
            if steering_field_4096 is not None:
                track_idx = min(int((step_idx / self.N) * horizon_steps), horizon_steps - 1)
                active_steering_vector = steering_field_4096[track_idx]
                
                # Align active steering vector with core hidden_dim
                if active_steering_vector.shape[-1] != self.hidden_dim:
                    if active_steering_vector.shape[-1] < self.hidden_dim:
                        padded = torch.zeros(self.hidden_dim, device=device, dtype=model_dtype)
                        padded[:active_steering_vector.shape[-1]] = active_steering_vector
                        active_steering_vector = padded
                    else:
                        active_steering_vector = active_steering_vector[:self.hidden_dim]
            else:
                active_steering_vector = swarm_trajectory.squeeze(0)

            # Predict the random noise contamination vector field using the unitary layers
            if hasattr(self.core, 'layers'):
                # ProprietaryHENRICore signature: forward(x, zone_c_attractor, temperature)
                predicted_noise, _ = self.core(canvas, active_steering_vector.unsqueeze(0), float(t))
            else:
                predicted_noise = self.core(canvas, t_tensor)

            # 3. Inject Zone C Trajectory Guidance Field
            trajectory_guidance = active_steering_vector.unsqueeze(0).unsqueeze(0).expand_as(canvas)
            
            # Synthesize the modified score function gradient
            total_score_direction = predicted_noise + (guidance_scale * trajectory_guidance)

            # 4. Execute the Euler-Maruyama Reverse Step
            canvas = canvas - (total_score_direction * dt)
            
            # Enforce Langevin Thermal Restabilization
            if t > 0.1:
                langevin_noise = torch.randn_like(canvas) * (t * 0.001)
                canvas += langevin_noise
            
            # Re-normalize immediately
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
        if self.blocked_indices and domain_tag == "ARC_Task":
            energy_all[:, self.blocked_indices] = float('inf')

        # Argmin on CPU, then move winning tokens back to GPU device
        winning_tokens = torch.argmin(energy_all, dim=-1).to(device) # [Sequence_Length]
        
        target_tokens = winning_tokens.unsqueeze(0) # Shape: [1, Sequence_Length]
        return target_tokens

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
