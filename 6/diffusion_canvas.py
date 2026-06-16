import torch
import torch.nn as nn
import torch.nn.functional as F

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

    @torch.no_grad()
    def crystallize_motif(self, swarm_trajectory, sequence_length=512, guidance_scale=4.5):
        """
        De-noises a raw high-entropy canvas into a structured, low-entropy English response matrix.
        
        Args:
            swarm_trajectory (Tensor): The optimal geometric path discovered by the 16 swarms [1, 4096].
            sequence_length (int): Fixed spatial token layout limit.
            guidance_scale (float): Strength of the Zone C attractor field injection.
        """
        self.core.eval()
        device = next(self.core.parameters()).device if list(self.core.parameters()) else torch.device("cpu")
        batch_size = 1

        # Adjust dimensions based on the core model parameters
        for p in self.core.parameters():
            self.hidden_dim = p.shape[-1]
            break

        # 1. Instantiate the High-Entropy Staging Canvas (Pure Gaussian Noise)
        # Shape: [1, Sequence_Length, hidden_dim]
        canvas = torch.randn(batch_size, sequence_length, self.hidden_dim, device=device)
        canvas = F.normalize(canvas, p=2, dim=-1) # Project cleanly onto the hypersphere

        # Construct a discrete variance schedule (cosinespace time-stepping)
        timesteps = torch.linspace(1.0, 0.001, self.N, device=device)
        dt = 1.0 / self.N

        print(f"[*] Initializing parallel thermodynamic relaxation phase over {self.N} steps...")

        # Ensure swarm_trajectory matches the hidden_dim
        if swarm_trajectory.shape[-1] != self.hidden_dim:
            if swarm_trajectory.shape[-1] < self.hidden_dim:
                padded = torch.zeros(swarm_trajectory.shape[0], self.hidden_dim, device=device)
                padded[:, :swarm_trajectory.shape[-1]] = swarm_trajectory
                swarm_trajectory = padded
            else:
                swarm_trajectory = swarm_trajectory[:, :self.hidden_dim]

        # 2. Reverse-Diffusion Denoising Loop
        for step_idx, t in enumerate(timesteps):
            # Broadcast scalar time position to match matrix shape requirements
            t_tensor = torch.full((batch_size, sequence_length, 1), t, device=device)
            
            # Predict the random noise contamination vector field using the unitary layers
            # Support both ProprietaryHENRICore and other standard models
            if hasattr(self.core, 'layers'):
                # ProprietaryHENRICore signature: forward(x, zone_c_attractor, temperature)
                predicted_noise, _ = self.core(canvas, swarm_trajectory, float(t))
            else:
                predicted_noise = self.core(canvas, t_tensor)

            # 3. Inject Zone C Trajectory Guidance Field
            # This warps the score vector field, pulling the text canvas down toward the swarm's solution path
            trajectory_guidance = swarm_trajectory.unsqueeze(1).expand_as(canvas)
            
            # Synthesize the modified score function gradient
            total_score_direction = predicted_noise + (guidance_scale * trajectory_guidance)

            # 4. Execute the Euler-Maruyama Reverse Step
            # Remove a slice of predicted noise while maintaining strict orthogonal magnitude rules
            canvas = canvas - (total_score_direction * dt)
            
            # Enforce Langevin Thermal Restabilization
            # Adding a decaying amount of thermal jitter to prevent early freezing in bad local modes
            if t > 0.1:
                langevin_noise = torch.randn_like(canvas) * (t * 0.001)
                canvas += langevin_noise
            
            # Re-normalize immediately to keep parameters anchored to the lossless hypersphere manifold
            canvas = F.normalize(canvas, p=2, dim=-1)

        print("[+] Canvas relaxation complete. Projecting latent phase-space to vocabulary strings...")

        # 5. Parallel Translation Gate
        # Map the entire stabilized 2D coordinate grid to token dimensions instantly
        logits = self.translation_head(canvas) # Shape: [1, Sequence_Length, Vocabulary_Size]
        target_tokens = torch.argmax(logits, dim=-1) # Shape: [1, Sequence_Length]

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
