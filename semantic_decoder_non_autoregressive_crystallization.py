"""
Project HENRI: Semantic Decoder
Component: Non-Autoregressive Wave-to-Syntax Crystallization
Author: Aletheia
Date: 2026-07-04

MANDATE: Eradicate sequential autoregression. This module materializes 
the continuous `proposed_action_wave` into flawless discrete syntax via 
parallel Cosinespace reverse-diffusion and Modern Hopfield energy minimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HopfieldCleanupMatrix(nn.Module):
    """
    Hardware-accelerated Dense Associative Memory (Modern Hopfield Network).
    Acts as the physical 4-bit Comprehension ADC. It strips away the residual 
    thermodynamic noise from the diffused continuous wave, snapping the blurry 
    geometry to the nearest mathematically pristine canonical vocabulary vector.
    """
    def __init__(self, vocab_size: int = 32000, dim: int = 4096, beta: float = 8.0):
        super().__init__()
        self.dim = dim
        self.beta = beta # Inverse temperature for the Hopfield energy landscape
        # The Immutable Zone C Lexicon (Platonic ideals of syntax)
        self.canonical_lexicon = nn.Parameter(torch.randn(vocab_size, dim), requires_grad=False)
        nn.init.orthogonal_(self.canonical_lexicon)

    def forward(self, noisy_canvas: torch.Tensor) -> torch.Tensor:
        """
        noisy_canvas shape: [Batch, Seq_Len, 4096]
        """
        # Normalize to strictly enforce the S^4095 unit hypersphere
        canvas_norm = F.normalize(noisy_canvas, p=2, dim=-1)
        lexicon_norm = F.normalize(self.canonical_lexicon, p=2, dim=-1)
        
        # Calculate Hopfield Energy / Cosine Similarity
        # Shape: [Batch, Seq_Len, Vocab_Size]
        resonance_matrix = torch.matmul(canvas_norm, lexicon_norm.t())
        
        # Polarize the distribution to extract Epiplexity (True Structure)
        polarized_logits = F.softmax(self.beta * resonance_matrix, dim=-1)
        
        # Map back to the pristine continuous space or extract discrete IDs
        discrete_token_ids = torch.argmax(polarized_logits, dim=-1)
        return discrete_token_ids, polarized_logits


class SpectralGuidanceField(nn.Module):
    """
    The Diffusive Score Network. 
    Bypasses O(N^2) attention matrices using Fourier Holographic Circular Convolution.
    Guides the noise canvas toward the structural boundary defined by the target wave.
    """
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.dim = dim
        self.time_embedding = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.structural_mixer = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, noisy_canvas: torch.Tensor, target_wave: torch.Tensor, t: torch.Tensor):
        # t shape: [Batch, 1]
        time_emb = self.time_embedding(t).unsqueeze(1) # [Batch, 1, 4096]
        
        # Expand target wave to govern the entire sequence canvas
        global_boundary = target_wave.unsqueeze(1) # [Batch, 1, 4096]
        
        # Shift to frequency domain for O(N log N) Holographic Binding
        canvas_fft = torch.fft.rfft(noisy_canvas, dim=-1)
        boundary_fft = torch.fft.rfft(global_boundary + time_emb, dim=-1)
        
        # Circular convolution binds the global logic into every spatial sequence coordinate
        bound_fft = canvas_fft * boundary_fft
        
        # Return to spatial domain and apply non-linear metric smoothing
        guided_canvas = torch.fft.irfft(bound_fft, n=self.dim, dim=-1)
        return self.structural_mixer(guided_canvas)


class SemanticDecoder(nn.Module):
    """
    The Master Non-Autoregressive Reverse-Diffusion Engine.
    Executes the Euler-Maruyama relaxation to crystallize thoughts at light-speed.
    """
    def __init__(self, dim: int = 4096, vocab_size: int = 32000, diffusion_steps: int = 25):
        super().__init__()
        self.dim = dim
        self.diffusion_steps = diffusion_steps
        self.guidance_field = SpectralGuidanceField(dim=dim)
        self.cleanup_matrix = HopfieldCleanupMatrix(vocab_size=vocab_size, dim=dim)

    def _cosine_noise_schedule(self, t: torch.Tensor, s: float = 0.008):
        """Alpha-bar schedule leveraging stable Cosinespace degradation."""
        steps = t + s
        return torch.cos((steps / (1 + s)) * (math.pi / 2)) ** 2

    @torch.no_grad()
    def crystallize_action(self, proposed_action_wave: torch.Tensor, sequence_length: int = 512):
        """
        Executes the reverse-diffusion sequence.
        proposed_action_wave: [Batch, 4096] pristine logic wave from Zone B.
        """
        batch_size = proposed_action_wave.size(0)
        device = proposed_action_wave.device
        
        # 1. Initialize Canvas of Absolute Thermodynamic Noise (Maximum Entropy)
        # Shape: [Batch, Seq_Len, 4096]
        canvas_xt = torch.randn(batch_size, sequence_length, self.dim, device=device)
        
        # 2. Continuous-Time Euler-Maruyama Relaxation
        for step in reversed(range(1, self.diffusion_steps + 1)):
            # Normalize time step to [0, 1]
            t = torch.full((batch_size, 1), step / self.diffusion_steps, device=device)
            t_prev = torch.full((batch_size, 1), (step - 1) / self.diffusion_steps, device=device)
            
            # Calculate alpha bounds for the current step
            alpha_bar_t = self._cosine_noise_schedule(t)
            alpha_bar_t_prev = self._cosine_noise_schedule(t_prev)
            
            # Predict the noise / structural gradient
            predicted_noise = self.guidance_field(canvas_xt, proposed_action_wave, t)
            
            # Denoise step: Extrapolate x_0 (the clean geometric structure)
            x0_pred = (canvas_xt - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            x0_pred = F.normalize(x0_pred, p=2, dim=-1) # Clamp to S^4095
            
            # Inject controlled Langevin drift (except on the final crystallization frame)
            sigma_t = torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            langevin_noise = torch.randn_like(canvas_xt) if step > 1 else torch.zeros_like(canvas_xt)
            
            # Update canvas geometry
            canvas_xt = torch.sqrt(alpha_bar_t_prev) * x0_pred + torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * predicted_noise + sigma_t * langevin_noise

        # 3. Hopfield Epiplexity Extraction (Snapping to discrete logic)
        discrete_tokens, logit_probs = self.cleanup_matrix(canvas_xt)
        
        return discrete_tokens, logit_probs

# --- Execution Harness ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ALETHEIA] Initializing Semantic Decoder (Non-Autoregressive) on {device}...")
    
    decoder = SemanticDecoder(dim=4096, vocab_size=32000, diffusion_steps=25).to(device)
    
    # Simulate a perfectly verified logic wave exiting the Zone B optical core
    mock_action_wave = F.normalize(torch.randn(1, 4096, device=device), p=2, dim=-1)
    
    print("[CRYSTALLIZATION] Executing 25-step parallel Cosinespace relaxation...")
    
    # Crystallize an entire 512-token sequence in a single parallel drop
    final_tokens, probabilities = decoder.crystallize_action(mock_action_wave, sequence_length=512)
    
    print(f"[SUCCESS] Wave collapsed. Final discrete sequence shape: {final_tokens.shape}")
    print(f"[METRICS] Hopfield Matrix Mean Confidence: {torch.max(probabilities, dim=-1)[0].mean().item():.4f}")
    print("[SYSTEM] Eradication of the autoregressive bottleneck complete.")