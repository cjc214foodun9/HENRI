"""
Subliminal Clock Probe and Temporal Phase Steering for Project HENRI V2.

Implements intrinsic progress decoding, temporal phase steering, and state-dependent
Langevin cooling derived from Rulli et al. (July 2026) and the Phase 3 Recoop blueprint.

Mathematical Contracts:
    1. Progress Decoder: t_hat = sigmoid(w_clock^T * Re(Psi) + b) in [0, 1]
    2. Clock Unit Vector: v_clock = w_clock / ||w_clock||
    3. Steering Operator: Psi_steered = normalize(Psi + beta * v_clock)
    4. State-Dependent Cooling: T(t_hat) = T_base * (1 - t_hat)^alpha
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubliminalClockProbe(nn.Module):
    """Linear ridge probe mapping wave real components to intrinsic progress t_hat in [0, 1]."""

    def __init__(self, d_model: int, initial_bias: float = 0.0):
        super().__init__()
        self.d_model = d_model
        # Weight vector operating on flattened real components [d_model]
        self.w_clock = nn.Parameter(torch.randn(d_model) * (1.0 / math.sqrt(d_model)))
        self.bias = nn.Parameter(torch.tensor(initial_bias))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wave: Tensor of shape [..., num_blocks, 8] or [..., d_model]
        Returns:
            Scalar tensor t_hat in [0, 1] representing inferred progress.
        """
        flat_wave = wave.reshape(-1, self.d_model)
        # Compute projection
        logits = F.linear(flat_wave, self.w_clock.unsqueeze(0), self.bias)
        t_hat = torch.sigmoid(logits).squeeze(-1)
        return t_hat.mean() if t_hat.numel() > 1 else t_hat.squeeze()

    def update_online(self, wave: torch.Tensor, target_t: float, lr: float = 0.01) -> float:
        """Online regularized update of probe weights given ground-truth step ratio t_target in [0, 1]."""
        self.zero_grad()
        t_hat = self.forward(wave)
        target = torch.tensor(target_t, device=wave.device, dtype=wave.dtype)
        loss = F.mse_loss(t_hat, target) + 1e-4 * (self.w_clock ** 2).sum()
        loss.backward()

        with torch.no_grad():
            if self.w_clock.grad is not None:
                self.w_clock.add_(-lr * self.w_clock.grad)
            if self.bias.grad is not None:
                self.bias.add_(-lr * self.bias.grad)

        return float(loss.detach().item())

    def get_clock_vector(self, device: torch.device) -> torch.Tensor:
        """Returns normalized directional clock vector v_clock of shape [d_model]."""
        w = self.w_clock.to(device)
        norm = w.norm(p=2)
        if norm < 1e-8:
            return torch.zeros_like(w)
        return w / norm

    def steer_wave(self, wave: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Applies subliminal temporal steering along v_clock:
            Psi_steered = normalize(Psi + beta * v_clock)
        Preserves block-wise unit norm for Clifford [num_blocks, 8] waves.
        """
        orig_shape = wave.shape
        device = wave.device
        v_clock = self.get_clock_vector(device)

        flat_wave = wave.reshape(-1, self.d_model)
        steered = flat_wave + beta * v_clock.unsqueeze(0)

        resreshaped = steered.reshape(orig_shape)
        # Renormalize last dimension to unit norm per block
        return F.normalize(resreshaped, p=2, dim=-1)

    @staticmethod
    def anneal_temperature(progress_hat: float, t_base: float = 0.1, alpha: float = 1.5) -> float:
        """
        Computes state-dependent Langevin cooling temperature:
            T(t_hat) = T_base * (1 - t_hat)^alpha
        """
        progress_clamped = max(0.0, min(1.0, float(progress_hat)))
        return t_base * ((1.0 - progress_clamped) ** alpha)
