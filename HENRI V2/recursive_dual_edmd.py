"""
Recursive Dual Extended Dynamic Mode Decomposition (R-EDMD) for Project HENRI V2.

Implements online, closed-form Koopman operator estimation with an exponential
forgetting factor (lambda_forget in [0.95, 0.99]).

Eliminates windowed batch replay by updating covariance and cross-covariance
matrices incrementally in O(r^2 * D) FLOPS without Backpropagation Through Time (BPTT).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecursiveDualEDMD(nn.Module):
    """
    Online Recursive Dual EDMD with exponential forgetting.
    Tracks low-rank covariance matrices C_t and G_t for state-action phase waves.
    """

    def __init__(
        self,
        d_model: int = 65536,
        r_rank: int = 16,
        lambda_forget: float = 0.98,
        regularization: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.r_rank = r_rank
        self.lambda_forget = lambda_forget
        self.regularization = regularization

        # Low-rank projection basis V: [d_model, r_rank]
        g = torch.Generator(device="cpu").manual_seed(42)
        v_init = torch.randn(d_model, r_rank, generator=g) / math.sqrt(d_model)
        self.register_buffer("V", F.normalize(v_init, p=2, dim=0))

        # Covariance matrices in rank-r subspace: [r_rank, r_rank]
        self.register_buffer("C_t", torch.eye(r_rank) * regularization)
        self.register_buffer("G_t", torch.zeros(r_rank, r_rank))

        # Reconstructed low-rank transition operator in ambient space: [r_rank, r_rank]
        self.register_buffer("A_sub", torch.eye(r_rank))

    def forward(self, state_wave: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
        """
        Predicts next state wave given current state and action wave in ambient space.
        Input shapes: [num_blocks, 8] or flat [d_model].
        """
        orig_shape = state_wave.shape
        flat_s = state_wave.view(-1).to(self.V.device)
        flat_a = action_wave.view(-1).to(self.V.device)

        # Combined phase input projected into r-rank subspace
        combined = F.normalize(flat_s + flat_a, p=2, dim=0)
        phi_t = self.V.T @ combined  # [r_rank]

        # Transition in rank-r subspace: phi_{t+1} = A_sub @ phi_t
        phi_next = self.A_sub @ phi_t

        # Map back to ambient d_model space and restore original block shape
        pred_flat = self.V @ phi_next
        pred_norm = F.normalize(pred_flat, p=2, dim=0)
        return pred_norm.view(orig_shape)

    def update_online_step(
        self,
        state_wave: torch.Tensor,
        action_wave: torch.Tensor,
        target_next_wave: torch.Tensor,
    ) -> float:
        """
        Updates C_t, G_t, and A_sub online in O(r^2 * D) FLOPS.
        Returns the instantaneous prediction loss (Sagnac MSE).
        """
        flat_s = state_wave.view(-1).to(self.V.device)
        flat_a = action_wave.view(-1).to(self.V.device)
        flat_next = target_next_wave.view(-1).to(self.V.device)

        x_t = F.normalize(flat_s + flat_a, p=2, dim=0)
        y_t = F.normalize(flat_next, p=2, dim=0)

        phi_x = self.V.T @ x_t  # [r_rank]
        phi_y = self.V.T @ y_t  # [r_rank]

        # Exponential forgetting update on covariance matrices
        self.C_t.mul_(self.lambda_forget).add_(torch.outer(phi_x, phi_x))
        self.G_t.mul_(self.lambda_forget).add_(torch.outer(phi_y, phi_x))

        # Closed-form regularized update for A_sub = G_t @ (C_t + reg*I)^{-1}
        reg_C = self.C_t + self.regularization * torch.eye(self.r_rank, device=self.C_t.device)
        self.A_sub.copy_(torch.linalg.solve(reg_C.T, self.G_t.T).T)

        # Compute instantaneous prediction loss
        with torch.no_grad():
            pred_y = self.V @ (self.A_sub @ phi_x)
            loss = float(F.mse_loss(pred_y, y_t).item())

        return loss
