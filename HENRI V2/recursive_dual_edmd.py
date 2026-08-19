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
        Input shapes: [num_blocks, 8] or flat [d_model] or batched [B, num_blocks, 8].
        """
        orig_shape = state_wave.shape
        if state_wave.dim() == 3:
            # Batched shape: [B, num_blocks, 8] -> [B, d_model]
            b_size = state_wave.shape[0]
            flat_s = state_wave.view(b_size, -1).to(self.V.device)
            flat_a = action_wave.view(b_size, -1).to(self.V.device)
            combined = F.normalize(flat_s + flat_a, p=2, dim=-1)  # [B, d_model]
            phi_t = combined @ self.V  # [B, r_rank]
            phi_next = phi_t @ self.A_sub.T  # [B, r_rank]
            pred_flat = phi_next @ self.V.T  # [B, d_model]
            pred_norm = F.normalize(pred_flat, p=2, dim=-1)
            return pred_norm.view(orig_shape)

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


class CoupledRecursiveDualEDMD(nn.Module):
    """Phase 8.34 Evolution II: online learned coupled transition operator.

    Extends the closed-form RecursiveDualEDMD subspace update with a GLOBAL
    low-rank field channel and a per-block local residual:

        pred = V (A_sub phi)          -- sealed subspace Koopman channel
             + B (C^T x)              -- global field channel (cross-block)
             + R_block x              -- per-block local residual (gap wiring)

    Every channel updates ONLINE with closed-form recursive least squares
    (no BPTT), matching the 8.31/8.32/8.33 calibration discipline.

    field_channel=False disables the global channel -> pure block-diagonal
    control arm (cross-block Jacobian exactly 0). The subspace Koopman
    channel is retained in both arms so the ONLY difference is coupling.

    Default-OFF: no production consumer; the 8.34 benchmark activates it.
    """

    def __init__(
        self,
        d_model: int = 65536,
        r_rank: int = 128,
        lambda_forget: float = 0.98,
        regularization: float = 1e-4,
        num_blocks: int = 8192,
        block_dim: int = 8,
        field_channel: bool = True,
    ):
        super().__init__()
        # Phase 5 P1 rank contract: validate + allocate at effective rank.
        if isinstance(r_rank, bool) or not isinstance(r_rank, int):
            raise TypeError(f"r_rank must be an int, got {type(r_rank).__name__}")
        if r_rank < 1:
            raise ValueError(f"r_rank must be >= 1, got {r_rank}")
        self.requested_rank = r_rank
        self.r_rank = min(r_rank, d_model)
        self.d_model = d_model
        self.lambda_forget = lambda_forget
        self.regularization = regularization
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.field_channel = field_channel

        g = torch.Generator(device="cpu").manual_seed(7)
        # Sealed subspace basis (mirrors RecursiveDualEDMD init).
        v_init = torch.randn(d_model, self.r_rank, generator=g) / math.sqrt(d_model)
        self.register_buffer("V", F.normalize(v_init, p=2, dim=0))

        # Subspace Koopman covariance (sealed baseline rule).
        self.register_buffer("C_t", torch.eye(self.r_rank) * regularization)
        self.register_buffer("G_t", torch.zeros(self.r_rank, self.r_rank))
        self.register_buffer("A_sub", torch.eye(self.r_rank))

        if field_channel:
            # Field read projector C [d, r] (Stiefel, fixed).
            c_init = torch.randn(d_model, self.r_rank, generator=g) / math.sqrt(d_model)
            self.register_buffer("C", F.normalize(c_init, p=2, dim=0))
            # Field write map B [d, r] + its recursive covariance (RLS).
            self.register_buffer("B", torch.zeros(d_model, self.r_rank))
            self.register_buffer("C_f", torch.eye(self.r_rank) * regularization)
            self.register_buffer("G_f", torch.zeros(d_model, self.r_rank))

        # Per-block local residual: [B, 8, 8] closed-form RLS covariances.
        self.register_buffer("X_cov", torch.zeros(num_blocks, block_dim, block_dim))
        self.register_buffer("Y_cov", torch.zeros(num_blocks, block_dim, block_dim))
        self.register_buffer("R_block", torch.zeros(num_blocks, block_dim, block_dim))

    def _predict_from(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t: [d] unit-norm combined wave. Returns unit-norm [d] prediction.
        Not no_grad-decorated so cross_block_jacobian can trace gradients;
        callers that need inference-only use torch.no_grad() context."""
        phi = self.V.T @ x_t
        pred_sub = self.V @ (self.A_sub @ phi)
        if self.field_channel:
            mode = self.C.T @ x_t
            pred_field = self.B @ mode
        else:
            pred_field = 0.0
        xb = x_t.view(self.num_blocks, self.block_dim)
        pred_local = torch.einsum("bij,bj->bi", self.R_block, xb).reshape(-1)
        pred = pred_sub + pred_field + pred_local
        return F.normalize(pred, p=2, dim=0)

    def forward(self, state_wave: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
        """Predicts next state wave. Input [num_blocks, 8] or flat [d] or [B, num_blocks, 8]."""
        orig_shape = state_wave.shape
        if state_wave.dim() == 3:
            b_size = state_wave.shape[0]
            flat_s = state_wave.reshape(b_size, -1).to(self.V.device)
            flat_a = action_wave.reshape(b_size, -1).to(self.V.device)
            combined = F.normalize(flat_s + flat_a, p=2, dim=-1)
            preds = torch.stack([self._predict_from(c) for c in combined])
            return preds.view(orig_shape)
        flat_s = state_wave.reshape(-1).to(self.V.device)
        flat_a = action_wave.reshape(-1).to(self.V.device)
        combined = F.normalize(flat_s + flat_a, p=2, dim=0)
        return self._predict_from(combined).view(orig_shape)

    @torch.no_grad()
    def update_online_step(
        self,
        state_wave: torch.Tensor,
        action_wave: torch.Tensor,
        target_next_wave: torch.Tensor,
    ) -> float:
        """Online closed-form updates for ALL channels. Returns MSE loss."""
        dev = self.V.device
        flat_s = state_wave.reshape(-1).to(dev)
        flat_a = action_wave.reshape(-1).to(dev)
        flat_next = target_next_wave.reshape(-1).to(dev)
        x_t = F.normalize(flat_s + flat_a, p=2, dim=0)
        y_t = F.normalize(flat_next, p=2, dim=0)

        # 1) Sealed subspace Koopman update.
        phi_x = self.V.T @ x_t
        phi_y = self.V.T @ y_t
        self.C_t.mul_(self.lambda_forget).add_(torch.outer(phi_x, phi_x))
        self.G_t.mul_(self.lambda_forget).add_(torch.outer(phi_y, phi_x))
        reg_C = self.C_t + self.regularization * torch.eye(self.r_rank, device=dev)
        self.A_sub.copy_(torch.linalg.solve(reg_C.T, self.G_t.T).T)

        if self.field_channel:
            # 2) Global field channel: closed-form RLS on C-projected mode.
            mode = self.C.T @ x_t
            self.C_f.mul_(self.lambda_forget).add_(torch.outer(mode, mode))
            self.G_f.mul_(self.lambda_forget).add_(torch.outer(y_t, mode))
            reg_Cf = self.C_f + self.regularization * torch.eye(self.r_rank, device=dev)
            self.B.copy_(torch.linalg.solve(reg_Cf.T, self.G_f.T).T)

        # 3) Per-block local residual: 8x8 closed-form RLS, batched.
        xb = x_t.view(self.num_blocks, self.block_dim)
        yb = y_t.view(self.num_blocks, self.block_dim)
        self.X_cov.mul_(self.lambda_forget).add_(torch.bmm(xb.unsqueeze(2), xb.unsqueeze(1)))
        self.Y_cov.mul_(self.lambda_forget).add_(torch.bmm(yb.unsqueeze(2), xb.unsqueeze(1)))
        regX = self.X_cov + self.regularization * torch.eye(self.block_dim, device=dev)
        self.R_block.copy_(
            torch.linalg.solve(regX.transpose(1, 2), self.Y_cov.transpose(1, 2)).transpose(1, 2)
        )

        with torch.no_grad():
            loss = float(F.mse_loss(self._predict_from(x_t), y_t).item())
        return loss

    def cross_block_jacobian(self, state_wave: torch.Tensor, action_wave: torch.Tensor,
                             block_a: int, block_b: int, include_field: bool = True) -> float:
        """d pred[block_a] / d state[block_b] (off-block).

        The dense-V subspace channel already couples blocks globally, so the
        FIELD-channel-attributable coupling is measured as
        include_field=True minus include_field=False (B zeroed). Callers
        should report the delta, not the raw value.
        """
        saved = None
        if self.field_channel and not include_field:
            saved = self.B.clone()
            self.B.zero_()
        try:
            s = state_wave.detach().to(self.V.device).requires_grad_(True)
            a = action_wave.detach().to(self.V.device)
            pred = self.forward(s, a)
            grad = torch.autograd.grad(pred[block_a].sum(), s, create_graph=False)[0]
            return float(grad[block_b].abs().sum().item())
        finally:
            if saved is not None:
                self.B.copy_(saved)
