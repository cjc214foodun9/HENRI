"""
HENRI V3 - Carrier K3: Empirical Koopman Operator Generator
Micro-architecturally optimized for Blackwell GB202 / RTX 5090.
Computes block-wise Ridge-regularized Koopman operators from empirical transition pairs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_covariance_accum_kernel(
    X_ptr,          # [N, M, 8] - Input states
    Y_ptr,          # [N, M, 8] - Next states
    A_ptr,          # [M, 8, 8] - Auto-covariance matrix output (X^T * X)
    B_ptr,          # [M, 8, 8] - Cross-covariance matrix output (Y^T * X)
    N_transitions: tl.constexpr,
    alpha: tl.constexpr,
    BLOCK_D: tl.constexpr, # Always 8 for Clifford blocks
):
    """
    Accumulates empirical auto-covariance and cross-covariance across transitions.
    Each Triton program instance processes one Clifford block m in parallel.
    """
    pid_m = tl.program_id(axis=0) # Index over M = 8,192 blocks
    
    # Initialize 8x8 accumulators in registers
    acc_A = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    acc_B = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)
    
    offs_d1 = tl.arange(0, BLOCK_D)
    offs_d2 = tl.arange(0, BLOCK_D)
    
    # Accumulate across all historical transitions for this action
    for i in range(N_transitions):
        # Compute memory offsets: [i, pid_m, d]
        row_offset = i * (8192 * BLOCK_D) + pid_m * BLOCK_D
        
        x_ptrs = X_ptr + row_offset + offs_d1
        y_ptrs = Y_ptr + row_offset + offs_d1
        
        x_vec = tl.load(x_ptrs) # [8]
        y_vec = tl.load(y_ptrs) # [8]
        
        # Outer products: A = x * x^T, B = y * x^T
        acc_A += x_vec[:, None] * x_vec[None, :]
        acc_B += y_vec[:, None] * x_vec[None, :]
        
    # Apply Ridge regularization: A = A + alpha * I_8
    diag_mask = offs_d1[:, None] == offs_d2[None, :]
    acc_A += tl.where(diag_mask, alpha, 0.0)
    
    # Write back to global memory: [M, 8, 8]
    out_offset = pid_m * (BLOCK_D * BLOCK_D) + offs_d1[:, None] * BLOCK_D + offs_d2[None, :]
    tl.store(A_ptr + out_offset, acc_A)
    tl.store(B_ptr + out_offset, acc_B)


class EmpiricalKoopmanGenerator(torch.nn.Module):
    """
    Carrier K3: Computes empirical Koopman operators from live transition pairs.
    Operates over M=8,192 factorized 8-dimensional Clifford blocks.
    """
    def __init__(self, num_blocks: int = 8192, block_dim: int = 8, alpha: float = 1e-4):
        super().__init__()
        self.M = num_blocks
        self.d = block_dim
        self.alpha = alpha
        self.D = num_blocks * block_dim # 65,536

    @torch.no_grad()
    def compute_action_operator(self, X_t: torch.Tensor, Y_t: torch.Tensor) -> torch.Tensor:
        """
        Computes K_a for a specific action from observed transition pairs.
        
        Args:
            X_t: Tensor of observed states [N, D] (FP32)
            Y_t: Tensor of observed subsequent states [N, D] (FP32)
            
        Returns:
            K_a: Block-diagonal Koopman operator [M, 8, 8] satisfying rho(K_a) <= 1.0
        """
        N = X_t.shape[0]
        assert N > 0, "Carrier K3 requires at least one transition pair."
        assert X_t.shape[1] == self.D, f"Expected state dimension {self.D}, got {X_t.shape[1]}"
        
        device = X_t.device
        X_reshaped = X_t.view(N, self.M, self.d).contiguous()
        Y_reshaped = Y_t.view(N, self.M, self.d).contiguous()
        
        A = torch.empty((self.M, self.d, self.d), device=device, dtype=torch.float32)
        B = torch.empty((self.M, self.d, self.d), device=device, dtype=torch.float32)
        
        # Launch fused Triton covariance kernel
        grid = (self.M,)
        _block_covariance_accum_kernel[grid](
            X_reshaped,
            Y_reshaped,
            A,
            B,
            N_transitions=N,
            alpha=self.alpha,
            BLOCK_D=self.d,
        )
        
        # Batched 8x8 Cholesky solve on GPU: K_m = B_m * (A_m)^(-1)
        # A_m is symmetric positive-definite due to Ridge regularizer alpha * I_8
        try:
            L = torch.linalg.cholesky(A) # [M, 8, 8]
            # Solve K A = B => K = B A^(-1) => K L L^T = B => K = B (L^T)^(-1) L^(-1)
            # Using torch.linalg.solve_triangular:
            # First solve W L^T = B  => W = B (L^T)^(-1)
            # Then solve K L = W     => K = W L^(-1)
            # Equiv to solving A^T K^T = B^T:
            K_t = torch.cholesky_solve(B.transpose(-1, -2), L) # [M, 8, 8]
            K_a = K_t.transpose(-1, -2) # [M, 8, 8]
        except torch.linalg.LinAlgError:
            # Fallback to pseudoinverse if matrix is ill-conditioned
            K_a = torch.matmul(B, torch.linalg.pinv(A))
            
        # Spectral radius check and contraction projection: rho(K_a) <= 1.0
        # For 8x8 matrices, spectral norm is evaluated via SVD
        spectral_norms = torch.linalg.matrix_norm(K_a, ord=2) # [M]
        scale_factors = torch.clamp(spectral_norms, min=1.0000).unsqueeze(-1).unsqueeze(-1)
        K_a_contractive = K_a / scale_factors
        
        return K_a_contractive

    @torch.no_grad()
    def forward_predict(self, psi_t: torch.Tensor, K_a: torch.Tensor) -> torch.Tensor:
        """
        Projects state psi_t through empirical Koopman operator K_a.
        
        Args:
            psi_t: State vector [D] or [B, D]
            K_a: Operator tensor [M, 8, 8]
            
        Returns:
            psi_pred: Predicted state [D] or [B, D] normalized to S^(D-1)
        """
        orig_shape = psi_t.shape
        if psi_t.dim() == 1:
            psi = psi_t.view(1, self.M, self.d, 1)
        else:
            psi = psi_t.view(-1, self.M, self.d, 1)
            
        # Batched matrix-vector product across M blocks: [M, 8, 8] x [B, M, 8, 1]
        K_expanded = K_a.unsqueeze(0) # [1, M, 8, 8]
        psi_next_blocks = torch.matmul(K_expanded, psi).squeeze(-1) # [B, M, 8]
        
        psi_pred = psi_next_blocks.view(-1, self.D)
        # Retract back onto complex hypersphere S^(D-1)
        psi_pred = torch.nn.functional.normalize(psi_pred, p=2, dim=-1)
        
        return psi_pred.squeeze(0) if len(orig_shape) == 1 else psi_pred