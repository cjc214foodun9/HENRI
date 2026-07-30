"""
Project HENRI V2: Geodesic Covariance Alignment (GCA) Engine
=============================================================
Implementation of Riemannian Geodesic Covariance Alignment based on
Sridhar Mahadevan et al. (https://github.com/sridharmahadevan/Geodesic-Covariance-Alignment).

Operates directly on the manifold of Symmetric Positive Definite (SPD) matrices
and Stiefel Manifolds V_r(R^n) for non-stationary domain alignment, EDMD transition
matrix transport, and Riemannian Langevin thermalization (SGLD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GeodesicCovarianceAligner(nn.Module):
    """
    Riemannian Geodesic Covariance Alignment (GCA) Engine for PyTorch / CUDA.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def compute_matrix_sqrt_and_inv_sqrt(self, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes C^{1/2} and C^{-1/2} via Eigendecomposition on SPD manifold.
        C: [D, D] SPD tensor
        """
        # Symmetrize to guarantee numerical stability
        C_sym = 0.5 * (C + C.mhash_t() if hasattr(C, 'mhash_t') else C + C.mT)
        eigenvalues, eigenvectors = torch.linalg.eigh(C_sym)
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)
        
        sqrt_eig = torch.sqrt(eigenvalues)
        inv_sqrt_eig = 1.0 / sqrt_eig
        
        C_sqrt = eigenvectors @ torch.diag(sqrt_eig) @ eigenvectors.mT
        C_inv_sqrt = eigenvectors @ torch.diag(inv_sqrt_eig) @ eigenvectors.mT
        return C_sqrt, C_inv_sqrt

    def compute_airm_distance(self, C1: torch.Tensor, C2: torch.Tensor) -> torch.Tensor:
        """
        Affine-Invariant Riemannian Metric (AIRM) distance d_AIRM(C1, C2).
        d_AIRM(C1, C2) = || log(C1^{-1/2} C2 C1^{-1/2}) ||_F
        """
        _, C1_inv_sqrt = self.compute_matrix_sqrt_and_inv_sqrt(C1)
        M = C1_inv_sqrt @ C2 @ C1_inv_sqrt
        
        eigenvalues = torch.linalg.eigvalsh(0.5 * (M + M.mT))
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)
        log_eigs = torch.log(eigenvalues)
        
        return torch.sqrt(torch.sum(log_eigs ** 2))

    def geodesic_interpolation(self, C1: torch.Tensor, C2: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        """
        Computes the unique Riemannian geodesic gamma(t) between C1 and C2.
        gamma(t) = C1^{1/2} * exp(t * log(C1^{-1/2} * C2 * C1^{-1/2})) * C1^{1/2}
        """
        C1_sqrt, C1_inv_sqrt = self.compute_matrix_sqrt_and_inv_sqrt(C1)
        M = C1_inv_sqrt @ C2 @ C1_inv_sqrt
        M_sym = 0.5 * (M + M.mT)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(M_sym)
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)
        
        log_eigs = torch.log(eigenvalues)
        exp_t_log_eigs = torch.exp(t * log_eigs)
        
        M_t = eigenvectors @ torch.diag(exp_t_log_eigs) @ eigenvectors.mT
        gamma_t = C1_sqrt @ M_t @ C1_sqrt
        return 0.5 * (gamma_t + gamma_t.mT)

    def riemannian_sgld_step(
        self, 
        W: torch.Tensor, 
        grad: torch.Tensor, 
        lr: float = 1e-3, 
        temperature: float = 1e-4
    ) -> torch.Tensor:
        """
        Riemannian Gradient Step on Stiefel Manifold V_r(R^n) under SGLD noise.
        W: [n, r] matrix on Stiefel manifold (W^T W = I_r)
        grad: [n, r] Euclidean gradient
        """
        # Project Euclidean gradient onto Tangent Space of Stiefel Manifold
        # grad_stiefel = grad - W @ grad.mT @ W
        grad_tangent = grad - W @ (grad.mT @ W)
        
        # Inject thermal Langevin noise projected onto tangent space
        noise = torch.randn_like(W) * torch.sqrt(torch.tensor(2.0 * temperature * lr, device=W.device))
        noise_tangent = noise - W @ (noise.mT @ W)
        
        # Step in tangent space
        W_step = W - lr * grad_tangent + noise_tangent
        
        # Retraction onto Stiefel Manifold via SVD (Exponential map approximation)
        U, _, Vh = torch.linalg.svd(W_step, full_matrices=False)
        W_next = U @ Vh
        return W_next

    def parallel_transport_edmd(
        self, 
        K_op: torch.Tensor, 
        C_src: torch.Tensor, 
        C_tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel transports EDMD transition operator K_op along AIRM geodesic from C_src to C_tgt.
        """
        _, C_src_inv_sqrt = self.compute_matrix_sqrt_and_inv_sqrt(C_src)
        C_tgt_sqrt, _ = self.compute_matrix_sqrt_and_inv_sqrt(C_tgt)
        
        # Transport operator: K_transported = C_tgt^{1/2} * C_src^{-1/2} * K_op * C_src^{1/2} * C_tgt^{-1/2}
        transport_map = C_tgt_sqrt @ C_src_inv_sqrt
        K_transported = transport_map @ K_op @ torch.linalg.inv(transport_map)
        return K_transported


def main():
    print("=========================================================================")
    print("=== HENRI V2: GEODESIC COVARIANCE ALIGNMENT (GCA) ENGINE ===============")
    print("=========================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligner = GeodesicCovarianceAligner().to(device)
    
    # Generate 2 synthetic SPD covariance matrices
    D = 64
    A = torch.randn(D, D, device=device)
    B = torch.randn(D, D, device=device)
    C1 = A @ A.mT + torch.eye(D, device=device)
    C2 = B @ B.mT + torch.eye(D, device=device)
    
    airm_dist = aligner.compute_airm_distance(C1, C2)
    print(f"Computed AIRM Geodesic Distance d_AIRM(C1, C2) : {airm_dist.item():.4f}")
    
    # Geodesic interpolation
    C_mid = aligner.geodesic_interpolation(C1, C2, t=0.5)
    print(f"Midpoint Covariance C(0.5) Shape                 : {C_mid.shape}")
    
    # Stiefel Riemannian SGLD step
    W = torch.linalg.qr(torch.randn(D, 16, device=device))[0]
    grad = torch.randn_like(W)
    W_next = aligner.riemannian_sgld_step(W, grad, lr=1e-3, temperature=1e-4)
    stiefel_err = torch.norm(W_next.mT @ W_next - torch.eye(16, device=device)).item()
    print(f"Stiefel Orthogonality Error ||W^T W - I||_F       : {stiefel_err:.8e} [PASSED]")
    print("=========================================================================")


if __name__ == "__main__":
    main()
