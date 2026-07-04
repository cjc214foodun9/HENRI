import torch
import torch.nn as nn
import math

class HenriDialogueIngress(nn.Module):
    """
    Implements Phase Sub-Space Partitioning on the S^4095 unit hypersphere.
    Projects incoming continuous wave phase angles into two strictly orthogonal subspaces:
    1. Conversational Subspace (Psi_chat)
    2. Operator/Code Subspace (Psi_code)
    Bypasses discrete text parsing gates by evaluating the relative energy in Psi_code.
    """
    def __init__(self, hrr_dim=4096, split_dim=2048, threshold=0.5):
        super().__init__()
        self.hrr_dim = hrr_dim
        self.split_dim = split_dim
        self.threshold = threshold

        # Initialize unitary projection matrix Q using Stiefel manifold initialization
        Q = torch.empty(hrr_dim, hrr_dim)
        nn.init.orthogonal_(Q)
        self.register_buffer("Q", Q.to(dtype=torch.float32))

    def partition_phases(self, phases: torch.Tensor):
        """
        Projects raw phase angles of shape [batch, hrr_dim] onto orthogonal subspaces.
        Returns:
            chat_phases: Projected conversational phase angles [batch, hrr_dim]
            code_phases: Projected operator/code phase angles [batch, hrr_dim]
            should_route_to_sandbox: Boolean mask [batch] indicating code-routing status
        """
        # Ensure input is float32 for stable matrix multiplication
        orig_dtype = phases.dtype
        phases_f32 = phases.to(dtype=torch.float32)
        
        # Project into orthogonal coordinate system
        projected = torch.matmul(phases_f32, self.Q) # [batch, hrr_dim]
        
        # Separate the subspaces in coordinate space
        chat_coords = projected.clone()
        chat_coords[..., self.split_dim:] = 0.0
        
        code_coords = projected.clone()
        code_coords[..., :self.split_dim] = 0.0
        
        # Project back to original embedding space
        chat_phases = torch.matmul(chat_coords, self.Q.t()).to(dtype=orig_dtype)
        code_phases = torch.matmul(code_coords, self.Q.t()).to(dtype=orig_dtype)
        
        # Calculate relative energy in the operator subspace
        total_energy = torch.norm(phases_f32, p=2, dim=-1, keepdim=True) + 1e-8
        code_energy = torch.norm(code_coords, p=2, dim=-1, keepdim=True)
        relative_code_energy = code_energy / total_energy
        
        # Determine code routing status
        should_route = (relative_code_energy.squeeze(-1) > self.threshold)
        
        return chat_phases, code_phases, should_route

    def partition_complex_wave(self, complex_wave: torch.Tensor):
        """
        Partitions a complex wave by separating its phase angles orthogonally.
        """
        phases = torch.angle(complex_wave)
        chat_p, code_p, should_route = self.partition_phases(phases)
        
        chat_wave = torch.polar(torch.ones_like(chat_p), chat_p)
        code_wave = torch.polar(torch.ones_like(code_p), code_p)
        
        return chat_wave, code_wave, should_route
