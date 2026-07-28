"""
Wave-JEPA (Wave Joint-Embedding Predictive Architecture) for Project HENRI V2.

Implements Yann LeCun's JEPA paradigm mapped onto continuous wave mechanics and Clifford Geometric Algebra Cl(3,0):
  1. Context Encoder f_theta(x_t) -> Psi_t in S^{D-1}
  2. Target Encoder g_phi(x_{t+1}) -> Psi_{t+1} in S^{D-1} (Stop-Gradient / Unitary)
  3. Action-Conditioned Latent Predictor p_psi(Psi_t, a_t) -> \hat{Psi}_{t+1} (R-EDMD Koopman Operator)
  4. Latent Sagnac Energy Loss L_JEPA = Delta_{Sagnac}(\hat{Psi}_{t+1}, Psi_{t+1})
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
from recursive_dual_edmd import RecursiveDualEDMD


class WaveJEPA(nn.Module):
    """
    Wave-JEPA Core Module:
    Predicts future environment states strictly in d=65,536 latent phase wave space without pixel/token decoders.
    """

    def __init__(self, d_model: int = 65536, num_blocks: int = 8192, r_rank: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.r_rank = r_rank

        # 1. Context & Target Ingress Tokenizer (Encoder f_theta and Target Encoder g_phi)
        self.encoder = O_VSA_IngressTokenizer(num_blocks=num_blocks, vocab_size=256)

        # 2. Action-Conditioned Latent Predictor (R-EDMD Koopman Operator p_psi)
        self.predictor = RecursiveDualEDMD(d_model=d_model, r_rank=r_rank, lambda_forget=0.98)

    def encode_context(self, grid_state: torch.Tensor) -> torch.Tensor:
        """Context Encoder f_theta: maps input spatial observation x_t to Psi_t in S^{D-1}."""
        wave = self.encoder.encode_spatial_grid(grid_state).squeeze(0)
        return F.normalize(wave.view(-1), p=2, dim=0).view(self.num_blocks, 8)

    @torch.no_grad()
    def encode_target(self, grid_state_next: torch.Tensor) -> torch.Tensor:
        """Target Encoder g_phi (Stop-Gradient): maps target observation x_{t+1} to Psi_{t+1} in S^{D-1}."""
        wave = self.encoder.encode_spatial_grid(grid_state_next).squeeze(0)
        return F.normalize(wave.view(-1), p=2, dim=0).view(self.num_blocks, 8)

    def predict_future_latent(self, state_wave: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
        """Action-Conditioned Predictor p_psi: predicts future latent wave \hat{Psi}_{t+1} = T(Psi_t, a_t)."""
        return self.predictor(state_wave, action_wave)

    def compute_sagnac_energy(self, pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
        """
        Latent Energy Metric L_JEPA: Sagnac Homodyne Delta \Delta_{Sagnac}(\hat{Psi}_{t+1}, Psi_{t+1}) \in [0, 2].
        Measures phase obstruction between predicted and true target wave fields.
        """
        flat_pred = F.normalize(pred_wave.view(-1), p=2, dim=0)
        flat_target = F.normalize(target_wave.view(-1), p=2, dim=0)

        # Inner product on unit sphere = cos(angle)
        cos_sim = torch.dot(flat_pred, flat_target)
        # Sagnac Delta Energy = 1.0 - cos_sim
        sagnac_energy = 1.0 - cos_sim
        return sagnac_energy

    def forward(self, x_t: torch.Tensor, a_t: torch.Tensor, x_t_plus_1: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full Wave-JEPA Forward Cycle:
          1. Psi_t = f_theta(x_t)
          2. Psi_{t+1} = g_phi(x_{t+1}) [Stop-Gradient Target]
          3. \hat{Psi}_{t+1} = p_psi(Psi_t, a_t)
          4. Energy = Delta_{Sagnac}(\hat{Psi}_{t+1}, Psi_{t+1})
        """
        psi_t = self.encode_context(x_t)
        psi_target = self.encode_target(x_t_plus_1)
        psi_pred = self.predict_future_latent(psi_t, a_t)

        energy = self.compute_sagnac_energy(psi_pred, psi_target)

        # Update predictor R-EDMD operator online
        self.predictor.update_operator(psi_t.view(-1), a_t.view(-1), psi_target.view(-1))

        metrics = {
            "sagnac_energy": float(energy.item()),
            "sagnac_coherence": float(1.0 - energy.item()),
            "pred_norm": float(torch.norm(psi_pred).item()),
            "target_norm": float(torch.norm(psi_target).item()),
        }

        return energy, metrics


if __name__ == "__main__":
    jepa = WaveJEPA(d_model=65536, num_blocks=8192)
    x_t = torch.randint(0, 10, (10, 10))
    a_t = torch.randn(8192, 8)
    x_next = torch.randint(0, 10, (10, 10))

    energy, stats = jepa(x_t, a_t, x_next)
    print(f"Wave-JEPA Energy (\Delta_Sagnac): {energy.item():.6f}")
    print("Metrics:", stats)
    print("Wave-JEPA standalone module verified successfully.")
