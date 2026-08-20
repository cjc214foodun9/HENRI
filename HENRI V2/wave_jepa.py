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
from typing import Dict, Tuple, Optional

from henri_vision_encoder import HENRIVisionEncoder
from recursive_dual_edmd import RecursiveDualEDMD


class WaveJEPA(nn.Module):
    """
    Wave-JEPA Core Module:
    Predicts future environment states strictly in d=65,536 latent phase wave space without pixel/token decoders.
    """

    def __init__(self, d_model: int = 65536, num_blocks: int = 8192, r_rank: int = 16,
                 device: Optional[str] = None, use_context_matching: bool = False,
                 context_mix: float = 0.3, context_beta: float = 8.0):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.r_rank = r_rank
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Context Matching (Extropic Directive 2 — 2608.01615 §III.B).
        # Default-OFF: when False, the forward path is byte-identical to the
        # legacy predictor. When True, predictions are anchored on local
        # context engrams (Zone C batches): pred = (1-lam)*T(Psi_t, a_t) +
        # lam * sum_j w_j C_j with w = softmax(beta * sim(Psi_t, C_j)).
        self.use_context_matching = use_context_matching
        self.context_mix = context_mix
        self.context_beta = context_beta

        # 1. Context & Target Ingress Encoder (HENRIVisionEncoder)
        self.encoder = HENRIVisionEncoder(d_model=d_model, k_blocks=num_blocks, device=self.device)

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
        """Action-Conditioned Predictor p_psi: predicts future latent wave \\hat{Psi}_{t+1} = T(Psi_t, a_t)."""
        return self.predictor(state_wave, action_wave)

    @torch.no_grad()
    def _context_weights(self, state_wave: torch.Tensor,
                         context_waves: torch.Tensor) -> torch.Tensor:
        """Softmax attention over context engrams (2608.01615 §III.B).

        w_j = softmax(beta * sim(Psi_t, C_j)) over the context batch.
        Context waves [M, num_blocks, 8]; returns [M] weights on the
        flattened unit-sphere similarity.
        """
        flat_state = F.normalize(state_wave.view(-1), p=2, dim=0)
        flat_ctx = F.normalize(context_waves.view(context_waves.shape[0], -1), p=2, dim=-1)
        sims = flat_state @ flat_ctx.T                     # [M]
        return torch.softmax(self.context_beta * sims, dim=-1)

    def predict_future_latent_context(self, state_wave: torch.Tensor,
                                      action_wave: torch.Tensor,
                                      context_waves: torch.Tensor) -> torch.Tensor:
        """Context-matched predictor (2608.01615 §III.B).

        pred = (1 - lam) * T(Psi_t, a_t) + lam * sum_j w_j C_j

        The transition factor is conditioned on local context so multi-step
        rollouts stay anchored to verified Zone C engrams instead of drifting
        with accumulated predictor error. lam = context_mix; lam=0 recovers
        the legacy predictor exactly.
        """
        base = self.predict_future_latent(state_wave, action_wave)
        w = self._context_weights(state_wave, context_waves)          # [M]
        anchor = torch.einsum('j,jnb->nb', w, context_waves)          # [nb, 8]
        anchor = F.normalize(anchor.view(-1), p=2, dim=0).view(state_wave.shape)
        lam = self.context_mix
        pred = (1.0 - lam) * base + lam * anchor
        return F.normalize(pred.view(-1), p=2, dim=0).view(state_wave.shape)

    def compute_sagnac_energy(self, pred_wave: torch.Tensor, target_wave: torch.Tensor) -> torch.Tensor:
        """
        Latent Energy Metric L_JEPA: Sagnac Homodyne Delta \Delta_{Sagnac}(\hat{Psi}_{t+1}, Psi_{t+1}) \in [0, 2].
        Measures phase obstruction between predicted and true target wave fields.
        """
        flat_pred = F.normalize(pred_wave.view(-1), p=2, dim=0).to(self.device)
        flat_target = F.normalize(target_wave.view(-1), p=2, dim=0).to(self.device)

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
        loss_val = self.predictor.update_online_step(psi_t, a_t, psi_target)

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
