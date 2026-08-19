"""
Phase 8.33 — Non-linear Macro-Option Phase-Attractor Wave-JEPA.

Inbox spec: G:\\My Drive\\HENRI_Inbox\\Wavejepa.txt (read 2026-08-19, 217
lines). Implemented faithfully with bounded corrections (recorded in the
preregistration `experiments/verification/arc_phase833_nonlinear_macrooption_prereg.md`):

  1. Per-component unit-modulus phasor normalization is KEPT (qFHRR/FHRR
     convention: S^1 per component; cosine over L = mean homodyne).
  2. W_compress is a FIXED Stiefel buffer (QR of randn, no gradient).
     Only the option codebook and the non-linear transition core train.
  3. All norm divisions clamp at 1e-8 (NaN guard).
  4. Egress seam `predict_full_wave` lifts the latent prediction back to a
     real [num_blocks, 8] planner wave (real part of the W_compress lift,
     per-block unit norm) — the ARC planner boundary contract.

Default-OFF: no production caller activates this module. Kill experiment and
verdict criteria are pre-registered (see prereg file). The module is a
DIAGNOSTIC PREDICTOR; it never grants score eligibility.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class MacroOptionAttractorBank(nn.Module):
    """Bank of trainable macro-option attractor phasors on S^1 per component."""

    def __init__(self, num_options: int = 32, opt_dim: int = 512):
        super().__init__()
        self.num_options = num_options
        self.opt_dim = opt_dim
        raw_phase = torch.randn(num_options, opt_dim) * math.pi
        self.option_embeddings = nn.Parameter(raw_phase)

    def forward(self, option_indices: torch.Tensor) -> torch.Tensor:
        """option_indices: (B,) or (B, H) LongTensor -> (..., opt_dim, 2)."""
        phases = self.option_embeddings[option_indices]
        real = torch.cos(phases)
        imag = torch.sin(phases)
        return torch.stack([real, imag], dim=-1)


class NonLinearWaveTransitionBlock(nn.Module):
    """Non-linear phase-coupling core: state+option -> delta in latent phasor space."""

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 1024, opt_dim: int = 512):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj_state = nn.Linear(in_dim * 2, hidden_dim)
        self.proj_opt = nn.Linear(opt_dim * 2, hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim * 2),
        )

    def forward(self, psi_complex: torch.Tensor, z_opt_complex: torch.Tensor) -> torch.Tensor:
        """psi_complex (B, in_dim, 2), z_opt_complex (B, opt_dim, 2) -> (B, in_dim, 2)."""
        B = psi_complex.size(0)
        psi_flat = psi_complex.view(B, -1)
        z_flat = z_opt_complex.view(B, -1)
        h_fused = self.proj_state(psi_flat) + self.proj_opt(z_flat)
        out_flat = self.net(h_fused)
        return out_flat.view(B, self.in_dim, 2)


class NonLinearWaveJEPA(nn.Module):
    """Continuous-time macro-option Wave-JEPA predictor on S^{D-1}."""

    def __init__(
        self,
        full_dim: int = 65536,
        compressed_dim: int = 2048,
        num_options: int = 32,
        opt_dim: int = 512,
        sagnac_lambda: float = 0.15,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.full_dim = full_dim
        self.compressed_dim = compressed_dim
        self.num_options = num_options
        self.opt_dim = opt_dim
        self.sagnac_lambda = sagnac_lambda
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Fixed Stiefel subspace projection [full_dim, compressed_dim].
        W_down = torch.randn(full_dim, compressed_dim)
        Q, _ = torch.linalg.qr(W_down)
        self.register_buffer("W_compress", Q)

        self.attractor_bank = MacroOptionAttractorBank(num_options, opt_dim)
        self.transition_core = NonLinearWaveTransitionBlock(
            in_dim=compressed_dim, hidden_dim=1024, opt_dim=opt_dim
        )

    @torch.no_grad()
    def select_option(self, action_wave: torch.Tensor) -> torch.Tensor:
        """Nearest attractor to the compressed action wave (cosine over the
        first 2*opt_dim latent components).

        action_wave: (B, D) real or (B, D, 2) complex -> (B,) LongTensor.
        """
        action_c = self.compress_wave(action_wave)  # (B, L, 2)
        slice_a = action_c[:, : self.opt_dim, :].reshape(action_c.size(0), -1)  # (B, 2*opt_dim)
        bank_phas = self.attractor_bank(
            torch.arange(self.num_options, device=action_wave.device)
        ).reshape(self.num_options, -1)  # (K, 2*opt_dim)
        sims = slice_a @ bank_phas.t()  # (B, K)
        return sims.argmax(dim=-1)

    def compress_wave(self, psi_full: torch.Tensor) -> torch.Tensor:
        """(B, D) real or (B, D, 2) -> (B, L, 2) unit-modulus phasors."""
        if psi_full.dim() == 2:
            real = psi_full
            imag = torch.zeros_like(real)
            psi_c = torch.stack([real, imag], dim=-1)
        else:
            psi_c = psi_full
        real_comp = torch.matmul(psi_c[..., 0], self.W_compress)
        imag_comp = torch.matmul(psi_c[..., 1], self.W_compress)
        compressed = torch.stack([real_comp, imag_comp], dim=-1)
        norm = torch.norm(compressed, dim=-1, keepdim=True).clamp(min=1e-8)
        return compressed / norm

    def predict_next_state(self, psi_t: torch.Tensor, option_id: torch.Tensor) -> torch.Tensor:
        """psi_t: (B, D) real or (B, D, 2); option_id (B,) -> (B, L, 2) latent pred."""
        psi_lat = self.compress_wave(psi_t)
        z_opt = self.attractor_bank(option_id)
        delta_psi = self.transition_core(psi_lat, z_opt)
        psi_next_unnorm = psi_lat + delta_psi
        norm = torch.norm(psi_next_unnorm, dim=-1, keepdim=True).clamp(min=1e-8)
        return psi_next_unnorm / norm

    @torch.no_grad()
    def predict_full_wave(self, psi_t: torch.Tensor, option_id: torch.Tensor,
                          num_blocks: int = 8192, block_dim: int = 8) -> torch.Tensor:
        """Lift latent prediction to a real [B, num_blocks, 8] planner wave."""
        psi_pred_lat = self.predict_next_state(psi_t, option_id)
        real_full = torch.matmul(psi_pred_lat[..., 0], self.W_compress.t())  # (B, D)
        wave = real_full.view(real_full.size(0), num_blocks, block_dim)
        norm = torch.norm(wave, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        return wave / norm

    def compute_sagnac_delta(self, psi_pred: torch.Tensor, psi_target: torch.Tensor) -> torch.Tensor:
        """1 - |<psi_pred, psi_target>|^2 / L^2, clamped to [0, 1]."""
        r1, i1 = psi_pred[..., 0], psi_pred[..., 1]
        r2, i2 = psi_target[..., 0], psi_target[..., 1]
        dot_real = torch.sum(r1 * r2 + i1 * i2, dim=-1)
        dot_imag = torch.sum(r1 * i2 - i1 * r2, dim=-1)
        inner_mag_sq = (dot_real ** 2 + dot_imag ** 2) / (self.compressed_dim ** 2)
        return (1.0 - inner_mag_sq).clamp(min=0.0, max=1.0)

    def forward(self, psi_t: torch.Tensor, option_id: torch.Tensor, psi_target: torch.Tensor) -> dict:
        psi_pred = self.predict_next_state(psi_t, option_id)
        psi_target_lat = self.compress_wave(psi_target)
        r_p, i_p = psi_pred[..., 0], psi_pred[..., 1]
        r_t, i_t = psi_target_lat[..., 0], psi_target_lat[..., 1]
        cosine_sim = torch.sum(r_p * r_t + i_p * i_t, dim=-1) / self.compressed_dim
        loss_jepa = 1.0 - cosine_sim.mean()
        sagnac_stress = self.compute_sagnac_delta(psi_pred, psi_target_lat).mean()
        total_loss = loss_jepa + self.sagnac_lambda * sagnac_stress
        return {
            "loss": total_loss,
            "jepa_loss": loss_jepa.item(),
            "sagnac_stress": sagnac_stress.item(),
            "psi_pred": psi_pred,
        }
