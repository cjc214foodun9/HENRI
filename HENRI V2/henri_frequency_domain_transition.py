"""
HENRI V2 Phase 8.9 — Frequency-Domain Diagonal Phase Rotators (qFHRR).

Action-conditioned diagonal phase transition (8.9-A) + closed-form Wirtinger
phase residual update (8.9-B), per the Phase 8.8 Postmortem / Phase 8.9
Blueprint (raw SHA ccacd145...).

Mechanism (Fourier Convolution Theorem; Plate 1995):
  spatial translation (dx, dy)  <->  diagonal phase multiplier
  Phi_a = exp(j * Theta_a), Theta_a = dx*Omega + dy*Theta  in [-pi, pi]^D
  Psi_{t+1} = Normalize(Psi_t (.) Phi_a)        [O(D) Hadamard]

This module is DIAGNOSTIC ONLY. Production LowRankCoupledTransition
(efe_planner.py:70) is untouched; this module is never imported by production.

Pre-registration: experiments/sweeps/phase89_diagonal_transition_design.md
Deviations from the blueprint sketch (pre-registered):
  1. update lr=1.0 (exact residual); sketch's 1e-2 cannot meet gate G3.
  2. Analytic complex CC-OS carriers (exp(j(r*Omega + c*Theta))), the canonical
     FHRR frequency-domain state; blueprint's own eq defines Psi = F{s_t} complex.
"""

import math

import torch
import torch.nn as nn

TAU = (math.sqrt(5.0) - 1.0) / 2.0  # golden-ratio conjugate: incommensurate spacing


def _band_limited_irrational(seed_base: float, dimension: int, carrier_scale: float) -> torch.Tensor:
    """Deterministic band-limited incommensurate frequency vector in [-2*pi*s, 2*pi*s].

    Omega_d = 2*pi*s * (2*frac(seed_base*(d+1)) - 1). Irrational seed_base makes
    frequencies incommensurate; the [0,1) fractional span is band-limited.
    """
    idx = torch.arange(1, dimension + 1, dtype=torch.float64)
    frac = (seed_base * idx) % 1.0
    return (2.0 * frac - 1.0) * (2.0 * math.pi * carrier_scale)


class FrequencyDomainDiagonalTransition(nn.Module):
    """Action-conditioned diagonal phase rotator (8.9-A) + Wirtinger update (8.9-B)."""

    def __init__(self, dimension: int = 65536, num_actions: int = 16, device: str = "cuda"):
        super().__init__()
        self.dimension = dimension
        self.num_actions = num_actions
        self.device = device
        # Action phase parameters Theta_a in [-pi, pi]; initialized to zero phase (identity).
        self.action_phases = nn.Parameter(
            torch.zeros(num_actions, dimension, dtype=torch.float32, device=device)
        )

    def phasor(self, action_idx: torch.Tensor) -> torch.Tensor:
        """Unit-modulus complex rotator exp(j*Theta_a); [B, D] complex64 or [D]."""
        theta_a = self.action_phases[action_idx]  # [B, D] or [D]
        return torch.polar(torch.ones_like(theta_a), theta_a)

    def forward(self, state_wave: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """Psi_{t+1} = Psi_t * exp(j*Theta_a). state_wave complex [B, D] or [D].

        NOTE (pre-registered deviation #3): the blueprint sketch's
        `Normalize(next_wave)` (vector L2) contradicts its own Sagnac formula
        1 - |<pred, actual>|/D — unit-L2 vectors would floor Sagnac at
        1 - 1/D ~= 0.99998 even for perfect prediction. FHRR phasor convention
        (per-element unit modulus, |z_d| = 1) keeps <a,a>/D = 1 -> Sagnac = 0 at
        perfect. Pure Hadamard phase rotation preserves per-element modulus, so
        no vector-level normalization is applied.
        """
        phi_a = self.phasor(action_idx)  # [B, D] or [D]
        next_wave = state_wave * phi_a  # Hadamard complex phase rotation
        return next_wave

    def update_online_wirtinger(
        self,
        state_t: torch.Tensor,
        state_tp1_actual: torch.Tensor,
        action_idx: int,
        lr: float = 1.0,
    ) -> float:
        """Closed-form frequency-domain phase learning (8.9-B).

        Theta_a <- Theta_a + lr * arg(Psi_{t+1} * conj(Psi_t) * exp(-j*Theta_a)).
        Returns Sagnac phase loss 1 - |<pred, actual>|/D after the update.
        """
        with torch.no_grad():
            theta_a = self.action_phases[action_idx]  # [D]
            phi_a = torch.polar(torch.ones_like(theta_a), theta_a)  # [D] complex
            # Target phase difference
            target_phase_diff = state_tp1_actual * torch.conj(state_t)  # [D] complex
            # Phase error residual
            phase_error = target_phase_diff * torch.conj(phi_a)
            angle_residual = torch.angle(phase_error)  # [-pi, pi]
            # Gradient update on phase parameters
            self.action_phases[action_idx] += lr * angle_residual.squeeze(0)
            # Measure Sagnac phase loss
            predicted_wave = self.forward(state_t, torch.tensor([action_idx], device=self.device))
            sagnac_loss = 1.0 - torch.abs(torch.sum(predicted_wave * torch.conj(state_tp1_actual))) / self.dimension
            return sagnac_loss.item()


class AnalyticSpatialCarriers:
    """Analytic complex CC-OS spatial carriers (8.9-C data source; deviation #2).

    Psi(r, c) = exp(j*(r*Omega + c*Theta)) in S^{D-1}, with band-limited
    incommensurate Omega/Theta. Translation is EXACT:
        Psi(r+dx, c+dy) = Psi(r, c) * exp(j*(dx*Omega + dy*Theta)).
    """

    def __init__(self, dimension: int = 65536, carrier_scale: float = 0.10, device: str = "cuda"):
        self.dimension = dimension
        self.carrier_scale = carrier_scale
        self.device = device
        self.omega = _band_limited_irrational(TAU, dimension, carrier_scale).to(
            device=device, dtype=torch.float32)
        self.theta = _band_limited_irrational(math.sqrt(3.0), dimension, carrier_scale).to(
            device=device, dtype=torch.float32)

    def encode(self, r: float, c: float) -> torch.Tensor:
        """Analytic complex carrier at integer/float position (r, c); [D] complex64."""
        phase = r * self.omega + c * self.theta  # [D] float32
        return torch.polar(torch.ones_like(phase), phase)

    def rotator(self, dx: float, dy: float) -> torch.Tensor:
        """Exact diagonal phase rotator for translation (dx, dy); [D] complex64."""
        return self.encode(dx, dy)

    def expected_sagnac(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        return 1.0 - float(torch.abs(torch.sum(predicted * torch.conj(actual))) / self.dimension)
