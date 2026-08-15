"""Phase 8.11 — Native Complex Wave Space Transition (default-OFF).

Blueprint: HENRI-POSTMORTEM-2026-08-PHASE8.10-FINAL (drive inbox), approved
2026-08-15. Pre-registration:
HENRI V2/experiments/sweeps/phase811_native_complex_transition_design.md

The Phase 8.10 kill showed that diagonal phase rotation is exact ONLY on
per-element unit-modulus complex phasors; the production REAL [8192, 8]
per-block L2-normalized wave type cannot carry phase
(acos(cos(phi+delta)/c) != phi+delta, 3000-step convergence instead of 3).

This module keeps the production transition interface
    forward(state_wave [B,8] real, action_wave [B,8] real) -> [B,8] real
while executing the world-model latent transition NATIVELY in C^D as
per-element unit-modulus phasors:

    lift:      z_t      = exp(j * acos(clamp(w_t)))      # real -> unit phasor
    rotate:    z_{t+1}  = z_t * exp(j * (Theta_a + phi_a))
    fit:       phi_a   += lr * angle(z_{t+1} * conj(z_t) * conj(phi_a))
    egress:    x_real   = per_block_normalize(Re(z_{t+1}))   # ONLY at egress

Design deviations from the blueprint (documented in pre-registration):
1. NO vector L2 normalization in forward_complex — FHRR keeps per-element
   unit modulus (8.9 lesson #3); vector norm floors Sagnac at 1 - 1/D.
2. Egress contract = real [B, 8] per-block unit, float32, finite (the
   blueprint's tautological ||x_egress - Re(Psi)|| < 1e-6 gate rejected).
3. No phantom-CLI self-test (wave_jepa.py --mode native_complex_test does
   not exist); verification = contract tests + CUDA matrix runner.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class NativeComplexWaveTransition(nn.Module):
    """Native complex-domain wave transition with real egress boundary.

    Transition dynamics execute in C^D (per-element unit-modulus phasors);
    real conversion occurs ONLY at the egress boundary. The production
    interface (EFEPlanner / LowRankCoupledTransition) is preserved:
    forward(state, action) -> [B, 8] real per-block unit.

    Action phases Theta_a [num_actions, D] are zero-init (identity rotation);
    phi_a corrections are learned by the closed-form elementwise angle
    residual (exact 1-step for any diagonal phase rotation; Fourier
    Convolution Theorem, Plate 1995). Lazy action indexing by cosine
    fingerprint of the deterministic decoder engram (bounded to
    num_actions; fail-closed vs learnable_actions).
    """

    def __init__(
        self,
        dimension: int = 65536,
        num_actions: int = 16,
        device: Optional[str] = None,
        num_blocks: int = 8192,
        block_dim: int = 8,
    ):
        super().__init__()
        self.dimension = dimension
        self.num_actions = num_actions
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.d = num_blocks * block_dim
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Action phase parameters Theta_a in [-pi, pi]; zero init = identity.
        self.action_phases = nn.Parameter(
            torch.zeros(num_actions, self.d, dtype=torch.float32, device=self._device)
        )
        # Lazy action-wave fingerprint prototypes (cosine identity).
        self.register_buffer(
            "_fp_buf", torch.zeros(num_actions, self.d, dtype=torch.float32)
        )
        self._fp_count = 0

    # -- production-compat surface -------------------------------------------
    @property
    def rank(self) -> int:
        return 0  # complex path has no low-rank field channel

    @property
    def requested_rank(self) -> int:
        return 0

    def _retract(self, residual_only: bool = False):
        """No-op for the complex path (phase params are unconstrained)."""
        return None

    # -- action indexing ------------------------------------------------------
    @torch.no_grad()
    def _action_index(self, action_wave: torch.Tensor) -> int:
        w = action_wave.detach().reshape(1, -1).to(self._fp_buf.device)
        w = w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)
        if self._fp_count == 0:
            self._fp_buf[0].copy_(w.reshape(-1))
            self._fp_count = 1
            return 0
        sims = torch.nn.functional.cosine_similarity(
            w, self._fp_buf[: self._fp_count], dim=-1
        )
        best = int(sims.argmax())
        if sims[best] > 0.999:
            return best
        if self._fp_count < self.num_actions:
            self._fp_buf[self._fp_count].copy_(w.reshape(-1))
            self._fp_count += 1
            return self._fp_count - 1
        return best  # capacity full -> nearest (bounded)

    # -- core ops -------------------------------------------------------------
    def _phasor(self, real_wave: torch.Tensor) -> torch.Tensor:
        """Real wave -> per-element unit-modulus complex phasor (acos lift)."""
        w = real_wave.reshape(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.polar(torch.ones_like(w), torch.acos(w))  # [d] complex

    def forward_complex(
        self, psi_complex: torch.Tensor, action_idx: int
    ) -> torch.Tensor:
        """Exact complex Hadamard phase rotation: Psi_{t+1} = Psi_t * exp(j*Theta_a).

        psi_complex: [d] per-element unit-modulus complex phasors.
        Returns [d] complex, per-element unit modulus preserved exactly
        (NO vector L2 normalization — FHRR convention).
        """
        theta_a = self.action_phases[action_idx].to(psi_complex.device)
        phi_a = torch.polar(torch.ones_like(theta_a), theta_a)  # unit phasor
        return psi_complex * phi_a

    def project_to_real_egress(
        self, psi_complex: torch.Tensor
    ) -> torch.Tensor:
        """Egress transduction: complex wave -> real L2-normalized [B, 8] blocks.

        Applied strictly at the readout/environment boundary.
        """
        real_part = torch.real(psi_complex).reshape(self.num_blocks, self.block_dim)
        block_norms = torch.norm(real_part, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return real_part / block_norms

    def forward(
        self, state_wave: torch.Tensor, action_wave: torch.Tensor
    ) -> torch.Tensor:
        """Production interface: real [B, 8] -> real [B, 8] per-block unit.

        lift -> rotate -> egress-projection back to real.
        """
        z = self._phasor(state_wave)                       # [d] complex
        idx = self._action_index(action_wave)              # int
        z_next = self.forward_complex(z, idx)              # [d] complex
        return self.project_to_real_egress(z_next)         # [B, 8] real

    # -- learning -------------------------------------------------------------
    @torch.no_grad()
    def update_phase_complex(
        self,
        psi_t: torch.Tensor,
        psi_next: torch.Tensor,
        action_idx: int,
        lr: float = 1.0,
    ) -> float:
        """Closed-form angle-residual update on complex phasors (exact 1-step).

        psi_t, psi_next: [d] per-element unit-modulus complex phasors.
        Delta_theta = angle(psi_next * conj(psi_t) * conj(phi_a)); then
        phi_a += lr * Delta_theta. Returns the pre-update mean |residual|.
        """
        theta = self.action_phases[action_idx].to(psi_t.device)
        phi_a = torch.polar(torch.ones_like(theta), theta)
        delta = torch.angle(psi_next * torch.conj(psi_t) * torch.conj(phi_a))
        pre = float(delta.abs().mean())
        self.action_phases[action_idx].add_(lr * delta.to(self.action_phases.device))
        return pre

    @torch.no_grad()
    def fit_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        observed_nexts: torch.Tensor,
        iters: int = 3,
        lr: float = 1.0,
    ) -> float:
        """Batched complex closed-form fit.

        states/actions/observed_nexts: [N, B, 8] real (production interface).
        Returns the mean pre-fit real-egress Sagnac loss.
        """
        N = states.shape[0]
        device = self.action_phases.device
        states = states.detach().to(device)
        actions = actions.detach().to(device)
        observed_nexts = observed_nexts.detach().to(device)

        preds = torch.stack([self.forward(states[i], actions[i]) for i in range(N)])
        p = preds.reshape(N, -1)
        o = observed_nexts.reshape(N, -1)
        pre = float(
            (1.0 - (p * o).sum(-1) / (p.norm(dim=-1) * o.norm(dim=-1)).clamp(min=1e-12)).mean()
        )

        for _ in range(iters):
            for i in range(N):
                idx = self._action_index(actions[i])
                z_t = self._phasor(states[i])
                z_n = self._phasor(observed_nexts[i])
                self.update_phase_complex(z_t, z_n, idx, lr=lr)
        return pre

    @torch.no_grad()
    def update_wirtinger(
        self,
        state_wave: torch.Tensor,
        action_wave: torch.Tensor,
        observed_next_wave: torch.Tensor,
        lr: float = 0.05,
        return_loss: bool = True,
    ) -> float:
        """Online update on the PRODUCTION real-egress Sagnac loss.

        Returns the pre-update real-metric Sagnac loss (runner semantics:
        lr=0.05 -> 1 damped step; lr=1.0 -> 20 steps).
        """
        with torch.no_grad():
            pred0 = self.forward(state_wave, action_wave)
            p = pred0.reshape(-1)
            o = observed_next_wave.detach().reshape(-1)
            pre = float(1.0 - torch.dot(p, o) / (torch.norm(p) * torch.norm(o)).clamp(min=1e-12))
        idx = self._action_index(action_wave)
        steps = max(1, int(round(lr * 20.0)))
        z_t = self._phasor(state_wave)
        z_n = self._phasor(observed_next_wave)
        for _ in range(steps):
            self.update_phase_complex(z_t, z_n, idx, lr=0.05)
        return pre

    def bind(self, state_wave: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
        """Placeholder for legacy EDMD callers (never invoked on complex path).

        Mirrors the legacy contract shape: returns complex [B, 8].
        """
        z = self._phasor(state_wave).reshape(self.num_blocks, self.block_dim)
        return z.unsqueeze(0)
