"""Viscoelastic memory kernel. Shared by Model 1 (swarm creep) and Model 2 (decay).

Document anchors:
  doc p4  : fractional viscoelastic memory kernel; effective weight w_i(t)
  doc p13 : Maxwell-Debye form, sum of P exponential relaxation times tau_1..tau_P
  doc p15 : class ViscoelasticMemoryKernel(nn.Module), stores P=4 decay factors
  doc p16 : non-Markovian convolution over temporal state history

Mechanism
    A Markovian system forgets instantly. This kernel keeps P delay slices and
    weights each by exp(-dt / tau_j), so dissipation depends on history. That is
    what lets the swarm creep across an energy saddle (doc p14, phase 2).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ViscoelasticMemoryKernel(nn.Module):
    """P exponential relaxation factors and a non-Markovian history convolution.

    Args:
        n_slices: P, the number of temporal delay slices (doc: P=4)
        dim:      wave dimension D (state width)
        learnable_tau: if True, log-tau is a parameter (Model 2 predicts it)
    """

    def __init__(self, n_slices: int = 4, dim: int = 65536, learnable_tau: bool = True):
        super().__init__()
        self.P = int(n_slices)
        self.dim = int(dim)
        # tau_j spread over decades: 1 ms, 10 ms, 100 ms, 1 s  (doc p4 tiers)
        taus = torch.logspace(-3, 0, self.P)
        if learnable_tau:
            self.log_tau = nn.Parameter(taus.log())
        else:
            self.register_buffer("log_tau", taus.log())
        # per-slice blend gate, logits init at 0 => uniform start
        self.blend = nn.Parameter(torch.zeros(self.P))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def relaxation_weights(self, dt: float) -> torch.Tensor:
        """gamma_j(dt) = exp(-dt / tau_j), normalized to sum 1."""
        g = torch.exp(-float(dt) / self.tau.clamp_min(1e-9))
        return g / g.sum().clamp_min(1e-12)

    def convolve(self, history: torch.Tensor, dt: float) -> torch.Tensor:
        """Non-Markovian memory convolution.

        history: [B, P, D] complex, most recent slice last.
        Returns [B, D] complex: the memory-weighted state.
        Both terms matter: `dt` sets the decay profile, `blend` reweights the
        slices. A slice with zero weight must not leak energy (gate S-2).
        """
        if history.dim() != 3:
            raise ValueError("history must be [B, P, D]")
        if history.shape[1] != self.P:
            raise ValueError(f"history has {history.shape[1]} slices, kernel has {self.P}")
        g = self.relaxation_weights(dt)                    # [P], sum 1
        w = g * torch.softmax(self.blend, dim=0)           # blend must engage
        w = w / w.sum().clamp_min(1e-12)
        return (history * w.view(1, -1, 1)).sum(dim=1)

    def friction(self, dt: float) -> torch.Tensor:
        """Effective viscosity eta(dt): high when fast modes have decayed.

        doc p14 phase 3: as synchronization tightens, effective viscosity rises,
        locking the swarm into the attractor. Returns a scalar tensor.
        """
        g = self.relaxation_weights(dt)
        # mass concentration in the SLOWEST slice
        return g[-1]

    def extra_repr(self) -> str:
        return f"P={self.P} dim={self.dim} tau={[round(float(t),4) for t in self.tau]}"
