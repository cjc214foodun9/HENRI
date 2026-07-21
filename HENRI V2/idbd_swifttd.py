"""
Project HENRI: IDBD step-size optimization + SwiftTD overshoot bound.

Implements per-parameter adaptive learning rates for the swarm's SGLD creep,
replacing the single global drift rate mu.

IDBD (Incremental Delta-Bar-Delta; Sutton 1992):
    Meta-learns log step-sizes beta_i = log alpha_i using the trace h_i:
        beta_i <- beta_i + theta * delta * x_i * h_i
        h_i    <- [1 - alpha_i x_i^2]_+ * h_i + alpha_i * delta * x_i
    Features tracking volatile variables keep high plasticity; stable
    invariants decay alpha_i -> 0 ("crystalline permanence").

SwiftTD overshoot bound:
    Correction ratio tau = sum_i alpha_i * phi_i^2. When tau > eta, scale the
    update by eta/tau so the correction never overshoots the target, and decay
    the offending step-sizes:  beta_i <- beta_i + phi_i^2 * ln(eps).

Here delta is the Sagnac prediction error (scalar per step), x_i is the SGLD
drift signal per parameter, and phi_i is the drift magnitude feature. The
per-parameter alpha then scales the applied drift, so fast-adapting experts
learn quickly and crystallized experts freeze.
"""

import math
import torch


class IDBDStepSizer:
    """
    Per-parameter IDBD meta-learning for a parameter tensor of arbitrary shape.
    State (beta, h) mirrors the parameter's shape and device.
    """

    def __init__(self, shape, init_alpha: float = 0.05, meta_theta: float = 0.01, device="cpu"):
        self.beta = torch.full(shape, math.log(init_alpha), device=device)
        self.h = torch.zeros(shape, device=device)
        self.meta_theta = meta_theta

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.beta)

    def update(self, delta: float, drift: torch.Tensor) -> torch.Tensor:
        """
        One IDBD meta-update. delta: scalar prediction error (Sagnac delta).
        drift: the raw drift tensor applied to the parameter (same shape).
        Returns the per-parameter step-size alpha to scale the next drift.
        """
        x = drift  # drift acts as the local feature/gradient signal
        a = self.alpha
        self.beta += self.meta_theta * delta * x * self.h
        a_new = self.alpha
        # Trace update with positive-part clipping
        self.h = torch.clamp(1.0 - a_new * x * x, min=0.0) * self.h + a_new * delta * x
        return a_new


class SwiftTDBound:
    """
    Overshoot guard: given per-parameter step-sizes and the drift features,
    bound the total correction ratio and decay over-eager step-sizes.
    """

    def __init__(self, eta: float = 1.0, eps: float = 0.1):
        self.eta = eta      # max allowed correction ratio
        self.log_eps = math.log(eps)  # decay strength for overshooting params

    def apply(self, alpha: torch.Tensor, drift: torch.Tensor, step_sizer: IDBDStepSizer) -> torch.Tensor:
        """
        Returns a bounded drift tensor. When tau = sum(alpha * drift^2) > eta,
        scales the update by eta/tau and decays contributing betas.
        """
        phi2 = drift * drift
        tau = (alpha * phi2).sum().item()
        if tau <= self.eta or tau <= 0.0:
            return drift
        scale = self.eta / tau
        # Decay step-sizes proportional to their contribution to overshoot
        step_sizer.beta += phi2 * self.log_eps
        return drift * scale


class AdaptiveCreepController:
    """
    Bundles IDBD + SwiftTD for one expert parameter tensor. Owns the state and
    produces the bounded, per-parameter-scaled drift for a creep step.
    """

    def __init__(self, shape, init_alpha: float = 0.05, meta_theta: float = 0.01,
                 eta: float = 1.0, eps: float = 0.1, device="cpu"):
        self.idbd = IDBDStepSizer(shape, init_alpha, meta_theta, device)
        self.swifttd = SwiftTDBound(eta, eps)

    def scaled_drift(self, delta: float, raw_drift: torch.Tensor) -> torch.Tensor:
        """
        Full pipeline: IDBD meta-update -> per-parameter alpha -> SwiftTD bound.
        delta: scalar Sagnac error for this step. raw_drift: -mu * grad F.
        """
        # Lazy device sync: controllers are created before .to(device) moves the
        # parent module, so migrate state on first drift call.
        if self.idbd.beta.device != raw_drift.device:
            self.idbd.beta = self.idbd.beta.to(raw_drift.device)
            self.idbd.h = self.idbd.h.to(raw_drift.device)
        alpha = self.idbd.update(delta, raw_drift)
        drift = alpha * raw_drift
        return self.swifttd.apply(alpha, drift, self.idbd)

    def plasticity_stats(self) -> dict:
        """Crystallization diagnostics: how much of the parameter is frozen."""
        a = self.idbd.alpha
        return {
            "mean_alpha": a.mean().item(),
            "max_alpha": a.max().item(),
            "frozen_fraction": (a < 1e-4).float().mean().item(),
        }

