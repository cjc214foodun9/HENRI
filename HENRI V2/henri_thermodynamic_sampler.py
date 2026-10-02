"""henri_thermodynamic_sampler.py -- Langevin generation with dark-port thermostat.

Defect 5 / arXiv:2506.15121 (Generative thermodynamic computing).

OBSERVED in source: structured samples arise from the natural time evolution of
a physical system under Langevin dynamics; training maximises the probability of
reversing a noising trajectory, so generation needs NO injected noise schedule
and NO active denoising control.  HENRI already owns the physical thermostat:
the Sagnac dark-port intensity

    I_dark = 0.5 * I0 * (1 - cos(delta_phi))

is the heat source.  A constructive candidate (delta_phi -> 0) is cold: it barely
perturbs.  A contradicting candidate drives the full dark port and receives
maximum thermal excitation -- exactly the exploration it needs, and exactly the
signal that later extinguishes it.

Discrete counterpart  theta <- theta - eta * grad + sqrt(2 * k * T) * xi.

Determinism: all noise derives from a caller-supplied torch.Generator seeded from
the RunManifest.  Default-OFF: constructors raise unless HENRI_THERMO_SAMPLER=1.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

ENV_ENABLE_FLAG = "HENRI_THERMO_SAMPLER"

DEFAULT_T_CRITICAL = 0.0431  # pre-Zone-C setpoint (rad); never the 0.35 search veto


def thermo_sampler_enabled() -> bool:
    return os.environ.get(ENV_ENABLE_FLAG, "").strip() in {"1", "true", "True", "yes"}


class ThermodynamicSamplerError(RuntimeError):
    """Base class for sampler failures."""


class ThermodynamicSamplerDisabledError(ThermodynamicSamplerError):
    """Raised when the sampler is used with HENRI_THERMO_SAMPLER unset or 0."""


def dark_port_intensity(delta_phi: float, i0: float = 1.0) -> float:
    """I_dark = 0.5 * I0 * (1 - cos(delta_phi)).  Constructive -> 0, orthogonal -> I0."""
    return 0.5 * float(i0) * (1.0 - math.cos(float(delta_phi)))


def dark_port_temperature(
    delta_phi: float, base_temperature: float = 1.0, i0: float = 1.0
) -> float:
    """T(delta_phi): the dark-port heat fed to the thermostat.

    Monotone in |delta_phi| over [0, pi]; T(0) = 0 (a coherent candidate is cold),
    T(pi) = base_temperature (a fully contradicting candidate is hottest).
    """
    return float(base_temperature) * dark_port_intensity(delta_phi, i0=i0)


@dataclass
class SamplerStep:
    step: int
    energy: float
    temperature: float
    grad_norm: float
    noise_norm: float


class ThermodynamicSampler:
    """Langevin sampler whose temperature is set by the Sagnac dark port."""

    def __init__(
        self,
        *,
        step_size: float = 0.05,
        base_temperature: float = 1.0,
        i0: float = 1.0,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not thermo_sampler_enabled():
            raise ThermodynamicSamplerDisabledError(
                f"{ENV_ENABLE_FLAG} is not set; thermodynamic sampler is disabled"
            )
        if step_size <= 0:
            raise ThermodynamicSamplerError("step_size must be positive")
        self.step_size = float(step_size)
        self.base_temperature = float(base_temperature)
        self.i0 = float(i0)
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed if seed is not None else 20261001) % (2**31))
        self.generator = generator

    def sample(
        self,
        theta: torch.Tensor,
        grad_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        delta_phi: float = 0.0,
        steps: int = 1,
        record: bool = False,
    ) -> Dict[str, Any]:
        """Run `steps` Langevin updates on theta.

        grad_fn(theta) -> dE/dtheta.  temperature is held at the dark-port value
        computed from the caller's Sagnac phase error.
        """
        if steps < 1:
            raise ThermodynamicSamplerError("steps must be >= 1")
        T = dark_port_temperature(delta_phi, self.base_temperature, self.i0)
        noise_scale = math.sqrt(2.0 * T)
        x = theta.detach().clone().requires_grad_(False)
        history: List[SamplerStep] = []
        for t in range(steps):
            with torch.enable_grad():
                xg = x.detach().clone().requires_grad_(True)
                e = grad_fn(xg)
                energy = float(e.detach())
                g = torch.autograd.grad(e, xg)[0]
            xi = torch.randn(x.shape, generator=self.generator, dtype=x.dtype)
            x = x - self.step_size * g + noise_scale * math.sqrt(self.step_size) * xi
            if record:
                history.append(
                    SamplerStep(
                        step=t,
                        energy=energy,
                        temperature=T,
                        grad_norm=float(g.norm()),
                        noise_norm=float(xi.norm()),
                    )
                )
        return {"theta": x, "temperature": T, "history": history}

    def reverse_trajectory_heat(
        self, energies: List[float], temperatures: List[float]
    ) -> float:
        """Accumulated heat proxy for the trajectory (lower is better).

        OBSERVED design goal in arXiv:2506.15121 is MINIMAL heat emission.  This
        reports the thermodynamic work proxy, not a calibrated physical quantity.
        """
        if len(energies) != len(temperatures):
            raise ThermodynamicSamplerError("energies and temperatures must align")
        return float(sum(abs(e) * t for e, t in zip(energies, temperatures)))
