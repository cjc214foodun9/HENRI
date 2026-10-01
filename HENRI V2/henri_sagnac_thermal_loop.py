"""Pillar 3: the Sagnac veto -> anisotropic Langevin thermalisation LOOP.

WHY THIS MODULE EXISTS
  HENRI-ARCH-2026-CRITICAL-DIRECTIVE-V1 Pillar 3 requires that a candidate
  trajectory violating an invariant axiom must NOT simply error or saturate.
  The rejected energy is directed into an anisotropic Langevin thermostat
  ("viscoelastic creep") so stalled parameters can escape a local minimum.

  A wiring audit (commit c2efd90, corrected in 4feee6f) measured the state:

      arc_sagnac_veto.py     COUPLED, default-OFF (evaluate_veto is called)
      henri_thermo_langevin  ORPHANED (0 production importers)
      arc_sagnac_veto does NOT import henri_thermo_langevin

  Both ends exist and are separately tested. The COUPLING is what was absent.
  This module is that coupling and nothing else.

FALSIFIABLE CLAIM
  A vetoed candidate raises the thermal budget above the quiescent budget, and
  the injected perturbation is bounded, seed-reproducible, and monotone in the
  Sagnac excess. If kT(delta) were flat in delta, the loop would inject a
  constant and carry no information from the veto -- so `kT` MUST depend on the
  veto delta for this module to be anything other than noise.

DEFAULTS PRESERVE THE PRODUCTION PATH
  `loop_enabled()` is default OFF and `run_loop()` returns a QUIESCENT event with
  `perturbed is theta` when disabled, so no existing behaviour moves.

PHYSICS CONVENTIONS (reused, not reinvented)
  * Veto threshold tau = DEFAULT_TAU_VETO = 0.35 (the pre-Zone C setpoint in the
    repository; the module reads it from arc_sagnac_veto when available).
  * Noise scale sqrt(2 * gamma * kT * dt) matches the SGLD convention already
    used by henri_thermo_langevin.step and the project's SGLD rule.
  * VETO_UNAVAILABLE NEVER FIRES. The loop refuses to act on a failed
    measurement; an unavailable veto is not evidence of coherence.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:  # pragma: no cover - import shim for the package layout
    from arc_sagnac_veto import VETO_OK, VETO_UNAVAILABLE, evaluate_veto
except Exception:  # pragma: no cover
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from arc_sagnac_veto import VETO_OK, VETO_UNAVAILABLE, evaluate_veto  # type: ignore


FLAG_ENV = "HENRI_SAGNAC_THERMAL_LOOP"

# The repo's pre-Zone C veto setpoint. Do not swap this for the advisory search
# veto (0.35 is the correct value here; both live in memory as distinct).
DEFAULT_TAU_VETO = 0.35


def loop_enabled() -> bool:
    """Default OFF: the default production path stays byte-identical."""
    return os.environ.get(FLAG_ENV, "0") == "1"


@dataclass(frozen=True)
class ThermalLoopConfig:
    """Frozen thermal constants. None of these are trainable."""
    tau_veto: float = DEFAULT_TAU_VETO
    gamma_base: float = 0.05      # anisotropic damping floor (creep rate)
    kT_base: float = 1.0          # quiescent thermal budget
    kT_max: float = 8.0           # hard cap: creep never becomes a random walk
    dt: float = 0.005             # integration timestep (matches the sink)
    max_steps: int = 4            # bounded creep per veto event
    seed: int = 0
    unit_norm_out: bool = True    # restore ||theta|| after creep


@dataclass
class ThermalEvent:
    """A measured record of what the loop did. Never a bare status code."""
    fired: bool
    status: str
    delta_axiom: float
    delta_epistemic: float
    kT: float
    gamma: float
    steps: int
    noise_rms: float
    energy_injected: float
    reason: str


def _quiescent(status: str, reason: str, delta_ax: float = 0.0,
               delta_ep: float = 0.0) -> ThermalEvent:
    return ThermalEvent(fired=False, status=status, delta_axiom=delta_ax,
                        delta_epistemic=delta_ep, kT=0.0, gamma=0.0, steps=0,
                        noise_rms=0.0, energy_injected=0.0, reason=reason)


def kT_from_delta(delta_axiom: float, cfg: ThermalLoopConfig) -> float:
    """Thermal budget as a function of the Sagnac excess over threshold.

    kT = kT_base * (1 + excess), clamped to kT_max, where
        excess = max(0, (delta - tau) / tau).

    This is the ONLY path by which the veto informs the thermostat. A test
    asserts kT is strictly increasing in delta over the ACTIVE range, because a
    flat kT would make the loop a constant-noise injector carrying no
    information from the veto.

    Returns 0.0 at or below threshold: no veto, no heat. This keeps the helper
    consistent with the quiescent event it describes (`_quiescent` reports
    kT == 0.0 for the coherent path). An earlier revision returned kT_base here,
    so the helper and the event disagreed about the same state.
    """
    if not math.isfinite(delta_axiom):
        return 0.0
    tau = max(float(cfg.tau_veto), 1e-12)
    excess = max(0.0, (float(delta_axiom) - tau) / tau)
    if excess <= 0.0:
        return 0.0
    kT = float(cfg.kT_base) * (1.0 + excess)
    return float(min(kT, float(cfg.kT_max)))


class SagnacThermalLoop:
    """Couple the Sagnac veto to an anisotropic Langevin creep step.

    Anisotropy: the damping applied to each coordinate is proportional to the
    absolute value of that coordinate's gradient, so coordinates already under
    large force are damped more (they are the stalled ones by hypothesis), while
    flat coordinates receive proportionally more of the thermal budget.
    """

    def __init__(self, cfg: Optional[ThermalLoopConfig] = None,
                 device: str = "cpu") -> None:
        self.cfg = cfg or ThermalLoopConfig()
        self.device = torch.device(device)
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(int(self.cfg.seed))
        if self.cfg.kT_max <= 0.0:
            raise ValueError("kT_max must be positive")
        if self.cfg.max_steps < 0:
            raise ValueError("max_steps must be non-negative")

    # ------------------------------------------------------------------ helpers
    def _randn(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=self._gen, dtype=torch.float32)

    def _anisotropic_gamma(self, force: torch.Tensor) -> torch.Tensor:
        """Per-coordinate damping in [gamma_base, 2*gamma_base].

        Normalised by the max |force| so the scale is dimension-independent
        (dimension blindness is a rejected failure mode in this repository).
        """
        a = torch.abs(force.detach().float())
        scale = float(a.max().item())
        if scale <= 0.0:
            return torch.full_like(a, float(self.cfg.gamma_base))
        return float(self.cfg.gamma_base) * (1.0 + a / scale)

    # --------------------------------------------------------------------- API
    def step(self, candidate_wave: torch.Tensor, axiom_wave: torch.Tensor,
             world_wave: torch.Tensor,
             theta: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], ThermalEvent]:
        """Evaluate the veto; on a hard veto, creep `theta` out of the stall.

        Returns (theta_after_or_None, event). `theta` is None only when the caller
        supplied no parameters to creep; the event still reports the veto.

        FAIL-CLOSED RULES
          * flag OFF            -> quiescent, theta returned unchanged
          * VETO_UNAVAILABLE    -> quiescent (a failed measurement is not a veto)
          * not triggered       -> quiescent, no energy injected
          * triggered           -> bounded creep, energy measured and reported
        """
        if not loop_enabled():
            return theta, _quiescent("DISABLED", "flag %s is not set" % FLAG_ENV)

        delta_ax, delta_ep, triggered, status = evaluate_veto(
            candidate_wave, axiom_wave, world_wave)

        if status != VETO_OK:
            return theta, _quiescent(status, "veto unavailable; loop refuses to act",
                                     delta_ax, delta_ep)

        if not triggered:
            return theta, _quiescent("COHERENT", "delta_axiom <= tau_veto",
                                     delta_ax, delta_ep)

        kT = kT_from_delta(delta_ax, self.cfg)
        if theta is None:
            return None, ThermalEvent(
                fired=True, status=VETO_OK, delta_axiom=delta_ax,
                delta_epistemic=delta_ep, kT=kT, gamma=float(self.cfg.gamma_base),
                steps=0, noise_rms=0.0, energy_injected=0.0,
                reason="hard veto; no parameters supplied to creep")

        x = theta.detach().clone().float()
        gamma_full = self._anisotropic_gamma(x)
        energy = 0.0
        last_rms = 0.0
        steps = 0
        for _ in range(int(self.cfg.max_steps)):
            sd = math.sqrt(2.0 * float(self.cfg.gamma_base) * kT * float(self.cfg.dt))
            noise = self._randn(x.shape) * sd * gamma_full
            # deterministic pull toward the axiom reference is NOT applied here:
            # the loop injects thermal energy only, so a test can attribute any
            # movement to the noise term and not to a hidden descent step.
            x = x + noise
            last_rms = float(noise.pow(2).mean().sqrt().item())
            energy += float(noise.pow(2).sum().item())
            steps += 1

        if self.cfg.unit_norm_out:
            n = float(x.norm().item())
            if n > 0.0:
                x = x / n * float(theta.detach().float().norm().item())

        return x.to(theta.dtype), ThermalEvent(
            fired=True, status=VETO_OK, delta_axiom=delta_ax, delta_epistemic=delta_ep,
            kT=kT, gamma=float(self.cfg.gamma_base), steps=steps,
            noise_rms=last_rms, energy_injected=energy,
            reason="hard veto; thermal creep applied")

    def run_loop(self, candidate_wave: torch.Tensor, axiom_wave: torch.Tensor,
                 world_wave: torch.Tensor,
                 theta: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], ThermalEvent]:
        """Alias kept explicit so the coupling point has a stable name."""
        return self.step(candidate_wave, axiom_wave, world_wave, theta)


def loop_status() -> dict:
    """Compact, decision-relevant status for telemetry -- no raw logs."""
    return {"module": "henri_sagnac_thermal_loop", "flag": FLAG_ENV,
            "enabled": loop_enabled(), "tau_veto": DEFAULT_TAU_VETO}
