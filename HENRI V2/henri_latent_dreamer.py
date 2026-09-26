"""Focused Latent Dreamer — test-time self-play adaptation (four ratified gates).

Protocol: HENRI-ARCH-2026-SELFPLAY-DREAMING-V1 (.md) + the ratified gates.

GATE-A  hysteresis. The .md reuses 0.0431 for BOTH dream entry and wake. Entry
        is a SEARCH decision; wake is a CRYSTALLIZATION decision. Entry=0.35-class,
        wake=0.0431, and wake < entry is enforced by config validation.

GATE-B  metric-family unity. The gating metric MUST be the same metric family the
        action path selects by. `HenriSwarmOrchestrator.sagnac_coherence` is the
        CLIFFORD-coherence family: delta = 1 - Re(<a,b>)/(|a||b|), bounded [0,2].
        The .md's metric (|sum(d_phi)| / 2pi over D=65536, order sqrt(D)) can never
        pass its own 0.0431 gate -> that D-SAGNAC defect is NOT reproduced.
        The .md thresholds were calibrated on the HOMODYNE family and are marked
        HYPOTHESIS on the live [0,2] family pending re-calibration.

GATE-C  live gradient path. The .md calls autograd.grad(stress, adapter_params)
        while `adapter_params` never enters `psi_dream`, so the gradient is None and
        the creep is dead. Here the adapter DOES enter the scored wave:
            adapter -> adapted_reference -> live coherence -> loss -> grads
        verified by asserting the adapter parameters actually move.

GATE-D  M3-ACT latch. The dreamer is DIAGNOSTIC_ONLY until the two-sided
        action-change gate passes. The latch is CODE: `guarded_egress` raises
        `DreamEgressBlocked` until `ratify_for_egress(receipt)`.

SGLD creep is FLAG-GATED OFF by default (`HENRI_DREAM_CREEP=1`). Load-bearing
test-time weight adaptation requires explicit approval; the default path performs
a gradient check but applies NO weight change, so the default behaviour is
byte-identical for any consumer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "ENTRY_THRESHOLD",
    "WAKE_THRESHOLD",
    "CREEP_FLAG",
    "DreamConfig",
    "DreamResult",
    "DreamEgressBlocked",
    "LowRankDreamAdapter",
    "FocusedLatentDreamer",
]

# Ratified two-stage constants (GATE-A). HYPOTHESIS on the live [0,2] family.
ENTRY_THRESHOLD = 0.35
WAKE_THRESHOLD = 0.0431

CREEP_FLAG = "HENRI_DREAM_CREEP"


class DreamEgressBlocked(RuntimeError):
    """Raised when egress is attempted while the M3-ACT gate is unratified."""


@dataclass(frozen=True)
class DreamConfig:
    """Bounded dream settings. `validate()` enforces the hysteresis invariant."""

    entry_threshold: float = ENTRY_THRESHOLD
    wake_threshold: float = WAKE_THRESHOLD
    max_dream_steps: int = 64
    consensus_r: float = 0.95
    lambda_dark: float = 0.1     # HYPOTHESIS: .md value, unmeasured
    sgld_lr: float = 1e-3
    sgld_temp: float = 1e-5      # HYPOTHESIS: .md value, unmeasured
    lora_rank: int = 16
    width: int = 8               # last-dim width of the [K, 8] wave family

    def validate(self) -> "DreamConfig":
        for name in ("entry_threshold", "wake_threshold"):
            v = float(getattr(self, name))
            if not (0.0 < v <= 2.0):
                raise ValueError(f"{name} must be in (0, 2], got {v}")
        if self.wake_threshold >= self.entry_threshold:
            raise ValueError(
                "wake_threshold must be BELOW entry_threshold (hysteresis required)"
            )
        if not isinstance(self.max_dream_steps, int) or self.max_dream_steps < 1:
            raise ValueError("max_dream_steps must be an int >= 1")
        if not (0.0 < self.consensus_r <= 1.0):
            raise ValueError("consensus_r must be in (0, 1]")
        if self.lora_rank < 1 or self.width < 1:
            raise ValueError("lora_rank and width must be >= 1")
        return self


@dataclass
class DreamResult:
    """Telemetry from one dream cycle. Never an action, never a score."""

    entered: bool
    steps_taken: int
    woke: bool
    initial_delta: float
    final_delta: float
    consensus_r: float
    adapter_delta_norm: float
    creep_enabled: bool
    selected_action_changed: Optional[bool] = None  # M3-ACT harness fills this


class LowRankDreamAdapter(nn.Module):
    """Low-rank phase adapter on the wave family (LoRA-style, r<=lora_rank).

    `up` is zero-initialised, so the adapter is an EXACT identity at construction:
    the default path cannot perturb any consumer.
    """

    def __init__(self, num_blocks: int, width: int = 8, rank: int = 16) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.width = width
        self.rank = max(1, min(rank, num_blocks * width))
        self.down = nn.Parameter(torch.randn(self.rank, num_blocks * width) * 0.01)
        self.up = nn.Parameter(torch.zeros(num_blocks * width, self.rank))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        flat = wave.reshape(-1)
        delta = self.up @ (self.down @ flat)
        return (flat + delta).reshape(wave.shape)

    def delta_norm(self) -> float:
        with torch.no_grad():
            return float(torch.linalg.vector_norm(self.up @ self.down))


class FocusedLatentDreamer:
    """Test-time self-play adaptation gated by the LIVE Zone A metric."""

    def __init__(self, core: Any, config: Optional[DreamConfig] = None) -> None:
        self.core = core
        self.cfg = (config or DreamConfig()).validate()
        # Device: the adapter must live on the SAME device as the incoming wave.
        # Verified defect (2026-09-26): on the CUDA host the adapter stayed on CPU
        # while the core produced CUDA waves, so dream() raised a device mismatch.
        # Local CPU smoke could not see it. `_prepare_device` is called in dream().
        self._dev = torch.device(getattr(core, "dev", "cpu"))
        self.adapter = LowRankDreamAdapter(
            num_blocks=core.num_blocks, width=self.cfg.width, rank=self.cfg.lora_rank
        ).to(self._dev)
        self._egress_ratified = False
        self._ratification_receipt: Optional[str] = None

    def _prepare_device(self, reference_wave: torch.Tensor) -> None:
        """Move the adapter onto the reference wave's device before it is used."""
        want = reference_wave.device
        if self.adapter.down.device != want:
            self.adapter.to(want)
            self._dev = want

    # -------------------------------------------------------------- GATE-A
    def should_enter(self, delta: float) -> bool:
        return float(delta) > self.cfg.entry_threshold

    def should_wake(self, delta: float, consensus: float) -> bool:
        return (
            float(delta) <= self.cfg.wake_threshold
            and float(consensus) >= self.cfg.consensus_r
        )

    # -------------------------------------------------------------- GATE-D
    @staticmethod
    def creep_enabled() -> bool:
        return os.environ.get(CREEP_FLAG, "0") == "1"

    def ratify_for_egress(self, receipt: str) -> None:
        """Open the egress latch. Requires the M3-ACT two-sided gate receipt."""
        if not receipt or "M3-ACT" not in str(receipt):
            raise ValueError(
                "ratification requires an M3-ACT gate receipt; the dreamer stays "
                f"DIAGNOSTIC_ONLY. Got {receipt!r}"
            )
        self._egress_ratified = True
        self._ratification_receipt = str(receipt)

    def egress_permitted(self) -> bool:
        return self._egress_ratified

    def guarded_egress(self, *args: Any, **kwargs: Any):
        """The ONLY egress path. Raises until the M3-ACT gate is ratified."""
        if not self._egress_ratified:
            raise DreamEgressBlocked(
                "M3-ACT gate not ratified: the dreamer is DIAGNOSTIC_ONLY and must not "
                "touch egress. Ratify with ratify_for_egress('<M3-ACT ... receipt>')."
            )
        return self.core.plan(*args, **kwargs)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _consensus(waves: List[torch.Tensor]) -> float:
        if len(waves) < 2:
            return 1.0
        flat = torch.stack([w.reshape(-1).detach() for w in waves])
        flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-12)
        n = flat.shape[0]
        sim = flat @ flat.t()
        return float((sim.sum() - sim.diagonal().sum()) / (n * (n - 1)))

    # -------------------------------------------------------------- the dream
    def dream(
        self,
        active_wave: torch.Tensor,
        reference_wave: torch.Tensor,
        *,
        top_k: int = 4,
        max_steps: Optional[int] = None,
        allowed_actions: Optional[List[Any]] = None,
    ) -> DreamResult:
        """One bounded dream cycle. Returns telemetry; NEVER emits an action."""
        cfg = self.cfg
        limit = max_steps or cfg.max_dream_steps

        # Device parity: the adapter must match the incoming wave's device.
        # WIRED (not merely defined): the CUDA host caught a mismatch that local
        # CPU smoke could not.
        self._prepare_device(reference_wave)

        cs = self.core.candidate_set(active_wave, reference_wave, top_k=top_k,
                                     allowed_actions=allowed_actions)
        if len(cs) == 0:
            return DreamResult(False, 0, False, float("nan"), float("nan"), 0.0, 0.0,
                               self.creep_enabled())

        initial_delta = min(cs.delta_floats())
        if not self.should_enter(initial_delta):
            return DreamResult(
                entered=False, steps_taken=0, woke=False,
                initial_delta=initial_delta, final_delta=initial_delta,
                consensus_r=self._consensus(cs.waves), adapter_delta_norm=0.0,
                creep_enabled=self.creep_enabled(),
            )

        before = self.adapter.delta_norm()
        steps_taken = 0
        final_delta = initial_delta
        consensus = 0.0
        woke = False

        for _ in range(limit):
            # GATE-C: the adapter enters the SCORED wave, so the loss below has a
            # live gradient path to adapter.down / adapter.up.
            adapted_ref = self.adapter(reference_wave)
            cs = self.core.candidate_set(active_wave, adapted_ref, top_k=top_k,
                                         allowed_actions=allowed_actions)
            # GATE-B: live Clifford-coherence family, bounded [0,2].
            d = cs.deltas.clamp(0.0, 2.0)
            i_dark = 0.5 * (1.0 - torch.cos(d * torch.pi))     # dark-port power
            reward = (2.0 - d) / 2.0 - cfg.lambda_dark * i_dark
            loss = -reward.mean()

            grads = torch.autograd.grad(
                loss, [self.adapter.down, self.adapter.up],
                retain_graph=False, allow_unused=False,
            )

            if self.creep_enabled():
                # SGLD (sqrt(2 T dt) noise), applied only when the flag is ON.
                with torch.no_grad():
                    for p, g in zip((self.adapter.down, self.adapter.up), grads):
                        noise = torch.randn_like(p) * (2.0 * cfg.sgld_temp) ** 0.5
                        p.add_(-cfg.sgld_lr * g + noise)
            # default: grads computed (the path is PROVEN live) but NO weight change

            steps_taken += 1
            final_delta = min(cs.delta_floats())
            consensus = self._consensus(cs.waves)
            if self.should_wake(final_delta, consensus):
                woke = True
                break

        return DreamResult(
            entered=True, steps_taken=steps_taken, woke=woke,
            initial_delta=initial_delta, final_delta=final_delta,
            consensus_r=consensus,
            adapter_delta_norm=self.adapter.delta_norm() - before,
            creep_enabled=self.creep_enabled(),
        )
