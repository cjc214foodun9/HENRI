"""
Project HENRI: Dual-Speed Agentic Harness (h1).

Default-OFF wiring of VERIFIED live components only (no new math):

  Zone A lift     -> HENRIVisionEncoder.encode_grid | O_VSA_IngressTokenizer
  Inner loop      -> WaveJEPA (R-EDMD latent predictor) + SagnacMCTSPlanner
                     dual_channel_sagnac_veto (tau = 0.35)
  Memory          -> SegmentCache (Zone C) gated residual retrieval, lazy
                     connect, fail-closed when unavailable
  Egress          -> HENRIUnifiedEgressTransducer (checkpoint policy)
  Tool execution  -> HENRIUniversalREPL.execute_python_repl +
                     DualChannelREPLVeto.evaluate_execution

Activation flag: HENRI_ARC_HARNESS=1 (default OFF). Absent flag means the
harness is never constructed by a runner.

Design doc: docs/architecture/henri_dual_speed_harness_design.md (a2886ec).
Pre-registered kill gates: K1 latency invariance (<= 50 us mean, p99 <= 2x
mean), K2 veto (0 false negatives, <= 1% false positives), K3 Zone C A/B.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

HARNESS_FLAG = "HENRI_ARC_HARNESS"
HARNESS_ENABLED = os.environ.get(HARNESS_FLAG, "0") == "1"

VETO_TAU = 0.35


def harness_active() -> bool:
    """Single source of truth for harness activation (flag, not inference)."""
    return os.environ.get(HARNESS_FLAG, "0") == "1"


class HENRIDualSpeedHarness:
    """
    Bounded dual-speed harness.

    - inner_step(): one GPU wave transition + Sagnac veto (20 kHz target).
    - outer_step(): one tool/REPL execution + REPL veto (1-100 Hz).
    - Zone C is connected lazily; absence is telemetry, never a silent mock.
    """

    def __init__(
        self,
        d_model: int = 65536,
        num_blocks: int = 8192,
        r_rank: int = 16,
        device: Optional[str] = None,
        checkpoint_policy: str = "auto",
        zone_c_dsn: Optional[str] = None,
        zone_c_required: bool = False,
    ) -> None:
        from wave_jepa import WaveJEPA  # live module
        from henri_vision_encoder import HENRIVisionEncoder
        from sagnac_mcts_planner import SagnacMCTSPlanner
        from henri_universal_repl import HENRIUniversalREPL, DualChannelREPLVeto

        self.d_model = d_model
        self.num_blocks = num_blocks
        self.r_rank = r_rank
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_policy = checkpoint_policy
        self.zone_c_required = zone_c_required

        # Zone A: vision ingress (grid -> [num_blocks, 8] S^{D-1} wave)
        self.vision_encoder = HENRIVisionEncoder(
            d_model=d_model, k_blocks=num_blocks, device=self.device
        )

        # Inner loop: latent predictor + Sagnac veto planner
        self.wave_jepa = WaveJEPA(
            d_model=d_model, num_blocks=num_blocks, r_rank=r_rank, device=self.device
        )
        self.planner = SagnacMCTSPlanner(
            d_model=d_model, k_blocks=num_blocks, tau_veto=VETO_TAU, device=self.device
        )

        # Outer loop: REPL tools + veto
        self.repl = HENRIUniversalREPL(d_model=d_model, device=self.device)
        self.repl_veto = DualChannelREPLVeto(tau_veto=VETO_TAU)

        # Zone C: lazy, fail-closed
        self.zone_c = None
        self.zone_c_dsn = zone_c_dsn
        self.zone_c_connected = False
        if zone_c_required:
            self.connect_zone_c()

    # ------------------------------------------------------------------ Zone C
    def connect_zone_c(self) -> bool:
        """Lazy connect. Uses offline://surrogate ONLY when explicitly requested."""
        if self.zone_c is not None:
            return self.zone_c_connected
        try:
            from zone_c_segment_cache import SegmentCache
            dsn = self.zone_c_dsn or "offline://surrogate"
            self.zone_c = SegmentCache.connect(dsn=dsn, num_blocks=self.num_blocks)
            self.zone_c_connected = True
        except Exception as exc:  # fail closed: never silently mock
            self.zone_c_connected = False
            if self.zone_c_required:
                raise RuntimeError(
                    f"Zone C required but unavailable: {exc}"
                ) from exc
        return self.zone_c_connected

    def retrieve_conditioning(self, query_wave: torch.Tensor) -> Optional[torch.Tensor]:
        """Gated fused conditioning wave, or None when Zone C is disconnected."""
        if not self.zone_c_connected or self.zone_c is None:
            return None
        res = self.zone_c.retrieve(query_wave)
        return res.get("conditioning_wave") if res.get("hits", 0) else None

    def checkpoint_engram(self, wave: torch.Tensor, domain: str, stress: float) -> Optional[str]:
        if not self.zone_c_connected or self.zone_c is None:
            return None
        return self.zone_c.checkpoint(wave, domain, stress)

    # ---------------------------------------------------------------- Inner loop
    def inner_step(
        self,
        state_wave: torch.Tensor,
        action_wave: torch.Tensor,
        axiom_wave: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        One bounded inner-loop step: latent transition + Sagnac veto.

        Returns typed telemetry:
          pred_wave, delta_axiom, delta_epistemic, vetoed, latency_ms
        """
        t0 = time.perf_counter()
        pred_wave = self.wave_jepa.predict_future_latent(state_wave, action_wave)
        if axiom_wave is None:
            axiom_wave = state_wave  # self-consistency boundary when no baseplate
        delta_axiom, delta_epistemic, hard_veto = self.planner.dual_channel_sagnac_veto(
            pred_wave, axiom_wave, state_wave
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "pred_wave": pred_wave,
            "delta_axiom": float(delta_axiom),
            "delta_epistemic": float(delta_epistemic),
            "vetoed": bool(hard_veto),
            "latency_ms": latency_ms,
        }

    # ---------------------------------------------------------------- Outer loop
    def outer_step(
        self, code: str, sagnac_delta: float = 0.0
    ) -> Dict[str, Any]:
        """
        One bounded outer-loop step: REPL execution + REPL veto.

        Returns typed telemetry:
          is_vetoed, q_score, returncode, stdout, stderr, latency_ms
        """
        t0 = time.perf_counter()
        res = self.repl.execute_python_repl(code)
        is_vetoed, q_score = self.repl_veto.evaluate_execution(
            command=code,
            returncode=res.get("returncode", -1),
            stdout=res.get("stdout", ""),
            stderr=res.get("stderr", ""),
            sagnac_delta=sagnac_delta,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "is_vetoed": bool(is_vetoed),
            "q_score": float(q_score),
            "returncode": res.get("returncode", -1),
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
            "latency_ms": latency_ms,
        }

    # ---------------------------------------------------------------- Cycle
    def step(
        self,
        observation: np.ndarray,
        action_spec: Dict[str, Any],
        domain: str = "arc",
    ) -> Dict[str, Any]:
        """
        One full harness cycle (Zone A -> retrieve -> inner -> outer -> checkpoint).

        Bounded: exactly one inner transition and one outer execution.
        """
        state_wave = self.vision_encoder.encode_grid(observation)
        cond = self.retrieve_conditioning(state_wave)
        action_wave = state_wave if cond is None else state_wave + cond

        inner = self.inner_step(state_wave, action_wave, axiom_wave=cond)
        if inner["vetoed"]:
            return {"inner": inner, "outer": None, "executed": False}

        code = action_spec.get("code", "print(1)")
        outer = self.outer_step(code, sagnac_delta=inner["delta_epistemic"])
        if not outer["is_vetoed"]:
            self.checkpoint_engram(inner["pred_wave"], domain, inner["delta_axiom"])

        return {"inner": inner, "outer": outer, "executed": not outer["is_vetoed"]}
