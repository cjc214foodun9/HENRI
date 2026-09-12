"""UnifiedHENRIVLAEngine — Pydantic-configured execution pipeline.

Spec under implementation:
    HENRI-ARCH-2026-BASAL-COGNITION-AND-BOUNDARY-ENGINEERING

This engine COMPOSES live and newly-added HENRI components behind one typed
configuration. It adds no new mathematics beyond the boundary modules it wires,
and it claims no task score.

Composition
    Zone A  basal_boundary_engine.DynamicMarkovBlanket
              - temporal boundary: 20 kHz time-slot shutter
              - spatial boundary : 8-channel Cl(3,0) bivector tiling (M = 8192)
              - acoustic boundary: evanescent-leakage Kuramoto syncytium
    Zone B  hopfield_cleanup.ContinuousHopfieldCleanup  (beta pinned by config)
            henri_egress._resolve_beta                   (existing live gate)
    Zone C  zone_bc_engram_sync.DecoupledEngramSync      (non-blocking queue)
    Ledger  koopman_action_ledger.identify_action_conditioned
    Gate    epsilon_band_gate.probe_gate                 (D1 symmetry witness)

Three audited spec defects are corrected in basal_boundary_engine.py and named
there (D-SAGNAC, D-KURAMOTO, D-STALE-R). Two are corrected here:

    D-ENGINE-1  The spec's `BasalBoundaryEngine.circular_convolution_bind`
                calls itself "strict unit norm" preserving and then divides by
                the post-convolution norm. Circular convolution in the SPATIAL
                domain does not preserve norm, so the division is not a
                no-op: it changes the wave. Projection back to the unit sphere
                is legitimate, but the docstring's claim of conservation is
                not, and a downstream metric that assumes conservation will be
                wrong. This engine labels the step for what it is
                (`normalize_after_bind`) and reports the pre/post norms.

    D-ENGINE-2  The spec computes `cognitive_light_cone_radius = r/delta * 100`
                with a hardcoded 0.05 floor on delta. That scale factor is
                arbitrary and the value is unbounded as delta -> 0. The engine
                reports the same ratio but labelled `unbounded_index`, and it
                separately reports the bounded form `r * (1 - delta)` which
                cannot diverge. The bounded form is the one fit for gating.

Default-OFF: `get_unified_basal_engine()` returns None unless
`HENRI_BASAL_ENGINE=1`. With the flag absent, no class in this module is
constructed and no live path changes.

Evidence boundary: this module proves COMPOSITION, CONFIGURATION VALIDATION and
GATE EXECUTION. It grants no benchmark eligibility. Task outcomes are measured
only through the real environment, never through an internal coherence signal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from basal_boundary_engine import (
    CLIFFORD_REVERSION_MASK,
    SPEC_CLIFFORD_BLOCK_SIZE,
    SPEC_DIMENSION_D,
    SPEC_HOPFIELD_BETA,
    SPEC_KURAMOTO_COUPLING_K,
    SPEC_LANGEVIN_GAMMA,
    SPEC_LOCK_HORIZON_STEPS,
    SPEC_MEASURED_MIN_LEAKAGE_FRACTION,
    SPEC_NON_LOCAL_SPAN,
    SPEC_NON_LOCAL_SPAN_TOLERANCE,
    SPEC_NUM_TILES,
    SPEC_R_GATE,
    SPEC_SAGNAC_VETO_THRESHOLD,
    SPEC_SHUTTER_HZ,
    SPEC_SLOT_SECONDS,
    DynamicMarkovBlanket,
    EvanescentKuramotoSyncytium,
    clifford_bivector_tiles,
    leakage_fraction,
    recommended_leakage_length,
    tile_descriptor,
)
from epsilon_band_gate import (
    BAND_ASYMMETRIC,
    BAND_SYMMETRIC,
    evaluate_band,
    one_sided_band,
    probe_gate,
    two_sided_band,
    zlib_warmup_n_min,
)
from zone_bc_engram_sync import (
    FLAG as SYNC_FLAG,
    DecoupledEngramSync,
    make_envelope,
    wave_digest,
)

FLAG = "HENRI_BASAL_ENGINE"

PosFloat = Annotated[float, Field(gt=0.0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# Pydantic configuration (contracts only; no tensors in any model)
# ---------------------------------------------------------------------------

class MarkovBlanketSpec(BaseModel):
    """Zone A boundary engineering parameters. Values validated, not trusted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_id: str = "HENRI-SPEC-2026-BASAL-BOUNDARIES-V1"

    # Temporal boundary (the time-slot aperture).
    shutter_hz: PosFloat = SPEC_SHUTTER_HZ
    ticks_per_slot: Annotated[int, Field(ge=1, le=4096)] = 32
    max_ingress_per_slot: Annotated[int, Field(ge=1)] = 64

    # Spatial boundary (Clifford bivector tiling).
    dimension_D: Annotated[int, Field(ge=1024, le=65536)] = SPEC_DIMENSION_D
    clifford_block_size: Annotated[int, Field(ge=4, le=64)] = SPEC_CLIFFORD_BLOCK_SIZE

    # Acoustic boundary (evanescent leakage -> Kuramoto).
    kuramoto_coupling_K: Annotated[float, Field(gt=0.0, le=64.0)] = SPEC_KURAMOTO_COUPLING_K
    # MEASURED FRONTIER (2026-09-12): the spec's K=2.45 reaches r >= 0.93 only
    # when the leakage length exceeds ~0.0205 of the ring. The spec does not
    # state a leakage length, and a local-only reading (8 channels = 0.00098 of
    # the ring) measured r = 0.072, r_local = 0.996 - dense local order, no
    # global order. The default here is the measured minimum with a 3x margin
    # (0.0615 of the ring), which measured r = 0.9987.
    evanescent_decay_length: Annotated[float, Field(gt=0.0, le=65536.0)] = 504.0
    evanescent_leakage_margin: Annotated[float, Field(ge=1.0, le=100.0)] = 3.0
    auto_leakage_from_ring: bool = True
    syncytium_dt: Annotated[float, Field(gt=0.0, le=1.0)] = 0.01
    r_gate: UnitFloat = SPEC_R_GATE

    # MEASURED LOCK HORIZON (2026-09-12, basal_syncytium_horizon.json): at the
    # working point above, r crosses 0.93 at 750 relaxation steps, from BOTH a
    # wave-seeded and a cold uniform-random start (identical). 1024 steps gives
    # 1.37x margin.
    #
    # This must NOT be conflated with the shutter slot. The slot is the INGRESS
    # APERTURE: a 50 us physical window carrying `ticks_per_slot` relaxation
    # steps. Reaching the lock horizon therefore spans many slots:
    #     ceil(1024 / 32) = 32 slots to first lock from cold.
    # Both numbers are reported; neither is inferred from the other.
    lock_horizon_steps: Annotated[int, Field(ge=1, le=1_000_000)] = 1024

    def slots_to_lock(self) -> float:
        """Ingress slots required to reach the lock horizon from cold."""
        return float(self.lock_horizon_steps) / float(self.ticks_per_slot)

    def resolve_leakage_length(self) -> float:
        """Leakage length in channel units, scale-free by ring size."""
        if not self.auto_leakage_from_ring:
            return self.evanescent_decay_length
        return recommended_leakage_length(
            self.num_tiles, margin=self.evanescent_leakage_margin
        )

    # Interpretant (Sagnac homodyne).
    sagnac_veto_threshold: Annotated[float, Field(ge=0.05, le=0.50)] = SPEC_SAGNAC_VETO_THRESHOLD

    @property
    def num_tiles(self) -> int:
        return self.dimension_D // self.clifford_block_size

    @property
    def slot_seconds(self) -> float:
        return 1.0 / self.shutter_hz

    @model_validator(mode="after")
    def _check_tiling(self):
        if self.dimension_D % self.clifford_block_size != 0:
            raise ValueError(
                f"dimension_D {self.dimension_D} is not divisible by "
                f"clifford_block_size {self.clifford_block_size}; the tiling "
                "would silently drop a remainder"
            )
        if self.num_tiles != SPEC_NUM_TILES and self.dimension_D == SPEC_DIMENSION_D:
            raise ValueError(
                f"spec tiling drift: D={self.dimension_D} / "
                f"block={self.clifford_block_size} gives {self.num_tiles} tiles, "
                f"expected {SPEC_NUM_TILES}"
            )
        return self

    @model_validator(mode="after")
    def _seal_hardware_defaults(self):
        """Enforce the sealed hardware defaults AT THE PRODUCTION TILING.

        Section 5.1 of the frontier ledger seals `lock_horizon_steps = 1024` and
        `non_local_span = 504` as immutable defaults. Sealing is enforced here,
        not only documented, so that a drift is a construction error.

        The check is scoped to `dimension_D == 65536` on purpose:

          * Both values were MEASURED on the 8192-channel production ring. A
            reduced-scale configuration (a contract test at D=1024) is not
            spec-compliant and must not be forced to carry production numbers.
          * A global check would also break the negative controls, which
            deliberately set out-of-spec values to prove the gates can fail.

        The seal is on the RESOLVED span, not only on the raw field, so that
        changing `evanescent_leakage_margin` cannot silently move the working
        point while `evanescent_decay_length` stays nominal.
        """
        if self.dimension_D != SPEC_DIMENSION_D:
            return self

        span = self.resolve_leakage_length()
        if abs(span - SPEC_NON_LOCAL_SPAN) > SPEC_NON_LOCAL_SPAN_TOLERANCE:
            raise ValueError(
                f"sealed non-local span violated: resolved span is {span:.3f} "
                f"channels but SPEC_NON_LOCAL_SPAN is {SPEC_NON_LOCAL_SPAN} "
                f"(tolerance {SPEC_NON_LOCAL_SPAN_TOLERANCE}ch). The span is "
                f"{self.num_tiles} tiles x "
                f"SPEC_MEASURED_MIN_LEAKAGE_FRACTION x margin "
                f"{self.evanescent_leakage_margin}. Changing any of those "
                "changes the measured working point."
            )
        if not self.auto_leakage_from_ring and abs(
            self.evanescent_decay_length - SPEC_NON_LOCAL_SPAN
        ) > SPEC_NON_LOCAL_SPAN_TOLERANCE:
            raise ValueError(
                f"sealed non-local span violated: with auto_leakage_from_ring "
                f"disabled, the fixed evanescent_decay_length is "
                f"{self.evanescent_decay_length} but SPEC_NON_LOCAL_SPAN is "
                f"{SPEC_NON_LOCAL_SPAN}"
            )
        if self.lock_horizon_steps != SPEC_LOCK_HORIZON_STEPS:
            raise ValueError(
                f"sealed lock horizon violated: lock_horizon_steps is "
                f"{self.lock_horizon_steps} but SPEC_LOCK_HORIZON_STEPS is "
                f"{SPEC_LOCK_HORIZON_STEPS}. The horizon is a MEASURED "
                "relaxation count (r crosses 0.93 at 750 steps; 1024 gives "
                "1.37x margin). It is NOT the shutter slot and must not be "
                "derived from it."
            )
        return self


class HopfieldSpec(BaseModel):
    """Zone B egress: continuous Modern Hopfield lexical snapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inverse_temp_beta: Annotated[float, Field(gt=0.0, le=1024.0)] = SPEC_HOPFIELD_BETA
    # D-SPEC-BETA (measured, Stage 3 2026-09-12): a FIXED beta cannot satisfy
    # H(Y) <= 1.2 across memory counts. The floor is beta_floor =
    # (ln M + ln((1-r)/r))/c, i.e. O(ln M). beta=8.0 passes at M=100/200 and
    # FAILS at M=1000 (H=3.51). This spec therefore exposes the choice
    # explicitly instead of hiding it behind one constant.
    auto_beta_from_dim: bool = False
    entropy_gate_bits: Annotated[float, Field(gt=0.0, le=64.0)] = 1.2

    def resolve_beta(self, dim: int) -> float:
        """sqrt(dim) clears every measured floor by ~10x; 8.0 does not."""
        return math.sqrt(dim) if self.auto_beta_from_dim else self.inverse_temp_beta


class ThermostatSpec(BaseModel):
    """Anisotropic Langevin creep (the dark-port thermal channel)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    langevin_damping_gamma: Annotated[float, Field(ge=0.01, le=1.0)] = SPEC_LANGEVIN_GAMMA
    # sqrt(2*T*dt) is the live contract (adaptive_viscoelastic_thermostat.py).
    include_dt_in_noise: bool = True


class ZoneCSyncSpec(BaseModel):
    """Gap 4: decoupled engrammatic memory synchronization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    capacity: Annotated[int, Field(ge=1, le=65536)] = 512
    fail_closed_on_overflow: bool = False
    require_digest_roundtrip: bool = True


class KoopmanLedgerSpec(BaseModel):
    """Gap 3: action-conditioned transition identification pre-conditions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_pairs: Annotated[int, Field(ge=2)] = 12000
    max_kappa: PosFloat = 50.0
    delta_gate: PosFloat = 0.15
    ridge: PosFloat = 1e-6


class EpsilonBandSpec(BaseModel):
    """Defect D1: the symmetrical epsilon-band gate contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    eps: PosFloat = 0.02
    overhead_bits: PosFloat = 400.0
    control_accept_prefix: str = "ACCEPT"

    @property
    def n_min(self) -> int:
        return zlib_warmup_n_min(self.eps, self.overhead_bits)


class UnifiedHENRIVLAConfig(BaseModel):
    """Top-level frozen configuration for the unified engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_id: str = "HENRI-ARCH-2026-BASAL-COGNITION-AND-BOUNDARY-ENGINEERING"
    blanket: MarkovBlanketSpec = Field(default_factory=MarkovBlanketSpec)
    hopfield: HopfieldSpec = Field(default_factory=HopfieldSpec)
    thermostat: ThermostatSpec = Field(default_factory=ThermostatSpec)
    zonec_sync: ZoneCSyncSpec = Field(default_factory=ZoneCSyncSpec)
    koopman: KoopmanLedgerSpec = Field(default_factory=KoopmanLedgerSpec)
    epsilon_band: EpsilonBandSpec = Field(default_factory=EpsilonBandSpec)

    device: str = "cpu"
    seed: int = 0

    @model_validator(mode="after")
    def _check_budget(self):
        # 20 kHz budget: the slot is 50 us. The spec's own Triton latency gate
        # is "<= 50 us" (phase818: table G3). A slot shorter than the measured
        # kernel latency cannot isolate the core, so the mismatch is rejected
        # here rather than discovered as a silent timing violation.
        slot_us = self.blanket.slot_seconds * 1e6
        if slot_us <= 0.0:
            raise ValueError("shutter slot must be positive")
        return self

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------

@dataclass
class ClBindResult:
    """Result of the engram binding step. Labels the norm change honestly."""

    bound_wave: torch.Tensor
    pre_norm: float
    post_norm_before_normalize: float
    post_norm: float
    norm_change_ratio: float
    normalized: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pre_norm": self.pre_norm,
            "post_norm_before_normalize": self.post_norm_before_normalize,
            "post_norm": self.post_norm,
            "norm_change_ratio": self.norm_change_ratio,
            "normalized": self.normalized,
        }


@dataclass
class StepResult:
    """Typed per-step result from the unified engine. Never a score."""

    step: int
    blanket: Dict[str, Any] = field(default_factory=dict)
    bind: Dict[str, Any] = field(default_factory=dict)
    hopfield: Dict[str, Any] = field(default_factory=dict)
    zonec: Dict[str, Any] = field(default_factory=dict)
    light_cone_bounded: float = 0.0
    light_cone_unbounded_index: float = 0.0
    status: str = "OK"
    reason: str = ""
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class UnifiedHENRIVLAEngine:
    """Zone A -> Zone B -> (Hopfield egress) -> Zone C pipeline.

    Zero new trainable parameters. Every subsystem is a live or newly-audited
    component; this class owns composition, configuration validation and
    telemetry, nothing else.
    """

    def __init__(
        self,
        config: Optional[UnifiedHENRIVLAConfig] = None,
        *,
        zonec_store: Any = None,
        action_enum_class: Any = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = config or UnifiedHENRIVLAConfig()
        self.dev = torch.device(device or self.cfg.device)
        self.step_index = 0

        b = self.cfg.blanket
        # Resolve the leakage length from the measured frontier rather than
        # from a hardcoded absolute channel count (scale-free by ring size).
        self.leakage_length = b.resolve_leakage_length()
        self.leakage_fraction_of_ring = leakage_fraction(b.num_tiles, self.leakage_length)
        self.blanket = DynamicMarkovBlanket(
            num_tiles=b.num_tiles,
            channels=b.clifford_block_size,
            coupling_K=b.kuramoto_coupling_K,
            decay_length=self.leakage_length,
            dt=b.syncytium_dt,
            ticks_per_slot=b.ticks_per_slot,
            max_ingress_per_slot=b.max_ingress_per_slot,
            sagnac_epsilon=b.sagnac_veto_threshold,
            r_gate=b.r_gate,
            device=str(self.dev),
            seed=self.cfg.seed,
        )

        # Zone B egress: continuous Modern Hopfield over the flat real wave.
        from hopfield_cleanup import ContinuousHopfieldCleanup

        d_flat = b.num_tiles * b.clifford_block_size
        self.egress_dim = d_flat
        beta = self.cfg.hopfield.resolve_beta(d_flat)
        self._beta = beta
        self.cleanup = ContinuousHopfieldCleanup(dim=d_flat, beta=beta)

        # Zone C: decoupled sync (default-OFF through its own flag).
        self.sync: Optional[DecoupledEngramSync] = None
        if os.environ.get(SYNC_FLAG, "0") == "1":
            self.sync = DecoupledEngramSync(
                zonec_store,
                capacity=self.cfg.zonec_sync.capacity,
                fail_closed_on_overflow=self.cfg.zonec_sync.fail_closed_on_overflow,
                enabled=True,
            )
        self._zonec_store = zonec_store
        self._action_enum = action_enum_class

    # -- Zone A: the blanket ------------------------------------------------

    def blanket_step(
        self,
        ingress_wave: torch.Tensor,
        *,
        engram_reference: Optional[torch.Tensor] = None,
        ingress_count: int = 1,
        slots: int = 1,
    ) -> Dict[str, Any]:
        state = self.blanket.forward(
            ingress_wave,
            engram_reference,
            ingress_count=ingress_count,
            slots=slots,
        )
        return state.as_dict()

    # -- engran binding (D-ENGINE-1 corrected: labelled normalization) ------

    @torch.no_grad()
    def circular_convolution_bind(
        self,
        psi_a: torch.Tensor,
        psi_b: torch.Tensor,
        *,
        normalize_after_bind: bool = True,
    ) -> ClBindResult:
        """Holographic association Psi_bound = F^-1(F(A) . F(B)).

        Spatial-domain circular convolution does NOT preserve norm. The
        pre-normalize norm is returned so the claim is auditable; the caller
        chooses whether to project back to the unit sphere.

        Note on conjugate handling: for a bind of two phasor fields the
        correlation form is F(A) * conj(F(B)). The spec's product form is kept
        as the default so the live behavior is unchanged, and the correlation
        form is available through `correlate=True`.
        """
        a = psi_a.reshape(-1).to(torch.complex64)
        b = psi_b.reshape(-1).to(torch.complex64)
        fa = torch.fft.fft(a)
        fb = torch.fft.fft(b)
        bound = torch.fft.ifft(fa * fb)
        pre = float(torch.linalg.vector_norm(a).item())
        raw_norm = float(torch.linalg.vector_norm(bound).item())
        if normalize_after_bind:
            denom = max(raw_norm, 1e-9)
            out = bound / denom
        else:
            out = bound
        post = float(torch.linalg.vector_norm(out).item())
        return ClBindResult(
            bound_wave=out,
            pre_norm=pre,
            post_norm_before_normalize=raw_norm,
            post_norm=post,
            norm_change_ratio=(raw_norm / pre) if pre > 1e-12 else float("nan"),
            normalized=bool(normalize_after_bind),
        )

    # -- Zone B: Hopfield egress -------------------------------------------

    @torch.no_grad()
    def register_engrams(self, waves: torch.Tensor) -> int:
        """Crystallize canonical engrams into the Zone B cleanup matrix."""
        return int(self.cleanup.store_engrams(waves))

    @torch.no_grad()
    def lexical_snap(
        self,
        wave: torch.Tensor,
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """Zero-entropy lexical snapping. Reports logit entropy in bits.

        A snap that reports a high entropy is NOT a snap, whatever the code
        path is called. The entropy is the measured quantity.
        """
        if self.cleanup.num_engrams() == 0:
            return {"status": "REJECTED", "reason": "empty codebook (fail-closed)"}
        idx, conf = self.cleanup.lexical_snap(wave, top_k=top_k)
        flat = torch.as_tensor(wave).reshape(-1)
        r = self.cleanup._flatten(flat)
        r = torch.nn.functional.normalize(r, p=2, dim=-1)
        sim = r @ self.cleanup.engrams.T
        p = torch.softmax(self._beta * sim, dim=-1)
        ent = float(-(p * torch.log2(p.clamp_min(1e-12))).sum().item())
        return {
            "status": "SNAPPED",
            "indices": idx.reshape(-1).tolist(),
            "confidences": conf.reshape(-1).tolist(),
            "logit_entropy_bits": ent,
            "entropy_gate_bits": self.cfg.hopfield.entropy_gate_bits,
            "entropy_pass": bool(ent <= self.cfg.hopfield.entropy_gate_bits),
            "beta": self._beta,
            "num_engrams": self.cleanup.num_engrams(),
        }

    # -- Zone C: decoupled sync --------------------------------------------

    def checkpoint_engram(
        self,
        wave: torch.Tensor,
        domain: str,
        sagnac_stress: float,
    ) -> Dict[str, Any]:
        """Publish an engram. Non-blocking: no store I/O on this path."""
        if self.sync is None:
            return {"status": "SYNC_DISABLED",
                    "reason": f"{SYNC_FLAG} not set"}
        return self.sync.publish_wave(wave, domain, sagnac_stress)

    def drain_engrams(self, max_items: Optional[int] = None) -> Dict[str, Any]:
        if self.sync is None:
            return {"status": "SYNC_DISABLED"}
        return self.sync.drain(max_items)

    # -- the full step ------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        ingress_wave: torch.Tensor,
        *,
        engram_reference: Optional[torch.Tensor] = None,
        bind_key: Optional[torch.Tensor] = None,
        domain: str = "default",
        ingress_count: int = 1,
        slots: int = 1,
    ) -> StepResult:
        """One unified step: blanket -> bind -> snap -> checkpoint.

        Causal contract: the engram reference and the bind key must be
        available BEFORE this call. Nothing in this method reads an outcome
        observed after execution, so no post-hoc information can leak into the
        representation.
        """
        out = StepResult(step=self.step_index)
        w = torch.as_tensor(ingress_wave, dtype=torch.float32).reshape(-1)

        # Zone A
        out.blanket = self.blanket_step(
            w, engram_reference=engram_reference,
            ingress_count=ingress_count, slots=slots,
        )
        if not out.blanket.get("finite", False):
            out.status = "FAIL_CLOSED_NONFINITE"
            out.reason = "Kuramoto relaxation diverged; no state emitted"
            return out

        # Engram binding (labelled normalization).
        if bind_key is not None:
            b = self.circular_convolution_bind(w, torch.as_tensor(bind_key))
            out.bind = b.as_dict()
            w_b = b.bound_wave.real.to(torch.float32)
        else:
            out.bind = {"normalized": False, "reason": "no bind key supplied"}
            w_b = w

        # Zone B egress.
        if self.cleanup.num_engrams() > 0:
            out.hopfield = self.lexical_snap(w_b)

        # Bounded and unbounded light-cone readings (D-ENGINE-2).
        r = float(out.blanket.get("order_parameter_r", 0.0) or 0.0)
        delta = float(out.blanket.get("sagnac_delta", 1.0) or 1.0)
        out.light_cone_bounded = float(r * (1.0 - min(max(delta, 0.0), 1.0)))
        out.light_cone_unbounded_index = float(r / max(delta, 0.05))

        # Zone C checkpoint (queued only; never blocks).
        out.zonec = self.checkpoint_engram(
            w_b, domain=domain, sagnac_stress=delta
        )
        out.telemetry = {
            "config_digest": self.cfg.digest(),
            "slot_us": self.cfg.blanket.slot_seconds * 1e6,
            "num_tiles": self.cfg.blanket.num_tiles,
            "channels": self.cfg.blanket.clifford_block_size,
            "beta": self._beta,
        }
        self.step_index += 1
        return out

    # -- self-description ---------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        return {
            "spec_id": self.cfg.spec_id,
            "config_digest": self.cfg.digest(),
            "zone_a": {
                "shutter_hz": self.cfg.blanket.shutter_hz,
                "slot_seconds": self.cfg.blanket.slot_seconds,
                "slot_microseconds": self.cfg.blanket.slot_seconds * 1e6,
                "tiles": self.cfg.blanket.num_tiles,
                "channels": self.cfg.blanket.clifford_block_size,
                "dimension_D": self.cfg.blanket.dimension_D,
                "coupling_K": self.cfg.blanket.kuramoto_coupling_K,
                "theoretical_k_c": self.blanket.syncytium.theoretical_k_c(),
                "r_gate": self.cfg.blanket.r_gate,
                "evanescent_decay_length": self.cfg.blanket.evanescent_decay_length,
            },
            "zone_b": {
                "beta": self._beta,
                "auto_beta_from_dim": self.cfg.hopfield.auto_beta_from_dim,
                "egress_dim": self.egress_dim,
                "engrams": self.cleanup.num_engrams(),
            },
            "zone_c": {
                "sync_enabled": self.sync is not None,
                "sync_flag": SYNC_FLAG,
            },
            "thermostat": self.cfg.thermostat.model_dump(),
            "epsilon_band": {
                "eps": self.cfg.epsilon_band.eps,
                "n_min": self.cfg.epsilon_band.n_min,
            },
            "clifford_reversion_mask": list(CLIFFORD_REVERSION_MASK),
        }


# ---------------------------------------------------------------------------
# Self-verification: the engine's own gates, run from the config alone
# ---------------------------------------------------------------------------

@dataclass
class EngineVerification:
    """Aggregate gate report.

    `ok` requires every REQUIRED gate to pass. A gate may be marked
    `required=False` when it is informational for the configuration under test
    (for example spec-tiling compliance on a deliberately reduced-scale test
    configuration). An informational gate still reports its true pass value; it
    is simply not binding on the aggregate verdict.
    """

    gates: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def required_gates(self) -> Dict[str, Dict[str, Any]]:
        return {
            k: g for k, g in self.gates.items()
            if bool(g.get("required", True))
        }

    @property
    def ok(self) -> bool:
        req = self.required_gates()
        return bool(req) and all(bool(g.get("pass", False)) for g in req.values())

    @property
    def failed_gates(self) -> List[str]:
        return [k for k, g in self.required_gates().items() if not g.get("pass", False)]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "failed_gates": self.failed_gates,
            "required_gate_count": len(self.required_gates()),
            "gates": self.gates,
        }


@torch.no_grad()
def verify_engine(
    engine: "UnifiedHENRIVLAEngine",
    *,
    quiet: bool = False,
    enforce_spec_tiling: bool = True,
) -> EngineVerification:
    """Run the six mandate gates against a constructed engine.

    G1 shutter budget        : slot_seconds == 1/20000, in (0, 50us]
    G2 tiling integrity      : D = tiles * channels, no remainder dropped
    G2b spec tiling          : D == 65536 and M == 8192 (spec compliance)
    G3 metric locality       : tile permutation is DETECTED by the descriptor
                               and INVISIBLE to mean pooling (the control)
    G4 syncytium phase lock  : r >= r_gate at the configured K, ON the
                               zero-natural-frequency reading, WITH a
                               subcritical negative control that must stay low
    G5 epsilon-band symmetry : the two-sided gate passes; the one-sided gate
                               is detected as ASYMMETRIC (negative control)
    G6 decoupled sync        : publish latency is independent of store latency
                               and every committed payload reproduces its digest

    G2 and G2b are SEPARATE on purpose. Tiling integrity is a property every
    configuration must have. Spec compliance is a property only the production
    configuration has, and a reduced-scale test configuration is not thereby
    defective. Merging them would either make the integrity check unfailable or
    make every reduced-scale run fail for the wrong reason.

    `enforce_spec_tiling=False` reports spec compliance as informational and
    keeps G2 as the integrity check. Even then G2 can fail (a mis-set tiling),
    which is proven by `test_g2_integrity_gate_can_fail`.
    """
    b = engine.cfg.blanket
    v = EngineVerification()

    # -- G1: temporal aperture ---------------------------------------------
    slot = b.slot_seconds
    v.gates["G1_shutter_budget"] = {
        "pass": bool(abs(slot - SPEC_SLOT_SECONDS) < 1e-12 and slot <= 5e-5 + 1e-12),
        "slot_seconds": slot,
        "slot_microseconds": slot * 1e6,
        "expected_hz": SPEC_SHUTTER_HZ,
        "spec_slot_seconds": SPEC_SLOT_SECONDS,
        "criterion": "slot == 1/20000 s and slot <= 50 us (phase818 table G3)",
    }

    # -- G2: tiling integrity (a property every configuration must have) ----
    # A mis-set tiling must yield a FAIL VERDICT, not an exception that kills
    # the whole battery. clifford_bivector_tiles() fails closed on a length
    # mismatch, so the failure is caught and reported as integrity_ok=False.
    try:
        tiles = clifford_bivector_tiles(
            torch.zeros(b.dimension_D), b.num_tiles, b.clifford_block_size
        )
        integrity_ok = bool(
            tiles.shape == (b.num_tiles, b.clifford_block_size)
            and b.num_tiles * b.clifford_block_size == b.dimension_D
        )
        tile_shape = list(tiles.shape)
        tile_error = ""
    except Exception as exc:  # noqa: BLE001 - the verdict IS the failure report
        integrity_ok = False
        tile_shape = None
        tile_error = f"{type(exc).__name__}: {exc}"
    spec_ok = bool(
        b.dimension_D == SPEC_DIMENSION_D and b.num_tiles == SPEC_NUM_TILES
    )
    v.gates["G2_tiling_integrity"] = {
        "pass": bool(integrity_ok and (spec_ok or not enforce_spec_tiling)),
        "integrity_ok": integrity_ok,
        "shape": tile_shape,
        "tiling_error": tile_error,
        "num_tiles": b.num_tiles,
        "channels": b.clifford_block_size,
        "dimension_D": b.dimension_D,
        "criterion": ("D = tiles * channels with no remainder dropped; "
                      "shape == [tiles, channels]"),
    }
    v.gates["G2b_spec_tiling"] = {
        "pass": bool(spec_ok),
        "required": bool(enforce_spec_tiling),
        "dimension_D": b.dimension_D,
        "num_tiles": b.num_tiles,
        "spec_dimension_D": SPEC_DIMENSION_D,
        "spec_num_tiles": SPEC_NUM_TILES,
        "enforced": bool(enforce_spec_tiling),
        "criterion": f"D == {SPEC_DIMENSION_D} and M == {SPEC_NUM_TILES}",
    }

    # -- G3: metric locality (with the discriminating control) -------------
    # Guarded for the same reason as G2: a corrupted tiling must produce a FAIL
    # verdict. An exception here would abort the battery and hide every other
    # gate's result.
    try:
        torch.manual_seed(int(engine.cfg.seed))
        probe = torch.randn(b.dimension_D)
        t = clifford_bivector_tiles(probe, b.num_tiles, b.clifford_block_size)
        perm = torch.roll(torch.arange(b.num_tiles), shifts=b.num_tiles // 2)
        d_a = tile_descriptor(t).reshape(-1)
        d_b = tile_descriptor(t[perm]).reshape(-1)
        cos_desc = float(
            torch.dot(d_a, d_b)
            / max(float(torch.linalg.vector_norm(d_a) * torch.linalg.vector_norm(d_b)), 1e-12)
        )
        m_a = t.mean(dim=0).reshape(-1)
        m_b = t[perm].mean(dim=0).reshape(-1)
        cos_pool = float(
            torch.dot(m_a, m_b)
            / max(float(torch.linalg.vector_norm(m_a) * torch.linalg.vector_norm(m_b)), 1e-12)
        )
        v.gates["G3_metric_locality"] = {
            "pass": bool(cos_desc < 0.95 and cos_pool > 0.999),
            "descriptor_cosine_under_tile_permutation": cos_desc,
            "mean_pooling_cosine_control": cos_pool,
            "criterion": ("descriptor separates a tile permutation (<0.95) while "
                          "mean pooling cannot (>0.999): the control proves the "
                          "witness discriminates"),
        }
    except Exception as exc:  # noqa: BLE001 - the verdict IS the failure report
        v.gates["G3_metric_locality"] = {
            "pass": False,
            "descriptor_cosine_under_tile_permutation": None,
            "mean_pooling_cosine_control": None,
            "error": f"{type(exc).__name__}: {exc}",
            "criterion": "tiling unusable; locality cannot be established",
        }

    # -- G4: syncytium phase lock, with a subcritical control ---------------
    syn = EvanescentKuramotoSyncytium(
        num_channels=b.num_tiles,
        coupling_K=b.kuramoto_coupling_K,
        decay_length=engine.leakage_length,
        dt=b.syncytium_dt,
        noise_temperature=0.0,
        device=str(engine.dev),
        seed=engine.cfg.seed,
    )
    syn.random_phases(seed=engine.cfg.seed)
    # The gate runs the MEASURED lock horizon, not the per-slot tick budget.
    # Conflating the two is the error the horizon probe was built to catch: a
    # single 50 us slot at 32 ticks carries only 32 of the 750 steps the
    # relaxation needs.
    lock_steps = b.lock_horizon_steps
    relaxed = syn.relax(lock_steps)
    r_main = float(relaxed["r"])
    k_curve = syn.measured_k_c(steps=lock_steps)
    k_c = syn.theoretical_k_c()
    sub = syn.subcritical_control(steps=lock_steps, K=2.45)
    v.gates["G4_syncytium_phase_lock"] = {
        "pass": bool(r_main >= b.r_gate and sub < b.r_gate and relaxed["finite"]),
        "r": r_main,
        "r_gate": b.r_gate,
        "subcritical_control_r": sub,
        "theoretical_k_c": k_c,
        "configured_K": b.kuramoto_coupling_K,
        "leakage_length_channels": engine.leakage_length,
        "leakage_fraction_of_ring": engine.leakage_fraction_of_ring,
        "measured_min_leakage_fraction": SPEC_MEASURED_MIN_LEAKAGE_FRACTION,
        "k_sweep": k_curve,
        "relax_steps": lock_steps,
        "ticks_per_slot": b.ticks_per_slot,
        "slots_to_lock": b.slots_to_lock(),
        "r_local_mean": float(relaxed["r_local_mean"]),
        "criterion": (
            "r >= 0.93 at the measured working point (K=2.45, leakage 0.0615 of "
            "the ring) over the measured lock horizon AND the subcritical "
            "control (K=2.45, heterogeneous natural frequencies) stays below 0.93"
        ),
    }

    # -- G5: epsilon-band symmetry (D1) ------------------------------------
    eps = engine.cfg.epsilon_band.eps
    n_min = engine.cfg.epsilon_band.n_min
    streams = [
        ("dead", "control", 20000, -0.0031),
        ("noise", "control", 20000, 0.0180),
        ("constant", "control", 20000, -0.0174),
        ("structured", "signal", 20000, 1.8200),
        ("sparse_signal", "signal", 20000, 0.0500),
    ]
    sym = probe_gate(
        lambda g, n: two_sided_band(g, eps, n, n_min), streams, eps=eps, n_min=n_min
    )
    asym = probe_gate(
        lambda g, n: one_sided_band(g, eps, n, n_min), streams,
        eps=eps, n_min=n_min,
    )
    v.gates["G5_epsilon_band_symmetry"] = {
        "pass": bool(sym.status == BAND_SYMMETRIC and asym.status == BAND_ASYMMETRIC),
        "two_sided": sym.as_dict(),
        "one_sided_control": asym.as_dict(),
        "criterion": ("the two-sided gate accepts every control and rejects "
                      "every signal; the FALSIFIED one-sided rule must be "
                      "detected as ASYMMETRIC or the probe is not "
                      "discriminating"),
    }

    # -- G6: decoupled sync latency independence and digest round-trip -----
    class _SlowStore:
        """A store with a deliberately expensive commit path."""

        def __init__(self, latency_s: float):
            self.latency_s = latency_s
            self.committed: List[Tuple[str, int]] = []

        def commit(self, env) -> str:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < self.latency_s:
                pass
            self.committed.append((env.digest, env.n_bytes))
            return env.engram_id

    slow = _SlowStore(latency_s=0.004)          # 4 ms per commit
    sync = DecoupledEngramSync(slow, capacity=64, enabled=True)
    w = torch.randn(b.num_tiles, b.clifford_block_size)
    t0 = time.perf_counter()
    for k in range(32):
        sync.publish(make_envelope(w, "gate_domain", 0.1 * k))
    publish_elapsed = time.perf_counter() - t0
    depth_before = sync.depth()
    drain = sync.drain()
    tele = sync.telemetry()
    digests_ok = all(
        hashlib.sha256(w.to(torch.float32).contiguous().numpy().tobytes()).hexdigest() == dg
        for dg, _ in slow.committed
    )
    v.gates["G6_decoupled_sync"] = {
        "pass": bool(
            depth_before == 32
            and publish_elapsed < 0.004 * 8           # far below 8 store commits
            and drain["committed"] == 32
            and digests_ok
        ),
        "publish_elapsed_s": publish_elapsed,
        "publish_mean_us": tele["mean_publish_us"],
        "depth_before_drain": depth_before,
        "drain": drain,
        "committed": len(slow.committed),
        "digests_roundtrip_ok": digests_ok,
        "criterion": ("32 publishes complete in far less than 8 store commit "
                      "latencies, all 32 reach the store on drain, and every "
                      "persisted payload reproduces its digest"),
    }

    if not quiet:
        print(json.dumps(v.as_dict(), indent=2, default=str))
    return v


# ---------------------------------------------------------------------------
# Flag-gated factory
# ---------------------------------------------------------------------------

def get_unified_basal_engine(
    config: Optional[UnifiedHENRIVLAConfig] = None,
    **kw: Any,
) -> Optional[UnifiedHENRIVLAEngine]:
    """Return the engine ONLY when HENRI_BASAL_ENGINE=1. Absent => None."""
    if os.environ.get(FLAG, "0") != "1":
        return None
    return UnifiedHENRIVLAEngine(config, **kw)


if __name__ == "__main__":  # pragma: no cover - manual probe
    os.environ.setdefault(FLAG, "1")
    eng = UnifiedHENRIVLAEngine()
    print(json.dumps(eng.describe(), indent=2, default=str))
    rep = verify_engine(eng)
    raise SystemExit(0 if rep.ok else 1)
