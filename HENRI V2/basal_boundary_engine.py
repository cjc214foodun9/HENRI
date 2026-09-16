"""Zone A dynamic Markov blanket + Zone B basal oscillator syncytium.

Spec under implementation:
    HENRI-ARCH-2026-BASAL-COGNITION-AND-BOUNDARY-ENGINEERING

This module engineers three operational boundaries. It does NOT restate the
spec's prose as code. Three audited spec defects are corrected here, and the
corrections are named at the site:

    D-SAGNAC  The spec's `evaluate_sagnac_interpretant` divides the real inner
              product of two UNIT-NORM waves by D. For unit-norm waves the
              inner product is a cosine in [-1, 1], so `1 - inner/D` is ~0.99998
              at D=65536 and the hard channel vetoes EVERY valid candidate.
              This is the same false-veto class FALSIFIED on 2026-08-12 and
              documented in `arc_sagnac_veto.py`. The canonical live metric is
              used instead: S = 0.5 * (1 + <a, b>), delta = 1 - S in [0, 1].

    D-KURAMOTO The spec budgets K=2.45 with dt=0.01 for 10 steps. For the
              uniform frequency distribution on [-pi, pi] the mean-field
              threshold is K_c = 4, so the spec's own default sits SUBCRITICAL
              and its own gate G-BASAL-1 (r >= 0.93) is unreachable.
              `darwinian_phase_swarm.py` already records this identical defect
              and floors its gain at 6.0. Here the coupling strength is an
              explicit, measured parameter and `theoretical_k_c()` reports the
              threshold so the gate can be read against it.

    D-STALE-R The spec reads `mean_sin` / `mean_cos` AFTER the relaxation loop,
              so `r_order` describes the phase vector BEFORE the final update.
              `relax()` recomputes the order parameter from the returned phases.

Locality contract (the "metric locality collapse" the spec names): no
reduction crosses a tile or a channel boundary. Tile identity is preserved by
CONCATENATION, never by summation. Channel identity inside a tile is preserved
by a Cl(3,0) grade decomposition, never by averaging.

Evidence boundary: this module is a simulation of the stated boundary
engineering. It is not a claim about physical BaTiO3 hardware. Every numeric
claim it makes is produced by the harness that imports it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Spec constants (PDF names and values preserved for traceability)
# ---------------------------------------------------------------------------

SPEC_ID = "HENRI-SPEC-2026-BASAL-BOUNDARIES-V1"
SPEC_DIMENSION_D = 65536
SPEC_CLIFFORD_BLOCK_SIZE = 8
SPEC_KURAMOTO_COUPLING_K = 2.45
SPEC_SAGNAC_VETO_THRESHOLD = 0.35
SPEC_HOPFIELD_BETA = 8.0
SPEC_LANGEVIN_GAMMA = 0.15

# 20 kHz time-slot aperture -> 50 microseconds.
SPEC_SHUTTER_HZ = 20_000.0
SPEC_SLOT_SECONDS = 1.0 / SPEC_SHUTTER_HZ            # 5.0e-5

# 65536 / 8 = 8192 tiles. M is the tile count, not an independent constant.
SPEC_NUM_TILES = SPEC_DIMENSION_D // SPEC_CLIFFORD_BLOCK_SIZE
SPEC_R_GATE = 0.93

# MEASURED 2026-09-12 (receipts: experiments/verification/
# basal_syncytium_frontier.json and basal_syncytium_boundary.json).
#
# On an 8192-channel ring, at the spec's own K = 2.45 and zero natural
# frequencies, the global order parameter r reaches the mandate gate r >= 0.93
# from a uniform-random initial condition only when the evanescent leakage
# length exceeds about 168 channels - a leakage FRACTION of about 0.0205 of the
# ring. Below that the system reaches r_local -> 0.996 (every channel's own
# neighbourhood is perfectly locked) while global r plateaus near 0.20 and then
# DECAYS.
#
# The transition is SHARP and the state is HYSTERETIC:
#   random  start: decay 160 -> r 0.165 FAIL ;  decay 168 -> r 0.940 PASS
#   ordered start: decay  32 -> r 1.0000 PASS (the locked state is
#                                 self-sustaining far below the random-start
#                                 threshold)
#
# Two consequences are load-bearing:
#   1. The spec's K = 2.45 is sufficient. Its implied leakage length is NOT
#      specified, and the local-only reading (decay = 8 channels) can never
#      pass. The gate is reachable, but only with wide leakage.
#   2. A working point chosen at the random-start threshold is a knife edge.
#      A default must carry margin, or the syncytium flaps.
#
# KNOWN BOUNDARY - the fraction is NOT uniformly valid across ring sizes.
# A direct ring-size sweep at the margin-3 working point (0.0615 of the ring)
# measured:
#     N=128  r=0.9926   N=1024 r=0.9309   N=4096 r=0.9920
#     N=256  r=0.9972   N=2048 r=0.9943   N=8192 r=0.9948
#     N=512  r=0.4670   <- ANOMALY: fails at 1024 steps, decays to 0.1454 at
#                          2048 steps, while every neighbouring size locks
# A single out-of-family size is a finite-size resonance, not a monotone law.
# Treat the fraction as an ORDER-OF-MAGNITUDE guide and always measure r at the
# target ring size before fixing a working point. The production ring is 8192
# channels, where the value is measured directly.
SPEC_MEASURED_MIN_LEAKAGE_FRACTION = 0.0205

# ---------------------------------------------------------------------------
# SEALED HARDWARE DEFAULTS (HENRI-ARCH-2026-FRONTIER-EVALUATION, section 5.1)
# ---------------------------------------------------------------------------
# These two values are the measured baseline. They are sealed as module-level
# constants and enforced at the PRODUCTION tiling by MarkovBlanketSpec, so that
# a later drift is a construction error rather than a silent behaviour change.
#
# SPEC_NON_LOCAL_SPAN = 504
#   The non-local coupling span in channel units, on the 8192-channel
#   production ring. This is 3x the measured percolation knee of 168 channels
#   (0.0205 of the ring). MEASURED: decay 160 -> r = 0.165 FAIL,
#   decay 168 -> r = 0.940 PASS. The sealed value is the ROUNDED form of the
#   auto-scaled 8192 * 0.0205 * 3.0 = 503.808, hence the 1.0-channel tolerance
#   in the enforcing validator.
#
#   HONEST BOUNDARY: 504 is a SIMULATION working point on the measured
#   ring-fraction rule. It is not a hardware measurement, and the rule is not
#   uniform across ring sizes (N=512 is a finite-size resonance that fails
#   while every neighbouring size locks). Do not restate it as a physical
#   fabrication specification.
#
# SPEC_LOCK_HORIZON_STEPS = 1024
#   Relaxation steps to reach r >= r_gate from cold. MEASURED: r crosses 0.93
#   at 750 steps from BOTH a wave-seeded and a cold uniform-random start
#   (identical). 1024 gives 1.37x margin.
#
#   HONEST BOUNDARY: this is a RELAXATION count, NOT a time-slot aperture. It
#   must stay decoupled from shutter_hz (20 kHz, 50 us). Reaching the horizon
#   from cold spans ceil(1024 / ticks_per_slot) ingress slots. The two numbers
#   must never be derived from one another.
SPEC_NON_LOCAL_SPAN = 504
SPEC_LOCK_HORIZON_STEPS = 1024
# Tolerance in channel units when enforcing SPEC_NON_LOCAL_SPAN at the
# production tiling. Absorbs the rounding of 503.808 -> 504 without weakening
# the seal: any real drift (a changed margin, fraction, or block size) moves
# the resolved span by far more than one channel.
SPEC_NON_LOCAL_SPAN_TOLERANCE = 1.0


def recommended_leakage_length(num_channels: int, margin: float = 3.0) -> float:
    """Leakage length in channel units for the given ring, with margin.

    Scale-free by construction: the threshold is a FRACTION of the ring, so a
    256-channel syncytium and an 8192-channel syncytium get proportional
    leakage lengths. `margin` multiplies the measured minimum fraction; the
    default 3.0 gives ~0.0615 of the ring, which measured r = 0.9987 at
    K = 2.45.
    """
    if num_channels < 2:
        raise BasalBoundaryError("num_channels must be >= 2")
    if margin < 1.0:
        raise BasalBoundaryError(
            "margin must be >= 1.0; a value below 1 places the operating point "
            "below the measured locking threshold"
        )
    return float(num_channels) * SPEC_MEASURED_MIN_LEAKAGE_FRACTION * float(margin)


def leakage_fraction(num_channels: int, decay_length: float) -> float:
    """Leakage length as a fraction of the ring (the scale-free coordinate)."""
    if num_channels < 2:
        raise BasalBoundaryError("num_channels must be >= 2")
    return float(decay_length) / float(num_channels)

# Cl(3,0) grade map over the basis ordering
# {1, e1, e2, e3, e12, e13, e23, e123}.
CLIFFORD_GRADES: Tuple[Tuple[int, ...], ...] = (
    (0,),          # grade 0: scalar
    (1, 2, 3),     # grade 1: vectors
    (4, 5, 6),     # grade 2: bivectors
    (7,),          # grade 3: pseudoscalar
)
# Reversion flips odd-grade-blade-count elements: bivectors and pseudoscalar.
# Byte-identical to ProductCliffordAlgebra3D.reversion_mask.
CLIFFORD_REVERSION_MASK: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0)


class BasalBoundaryError(RuntimeError):
    """Raised on a boundary-contract violation (fail-closed)."""


# ---------------------------------------------------------------------------
# Boundary 1 - TEMPORAL (the time-slot aperture)
# ---------------------------------------------------------------------------

@dataclass
class ShutterSlots:
    """Result of one shutter cycle."""

    slot_index: int
    slot_seconds: float
    isolated: bool
    ingress_admitted: int
    ingress_throttled: int
    relaxation_steps: int


class ShutterClock:
    """20 kHz time-slot aperture for Zone A.

    During a slot the optical core is isolated from sensory perturbation so the
    basal oscillators can settle. The clock is the temporal Markov blanket: it
    decides WHEN the core observes.

    Contract:
      - `slot_seconds` is 1/20000 = 5e-5 s; a caller cannot widen it silently.
      - `slot_index` is strictly monotonic within a clock instance.
      - `ticks_per_slot` relaxation steps happen per slot; ingress is admitted
        only on the slot boundary, never mid-slot.
    """

    def __init__(
        self,
        slot_hz: float = SPEC_SHUTTER_HZ,
        ticks_per_slot: int = 32,
        max_ingress_per_slot: int = 64,
    ) -> None:
        if slot_hz <= 0.0:
            raise BasalBoundaryError(f"slot_hz must be positive; got {slot_hz}")
        if ticks_per_slot < 1:
            raise BasalBoundaryError("ticks_per_slot must be >= 1")
        if max_ingress_per_slot < 1:
            raise BasalBoundaryError("max_ingress_per_slot must be >= 1")
        self.slot_hz = float(slot_hz)
        self.slot_seconds = 1.0 / self.slot_hz
        self.ticks_per_slot = int(ticks_per_slot)
        self.max_ingress_per_slot = int(max_ingress_per_slot)
        self._slot_index = -1
        self._last_wall: Optional[float] = None

    @property
    def slot_index(self) -> int:
        return self._slot_index

    def advance(
        self,
        ingress_count: int = 0,
        *,
        wall_now: Optional[float] = None,
    ) -> ShutterSlots:
        """Open the next ingress aperture, then isolate.

        Returns the admitted / throttled ingress counts. Throttled ingress is
        RECORDED, never silently discarded: silent discard is how sensory
        flooding becomes invisible.
        """
        if ingress_count < 0:
            raise BasalBoundaryError("ingress_count must be >= 0")
        self._slot_index += 1
        admitted = min(int(ingress_count), self.max_ingress_per_slot)
        throttled = int(ingress_count) - admitted
        now = time.perf_counter() if wall_now is None else float(wall_now)
        if self._last_wall is not None and now < self._last_wall:
            # A monotonic violation means the caller is not driving slots in
            # time order; the isolation budget cannot be trusted.
            raise BasalBoundaryError(
                f"non-monotonic slot clock: {now} < {self._last_wall}"
            )
        self._last_wall = now
        return ShutterSlots(
            slot_index=self._slot_index,
            slot_seconds=self.slot_seconds,
            isolated=True,
            ingress_admitted=admitted,
            ingress_throttled=throttled,
            relaxation_steps=self.ticks_per_slot,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "slot_hz": self.slot_hz,
            "slot_seconds": self.slot_seconds,
            "slot_microseconds": self.slot_seconds * 1e6,
            "ticks_per_slot": self.ticks_per_slot,
            "max_ingress_per_slot": self.max_ingress_per_slot,
            "slots_elapsed": self._slot_index + 1,
        }


# ---------------------------------------------------------------------------
# Boundary 2 - SPATIAL (Cl(3,0) bivector tiling)
# ---------------------------------------------------------------------------

def clifford_bivector_tiles(
    wave: torch.Tensor,
    num_tiles: int = SPEC_NUM_TILES,
    channels: int = SPEC_CLIFFORD_BLOCK_SIZE,
) -> torch.Tensor:
    """Split a flat wave into [num_tiles, channels] Cl(3,0) tiles.

    Locality: this is a pure reshape. No reduction happens here, so no tile and
    no channel can leak into another. The spec's failure mode (mean pooling
    across tiles) is introduced only by a later reduction that crosses this
    axis - see `tile_descriptor`.
    """
    w = torch.as_tensor(wave, dtype=torch.float32).reshape(-1)
    total = num_tiles * channels
    if w.numel() != total:
        raise BasalBoundaryError(
            f"wave has {w.numel()} elements; the {num_tiles}-tile / "
            f"{channels}-channel tiling requires exactly {total}"
        )
    return w.view(num_tiles, channels)


def tile_grade_decompose(tiles: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Decompose each tile into Cl(3,0) grades.

    tiles: [num_tiles, 8] -> scalar [T,1], vector [T,3], bivector [T,3],
    pseudoscalar [T,1]. Grade identity is retained per tile.
    """
    if tiles.ndim != 2 or tiles.shape[-1] != SPEC_CLIFFORD_BLOCK_SIZE:
        raise BasalBoundaryError(
            f"tiles must be [T, {SPEC_CLIFFORD_BLOCK_SIZE}]; got "
            f"{tuple(tiles.shape)}"
        )
    return {
        "scalar": tiles[:, 0:1],
        "vector": tiles[:, 1:4],
        "bivector": tiles[:, 4:7],
        "pseudoscalar": tiles[:, 7:8],
    }


def tile_descriptor(tiles: torch.Tensor) -> torch.Tensor:
    """Locality-preserving tile descriptor: [num_tiles * 8] -> [num_tiles, 8].

    Identity of the descriptor:
      - the BIVECTOR grade is L1-reduced inside the tile (a local reduction),
      - the reversion mask is applied (bivectors and pseudoscalar flip sign),
      - the tile axis is CONCATENATED, never summed.

    Consequence, which is the measurable claim: tile-permuting the input
    changes the descriptor. Mean pooling over the tile axis does not.
    """
    grades = tile_grade_decompose(tiles)
    mask = torch.tensor(
        CLIFFORD_REVERSION_MASK, dtype=tiles.dtype, device=tiles.device
    )
    biv = grades["bivector"] * mask[4:7].unsqueeze(0)
    # Local scalar reductions, one per grade block, per tile. No cross-tile op.
    desc = torch.cat(
        [
            grades["scalar"].abs(),
            grades["vector"].abs().mean(dim=1, keepdim=True),
            biv.abs().mean(dim=1, keepdim=True),
            torch.linalg.vector_norm(grades["bivector"], dim=1, keepdim=True),
            grades["pseudoscalar"].abs(),
            tiles[:, 0:1] * tiles[:, 4:5],            # scalar-bivector product
            (tiles[:, 1:2] * tiles[:, 5:6]),          # e1 * e13 coupling
            torch.linalg.vector_norm(tiles, dim=1, keepdim=True),
        ],
        dim=1,
    )
    return desc


def bivector_tile_signature(tiles: torch.Tensor) -> torch.Tensor:
    """Per-tile bivector signature: [num_tiles]. Locality witness.

    Two waves that differ only by a tile permutation produce different
    signatures under this function and identical signatures under global mean
    pooling. That difference is the metric-locality witness.
    """
    grades = tile_grade_decompose(tiles)
    return grades["bivector"].abs().mean(dim=1)


# ---------------------------------------------------------------------------
# Boundary 3 - ACOUSTIC (evanescent leakage -> Kuramoto syncytium)
# ---------------------------------------------------------------------------

def evanescent_kernel(
    num_channels: int,
    decay_length: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Row-sum-1 exponential coupling kernel on a periodic 1-D channel ring.

    J_j = exp(-d_j / decay_length), d_j = min(j, N - j), normalized to sum 1.
    `decay_length` is the evanescent leakage length in channel units: the
    distance over which the coupling falls by 1/e. Small decay_length => local
    coupling only. This is the Optical Syncytium's gap-junction analogue.
    """
    if num_channels < 2:
        raise BasalBoundaryError("num_channels must be >= 2")
    if decay_length <= 0.0:
        raise BasalBoundaryError("decay_length must be positive")
    idx = torch.arange(num_channels, dtype=dtype)
    d = torch.minimum(idx, num_channels - idx)
    k = torch.exp(-d / float(decay_length))
    return k / k.sum()


class EvanescentKuramotoSyncytium:
    """Coupled basal phase oscillators on a channel ring.

    Equation (per channel k, in radians):

        d(theta_k)/dt = omega_k + K * Im[ e^{-i theta_k} * Z_k ]
        Z_k           = sum_j J_kj * e^{i theta_j}

    Z is the exponentially-weighted phase sum over the channel neighbourhood
    (the evanescent leakage sum). O(N log N) by circular convolution: no
    [N, N] tensor is ever materialized.

    Interpretation, stated as an assumption rather than a result: each channel
    is a basal agent whose intrinsic goal is to minimise its phase divergence
    from its leaking neighbours. The global order parameter r measures whether
    the individual channels have surrendered their private phase freedom to
    form an optical syncytium.

    `natural_frequency_scale = 0.0` is the mandate's reading: pure local
    phase-divergence minimisation, no intrinsic frequency. A non-zero scale
    reintroduces heterogeneous natural frequencies, and then the coupling must
    exceed K_c = 4 (uniform distribution) or the lattice cannot lock. That
    contrast is the discriminating control, not an implementation detail.
    """

    def __init__(
        self,
        num_channels: int = SPEC_NUM_TILES,
        coupling_K: float = SPEC_KURAMOTO_COUPLING_K,
        decay_length: float = 8.0,
        dt: float = 0.01,
        natural_frequency_scale: float = 0.0,
        noise_temperature: float = 0.0,
        langevin_gamma: float = SPEC_LANGEVIN_GAMMA,
        seed: int = 0,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_channels < 2:
            raise BasalBoundaryError("num_channels must be >= 2")
        if dt <= 0.0:
            raise BasalBoundaryError("dt must be positive")
        if noise_temperature < 0.0:
            raise BasalBoundaryError("noise_temperature must be >= 0")
        self.num_channels = int(num_channels)
        self.coupling_K = float(coupling_K)
        self.decay_length = float(decay_length)
        self.dt = float(dt)
        self.noise_temperature = float(noise_temperature)
        self.langevin_gamma = float(langevin_gamma)
        self.device = torch.device(device or "cpu")
        self.dtype = dtype

        self.kernel = evanescent_kernel(
            self.num_channels, self.decay_length, dtype=dtype
        ).to(self.device)
        # Full complex FFT: the state exp(i*theta) is COMPLEX, so rfft/irfft is
        # not applicable (irfft rejects a complex input). A cyclic convolution
        # of a complex sequence with a real kernel is fft(z) * fft(k) -> ifft.
        self._kernel_fft = torch.fft.fft(self.kernel.to(torch.float32))

        g = torch.Generator(device="cpu").manual_seed(int(seed))
        if natural_frequency_scale:
            self.natural_frequencies = (
                (torch.rand(self.num_channels, generator=g) * 2.0 - 1.0)
                * math.pi
                * float(natural_frequency_scale)
            ).to(self.device)
        else:
            self.natural_frequencies = torch.zeros(
                self.num_channels, dtype=dtype, device=self.device
            )
        self.phases: Optional[torch.Tensor] = None
        self.steps_taken = 0

    # -- initialization -----------------------------------------------------

    def seed_phases_from_wave(self, wave: torch.Tensor) -> torch.Tensor:
        """Set the phase vector from a real wave's tile-level phase.

        Each tile [8] is treated as (real, imag) pairs of a 4-component complex
        phasor; the tile phase is the argument of its complex mean. This keeps
        the ingress wave's block structure intact: tile k seeds channel k.
        """
        tiles = clifford_bivector_tiles(wave, self.num_channels, 8)
        pairs = tiles.view(self.num_channels, 4, 2)
        z = torch.complex(pairs[..., 0], pairs[..., 1]).mean(dim=1)
        self.phases = torch.angle(z).to(dtype=self.dtype, device=self.device)
        self.steps_taken = 0
        return self.phases

    def random_phases(self, seed: int = 0) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.phases = (
            (torch.rand(self.num_channels, generator=g) * 2.0 - 1.0) * math.pi
        ).to(dtype=self.dtype, device=self.device)
        self.steps_taken = 0
        return self.phases

    # -- dynamics -----------------------------------------------------------

    def coupled_field(self, phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Z_real, Z_imag) of the evanescent neighbourhood phase sum.

        Z = J * exp(i theta), a cyclic convolution. The kernel is stationary and
        row-sum 1, so Z is the exponentially weighted neighbourhood mean phasor
        of each channel. O(N log N); no [N, N] tensor is materialized.
        """
        z = torch.exp(1j * phases.to(torch.float32))
        z_k = torch.fft.ifft(torch.fft.fft(z) * self._kernel_fft)
        return z_k.real, z_k.imag

    def local_order_parameter(self, phases: torch.Tensor) -> torch.Tensor:
        """Per-channel |Z_k|: how coherent each channel's neighbourhood is."""
        zr, zi = self.coupled_field(phases)
        return torch.sqrt(zr * zr + zi * zi)

    @staticmethod
    def order_parameter(phases: torch.Tensor) -> float:
        """Global Kuramoto order parameter r = |mean(exp(i theta))| in [0, 1]."""
        z = torch.exp(1j * phases.to(torch.float32))
        return float(torch.abs(z.mean()).item())

    def relax(
        self,
        steps: int,
        *,
        dt: Optional[float] = None,
        noise: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Integrate `steps` Euler-Maruyama steps of the basal relaxation.

        Returns {r, r_local_mean, phases, steps_taken, finite}. `finite` False
        means the integration diverged and the caller must fail closed.
        """
        if self.phases is None:
            raise BasalBoundaryError("phases unset; call seed_phases_from_wave first")
        if steps < 1:
            raise BasalBoundaryError("steps must be >= 1")
        dt = self.dt if dt is None else float(dt)
        noise = self.noise_temperature if noise is None else float(noise)

        theta = self.phases
        for _ in range(int(steps)):
            zr, zi = self.coupled_field(theta)
            # Im[e^{-i theta} Z] = cos(theta)*Zi - sin(theta)*Zr
            force = torch.cos(theta) * zi - torch.sin(theta) * zr
            d_theta = self.natural_frequencies + self.coupling_K * force
            if noise > 0.0:
                d_theta = d_theta + torch.randn_like(theta) * math.sqrt(
                    2.0 * noise * dt
                )
            theta = theta + d_theta * dt

        theta = torch.atan2(torch.sin(theta), torch.cos(theta))
        finite = bool(torch.isfinite(theta).all().item())
        self.phases = theta
        self.steps_taken += int(steps)
        return {
            "r": self.order_parameter(theta),
            "r_local_mean": float(self.local_order_parameter(theta).mean().item()),
            "phases": theta,
            "steps_taken": self.steps_taken,
            "finite": finite,
        }

    # -- threshold and controls --------------------------------------------

    def theoretical_k_c(self) -> float:
        """Mean-field Kuramoto threshold for the uniform frequency family.

        K_c = 2 / (pi * g(0) * R(0)). For g uniform on [-pi, pi], g(0) = 1/(2pi)
        and the row-normalized kernel has R(0) = 1, so K_c = 4.
        """
        return 4.0

    def subcritical_control(self, steps: int, K: Optional[float] = None) -> float:
        """Run a deliberately subcritical relaxation and return r.

        This is the negative control for the syncytium gate: r must stay low
        when K < K_c AND the natural frequencies are heterogeneous. A gate that
        passes here is measuring something other than phase locking.
        """
        keep_phases, keep_freqs, keep_K = self.phases, self.natural_frequencies, self.coupling_K
        try:
            g = torch.Generator(device="cpu").manual_seed(1234)
            self.natural_frequencies = (
                (torch.rand(self.num_channels, generator=g) * 2.0 - 1.0) * math.pi
            ).to(self.device)
            self.coupling_K = float(K if K is not None else self.coupling_K)
            self.random_phases(seed=1234)
            out = self.relax(steps)
            return float(out["r"])
        finally:
            self.phases, self.natural_frequencies, self.coupling_K = keep_phases, keep_freqs, keep_K

    def measured_k_c(self, steps: int, K_grid: Optional[List[float]] = None) -> List[Dict[str, float]]:
        """Sweep K with heterogeneous frequencies and report r(K).

        The threshold is read off as the smallest K whose r clears the gate.
        """
        grid = K_grid or [1.0, 2.0, 2.45, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
        rows: List[Dict[str, float]] = []
        keep_phases, keep_freqs = self.phases, self.natural_frequencies
        g = torch.Generator(device="cpu").manual_seed(4321)
        hetero = ((torch.rand(self.num_channels, generator=g) * 2.0 - 1.0) * math.pi).to(self.device)
        try:
            for K in grid:
                self.natural_frequencies = hetero
                self.coupling_K = float(K)
                self.random_phases(seed=99)
                r = float(self.relax(steps)["r"])
                rows.append({"K": float(K), "r": r, "clears_gate": bool(r >= SPEC_R_GATE)})
        finally:
            self.phases, self.natural_frequencies = keep_phases, keep_freqs
        return rows

    # -- Sagnac interpretant (canonical metric) -----------------------------

    @staticmethod
    def sagnac_delta(candidate: torch.Tensor, reference: torch.Tensor) -> float:
        """Canonical Sagnac homodyne delta in [0, 1] for unit-norm waves.

        S = 0.5 * (1 + <a, b>); delta = 1 - S. Identical waves -> 0, opposite
        -> 1. This replaces the spec's `1 - <a,b>/D`, which is ~1 for every
        unit-norm pair and therefore vetoes everything (D-SAGNAC).
        """
        a = candidate.reshape(-1).to(torch.float32)
        b = reference.reshape(-1).to(torch.float32)
        na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
        if float(na) <= 1e-12 or float(nb) <= 1e-12:
            return 1.0
        cos = float(torch.dot(a, b) / (na * nb))
        cos = max(-1.0, min(1.0, cos))
        s = 0.5 * (1.0 + cos)
        return float(max(0.0, min(1.0, 1.0 - s)))

    @staticmethod
    def is_vetoed(delta: float, epsilon_hard: float = SPEC_SAGNAC_VETO_THRESHOLD) -> bool:
        return bool(delta > float(epsilon_hard))


# ---------------------------------------------------------------------------
# Zone A dynamic Markov blanket: the three boundaries in one object
# ---------------------------------------------------------------------------

@dataclass
class BlanketState:
    """Typed per-step boundary state. Every field is measured, none inferred."""

    slot_index: int
    slot_seconds: float
    ingress_admitted: int
    ingress_throttled: int
    tiles: int
    channels: int
    tile_locality_cosine: float
    order_parameter_r: float
    r_local_mean: float
    sagnac_delta: float
    boundary_violation: bool
    is_crystallized: bool
    cognitive_light_cone_radius: float
    relaxed_steps: int
    finite: bool
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        return d


class DynamicMarkovBlanket:
    """Zone A blanket: temporal aperture, spatial tiling, acoustic syncytium.

    The blanket does not compute a solution. It decides WHERE (tile), WHEN
    (slot) and HOW LONG (relaxation steps) the basal channels observe and
    update against the external task.
    """

    def __init__(
        self,
        *,
        num_tiles: int = SPEC_NUM_TILES,
        channels: int = SPEC_CLIFFORD_BLOCK_SIZE,
        coupling_K: float = SPEC_KURAMOTO_COUPLING_K,
        decay_length: float = 8.0,
        dt: float = 0.01,
        ticks_per_slot: int = 32,
        max_ingress_per_slot: int = 64,
        sagnac_epsilon: float = SPEC_SAGNAC_VETO_THRESHOLD,
        r_gate: float = SPEC_R_GATE,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.num_tiles = int(num_tiles)
        self.channels = int(channels)
        self.sagnac_epsilon = float(sagnac_epsilon)
        self.r_gate = float(r_gate)
        self.clock = ShutterClock(
            ticks_per_slot=ticks_per_slot, max_ingress_per_slot=max_ingress_per_slot
        )
        self.syncytium = EvanescentKuramotoSyncytium(
            num_channels=self.num_tiles,
            coupling_K=coupling_K,
            decay_length=decay_length,
            dt=dt,
            device=device,
            seed=seed,
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def tile_locality_cosine(wave: torch.Tensor, num_tiles: int = SPEC_NUM_TILES) -> float:
        """Cosine between a wave and its tile-permuted twin.

        1.0 means the descriptor cannot see tile identity (locality destroyed).
        A value well below 1.0 means tile identity survives.
        """
        tiles = clifford_bivector_tiles(wave, num_tiles, SPEC_CLIFFORD_BLOCK_SIZE)
        desc_a = tile_descriptor(tiles).reshape(-1)
        perm = torch.roll(torch.arange(num_tiles), shifts=num_tiles // 2)
        desc_b = tile_descriptor(tiles[perm]).reshape(-1)
        na, nb = torch.linalg.vector_norm(desc_a), torch.linalg.vector_norm(desc_b)
        if float(na) <= 1e-12 or float(nb) <= 1e-12:
            return 1.0
        return float(torch.dot(desc_a, desc_b) / (na * nb))

    @staticmethod
    def mean_pooling_locality_cosine(wave: torch.Tensor, num_tiles: int = SPEC_NUM_TILES) -> float:
        """The same witness under the spec's mean-pooling reduction.

        Expected: exactly 1.0, because summation is permutation-invariant. This
        is the control that proves the witness discriminates.
        """
        tiles = clifford_bivector_tiles(wave, num_tiles, SPEC_CLIFFORD_BLOCK_SIZE)
        perm = torch.roll(torch.arange(num_tiles), shifts=num_tiles // 2)
        a = tiles.mean(dim=0).reshape(-1)
        b = tiles[perm].mean(dim=0).reshape(-1)
        na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
        if float(na) <= 1e-12 or float(nb) <= 1e-12:
            return 1.0
        return float(torch.dot(a, b) / (na * nb))

    # -- the blanket step ---------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        ingress_wave: torch.Tensor,
        engram_reference: Optional[torch.Tensor] = None,
        *,
        ingress_count: int = 1,
        slots: int = 1,
        wall_now: Optional[float] = None,
    ) -> BlanketState:
        """Run `slots` shutter cycles over an ingress wave.

        Data path: ingress wave -> tile -> seed basal phases -> relax per slot
        (isolated) -> canonical Sagnac check against the engram reference ->
        typed state. Fail-closed on a non-finite relaxation.
        """
        if slots < 1:
            raise BasalBoundaryError("slots must be >= 1")
        tiles = clifford_bivector_tiles(
            ingress_wave, self.num_tiles, self.channels
        )
        self.syncytium.seed_phases_from_wave(ingress_wave)

        last: Optional[Dict[str, Any]] = None
        slot_rec = None
        admitted = throttled = 0
        relax_steps = 0
        for _ in range(int(slots)):
            slot_rec = self.clock.advance(ingress_count, wall_now=wall_now)
            admitted += slot_rec.ingress_admitted
            throttled += slot_rec.ingress_throttled
            last = self.syncytium.relax(slot_rec.relaxation_steps)
            relax_steps += slot_rec.relaxation_steps
            if not last["finite"]:
                break

        r = float(last["r"]) if last else float("nan")
        r_local = float(last["r_local_mean"]) if last else float("nan")
        phases = last["phases"] if last else None

        # The Sagnac interpretant: compare the relaxed phase carrier against the
        # Zone C engram reference. No reference -> delta is 1.0 (unknown), which
        # is a veto, never a silent pass.
        if engram_reference is None or phases is None:
            delta = 1.0
        else:
            candidate = torch.cat([torch.cos(phases), torch.sin(phases)])
            delta = self.syncytium.sagnac_delta(candidate, engram_reference)
        veto = self.syncytium.is_vetoed(delta, self.sagnac_epsilon)

        # Cognitive light cone: the spatio-temporal radius over which the
        # coupled oscillators hold coherence. r=1 and delta=0 give the widest
        # cone; the scale is a reported index, not a physical length.
        cone = float(r) / max(delta, 0.05) if math.isfinite(r) else 0.0

        locality = self.tile_locality_cosine(ingress_wave, self.num_tiles)

        state = BlanketState(
            slot_index=self.clock.slot_index,
            slot_seconds=self.clock.slot_seconds,
            ingress_admitted=admitted,
            ingress_throttled=throttled,
            tiles=self.num_tiles,
            channels=self.channels,
            tile_locality_cosine=locality,
            order_parameter_r=r,
            r_local_mean=r_local,
            sagnac_delta=delta,
            boundary_violation=veto,
            is_crystallized=bool((not veto) and (r >= self.r_gate)),
            cognitive_light_cone_radius=cone,
            relaxed_steps=relax_steps,
            finite=bool(last["finite"]) if last else False,
            telemetry={
                "action": "CONSTRUCTIVE_LATCH" if not veto else "LANGEVIN_THERMAL_VETO",
                "coupling_K": self.syncytium.coupling_K,
                "decay_length": self.syncytium.decay_length,
                "r_gate": self.r_gate,
                "sagnac_epsilon": self.sagnac_epsilon,
                "mean_pooling_locality_cosine": self.mean_pooling_locality_cosine(
                    ingress_wave, self.num_tiles
                ),
            },
        )
        return state
