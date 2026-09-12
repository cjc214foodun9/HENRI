"""Step 5.2: the GPU carrier `_fused_autopoietic_kuramoto_kernel`.

Spec: HENRI-ARCH-2026-FRONTIER-EVALUATION-AND-DISCOVERY-LEDGER, section 5.2

WHAT IS HERE
------------
1. `_fused_autopoietic_kuramoto_kernel` — the Triton kernel, BLOCK_SIZE = 1024,
   carrying the verified non-local span as an explicit tap set.
2. THREE independent CPU implementations of the SAME arithmetic, so the
   algorithm can be proven equivalent before any GPU exists:

     span_coupled_field / relax_span   direct tap sum (what the kernel does)
     fft_relax                         circular convolution (the FFT path)

   Agreement between those two IS the parity proof. It runs anywhere, and it is
   the check that proves the kernel is not quietly solving a different problem.
3. `tau_budget_analysis()` — a DERIVED structural bound on tau_relax.

THE 12.8 us BOUND, CHECKED BEFORE WRITING THE KERNEL
----------------------------------------------------
The spec asks for 1024 relaxation steps inside 12.8 us. That is 12.5 ns/step.
A Kuramoto step is GLOBAL: every channel reads its neighbours out to the span,
so the updated phases must be visible to all readers before the next step
begins. That visibility costs one synchronization per step:

    tau_relax >= steps * (tau_sync + tau_compute)

  design A   the spec's layout: 8 blocks of 1024 threads, halo across the span.
             The halo crosses block boundaries, so the barrier must be a
             grid-wide cooperative sync: ~1-3 us  =>  1.0-3.1 ms
  design B   one persistent block, whole ring resident, no halo at all.
             Still one block barrier per step: ~20-40 ns  =>  20-41 us

One block barrier alone already exceeds 12.5 ns/step. So the 12.8 us SUB-BUDGET
is not reachable by either design, before any compute is counted. The 50 us
SHUTTER, however, IS reachable at the design-B floor. Both statements are
reported; neither is assumed. This is DERIVED from published latency constants,
not measured.

Note the launch cost consequence: 1024 separate kernel launches is impossible
(< 100 ns per launch does not exist). The relaxation must loop INSIDE one
persistent kernel, and the barrier is what a step costs.

EVIDENCE BOUNDARY
-----------------
Triton is not installed and no CUDA device is present on this host
(`torch 2.13.0+cpu`). Nothing in this module is a GPU measurement. The Triton
kernel is authored and AST-checked; every latency figure is DERIVED.
`measured_tau_us` is None and stays None until a real GPU run fills it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from basal_boundary_engine import (
    SPEC_KURAMOTO_COUPLING_K,
    SPEC_LOCK_HORIZON_STEPS,
    SPEC_NON_LOCAL_SPAN,
    SPEC_NUM_TILES,
    SPEC_R_GATE,
)

# ---------------------------------------------------------------------------
# Spec constants (section 5.2)
# ---------------------------------------------------------------------------
SPEC_BLOCK_SIZE = 1024
SPEC_TAU_BUDGET_US = 12.8
SPEC_SHUTTER_US = 50.0
SPEC_NUM_BLOCKS = SPEC_NUM_TILES // SPEC_BLOCK_SIZE      # 8192 / 1024 = 8

# ---------------------------------------------------------------------------
# Hardware latency inputs to the DERIVED bound.
#
# Published order-of-magnitude figures for NVIDIA SMs, NOT measurements taken
# here. They are the inputs to the bound, and the bound is only as sound as they
# are. Kept as RANGES so the report cannot hide the uncertainty.
# ---------------------------------------------------------------------------
SYNCTHREADS_LATENCY_NS = (20.0, 40.0)
GRID_SYNC_LATENCY_US = (1.0, 3.0)
BLACKWELL_SM_FMA_PER_CLK = 128
BLACKWELL_SM_COUNT = 128

# ---------------------------------------------------------------------------
# Optional Triton import. The module must import cleanly on a CPU-only host so
# that the parity proof and the bound analysis stay runnable everywhere.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the host, not on the logic
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
    TRITON_IMPORT_ERROR = ""
except Exception as _exc:  # noqa: BLE001
    triton = None
    tl = None
    TRITON_AVAILABLE = False
    TRITON_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def taps_for_reach(half_width: int) -> int:
    """Number of taps for a symmetric +/-half_width reach. Always ODD.

    `2H+1` taps cover distances -H..+H exactly once each, so the tap set is
    symmetric about the channel. An even tap count would place the kernel's
    centre between taps and break that symmetry, which is why the reach is the
    parameter and the tap count is derived from it.
    """
    h = int(half_width)
    if h < 0:
        raise ValueError("half_width must be >= 0")
    return 2 * h + 1


# ---------------------------------------------------------------------------
# 1. Kernel construction
# ---------------------------------------------------------------------------

def span_evanescent_weights(
    num_channels: int,
    decay_length: float,
    half_width: int,
) -> torch.Tensor:
    """Evanescent coupling taps over +/-half_width, renormalized.

    Tap `w` has signed distance `|w - half_width|` from its own channel, so tap
    order is exactly the order the Triton kernel walks. Returns 2H+1 taps.

    Renormalization matters. Dropping taps and NOT renormalizing would silently
    scale the effective coupling strength; renormalizing keeps the weights a
    convex combination, so the same K means the same thing at every reach.
    """
    if num_channels < 2:
        raise ValueError("num_channels must be >= 2")
    if decay_length <= 0.0:
        raise ValueError("decay_length must be positive")
    H = int(min(int(half_width), num_channels // 2))
    w = torch.arange(taps_for_reach(H), dtype=torch.float32)
    distance = (w - H).abs()
    weights = torch.exp(-distance / float(decay_length))
    total = float(weights.sum().item())
    if total <= 0.0:
        raise ValueError("degenerate reach: all weights are zero")
    return weights / total


def ring_kernel(
    num_channels: int,
    decay_length: float,
    half_width: Optional[int] = None,
) -> torch.Tensor:
    """Full-ring periodic kernel, truncated at +/-half_width and renormalized.

    Returns N weights indexed by cyclic offset j, with the coupling value
    determined by the distance d = min(j, N-j):

        k[j] = exp(-d / decay_length)  for d <= half_width, else 0,
        renormalized to sum 1.

    With `half_width = N//2` every cyclic distance is inside the reach, so the
    result is IDENTICAL to `evanescent_kernel(N, decay_length)`. That identity is
    the exact parity anchor between the FFT path and the span path.

    Because this is an ordinary periodic kernel, it can be applied by circular
    convolution. That makes the truncation sweep in
    `experiments/verification/basal_span_fidelity.py` cost O(N log N) per step
    instead of O(N * taps), which is what makes the sweep affordable.
    """
    n = int(num_channels)
    if n < 2:
        raise ValueError("num_channels must be >= 2")
    if decay_length <= 0.0:
        raise ValueError("decay_length must be positive")
    half = n // 2 if half_width is None else int(
        min(max(int(half_width), 0), n // 2)
    )
    j = torch.arange(n, dtype=torch.float32)
    d = torch.minimum(j, n - j)
    w = torch.exp(-d / float(decay_length))
    w = torch.where(d <= half, w, torch.zeros_like(w))
    total = float(w.sum().item())
    if total <= 0.0:
        raise ValueError("degenerate reach: the kernel is empty")
    return w / total


def kernel_l1_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """L1 distance between two normalized kernels. 0 = identical."""
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a - b).abs().sum().item())


# ---------------------------------------------------------------------------
# 2. CPU reference implementations (the parity proof)
# ---------------------------------------------------------------------------

def span_coupled_field(
    phases: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(Z_real, Z_imag) by DIRECT tap sum — the arithmetic the kernel performs.

        Z_k = sum_w weights[w] * exp(i * theta_{k - (w - H)}),  H = taps // 2

    O(N * taps) and materializes an [N, taps] gather. On a GPU the kernel tiles
    that gather across programs; here it is materialized for clarity. No FFT is
    used anywhere in this function, which is the point: this path and the FFT
    path are independent, so their agreement is evidence.
    """
    phases = torch.as_tensor(phases, dtype=torch.float32)
    n = int(phases.shape[0])
    taps = int(weights.shape[0])
    if taps % 2 == 0:
        raise ValueError(f"taps must be odd for a symmetric kernel; got {taps}")
    H = taps // 2
    idx = torch.arange(n)
    dist = torch.arange(taps) - H
    j = (idx[:, None] - dist[None, :]) % n
    th = phases[j]
    zr = (torch.cos(th) * weights[None, :]).sum(dim=1)
    zi = (torch.sin(th) * weights[None, :]).sum(dim=1)
    return zr, zi


def _step(theta: torch.Tensor, omega: torch.Tensor, zr: torch.Tensor,
          zi: torch.Tensor, coupling_K: float, dt: float) -> torch.Tensor:
    """One Euler step. Identical arithmetic to EvanescentKuramotoSyncytium."""
    force = torch.cos(theta) * zi - torch.sin(theta) * zr
    return theta + (omega + float(coupling_K) * force) * float(dt)


def relax_span(
    phases: torch.Tensor,
    weights: torch.Tensor,
    *,
    coupling_K: float = SPEC_KURAMOTO_COUPLING_K,
    dt: float = 0.01,
    steps: int = SPEC_LOCK_HORIZON_STEPS,
    natural_frequencies: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Euler integration through the direct tap sum."""
    theta = torch.as_tensor(phases, dtype=torch.float32).clone()
    n = int(theta.shape[0])
    if natural_frequencies is None:
        omega = torch.zeros(n, dtype=torch.float32)
    else:
        omega = torch.as_tensor(natural_frequencies, dtype=torch.float32)
    for _ in range(int(steps)):
        zr, zi = span_coupled_field(theta, weights)
        theta = _step(theta, omega, zr, zi, coupling_K, dt)
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def fft_relax(
    phases: torch.Tensor,
    kernel: torch.Tensor,
    *,
    coupling_K: float = SPEC_KURAMOTO_COUPLING_K,
    dt: float = 0.01,
    steps: int = SPEC_LOCK_HORIZON_STEPS,
    natural_frequencies: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Euler integration through circular convolution.

    The step is byte-for-byte the same arithmetic as
    `EvanescentKuramotoSyncytium.relax`, so agreement with that live class
    measures the COUPLING and not a difference in the integrator.
    """
    theta = torch.as_tensor(phases, dtype=torch.float32).clone()
    n = int(theta.shape[0])
    kernel = torch.as_tensor(kernel, dtype=torch.float32)
    if int(kernel.shape[0]) != n:
        raise ValueError(
            f"kernel length {int(kernel.shape[0])} != channel count {n}"
        )
    if natural_frequencies is None:
        omega = torch.zeros(n, dtype=torch.float32)
    else:
        omega = torch.as_tensor(natural_frequencies, dtype=torch.float32)
    kf = torch.fft.fft(kernel)
    for _ in range(int(steps)):
        z = torch.fft.ifft(torch.fft.fft(torch.exp(1j * theta)) * kf)
        theta = _step(theta, omega, z.real, z.imag, coupling_K, dt)
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def order_parameter(phases: torch.Tensor) -> float:
    """Global Kuramoto order parameter r = |mean(exp(i theta))| in [0, 1]."""
    z = torch.exp(1j * torch.as_tensor(phases, dtype=torch.float32))
    return float(torch.abs(z.mean()).item())


# ---------------------------------------------------------------------------
# 3. The Triton kernel
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:  # pragma: no cover - no Triton in this environment

    @triton.jit
    def _fused_autopoietic_kuramoto_kernel(
        theta_ptr,          # *fp32 [N] phase angles, read
        omega_ptr,          # *fp32 [N] natural frequencies, read
        weight_ptr,         # *fp32 [2H+1] coupling taps, read
        out_ptr,            # *fp32 [N] updated phases, written
        num_channels,       # i32  N
        span,               # i32  2H+1
        coupling_K,         # fp32
        dt,                 # fp32
        BLOCK_SIZE: tl.constexpr,
        BLOCK_SPAN: tl.constexpr,
    ):
        """One autopoietic Kuramoto relaxation step over a channel tile.

        Layout (spec 5.2): BLOCK_SIZE = 1024 channels per program, 8 programs
        for the 8192-channel ring. The span taps are read ACROSS the tile
        boundary — that halo is what makes the step global, and it is why a
        cross-block barrier is unavoidable in this layout.

        Tap `w` has signed distance `w - span//2`, so the reference weight
        vector is consumed here with no reordering. Neighbour indices wrap on the
        ring; the wrap is what makes the ring periodic rather than clipped.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_channels
        theta = tl.load(theta_ptr + offs, mask=mask, other=0.0)

        zr = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        zi = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        half = span // 2

        for w0 in range(0, span, BLOCK_SPAN):
            offs_w = w0 + tl.arange(0, BLOCK_SPAN)
            w_mask = offs_w < span
            dist = offs_w[None, :] - half
            j = offs[:, None] - dist
            j = (j + num_channels) % num_channels
            th_j = tl.load(
                theta_ptr + j,
                mask=w_mask[None, :] & mask[:, None],
                other=0.0,
            )
            wt = tl.load(weight_ptr + offs_w, mask=w_mask, other=0.0)
            zr += tl.sum(tl.cos(th_j) * wt[None, :], axis=1)
            zi += tl.sum(tl.sin(th_j) * wt[None, :], axis=1)

        omega = tl.load(omega_ptr + offs, mask=mask, other=0.0)
        force = tl.cos(theta) * zi - tl.sin(theta) * zr
        new_theta = theta + (omega + coupling_K * force) * dt
        tl.store(out_ptr + offs, new_theta, mask=mask)


def fused_relax(
    phases: torch.Tensor,
    weights: torch.Tensor,
    *,
    coupling_K: float = SPEC_KURAMOTO_COUPLING_K,
    dt: float = 0.01,
    steps: int = SPEC_LOCK_HORIZON_STEPS,
    block_size: int = SPEC_BLOCK_SIZE,
    block_span: int = 64,
    natural_frequencies: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drive the Triton kernel for `steps` iterations on CUDA.

    THIS IS THE CORRECTNESS PATH, NOT THE LATENCY PATH. One launch per step; the
    launch boundary supplies the global barrier, so correctness never depends on
    an in-kernel grid sync. It is how the kernel's numerics are checked against
    the CPU reference on a real device.

    For latency, `steps` launches is the problem: at any real launch cost this
    path cannot approach 12.8 us. The latency design is one PERSISTENT kernel
    that loops internally (see `tau_budget_analysis`), because a step must still
    cost at least one barrier no matter how it is scheduled.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            f"Triton is not available on this host ({TRITON_IMPORT_ERROR}). "
            "Use relax_span() or fft_relax() for the CPU path."
        )
    if not cuda_available():
        raise RuntimeError("fused_relax requires a CUDA device")

    n = int(phases.shape[0])
    span = int(weights.shape[0])
    theta = phases.detach().to("cuda", torch.float32).clone()
    out = torch.empty_like(theta)
    wt = weights.detach().to("cuda", torch.float32)
    omega = (
        torch.zeros(n, dtype=torch.float32, device="cuda")
        if natural_frequencies is None
        else natural_frequencies.to("cuda", torch.float32)
    )
    grid = (triton.cdiv(n, block_size),)
    for _ in range(int(steps)):
        _fused_autopoietic_kuramoto_kernel[grid](
            theta, omega, wt, out,
            n, span, float(coupling_K), float(dt),
            BLOCK_SIZE=block_size, BLOCK_SPAN=block_span,
        )
        theta, out = out, theta
    return torch.atan2(torch.sin(theta), torch.cos(theta))


# ---------------------------------------------------------------------------
# 4. The derived execution bound
# ---------------------------------------------------------------------------

@dataclass
class TauBudget:
    """Structural bound on tau_relax. DERIVED, never measured here."""

    steps: int
    num_channels: int
    num_blocks: int
    per_step_budget_ns: float
    macs_per_step: float
    compute_cycles_single_block: float
    sync_floor_us_design_a: Tuple[float, float]
    sync_floor_us_design_b: Tuple[float, float]
    compute_floor_us_single_block: float
    compute_floor_us_multi_block: float
    compute_floor_us_fft_multi_block: float
    floor_us_design_a: Tuple[float, float]
    floor_us_design_b: Tuple[float, float]
    binding_constraint: str
    verdict: str
    sub_budget_reachable: bool
    shutter_reachable: bool
    # Compute-only reachability, with synchronization cost set to ZERO. Reported
    # separately because the two answers differ, and collapsing them would hide
    # which constraint actually binds.
    shutter_reachable_compute_only: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "num_channels": self.num_channels,
            "num_blocks": self.num_blocks,
            "per_step_budget_ns": self.per_step_budget_ns,
            "sync_floor_us_design_a": list(self.sync_floor_us_design_a),
            "sync_floor_us_design_b": list(self.sync_floor_us_design_b),
            "compute_floor_us_single_block": self.compute_floor_us_single_block,
            "compute_floor_us_multi_block": self.compute_floor_us_multi_block,
            "compute_floor_us_fft_multi_block": self.compute_floor_us_fft_multi_block,
            "floor_us_design_a": list(self.floor_us_design_a),
            "floor_us_design_b": list(self.floor_us_design_b),
            "binding_constraint": self.binding_constraint,
            "macs_per_step": self.macs_per_step,
            "compute_cycles_single_block": self.compute_cycles_single_block,
            "sub_budget_us": SPEC_TAU_BUDGET_US,
            "shutter_us": SPEC_SHUTTER_US,
            "sub_budget_reachable": self.sub_budget_reachable,
            "shutter_reachable": self.shutter_reachable,
            "shutter_reachable_compute_only": self.shutter_reachable_compute_only,
            "verdict": self.verdict,
            "evidence_class": "DERIVED",
            "measured_tau_us": None,
        }


def tau_budget_analysis(
    *,
    steps: int = SPEC_LOCK_HORIZON_STEPS,
    num_channels: int = SPEC_NUM_TILES,
    block_size: int = SPEC_BLOCK_SIZE,
    sm_count: int = BLACKWELL_SM_COUNT,
    clock_ghz: float = 2.0,
) -> TauBudget:
    """Derive the lower bound on the 1024-step relaxation.

    Each term is shown so the arithmetic can be checked by hand.

    per-step budget
        SPEC_TAU_BUDGET_US * 1000 / steps ns.

    design A (the spec's 8-block layout with a span halo)
        tau >= steps * tau_grid_sync
        A halo crossing block boundaries is only correct if every block sees a
        consistent phase vector, which requires a grid-wide barrier per step.

    design B (persistent single block, whole ring resident)
        tau >= steps * tau_syncthreads
        A single block removes the halo entirely, but the shared phase vector is
        still read by every thread, so one block barrier per step remains. The
        barrier is the floor; it cannot be optimized away.

    compute term
        MACs per step = 2 * N * taps (one multiply-accumulate for each of Zr and
        Zi, per tap). Reported as cycles on ONE SM, which is the binding case for
        the persistent single-block design.
    """
    per_step_ns = (SPEC_TAU_BUDGET_US * 1000.0) / float(steps)
    a_lo, a_hi = GRID_SYNC_LATENCY_US
    b_lo_ns, b_hi_ns = SYNCTHREADS_LATENCY_NS

    design_a = (steps * a_lo, steps * a_hi)
    design_b = (steps * b_lo_ns / 1000.0, steps * b_hi_ns / 1000.0)

    taps = taps_for_reach(int(SPEC_NON_LOCAL_SPAN) // 2)
    macs = 2.0 * float(num_channels) * float(taps)
    cycles_single = macs / float(BLACKWELL_SM_FMA_PER_CLK)
    total_macs = macs * float(steps)
    lanes_all = float(BLACKWELL_SM_FMA_PER_CLK) * float(sm_count)

    def _macs_to_us(total: float, lanes: float) -> float:
        """MACs on `lanes` FMA lanes at clock_ghz, in microseconds."""
        return (total / lanes) / (clock_ghz * 1e9) * 1e6

    # Compute floors. One MAC = one FMA lane-cycle.
    compute_single = _macs_to_us(total_macs, float(BLACKWELL_SM_FMA_PER_CLK))
    # The spec's 8 blocks, one per SM, each block's work in parallel.
    compute_spec = _macs_to_us(
        total_macs / float(max(1, num_channels // block_size)),
        float(BLACKWELL_SM_FMA_PER_CLK),
    )
    # Perfect distribution over every SM: the absolute compute floor of the
    # tap-sum path, with ZERO synchronization cost.
    compute_dist = _macs_to_us(total_macs, lanes_all)
    # The FFT alternative is cheaper per step. This is the absolute floor of
    # ANY algorithm that applies the coupling, sync still excluded.
    fft_macs = (
        2.0 * (5.0 * float(num_channels) * math.log2(float(num_channels)))
        + 4.0 * float(num_channels)
    ) * float(steps)
    compute_fft = _macs_to_us(fft_macs, lanes_all)

    # Total floors = synchronization + compute, per design.
    floor_a = (design_a[0] + compute_spec, design_a[1] + compute_spec)
    floor_b = (design_b[0] + compute_single, design_b[1] + compute_single)

    # The best case anywhere: distribute the work over every SM AND pay a grid
    # barrier per step. Sync dominates the distributed compute.
    floor_best = design_a[0] + compute_dist
    pure_compute_floor = min(compute_dist, compute_fft)

    sub_ok = floor_best <= SPEC_TAU_BUDGET_US
    shutter_ok = floor_best <= SPEC_SHUTTER_US
    # Compute-only: if synchronization were free (it is not), does the work fit
    # the shutter? Reported separately so the two answers cannot be conflated.
    shutter_compute_only = pure_compute_floor <= SPEC_SHUTTER_US

    # A gate that cannot fail is not a gate. The 12.8 us budget is unreachable
    # even with the compute of the whole device and no synchronization at all.
    binding = "synchronization" if design_a[0] > compute_dist else "compute"

    verdict = (
        f"SUB-BUDGET NOT REACHABLE. {SPEC_TAU_BUDGET_US} us over {steps} steps "
        f"is {per_step_ns:.2f} ns/step. A Kuramoto step is GLOBAL: every channel "
        f"reads its neighbours out to the span, so the updated phases must be "
        f"visible to all readers before the next step. That costs one barrier "
        f"per step, and {per_step_ns:.2f} ns is below the cost of a single "
        f"barrier ({b_lo_ns:.0f}-{b_hi_ns:.0f} ns for __syncthreads). "
        f"THE DECISIVE TERM: even if ALL synchronization is ignored and the "
        f"compute of the whole device is used, the floor is "
        f"{pure_compute_floor:.1f} us ({compute_fft:.1f} us via FFT, "
        f"{compute_dist:.1f} us via the tap sum) -- already above the budget. "
        f"Adding the grid barrier a distributed design requires: "
        f"{floor_best:.0f} us, over {SPEC_TAU_BUDGET_US} us and over the "
        f"{SPEC_SHUTTER_US:.0f} us shutter. The spec's own 8-block halo layout "
        f"costs {floor_a[0]:.0f}-{floor_a[1]:.0f} us (grid sync per step); a "
        f"persistent single block costs {floor_b[0]:.0f}-{floor_b[1]:.0f} us "
        f"(compute-bound on one SM). Binding constraint: {binding}."
    )

    return TauBudget(
        steps=int(steps),
        num_channels=int(num_channels),
        num_blocks=int(num_channels) // int(block_size),
        per_step_budget_ns=per_step_ns,
        sync_floor_us_design_a=design_a,
        sync_floor_us_design_b=design_b,
        compute_floor_us_single_block=compute_single,
        compute_floor_us_multi_block=compute_spec,
        compute_floor_us_fft_multi_block=compute_fft,
        floor_us_design_a=floor_a,
        floor_us_design_b=floor_b,
        binding_constraint=binding,
        macs_per_step=macs,
        compute_cycles_single_block=cycles_single,
        verdict=verdict,
        sub_budget_reachable=bool(sub_ok),
        shutter_reachable=bool(shutter_ok),
        shutter_reachable_compute_only=bool(shutter_compute_only),
    )


def slot_budget_analysis(
    *,
    ticks_per_slot: int = 32,
    num_channels: int = SPEC_NUM_TILES,
    sm_count: int = BLACKWELL_SM_COUNT,
    clock_ghz: float = 2.0,
) -> Dict[str, Any]:
    """Per-SLOT feasibility: the resolution when a sub-budget fails.

    The mandate asks for the full 1024-step relaxation inside 12.8 us, "fitting
    inside the 50 us shutter window". That conflates two DIFFERENT quantities:

      LOCK HORIZON   750-1024 relaxation steps from cold (MEASURED). A one-time
                     cold-start cost. It does NOT fit one slot and was never
                     required to: `slots_to_lock()` already reports 32 slots at
                     32 ticks/slot.
      SHUTTER SLOT   50 us of ingress aperture carrying `ticks_per_slot`
                     relaxation steps.

    `tau_budget_analysis()` shows the 1024-step horizon cannot fit a single
    50 us slot. This function asks the question the architecture actually
    needs: does ONE SLOT of relaxation fit the aperture?

    Two coupling algorithms are costed, because they differ by ~8x and the
    cheaper one is the correct GPU choice:

      tap sum   Z_k = sum_w J_w exp(i theta_{k-w})   O(N * taps)   ~8.27e6 MAC
      FFT       Z   = ifft(fft(exp(i theta)) * J)    O(N log N)    ~1.10e6 MAC

    The Triton kernel in this module implements the TAP SUM. On GPU the FFT
    form is materially cheaper, so a production kernel should use cuFFT for the
    coupling and keep the tap sum as the parity reference.

    Synchronization is costed explicitly and is the binding term: a coupling
    step is global, so every step needs a barrier before the next.
    """
    taps = taps_for_reach(int(SPEC_NON_LOCAL_SPAN) // 2)
    n = float(num_channels)
    lanes_all = float(BLACKWELL_SM_FMA_PER_CLK) * float(sm_count)

    macs_step_tap = 2.0 * n * float(taps)
    macs_step_fft = 2.0 * (5.0 * n * math.log2(n)) + 4.0 * n

    def _us(total_macs: float) -> float:
        return (total_macs / lanes_all) / (clock_ghz * 1e9) * 1e6

    ticks = float(ticks_per_slot)
    g_lo, g_hi = GRID_SYNC_LATENCY_US
    b_lo, b_hi = (v / 1000.0 for v in SYNCTHREADS_LATENCY_NS)  # ns -> us

    compute_slot_tap = _us(macs_step_tap * ticks)
    compute_slot_fft = _us(macs_step_fft * ticks)

    # Synchronization per slot, both barrier kinds.
    sync_slot_grid = (ticks * g_lo, ticks * g_hi)
    sync_slot_block = (ticks * b_lo, ticks * b_hi)

    # Totals. Grid sync pairs with a distributed (multi-SM) coupling; the block
    # barrier pairs with a single-block layout, whose compute is then confined
    # to ONE SM.
    compute_slot_tap_one_sm = _us(macs_step_tap * ticks) * float(sm_count)
    floor_grid_tap = (sync_slot_grid[0] + compute_slot_tap,
                      sync_slot_grid[1] + compute_slot_tap)
    floor_grid_fft = (sync_slot_grid[0] + compute_slot_fft,
                      sync_slot_grid[1] + compute_slot_fft)
    floor_block_tap = (sync_slot_block[0] + compute_slot_tap_one_sm,
                       sync_slot_block[1] + compute_slot_tap_one_sm)

    best = min(floor_grid_fft[0], floor_grid_fft[1])
    fits_50 = floor_grid_fft[0] <= SPEC_SHUTTER_US
    fits_128 = floor_grid_fft[0] <= SPEC_TAU_BUDGET_US

    return {
        "ticks_per_slot": int(ticks_per_slot),
        "macs_per_step_tap_sum": macs_step_tap,
        "macs_per_step_fft": macs_step_fft,
        "compute_slot_us_tap_sum": compute_slot_tap,
        "compute_slot_us_fft": compute_slot_fft,
        "sync_slot_us_grid": list(sync_slot_grid),
        "sync_slot_us_block": list(sync_slot_block),
        "floor_slot_us_grid_fft": list(floor_grid_fft),
        "floor_slot_us_grid_tap": list(floor_grid_tap),
        "floor_slot_us_block_tap": list(floor_block_tap),
        "best_configuration": "multi-block FFT coupling + grid sync per step",
        "best_floor_us": best,
        "fits_shutter_50us": bool(fits_50),
        "fits_sub_budget_12p8us": bool(fits_128),
        "slots_to_lock_from_cold": int(
            math.ceil(SPEC_LOCK_HORIZON_STEPS / max(1, int(ticks_per_slot)))
        ),
        "evidence_class": "DERIVED",
        "measured_tau_us": None,
        "verdict": (
            f"PER-SLOT: the best configuration (FFT coupling spread over "
            f"{sm_count} SMs, one grid barrier per step) floors at "
            f"{floor_grid_fft[0]:.1f}-{floor_grid_fft[1]:.1f} us for "
            f"{int(ticks_per_slot)} ticks. That "
            f"{'FITS' if fits_50 else 'DOES NOT FIT'} the "
            f"{SPEC_SHUTTER_US:.0f} us shutter and "
            f"{'fits' if fits_128 else 'DOES NOT fit'} the "
            f"{SPEC_TAU_BUDGET_US} us sub-budget. So the aperture is "
            f"satisfiable PER SLOT but the 1024-step COLD LOCK is not a "
            f"single-slot operation: it spans "
            f"{int(math.ceil(SPEC_LOCK_HORIZON_STEPS / max(1, int(ticks_per_slot))))} "
            f"slots. The mandate's '1024 steps inside one 50 us shutter' is "
            f"FALSIFIED; '32 ticks inside one 50 us shutter' is feasible at the "
            f"optimistic grid-sync figure. The tap-sum kernel as written costs "
            f"{compute_slot_tap:.1f} us of pure compute per slot and does NOT "
            f"fit the shutter, which is why the FFT form is the production "
            f"choice."
        ),
    }


def backend_report() -> Dict[str, Any]:
    """State of the GPU path. Never claims a run that did not happen."""
    ok = cuda_available() and TRITON_AVAILABLE
    return {
        "triton_available": TRITON_AVAILABLE,
        "triton_import_error": TRITON_IMPORT_ERROR,
        "cuda_available": cuda_available(),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if cuda_available() else None,
        "measured_tau_us": None,
        "status": "OBSERVED" if ok else "BLOCKED",
        "note": (
            "measured_tau_us stays None until a real GPU run fills it. The "
            "tau figures from tau_budget_analysis() are DERIVED."
        ),
    }
