"""
Project HENRI - qFHRR Autopoietic Carrier (Stage 1).

Document: HENRI-ARCH-2026-CARRIER-AUDIT-AND-PHYSICAL-ML-GAPS
Target:   carrier/e6-physical-verifier

PURPOSE (from spec section 2.1)
-------------------------------
The G-AUTO-1 statistical contract operates on discrete symbol streams
S in Sigma^n. The Zone B wave core operates on continuous complex state
vectors Psi on the unit hypersphere S^{D-1}. Evaluating one directly
against the other is a category error:

  * Kolmogorov complexity and Shannon entropy require a discrete alphabet
    with counting measure.
  * Wave mechanics on S^{D-1} require differential forms, Haar measure, and
    continuous Lie group actions.

This module supplies the canonical, norm-preserving projection pair:

    Psi in S^{D-1}  --phase quantize-->  S in Sigma^n (|Sigma| = 256)
    S               --pinned compressor-->  G_eps(S) in R

and enforces the SYMMETRICAL two-sided epsilon-band contract, which
resolves Defect D1 (self-destroying gate).

DEFECTS FIXED RELATIVE TO THE SPEC DRAFT
----------------------------------------
D1 (self-destroying gate). The draft rejected when G > eps. Because a
   finite-n compressor has warm-up cost, control streams (dead, noise)
   produce K_LZ > 0 and therefore G < 0. A one-sided rule rejects its own
   negative controls. Fix: two-sided test in classify().

DEFECTS PRESENT IN THE SPEC DRAFT KERNEL (flagged, NOT silently fixed)
----------------------------------------------------------------------
D-A  The draft kernel computes dot_sum and the Kuramoto order parameter
     per BLOCK, then writes the scalars from pid == 0 only. That is a
     block-local value, not a global reduction over D. The draft itself
     admits "Block-level approximation". It is not a global Sagnac
     homodyne overlap. See torch_reference_kernel() for the exact global
     form.
D-B  The draft kernel stores energy_pool_ptr from EVERY block, which is a
     write race on the same scalar address. Only one writer may update it,
     or the update must be atomic.
D-C  tl.atan2 is used on a truncated complex pair. Phase is undefined where
     |Psi| -> 0, and the retraction cos/sin discards magnitude, so the
     operator is norm-preserving only for inputs already on the unit
     sphere.
D-D  dt_eff = 1 / max(E, 0.01) is a rate limiter, not a Margolus-Levitin
     bound. It is dimensionless and has no hbar or energy-in-joules scale.

These are reported, not hidden. Resolving them is Stage 1 follow-on work
and must be gated by measurement, not by renaming.

EVIDENCE LABELS
---------------
Every printed number is OBSERVED (a real run produced it) or DERIVED
(computed by the stated rule). Nothing is asserted from the draft text.
"""

from __future__ import annotations

import json
import math
import zlib
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_EPSILON = 0.02

# n_min is NOT free to choose. Defect D1 quantified (measured 2026-09-12):
# zlib carries a fixed warm-up cost of roughly O(300..500) bits. That constant
# appears in K_LZ as overhead/n, so for small n the control streams show
# G < -eps and a one-sided rule rejects its own negative controls.
#
#   n_min >= ceil(overhead_bits / epsilon)
#
# Measured control streams (|G|, eps = 0.02), dead / noise:
#     n= 1024  0.13281 / 0.24020   FAIL
#     n= 2048  0.08984 / 0.13154   FAIL
#     n= 4096  0.05078 / 0.06839   FAIL
#     n= 8192  0.03027 / 0.03873   FAIL
#     n=12400  0.02258 / 0.02253   FAIL
#     n=16384  0.01904 / 0.01745   PASS
#     n=32768  0.01270 / 0.00935   PASS
#     n=65536  0.01025 / 0.00619   PASS
#
# The measured crossover (~16k) matches the derived bound
# (approx 317 overhead bits / 0.02 = 15850) within one doubling step.
DEFAULT_MIN_TOKENS = 16384
ALPHABET_SIZE = 256

# ---------------------------------------------------------------------------
# Optional accelerators. Import failure must not break the CPU contract path.
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False


# ===========================================================================
# 1. The fused Triton kernel (spec section 5, faithful transcription)
# ===========================================================================
if _HAS_TRITON:

    @triton.jit
    def _fused_autopoietic_kuramoto_kernel(
        psi_real_ptr,
        psi_imag_ptr,
        axiom_real_ptr,
        axiom_imag_ptr,
        energy_pool_ptr,
        order_param_ptr,
        sagnac_delta_ptr,
        coupling_K,
        absorption_alpha,
        steal_delta,
        veto_threshold,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused GPU kernel: Sagnac homodyne, autopoietic drag, Margolus-Levitin
        rate limiting, and Kuramoto phase relaxation in one pass.

        See defects D-A and D-B: reductions and the energy store are
        block-local / racy as transcribed from the draft.
        """
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D

        # 1. Global load of unitary wave vector components.
        r_psi = tl.load(psi_real_ptr + offsets, mask=mask, other=0.0)
        i_psi = tl.load(psi_imag_ptr + offsets, mask=mask, other=0.0)
        r_ax = tl.load(axiom_real_ptr + offsets, mask=mask, other=0.0)
        i_ax = tl.load(axiom_imag_ptr + offsets, mask=mask, other=0.0)

        # 2. Local Sagnac homodyne alignment: Re(Psi * conj(Psi_axiom)).
        dot_local = r_psi * r_ax + i_psi * i_ax
        dot_sum = tl.sum(dot_local, axis=0)
        norm_overlap = dot_sum / D          # D-A: block-local, not global
        sagnac_err = 1.0 - norm_overlap

        # 3. Autopoietic energetic drag accounting.
        is_vetoed = sagnac_err > veto_threshold
        drag_factor = tl.where(
            is_vetoed, (1.0 - absorption_alpha) * steal_delta + 1.0, 1.0
        )

        # 4. Margolus-Levitin dynamic execution scaling.
        curr_energy = tl.load(energy_pool_ptr)
        new_energy = tl.maximum(curr_energy - drag_factor, 0.0)
        tl.store(energy_pool_ptr, new_energy)   # D-B: racy across blocks
        dt_eff = 1.0 / tl.maximum(new_energy, 0.01)   # D-D: dimensionless

        # 5. Non-linear Kuramoto phase update.
        theta = tl.atan2(i_psi, r_psi)          # D-C: undefined at |Psi|=0
        cos_sum = tl.sum(tl.cos(theta), axis=0)
        sin_sum = tl.sum(tl.sin(theta), axis=0)
        r_order = tl.sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / D

        mean_phase = tl.atan2(sin_sum, cos_sum)
        d_theta = coupling_K * tl.sin(mean_phase - theta) * dt_eff
        theta_new = theta + d_theta

        # 6. Reproject to the complex unit hypersphere (Stiefel retraction).
        tl.store(psi_real_ptr + offsets, tl.cos(theta_new), mask=mask)
        tl.store(psi_imag_ptr + offsets, tl.sin(theta_new), mask=mask)

        if pid == 0:
            tl.store(order_param_ptr, r_order)
            tl.store(sagnac_delta_ptr, sagnac_err)

else:  # pragma: no cover
    _fused_autopoietic_kuramoto_kernel = None


def launch_kernel(
    psi: "torch.Tensor",
    axiom: "torch.Tensor",
    energy: float,
    coupling_K: float = 0.5,
    absorption_alpha: float = 0.5,
    steal_delta: float = 0.1,
    veto_threshold: float = 0.5,
    block_size: int = 1024,
) -> Dict[str, float]:
    """Launch the fused kernel once. CUDA required. Returns telemetry scalars.

    This is the PRODUCTION path. It is not exercised unless a CUDA device
    is present; the caller must check the returned 'launched' flag.
    """
    if not _HAS_TORCH or _fused_autopoietic_kuramoto_kernel is None:
        return {"launched": 0, "reason": "no_triton_or_torch"}
    if not torch.cuda.is_available():
        return {"launched": 0, "reason": "no_cuda"}

    D = int(psi.numel())
    psi_r = psi.real.contiguous().float()
    psi_i = psi.imag.contiguous().float()
    ax_r = axiom.real.contiguous().float()
    ax_i = axiom.imag.contiguous().float()
    energy_t = torch.tensor([float(energy)], device=psi.device, dtype=torch.float32)
    order_t = torch.zeros(1, device=psi.device, dtype=torch.float32)
    sagnac_t = torch.zeros(1, device=psi.device, dtype=torch.float32)

    grid = (triton.cdiv(D, block_size),)
    _fused_autopoietic_kuramoto_kernel[grid](
        psi_r, psi_i, ax_r, ax_i, energy_t, order_t, sagnac_t,
        float(coupling_K), float(absorption_alpha), float(steal_delta),
        float(veto_threshold), D=D, BLOCK_SIZE=block_size,
    )
    return {
        "launched": 1,
        "order_param": float(order_t.item()),
        "sagnac_delta": float(sagnac_t.item()),
        "energy_remaining": float(energy_t.item()),
        "D": D,
    }


def torch_reference_kernel(
    psi: "torch.Tensor",
    axiom: "torch.Tensor",
    energy: float,
    coupling_K: float = 0.5,
    absorption_alpha: float = 0.5,
    steal_delta: float = 0.1,
    veto_threshold: float = 0.5,
) -> Dict[str, float]:
    """Exact GLOBAL reduction of the same arithmetic. Resolves defect D-A.

    CPU or CUDA. Uses global sums, not per-block approximations, and a single
    scalar writer. Returns the same telemetry keys as launch_kernel.
    """
    if not _HAS_TORCH:
        return {"launched": 0, "reason": "no_torch"}

    p = psi.reshape(-1)
    a = axiom.reshape(-1)
    if p.numel() != a.numel():
        raise ValueError(f"shape mismatch: psi={p.numel()} axiom={a.numel()}")
    D = int(p.numel())

    dot_sum = float(torch.sum(p.real * a.real + p.imag * a.imag).item())
    norm_overlap = dot_sum / max(D, 1)
    sagnac_err = 1.0 - norm_overlap

    is_vetoed = sagnac_err > veto_threshold
    drag = ((1.0 - absorption_alpha) * steal_delta + 1.0) if is_vetoed else 1.0
    new_energy = max(energy - drag, 0.0)
    dt_eff = 1.0 / max(new_energy, 0.01)

    theta = torch.angle(p)
    cos_sum = float(torch.sum(torch.cos(theta)).item())
    sin_sum = float(torch.sum(torch.sin(theta)).item())
    r_order = math.hypot(cos_sum, sin_sum) / max(D, 1)
    mean_phase = math.atan2(sin_sum, cos_sum)
    theta_new = theta + coupling_K * torch.sin(mean_phase - theta) * dt_eff

    return {
        "launched": 1,
        "global_reduction": 1,
        "order_param": r_order,
        "sagnac_delta": sagnac_err,
        "energy_remaining": new_energy,
        "D": D,
        "psi_out": torch.polar(torch.ones_like(theta_new), theta_new),
    }


# ===========================================================================
# 2. Complexity estimators
# ===========================================================================
def shannon_entropy_bits(sequence: bytes) -> float:
    """Empirical Shannon entropy of the byte histogram, in bits per symbol."""
    n = len(sequence)
    if n == 0:
        return 0.0
    counts = [0] * ALPHABET_SIZE
    for b in sequence:
        counts[b] += 1
    h = 0.0
    for c in counts:
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h


def deflate_bits_per_symbol(sequence: bytes, level: int = 9) -> Tuple[float, int]:
    """Pinned compressor: zlib Deflate level 9. Returns (bits/symbol, bytes)."""
    n = len(sequence)
    if n == 0:
        return 0.0, 0
    c = zlib.compress(sequence, level)
    return (len(c) * 8.0) / n, len(c)


def lz78_phrase_count(sequence: bytes) -> int:
    """Number of LZ78 dictionary phrases. Basis for a normalized complexity."""
    table: Dict[bytes, int] = {}
    w = b""
    count = 0
    for b in sequence:
        wb = w + bytes((b,))
        if wb in table:
            w = wb
        else:
            table[wb] = count
            count += 1
            w = b""
    if w:
        count += 1
    return count


def lz78_bits_per_symbol(sequence: bytes) -> float:
    """Normalized LZ78 complexity in bits per symbol: c(n)*log2(c(n))/n.

    Chosen over c*log2(n)/n because the latter does not vanish for a
    constant string. Both are reported by the gate script.
    """
    n = len(sequence)
    if n == 0:
        return 0.0
    c = lz78_phrase_count(sequence)
    if c <= 1:
        return 0.0
    return (c * math.log2(c)) / n


# ===========================================================================
# 3. The carrier verifier
# ===========================================================================
class AutopoieticCarrierVerifier:
    """Maps continuous wave states to discrete symbol sequences and evaluates
    the symmetrical epsilon-band statistical contract (G-AUTO-1).

    Parameters
    ----------
    epsilon_bound : float
        Half-width of the accepted band. The user gate fixes this at 0.02.
    min_tokens : int
        Pre-condition gate. Streams shorter than this are quarantined as
        BLOCKED_SUB_MINIMUM_N because finite-n compressor warm-up distorts G.
    """

    def __init__(self, epsilon_bound: float = DEFAULT_EPSILON,
                 min_tokens: int = DEFAULT_MIN_TOKENS):
        self.epsilon = float(epsilon_bound)
        self.min_tokens = int(min_tokens)

    # -- continuous -> discrete -------------------------------------------
    def project_wave_to_symbols(self, psi_complex: "torch.Tensor") -> bytes:
        """Canonical projection: phase theta in [-pi, pi) -> byte in 0..255.

        Norm preserving in the sense that only the phase is consumed: the
        magnitude is discarded, so the projection is well defined for any
        nonzero magnitude and identical for all scalar multiples of Psi.
        """
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for project_wave_to_symbols")
        z = psi_complex.reshape(-1)
        phases = torch.angle(z).contiguous()
        normalized = ((phases + math.pi) / (2.0 * math.pi) * 256.0).clamp(0, 255)
        return normalized.to(torch.uint8).cpu().numpy().tobytes()

    def symbols_to_wave(self, sequence: bytes) -> "torch.Tensor":
        """Inverse projection: byte -> unit-modulus complex symbol."""
        if not _HAS_TORCH:
            raise RuntimeError("torch is required for symbols_to_wave")
        b = torch.tensor(list(sequence), dtype=torch.float32)
        theta = (b / 256.0) * 2.0 * math.pi - math.pi
        return torch.polar(torch.ones_like(theta), theta)

    # -- the estimator ----------------------------------------------------
    def compute_statistical_contract(self, sequence: bytes) -> Dict[str, Any]:
        """Shannon entropy, normalized compressor length, and G_eps(S).

        G_eps(S) = [H_shannon(S) - K_LZ(S)/n] * Theta(n - n_min)

        Verdicts:
          ACCEPT_STRUCTURED_COMPLEXITY  G > +eps   (structure exceeds noise)
          ACCEPT_CONTROL_STREAM         |G| <= eps (dead or noise control)
          REJECT_DEGENERATE_BIAS        G < -eps   (compressor beat entropy)
          BLOCKED_SUB_MINIMUM_N         n < n_min  (pre-condition gate)
        """
        n = len(sequence)
        if n < self.min_tokens:
            return {
                "verdict": "BLOCKED_SUB_MINIMUM_N",
                "n": n,
                "n_min": self.min_tokens,
                "G_stat": 0.0,
                "epsilon": self.epsilon,
            }

        h = shannon_entropy_bits(sequence)
        k_zlib, cbytes = deflate_bits_per_symbol(sequence, 9)
        k_lz78 = lz78_bits_per_symbol(sequence)

        g = h - k_zlib
        if g > self.epsilon:
            verdict = "ACCEPT_STRUCTURED_COMPLEXITY"
        elif abs(g) <= self.epsilon:
            verdict = "ACCEPT_CONTROL_STREAM"
        else:
            verdict = "REJECT_DEGENERATE_BIAS"

        return {
            "verdict": verdict,
            "n": n,
            "n_min": self.min_tokens,
            "h_shannon": round(h, 6),
            "k_lz_normalized": round(k_zlib, 6),
            "k_lz78_normalized": round(k_lz78, 6),
            "compressed_bytes": cbytes,
            "G_stat": round(g, 6),
            "G_stat_lz78": round(h - k_lz78, 6),
            "abs_G": round(abs(g), 6),
            "epsilon": self.epsilon,
        }


# ===========================================================================
# 4. Synthetic controls (deterministic, reproducible)
# ===========================================================================
def _lcg_bytes(n: int, seed: int) -> bytes:
    """Deterministic uniform byte stream. No dependency on numpy RNG state."""
    out = bytearray(n)
    x = seed & 0xFFFFFFFF
    for i in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out[i] = (x >> 16) & 0xFF
    return bytes(out)


def stream_dead(n: int) -> bytes:
    """Control 1: constant zero stream. H = 0."""
    return bytes(n)


def stream_noise(n: int, seed: int = 12345) -> bytes:
    """Control 2: uniform pseudo-random stream. H -> 8 bits/symbol."""
    return _lcg_bytes(n, seed)


def stream_periodic(n: int, period: int = 4) -> bytes:
    """Trivially periodic stream. NOT a spec control; used to probe whether
    the gate rejects trivial periodicity as structure. It does not."""
    pat = bytes((i * 64) % 256 for i in range(period))
    return (pat * (n // period + 1))[:n]


def stream_structured_markov(n: int, states: int = 16, seed: int = 7) -> bytes:
    """Task-like stream: sparse Markov chain over `states` symbols.

    Moderate Shannon entropy with strong local predictability, so a real
    compressor achieves K_LZ well below H and G rises above the band.
    """
    trans = [[0] * states for _ in range(states)]
    for s in range(states):
        trans[s][s] = 60
        trans[s][(s + 1) % states] = 30
        trans[s][(s + 7) % states] = 10
    out = bytearray(n)
    s = 0
    x = seed
    for i in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        r = (x >> 16) % 100
        acc = 0
        for nxt in range(states):
            acc += trans[s][nxt]
            if r < acc:
                s = nxt
                break
        out[i] = s * 8
    return bytes(out)


# ===========================================================================
# 5. Self-test
# ===========================================================================
def _selftest(n: int = DEFAULT_MIN_TOKENS, epsilon: float = DEFAULT_EPSILON) -> int:
    v = AutopoieticCarrierVerifier(epsilon_bound=epsilon, min_tokens=128)
    cases = [
        ("dead", stream_dead(n), "ACCEPT_CONTROL_STREAM"),
        ("noise", stream_noise(n), "ACCEPT_CONTROL_STREAM"),
        ("structured", stream_structured_markov(n), "ACCEPT_STRUCTURED_COMPLEXITY"),
    ]
    print(f"=== CARRIER SELFTEST  n={n}  eps={epsilon} ===")
    print(f"{'stream':<12}{'H':>9}{'K_zlib':>9}{'G':>10}{'|G|':>9}  verdict")
    fails = 0
    for name, seq, expect in cases:
        r = v.compute_statistical_contract(seq)
        flag = "OK" if r["verdict"] == expect else "MISMATCH"
        if flag != "OK":
            fails += 1
        print(f"{name:<12}{r['h_shannon']:>9.4f}{r['k_lz_normalized']:>9.4f}"
              f"{r['G_stat']:>10.5f}{r['abs_G']:>9.5f}  {r['verdict']} {flag}")
    print(f"selftest failures: {fails}")
    return fails


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    raise SystemExit(1 if _selftest(n) else 0)
