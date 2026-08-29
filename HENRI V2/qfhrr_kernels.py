"""
Project HENRI: Quantized FHRR (qFHRR) phase-domain similarity kernels.

Maps continuous [num_blocks, 8] Clifford waves to 8-bit phase indices on the
Z_256 ring (4 complex phase pairs per block -> 4 INT8 codes per block), and
evaluates cosine similarity as modular phase-difference lookup:

    sim(q_a, q_b) = (1/D) * sum_i LUT_cos[(q_a_i - q_b_i) mod 256]

The heavy path is a Triton fused kernel (GPU, INT32 accumulation, block
partials -> final divide; no atomic contention). A pure-torch fallback keeps
the CPU suite green. Behind the similarity contract sits a batch form for
engram stores: sim against M stored codes in one call.

Engineering notes:
- The real 8-dim Clifford block is viewed as 4 complex pairs (re,im interleaved
  as stored: first 4 = real parts, last 4 = imag parts — matching
  efe_planner.UnitaryWaveTransition.bind's convention). Quantization takes
  atan2 of each pair; degenerate pairs (norm ~ 0) map to code 0.
- LUT holds INT32 cosine values scaled by 127; accumulation in INT32 is
  overflow-safe up to D ~ 2^31/127 ~ 16.9M (d=65536 is 130x under the wall).
- Similarity is NOT the valence source. It accelerates retrieval/resonance
  sites (preference store, novelty memory, Hopfield cleanup, Zone C cosine)
  where targets legitimately exist. No T_g anchor lives here.
"""

import math
import os

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
    # atan2 lives in libdevice on Triton >= 3.3 (tl.math.atan2 removed).
    try:
        from triton.language.extra import libdevice as _tlib
    except ImportError:
        from triton.language.extra.cuda import libdevice as _tlib
except ImportError:  # CPU-only box
    _HAS_TRITON = False


K_PHASE = 256          # Z_256 phase ring
LUT_SCALE = 127.0      # INT8-range cosine scaling


# ---------------------------------------------------------------------------
# LUT
# ---------------------------------------------------------------------------

def build_cos_lut(device="cpu") -> torch.Tensor:
    """256-entry cosine LUT, INT32 scaled by 127.

    LUT[d] = round(127 * cos(2*pi*d/256)); index is the modular phase
    difference in [0, 255]. Signed range [-127, 127].
    """
    idx = torch.arange(K_PHASE, dtype=torch.float64)
    lut = torch.round(LUT_SCALE * torch.cos(2.0 * math.pi * idx / K_PHASE))
    return lut.to(torch.int32).to(device)


# ---------------------------------------------------------------------------
# Codec: continuous wave <-> quantized phase codes
# ---------------------------------------------------------------------------

def wave_to_phase_codes(wave: torch.Tensor) -> torch.Tensor:
    """[num_blocks, 8] real Clifford wave -> [num_blocks, 4] uint8 phase codes.

    Convention: block layout is [re_0..3 | im_0..3] (efe_planner.bind).
    theta = atan2(im, re) in [-pi, pi) -> q = floor((theta + pi) * 256 / 2pi),
    clamped to [0, 255]. Degenerate pairs (|re|+|im| ~ 0) map to 0.
    """
    nb = wave.shape[0]
    re = wave[..., :4]
    im = wave[..., 4:]
    theta = torch.atan2(im, re)                              # [nb, 4] in [-pi, pi)
    q = torch.floor((theta + math.pi) * (K_PHASE / (2.0 * math.pi)))
    q = q.clamp(0, K_PHASE - 1)
    # Degenerate (near-zero) pairs carry no phase information
    dead = (re.abs() + im.abs()) < 1e-9
    q = torch.where(dead, torch.zeros_like(q), q)
    return q.to(torch.uint8).view(nb, 4)


def phase_codes_to_wave(q: torch.Tensor) -> torch.Tensor:
    """[num_blocks, 4] uint8 codes -> [num_blocks, 8] unit-modulus wave.

    Inverse of wave_to_phase_codes up to quantization error: each code maps
    back to the bin-center phase; complex pairs are unit modulus.
    """
    nb = q.shape[0]
    theta = (q.to(torch.float32) + 0.5) * (2.0 * math.pi / K_PHASE) - math.pi
    re, im = torch.cos(theta), torch.sin(theta)
    return torch.cat([re, im], dim=-1).view(nb, 8)


def quantization_roundtrip_error(wave: torch.Tensor) -> float:
    """Max phase error (radians) between the source wave's per-pair phase and
    the dequantized codes. The codec is phase-only: per-pair amplitude is NOT
    preserved (each code decodes to a unit-modulus pair). Bound: bin
    half-width 2pi/(2*256) ≈ 0.01227 rad, plus bin-center decode."""
    re, im = wave[..., :4], wave[..., 4:]
    theta_src = torch.atan2(im, re)
    q = wave_to_phase_codes(wave)
    theta_dec = (q.to(torch.float32) + 0.5) * (2.0 * math.pi / K_PHASE) - math.pi
    # wrap-aware phase distance
    d = (theta_dec - theta_src + math.pi) % (2.0 * math.pi) - math.pi
    return d.abs().max().item()


# ---------------------------------------------------------------------------
# Similarity: torch fallback (always available)
# ---------------------------------------------------------------------------

def qfhrr_similarity_torch(q_a: torch.Tensor, q_b: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """sim(q_a, q_b) via modular-difference LUT, pure torch.

    q_a: [D] uint8; q_b: [D] or [M, D] uint8. Returns scalar or [M] float32
    in approximately [-1, 1] (scaled by LUT_SCALE then normalized).
    """
    a = q_a.to(torch.int16).view(-1)
    b = q_b.to(torch.int16)
    if b.dim() == 1:
        b = b.view(1, -1)
    diff = (a.unsqueeze(0) - b) & (K_PHASE - 1)              # [M, D]
    acc = lut[diff.long()].to(torch.int64).sum(dim=-1)       # INT64-safe sum
    return acc.to(torch.float32) / (LUT_SCALE * a.numel())


# ---------------------------------------------------------------------------
# Triton fused kernel (GPU)
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _qfhrr_sim_kernel(
        Q_ptr, M_ptr, LUT_ptr, Part_ptr,
        D: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Per-(engram, block) partial sums; host divides by LUT_SCALE*D.

        Grid: (M, cdiv(D, BLOCK)). Part_ptr: [M, num_blocks_d] int32.
        Each program handles one engram and one BLOCK-slice of the D axis:
        load q slice (uint8), load engram slice, modular diff, LUT gather,
        INT32 block reduction, store partial. No atomics.
        """
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs = pid_b * BLOCK + tl.arange(0, BLOCK)
        mask = offs < D

        q = tl.load(Q_ptr + offs, mask=mask, other=0).to(tl.int16)
        m = tl.load(M_ptr + pid_m * D + offs, mask=mask, other=0).to(tl.int16)
        diff = (q - m) & 255
        cos_val = tl.load(LUT_ptr + diff, mask=mask, other=0)
        partial = tl.sum(cos_val, axis=0)
        tl.store(Part_ptr + pid_m * tl.num_programs(1) + pid_b, partial)

    def qfhrr_similarity_triton(q_a: torch.Tensor, q_b: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
        """GPU fused similarity. q_a: [D] uint8 cuda; q_b: [D] or [M, D] uint8
        cuda; lut: [256] int32 cuda. Returns scalar or [M] float32."""
        a = q_a.contiguous().view(-1)
        b = q_b.contiguous()
        if b.dim() == 1:
            b = b.view(1, -1)
        M, D = b.shape
        BLOCK = 4096
        n_blocks = triton.cdiv(D, BLOCK)
        partials = torch.zeros((M, n_blocks), dtype=torch.int32, device=a.device)
        _qfhrr_sim_kernel[(M, n_blocks)](
            a, b, lut, partials, D=D, BLOCK=BLOCK,
        )
        acc = partials.to(torch.int64).sum(dim=-1)
        out = acc.to(torch.float32) / (LUT_SCALE * D)
        return out if M > 1 else out.squeeze(0)

    # ------------------------------------------------------------------
    # Triton 3x3 complex matrix logarithm (Phase 8.18 C3, spec 158c02c7).
    # Closed-form characteristic-polynomial roots (Cardano cubic +
    # Lagrange projectors), scalar float32 arithmetic only (no complex eig
    # in Triton). Vectorized over BLOCK_N matrices per program.
    # Layout: [N, 18] float32 = real 9 | imag 9 (house convention).
    # ------------------------------------------------------------------
    @triton.jit
    def _su3_log_kernel(a_ptr, o_ptr, n, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < n
        base = offs
        # [N,18] row-major: row r starts at flat r*18 (stride-18 addressing)
        base18 = base * 18
        # ---- load 9 real + 9 imag (fully unrolled, no comprehensions) ----
        mre00 = tl.load(a_ptr + base18 + 0, mask=mask, other=0.0)
        mre01 = tl.load(a_ptr + base18 + 1, mask=mask, other=0.0)
        mre02 = tl.load(a_ptr + base18 + 2, mask=mask, other=0.0)
        mre10 = tl.load(a_ptr + base18 + 3, mask=mask, other=0.0)
        mre11 = tl.load(a_ptr + base18 + 4, mask=mask, other=0.0)
        mre12 = tl.load(a_ptr + base18 + 5, mask=mask, other=0.0)
        mre20 = tl.load(a_ptr + base18 + 6, mask=mask, other=0.0)
        mre21 = tl.load(a_ptr + base18 + 7, mask=mask, other=0.0)
        mre22 = tl.load(a_ptr + base18 + 8, mask=mask, other=0.0)
        mim00 = tl.load(a_ptr + 9 + base18 + 0, mask=mask, other=0.0)
        mim01 = tl.load(a_ptr + 9 + base18 + 1, mask=mask, other=0.0)
        mim02 = tl.load(a_ptr + 9 + base18 + 2, mask=mask, other=0.0)
        mim10 = tl.load(a_ptr + 9 + base18 + 3, mask=mask, other=0.0)
        mim11 = tl.load(a_ptr + 9 + base18 + 4, mask=mask, other=0.0)
        mim12 = tl.load(a_ptr + 9 + base18 + 5, mask=mask, other=0.0)
        mim20 = tl.load(a_ptr + 9 + base18 + 6, mask=mask, other=0.0)
        mim21 = tl.load(a_ptr + 9 + base18 + 7, mask=mask, other=0.0)
        mim22 = tl.load(a_ptr + 9 + base18 + 8, mask=mask, other=0.0)
        # ---- trace tr = sum diag ----
        tr_r = mre00 + mre11 + mre22
        tr_i = mim00 + mim11 + mim22
        # ---- cubic x^3 + a x^2 + b x + c: a = -tr, b = conj(tr), c = -1 ----
        a_r = -tr_r
        a_i = -tr_i
        b_r = tr_r
        b_i = -tr_i
        c_r = tl.full([BLOCK_N], -1.0, dtype=tl.float32)
        # p = b - a^2/3
        a2_r = a_r * a_r - a_i * a_i
        a2_i = 2.0 * a_r * a_i
        p_r = b_r - a2_r / 3.0
        p_i = b_i - a2_i / 3.0
        # q = 2 a^3 / 27 - a b / 3 + c
        a3_r = a2_r * a_r - a2_i * a_i
        a3_i = a2_r * a_i + a2_i * a_r
        ab_r = a_r * b_r - a_i * b_i
        ab_i = a_r * b_i + a_i * b_r
        q_r = 2.0 * a3_r / 27.0 - ab_r / 3.0 + c_r
        q_i = 2.0 * a3_i / 27.0 - ab_i / 3.0
        # D = (q/2)^2 + (p/3)^3
        q2_r = q_r / 2.0
        q2_i = q_i / 2.0
        p3_r = p_r / 3.0
        p3_i = p_i / 3.0
        p3c_r = p3_r * p3_r * p3_r - 3.0 * p3_r * p3_i * p3_i
        p3c_i = 3.0 * p3_r * p3_r * p3_i - p3_i * p3_i * p3_i
        D_r = q2_r * q2_r - q2_i * q2_i + p3c_r
        D_i = 2.0 * q2_r * q2_i + p3c_i
        # sqrt(D) — radicands are >= 0 by construction; clamp guards float32
        # rounding where D_mag can land 1 ulp below |D_r| (sqrt(-eps) = NaN).
        D_mag = tl.sqrt(D_r * D_r + D_i * D_i)
        sD_r = tl.sqrt(tl.maximum((D_mag + D_r) / 2.0, 0.0))
        sD_i = tl.where(D_i >= 0.0, 1.0, -1.0) * tl.sqrt(tl.maximum((D_mag - D_r) / 2.0, 0.0))
        # u^3 = -q/2 + sqrt(D); cube root (3 branches)
        u3_r = -q2_r + sD_r
        u3_i = -q2_i + sD_i
        u3_mag = tl.sqrt(u3_r * u3_r + u3_i * u3_i)
        u3_ang = _tlib.atan2(u3_i, u3_r)
        u_mag = tl.exp2(tl.log2(tl.maximum(u3_mag, 1e-30)) * (1.0 / 3.0))
        # ---- U^2 = U @ U (explicit 3x3 complex) ----
        u2r00 = mre00 * mre00 - mim00 * mim00 + mre01 * mre10 - mim01 * mim10 + mre02 * mre20 - mim02 * mim20
        u2r01 = mre00 * mre01 - mim00 * mim01 + mre01 * mre11 - mim01 * mim11 + mre02 * mre21 - mim02 * mim21
        u2r02 = mre00 * mre02 - mim00 * mim02 + mre01 * mre12 - mim01 * mim12 + mre02 * mre22 - mim02 * mim22
        u2r10 = mre10 * mre00 - mim10 * mim00 + mre11 * mre10 - mim11 * mim10 + mre12 * mre20 - mim12 * mim20
        u2r11 = mre10 * mre01 - mim10 * mim01 + mre11 * mre11 - mim11 * mim11 + mre12 * mre21 - mim12 * mim21
        u2r12 = mre10 * mre02 - mim10 * mim02 + mre11 * mre12 - mim11 * mim12 + mre12 * mre22 - mim12 * mim22
        u2r20 = mre20 * mre00 - mim20 * mim00 + mre21 * mre10 - mim21 * mim10 + mre22 * mre20 - mim22 * mim20
        u2r21 = mre20 * mre01 - mim20 * mim01 + mre21 * mre11 - mim21 * mim11 + mre22 * mre21 - mim22 * mim21
        u2r22 = mre20 * mre02 - mim20 * mim02 + mre21 * mre12 - mim21 * mim12 + mre22 * mre22 - mim22 * mim22
        u2i00 = mre00 * mim00 + mim00 * mre00 + mre01 * mim10 + mim01 * mre10 + mre02 * mim20 + mim02 * mre20
        u2i01 = mre00 * mim01 + mim00 * mre01 + mre01 * mim11 + mim01 * mre11 + mre02 * mim21 + mim02 * mre21
        u2i02 = mre00 * mim02 + mim00 * mre02 + mre01 * mim12 + mim01 * mre12 + mre02 * mim22 + mim02 * mre22
        u2i10 = mre10 * mim00 + mim10 * mre00 + mre11 * mim10 + mim11 * mre10 + mre12 * mim20 + mim12 * mre20
        u2i11 = mre10 * mim01 + mim10 * mre01 + mre11 * mim11 + mim11 * mre11 + mre12 * mim21 + mim12 * mre21
        u2i12 = mre10 * mim02 + mim10 * mre02 + mre11 * mim12 + mim11 * mre12 + mre12 * mim22 + mim12 * mre22
        u2i20 = mre20 * mim00 + mim20 * mre00 + mre21 * mim10 + mim21 * mre10 + mre22 * mim20 + mim22 * mre20
        u2i21 = mre20 * mim01 + mim20 * mre01 + mre21 * mim11 + mim21 * mre11 + mre22 * mim21 + mim22 * mre21
        u2i22 = mre20 * mim02 + mim20 * mre02 + mre21 * mim12 + mim21 * mre12 + mre22 * mim22 + mim22 * mre22
        # ---- 3 roots via per-branch pairing v = -p/(3u) ----
        # root k = 0, 1, 2 (static unroll)
        ang0 = (u3_ang + 0.0) / 3.0
        ang1 = (u3_ang + 6.283185307179586) / 3.0
        ang2 = (u3_ang + 12.566370614359172) / 3.0
        u0_r = u_mag * tl.cos(ang0)
        u0_i = u_mag * tl.sin(ang0)
        u1_r = u_mag * tl.cos(ang1)
        u1_i = u_mag * tl.sin(ang1)
        u2_r = u_mag * tl.cos(ang2)
        u2_i = u_mag * tl.sin(ang2)
        den0 = 3.0 * (u0_r * u0_r + u0_i * u0_i)
        den1 = 3.0 * (u1_r * u1_r + u1_i * u1_i)
        den2 = 3.0 * (u2_r * u2_r + u2_i * u2_i)
        v0_r = (-p_r * u0_r - p_i * u0_i) / den0
        v0_i = (p_r * u0_i - p_i * u0_r) / den0
        v1_r = (-p_r * u1_r - p_i * u1_i) / den1
        v1_i = (p_r * u1_i - p_i * u1_r) / den1
        v2_r = (-p_r * u2_r - p_i * u2_i) / den2
        v2_i = (p_r * u2_i - p_i * u2_r) / den2
        x0_r = u0_r + v0_r - a_r / 3.0
        x0_i = u0_i + v0_i - a_i / 3.0
        x1_r = u1_r + v1_r - a_r / 3.0
        x1_i = u1_i + v1_i - a_i / 3.0
        x2_r = u2_r + v2_r - a_r / 3.0
        x2_i = u2_i + v2_i - a_i / 3.0
        th0 = _tlib.atan2(x0_i, x0_r)
        th1 = _tlib.atan2(x1_i, x1_r)
        th2 = _tlib.atan2(x2_i, x2_r)
        # ---- Lagrange projectors via Cayley-Hamilton: Pk = (U^2 - S_k U + P_k I) * s_k ----
        # k=0: j1=1, j2=2
        S_r = x1_r + x2_r
        S_i = x1_i + x2_i
        P_r = x1_r * x2_r - x1_i * x2_i
        P_i = x1_r * x2_i + x1_i * x2_r
        d1_r = x0_r - x1_r
        d1_i = x0_i - x1_i
        d2_r = x0_r - x2_r
        d2_i = x0_i - x2_i
        de_r = d1_r * d2_r - d1_i * d2_i
        de_i = d1_r * d2_i + d1_i * d2_r
        de_mag = tl.maximum(de_r * de_r + de_i * de_i, 1e-30)
        s_r = de_r / de_mag
        s_i = -de_i / de_mag
        # T1 = U2 * s
        t1r00 = u2r00 * s_r - u2i00 * s_i
        t1i00 = u2r00 * s_i + u2i00 * s_r
        t1r01 = u2r01 * s_r - u2i01 * s_i
        t1i01 = u2r01 * s_i + u2i01 * s_r
        t1r02 = u2r02 * s_r - u2i02 * s_i
        t1i02 = u2r02 * s_i + u2i02 * s_r
        t1r10 = u2r10 * s_r - u2i10 * s_i
        t1i10 = u2r10 * s_i + u2i10 * s_r
        t1r11 = u2r11 * s_r - u2i11 * s_i
        t1i11 = u2r11 * s_i + u2i11 * s_r
        t1r12 = u2r12 * s_r - u2i12 * s_i
        t1i12 = u2r12 * s_i + u2i12 * s_r
        t1r20 = u2r20 * s_r - u2i20 * s_i
        t1i20 = u2r20 * s_i + u2i20 * s_r
        t1r21 = u2r21 * s_r - u2i21 * s_i
        t1i21 = u2r21 * s_i + u2i21 * s_r
        t1r22 = u2r22 * s_r - u2i22 * s_i
        t1i22 = u2r22 * s_i + u2i22 * s_r
        # T2 = S * U * s
        su00_r = S_r * mre00 - S_i * mim00
        su00_i = S_r * mim00 + S_i * mre00
        su01_r = S_r * mre01 - S_i * mim01
        su01_i = S_r * mim01 + S_i * mre01
        su02_r = S_r * mre02 - S_i * mim02
        su02_i = S_r * mim02 + S_i * mre02
        su10_r = S_r * mre10 - S_i * mim10
        su10_i = S_r * mim10 + S_i * mre10
        su11_r = S_r * mre11 - S_i * mim11
        su11_i = S_r * mim11 + S_i * mre11
        su12_r = S_r * mre12 - S_i * mim12
        su12_i = S_r * mim12 + S_i * mre12
        su20_r = S_r * mre20 - S_i * mim20
        su20_i = S_r * mim20 + S_i * mre20
        su21_r = S_r * mre21 - S_i * mim21
        su21_i = S_r * mim21 + S_i * mre21
        su22_r = S_r * mre22 - S_i * mim22
        su22_i = S_r * mim22 + S_i * mre22
        t2r00 = su00_r * s_r - su00_i * s_i
        t2i00 = su00_r * s_i + su00_i * s_r
        t2r01 = su01_r * s_r - su01_i * s_i
        t2i01 = su01_r * s_i + su01_i * s_r
        t2r02 = su02_r * s_r - su02_i * s_i
        t2i02 = su02_r * s_i + su02_i * s_r
        t2r10 = su10_r * s_r - su10_i * s_i
        t2i10 = su10_r * s_i + su10_i * s_r
        t2r11 = su11_r * s_r - su11_i * s_i
        t2i11 = su11_r * s_i + su11_i * s_r
        t2r12 = su12_r * s_r - su12_i * s_i
        t2i12 = su12_r * s_i + su12_i * s_r
        t2r20 = su20_r * s_r - su20_i * s_i
        t2i20 = su20_r * s_i + su20_i * s_r
        t2r21 = su21_r * s_r - su21_i * s_i
        t2i21 = su21_r * s_i + su21_i * s_r
        t2r22 = su22_r * s_r - su22_i * s_i
        t2i22 = su22_r * s_i + su22_i * s_r
        # T3 = P * s * I (diagonal only)
        diag_r = P_r * s_r - P_i * s_i
        diag_i = P_r * s_i + P_i * s_r
        # Pk0 = T1 - T2 + T3; accumulate i*th0*Pk0: re = -th*im, im = th*re
        lr00 = -th0 * (t1i00 - t2i00 + diag_i)
        li00 = th0 * (t1r00 - t2r00 + diag_r)
        lr01 = -th0 * (t1i01 - t2i01)
        li01 = th0 * (t1r01 - t2r01)
        lr02 = -th0 * (t1i02 - t2i02)
        li02 = th0 * (t1r02 - t2r02)
        lr10 = -th0 * (t1i10 - t2i10)
        li10 = th0 * (t1r10 - t2r10)
        lr11 = -th0 * (t1i11 - t2i11 + diag_i)
        li11 = th0 * (t1r11 - t2r11 + diag_r)
        lr12 = -th0 * (t1i12 - t2i12)
        li12 = th0 * (t1r12 - t2r12)
        lr20 = -th0 * (t1i20 - t2i20)
        li20 = th0 * (t1r20 - t2r20)
        lr21 = -th0 * (t1i21 - t2i21)
        li21 = th0 * (t1r21 - t2r21)
        lr22 = -th0 * (t1i22 - t2i22 + diag_i)
        li22 = th0 * (t1r22 - t2r22 + diag_r)
        # k=1: j1=0, j2=2
        S_r = x0_r + x2_r
        S_i = x0_i + x2_i
        P_r = x0_r * x2_r - x0_i * x2_i
        P_i = x0_r * x2_i + x0_i * x2_r
        d1_r = x1_r - x0_r
        d1_i = x1_i - x0_i
        d2_r = x1_r - x2_r
        d2_i = x1_i - x2_i
        de_r = d1_r * d2_r - d1_i * d2_i
        de_i = d1_r * d2_i + d1_i * d2_r
        de_mag = tl.maximum(de_r * de_r + de_i * de_i, 1e-30)
        s_r = de_r / de_mag
        s_i = -de_i / de_mag
        t1r00 = u2r00 * s_r - u2i00 * s_i
        t1i00 = u2r00 * s_i + u2i00 * s_r
        t1r01 = u2r01 * s_r - u2i01 * s_i
        t1i01 = u2r01 * s_i + u2i01 * s_r
        t1r02 = u2r02 * s_r - u2i02 * s_i
        t1i02 = u2r02 * s_i + u2i02 * s_r
        t1r10 = u2r10 * s_r - u2i10 * s_i
        t1i10 = u2r10 * s_i + u2i10 * s_r
        t1r11 = u2r11 * s_r - u2i11 * s_i
        t1i11 = u2r11 * s_i + u2i11 * s_r
        t1r12 = u2r12 * s_r - u2i12 * s_i
        t1i12 = u2r12 * s_i + u2i12 * s_r
        t1r20 = u2r20 * s_r - u2i20 * s_i
        t1i20 = u2r20 * s_i + u2i20 * s_r
        t1r21 = u2r21 * s_r - u2i21 * s_i
        t1i21 = u2r21 * s_i + u2i21 * s_r
        t1r22 = u2r22 * s_r - u2i22 * s_i
        t1i22 = u2r22 * s_i + u2i22 * s_r
        su00_r = S_r * mre00 - S_i * mim00
        su00_i = S_r * mim00 + S_i * mre00
        su01_r = S_r * mre01 - S_i * mim01
        su01_i = S_r * mim01 + S_i * mre01
        su02_r = S_r * mre02 - S_i * mim02
        su02_i = S_r * mim02 + S_i * mre02
        su10_r = S_r * mre10 - S_i * mim10
        su10_i = S_r * mim10 + S_i * mre10
        su11_r = S_r * mre11 - S_i * mim11
        su11_i = S_r * mim11 + S_i * mre11
        su12_r = S_r * mre12 - S_i * mim12
        su12_i = S_r * mim12 + S_i * mre12
        su20_r = S_r * mre20 - S_i * mim20
        su20_i = S_r * mim20 + S_i * mre20
        su21_r = S_r * mre21 - S_i * mim21
        su21_i = S_r * mim21 + S_i * mre21
        su22_r = S_r * mre22 - S_i * mim22
        su22_i = S_r * mim22 + S_i * mre22
        t2r00 = su00_r * s_r - su00_i * s_i
        t2i00 = su00_r * s_i + su00_i * s_r
        t2r01 = su01_r * s_r - su01_i * s_i
        t2i01 = su01_r * s_i + su01_i * s_r
        t2r02 = su02_r * s_r - su02_i * s_i
        t2i02 = su02_r * s_i + su02_i * s_r
        t2r10 = su10_r * s_r - su10_i * s_i
        t2i10 = su10_r * s_i + su10_i * s_r
        t2r11 = su11_r * s_r - su11_i * s_i
        t2i11 = su11_r * s_i + su11_i * s_r
        t2r12 = su12_r * s_r - su12_i * s_i
        t2i12 = su12_r * s_i + su12_i * s_r
        t2r20 = su20_r * s_r - su20_i * s_i
        t2i20 = su20_r * s_i + su20_i * s_r
        t2r21 = su21_r * s_r - su21_i * s_i
        t2i21 = su21_r * s_i + su21_i * s_r
        t2r22 = su22_r * s_r - su22_i * s_i
        t2i22 = su22_r * s_i + su22_i * s_r
        diag_r = P_r * s_r - P_i * s_i
        diag_i = P_r * s_i + P_i * s_r
        lr00 += -th1 * (t1i00 - t2i00 + diag_i)
        li00 += th1 * (t1r00 - t2r00 + diag_r)
        lr01 += -th1 * (t1i01 - t2i01)
        li01 += th1 * (t1r01 - t2r01)
        lr02 += -th1 * (t1i02 - t2i02)
        li02 += th1 * (t1r02 - t2r02)
        lr10 += -th1 * (t1i10 - t2i10)
        li10 += th1 * (t1r10 - t2r10)
        lr11 += -th1 * (t1i11 - t2i11 + diag_i)
        li11 += th1 * (t1r11 - t2r11 + diag_r)
        lr12 += -th1 * (t1i12 - t2i12)
        li12 += th1 * (t1r12 - t2r12)
        lr20 += -th1 * (t1i20 - t2i20)
        li20 += th1 * (t1r20 - t2r20)
        lr21 += -th1 * (t1i21 - t2i21)
        li21 += th1 * (t1r21 - t2r21)
        lr22 += -th1 * (t1i22 - t2i22 + diag_i)
        li22 += th1 * (t1r22 - t2r22 + diag_r)
        # k=2: j1=0, j2=1
        S_r = x0_r + x1_r
        S_i = x0_i + x1_i
        P_r = x0_r * x1_r - x0_i * x1_i
        P_i = x0_r * x1_i + x0_i * x1_r
        d1_r = x2_r - x0_r
        d1_i = x2_i - x0_i
        d2_r = x2_r - x1_r
        d2_i = x2_i - x1_i
        de_r = d1_r * d2_r - d1_i * d2_i
        de_i = d1_r * d2_i + d1_i * d2_r
        de_mag = tl.maximum(de_r * de_r + de_i * de_i, 1e-30)
        s_r = de_r / de_mag
        s_i = -de_i / de_mag
        t1r00 = u2r00 * s_r - u2i00 * s_i
        t1i00 = u2r00 * s_i + u2i00 * s_r
        t1r01 = u2r01 * s_r - u2i01 * s_i
        t1i01 = u2r01 * s_i + u2i01 * s_r
        t1r02 = u2r02 * s_r - u2i02 * s_i
        t1i02 = u2r02 * s_i + u2i02 * s_r
        t1r10 = u2r10 * s_r - u2i10 * s_i
        t1i10 = u2r10 * s_i + u2i10 * s_r
        t1r11 = u2r11 * s_r - u2i11 * s_i
        t1i11 = u2r11 * s_i + u2i11 * s_r
        t1r12 = u2r12 * s_r - u2i12 * s_i
        t1i12 = u2r12 * s_i + u2i12 * s_r
        t1r20 = u2r20 * s_r - u2i20 * s_i
        t1i20 = u2r20 * s_i + u2i20 * s_r
        t1r21 = u2r21 * s_r - u2i21 * s_i
        t1i21 = u2r21 * s_i + u2i21 * s_r
        t1r22 = u2r22 * s_r - u2i22 * s_i
        t1i22 = u2r22 * s_i + u2i22 * s_r
        su00_r = S_r * mre00 - S_i * mim00
        su00_i = S_r * mim00 + S_i * mre00
        su01_r = S_r * mre01 - S_i * mim01
        su01_i = S_r * mim01 + S_i * mre01
        su02_r = S_r * mre02 - S_i * mim02
        su02_i = S_r * mim02 + S_i * mre02
        su10_r = S_r * mre10 - S_i * mim10
        su10_i = S_r * mim10 + S_i * mre10
        su11_r = S_r * mre11 - S_i * mim11
        su11_i = S_r * mim11 + S_i * mre11
        su12_r = S_r * mre12 - S_i * mim12
        su12_i = S_r * mim12 + S_i * mre12
        su20_r = S_r * mre20 - S_i * mim20
        su20_i = S_r * mim20 + S_i * mre20
        su21_r = S_r * mre21 - S_i * mim21
        su21_i = S_r * mim21 + S_i * mre21
        su22_r = S_r * mre22 - S_i * mim22
        su22_i = S_r * mim22 + S_i * mre22
        t2r00 = su00_r * s_r - su00_i * s_i
        t2i00 = su00_r * s_i + su00_i * s_r
        t2r01 = su01_r * s_r - su01_i * s_i
        t2i01 = su01_r * s_i + su01_i * s_r
        t2r02 = su02_r * s_r - su02_i * s_i
        t2i02 = su02_r * s_i + su02_i * s_r
        t2r10 = su10_r * s_r - su10_i * s_i
        t2i10 = su10_r * s_i + su10_i * s_r
        t2r11 = su11_r * s_r - su11_i * s_i
        t2i11 = su11_r * s_i + su11_i * s_r
        t2r12 = su12_r * s_r - su12_i * s_i
        t2i12 = su12_r * s_i + su12_i * s_r
        t2r20 = su20_r * s_r - su20_i * s_i
        t2i20 = su20_r * s_i + su20_i * s_r
        t2r21 = su21_r * s_r - su21_i * s_i
        t2i21 = su21_r * s_i + su21_i * s_r
        t2r22 = su22_r * s_r - su22_i * s_i
        t2i22 = su22_r * s_i + su22_i * s_r
        diag_r = P_r * s_r - P_i * s_i
        diag_i = P_r * s_i + P_i * s_r
        lr00 += -th2 * (t1i00 - t2i00 + diag_i)
        li00 += th2 * (t1r00 - t2r00 + diag_r)
        lr01 += -th2 * (t1i01 - t2i01)
        li01 += th2 * (t1r01 - t2r01)
        lr02 += -th2 * (t1i02 - t2i02)
        li02 += th2 * (t1r02 - t2r02)
        lr10 += -th2 * (t1i10 - t2i10)
        li10 += th2 * (t1r10 - t2r10)
        lr11 += -th2 * (t1i11 - t2i11 + diag_i)
        li11 += th2 * (t1r11 - t2r11 + diag_r)
        lr12 += -th2 * (t1i12 - t2i12)
        li12 += th2 * (t1r12 - t2r12)
        lr20 += -th2 * (t1i20 - t2i20)
        li20 += th2 * (t1r20 - t2r20)
        lr21 += -th2 * (t1i21 - t2i21)
        li21 += th2 * (t1r21 - t2r21)
        lr22 += -th2 * (t1i22 - t2i22 + diag_i)
        li22 += th2 * (t1r22 - t2r22 + diag_r)
        # ---- store 9 real + 9 imag ----
        tl.store(o_ptr + base18 + 0, lr00, mask=mask)
        tl.store(o_ptr + base18 + 1, lr01, mask=mask)
        tl.store(o_ptr + base18 + 2, lr02, mask=mask)
        tl.store(o_ptr + base18 + 3, lr10, mask=mask)
        tl.store(o_ptr + base18 + 4, lr11, mask=mask)
        tl.store(o_ptr + base18 + 5, lr12, mask=mask)
        tl.store(o_ptr + base18 + 6, lr20, mask=mask)
        tl.store(o_ptr + base18 + 7, lr21, mask=mask)
        tl.store(o_ptr + base18 + 8, lr22, mask=mask)
        tl.store(o_ptr + 9 + base18 + 0, li00, mask=mask)
        tl.store(o_ptr + 9 + base18 + 1, li01, mask=mask)
        tl.store(o_ptr + 9 + base18 + 2, li02, mask=mask)
        tl.store(o_ptr + 9 + base18 + 3, li10, mask=mask)
        tl.store(o_ptr + 9 + base18 + 4, li11, mask=mask)
        tl.store(o_ptr + 9 + base18 + 5, li12, mask=mask)
        tl.store(o_ptr + 9 + base18 + 6, li20, mask=mask)
        tl.store(o_ptr + 9 + base18 + 7, li21, mask=mask)
        tl.store(o_ptr + 9 + base18 + 8, li22, mask=mask)

    def su3_matrix_log_triton(field: torch.Tensor) -> torch.Tensor:
        """GPU 3x3 complex matrix log via Triton. field: [N,3,3] complex cuda."""
        n = field.shape[0]
        fb = torch.view_as_real(field).permute(0, 3, 1, 2).reshape(n, 18).contiguous()
        ob = torch.empty_like(fb)
        grid = (triton.cdiv(n, 64),)
        _su3_log_kernel[grid](fb, ob, n, BLOCK_N=64)
        return torch.view_as_complex(ob.reshape(n, 2, 3, 3).permute(0, 2, 3, 1).contiguous())


def qfhrr_similarity(q_a: torch.Tensor, q_b: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Dispatch: Triton on CUDA when available, torch fallback otherwise."""
    if _HAS_TRITON and q_a.is_cuda:
        return qfhrr_similarity_triton(q_a, q_b, lut)
    return qfhrr_similarity_torch(q_a, q_b, lut)


# ---------------------------------------------------------------------------
# Potts/Ising translation (Extropic Directive 1 — 2606.17327 §2)
# ---------------------------------------------------------------------------

def potts_onehot_spins(q: torch.Tensor, k: int = K_PHASE) -> torch.Tensor:
    """Potts variable q in Z_K -> binary Ising spins sigma in {-1, +1}^K.

    Standard Potts->Ising embedding (Jelincic & Walker, 2606.17327 §2): each
    Z_K value is one-hot encoded, the active bin maps to +1 and all other bins
    to -1. Exactly one +1 per row. Per-row constraint:
        sum_a sigma_{i,a} = 2 - K
    """
    q = q.to(torch.long)
    onehot = torch.eye(k, dtype=torch.float32, device=q.device)[q]
    return 2.0 * onehot - 1.0


def ising_spin_constraint(spins: torch.Tensor) -> torch.Tensor:
    """Per-row spin sum of a valid one-hot Ising block (must equal 2 - K)."""
    return spins.sum(dim=-1)


class IsingHamiltonian:
    """Factorized Ising spin-glass Hamiltonian for qFHRR unbinding.

    Potts->Ising embedding (2606.17327 §2): each Z_K phase bin becomes a K-slot
    spin block with exactly one +1 (the active bin) and K-1 entries of -1.

    Coupling J = (1/D) * phi (x) phi is rank-1 and is stored FACTORIZED as the
    phase vector only — a dense [D, D] J is 34 GiB at D=65,536 and is never
    materialized. Energy:

        E(sigma) = -(1/D) * (sum_{i,a} sigma_{i,a} * phi_i)^2 - h . sigma

    Exact one-hot constraint note (2606.17327 §2): every valid configuration
    has exactly one +1 and K-1 entries of -1 per row, so each row contributes
    (2 - K) * phi_i to the coupling sum regardless of which slot is active.
    The coupling term is therefore CONSTANT over all valid configurations;
    the discriminating term is the external field h, which carries the qFHRR
    phase-difference LUT:

        h_{i,a} = h_scale * cos(theta_i - bin_center_a)

    The ground state (argmax_a h_{i,a} per row) recovers the nearest-bin
    argmax — the same unbinding result as the cosine-LUT kernel up to bin
    rounding (floor binning vs nearest bin center). The value of the
    Hamiltonian construction is the TSU digital-twin: a well-formed factor
    graph whose sampling converges to the LUT ground state on hardware-like
    thermal noise, verified by the sampler-convergence contract test.
    """

    def __init__(self, phase_vec: torch.Tensor, h_field: torch.Tensor,
                 max_sagnac_delta: float = 0.35, spins: torch.Tensor = None):
        self.phase_vec = phase_vec          # [P] flattened per-pair bin angles
        self.h_field = h_field              # [P, K] phase-difference LUT
        self.max_sagnac_delta = max_sagnac_delta
        self.D = phase_vec.numel()
        self.spins = spins                  # initial configuration [P, K]

    @property
    def dense_coupling_bytes(self) -> int:
        """Audit guard: cost of a naive dense [D, D] J (fp32)."""
        return 4 * self.D * self.D

    def energy(self, spins: torch.Tensor) -> torch.Tensor:
        """Factorized Ising energy (never forms [D, D])."""
        spins = spins.to(torch.float32)
        h_term = -(spins * self.h_field).sum()
        sdot = (spins * self.phase_vec.unsqueeze(-1)).sum()
        coupling = -(sdot * sdot) / self.D
        return coupling + h_term

    def decode(self, spins: torch.Tensor) -> torch.Tensor:
        """Argmax slot per row -> Z_K codes [P] -> [P // 4, 4] uint8."""
        codes = spins.argmax(dim=-1).to(torch.uint8)
        return codes.reshape(-1, 4)


def qfhrr_to_ising_hamiltonian(wave: torch.Tensor, h_scale: float = 1.0,
                               max_sagnac_delta: float = 0.35) -> IsingHamiltonian:
    """Extropic Directive 1 (2606.17327 §2): qFHRR wave -> Ising spin glass.

    wave: [num_blocks, 8] real Clifford wave (query).
    Returns an IsingHamiltonian whose h-field is the phase-difference LUT and
    whose initial spin configuration is the wave's own quantized codes.
    """
    codes = wave_to_phase_codes(wave)                       # [nb, 4] uint8
    re, im = wave[..., :4], wave[..., 4:]
    theta = torch.atan2(im, re)                             # [nb, 4] in [-pi, pi)
    theta_flat = theta.reshape(-1)
    bin_centers = (torch.arange(K_PHASE, dtype=torch.float32, device=wave.device)
                   + 0.5) * (2.0 * math.pi / K_PHASE) - math.pi
    h_field = h_scale * torch.cos(theta_flat.unsqueeze(-1) - bin_centers.unsqueeze(0))
    spins = potts_onehot_spins(codes.reshape(-1), k=K_PHASE)
    return IsingHamiltonian(theta_flat, h_field,
                            max_sagnac_delta=max_sagnac_delta, spins=spins)


def sample_ising_gibbs(hamiltonian: IsingHamiltonian, n_samples: int = 1,
                       temperature: float = 0.1, steps: int = 50,
                       seed: int = 0) -> dict:
    """Potts heat-bath (Glauber) sampling sweeps — TSU digital-twin kernel.

    Each sweep re-samples every row's active slot from the row Gibbs
    distribution (exact detailed balance, the standard Potts heat-bath):

        P(a | row i) ∝ exp(2 * h[i, a] / T)

    because the one-hot row constraint makes the coupling contribution
    constant, leaving only the field term. At T -> 0 the sampler returns the
    h-field ground state (argmax_a h[i, a]); at finite T it draws the exact
    Boltzmann distribution. No proposal starvation (Metropolis's 1/255
    proposal-hit-rate problem), fully vectorized (row-wise softmax over K).

    Returns {"spins": [n, P, K], "codes": [n, nb, 4] uint8, "energies": [n]}.
    """
    g = torch.Generator(device=hamiltonian.h_field.device).manual_seed(seed)
    P, K = hamiltonian.h_field.shape
    nb = P // 4
    h = hamiltonian.h_field
    # Initial configuration: the constructor's spins argument is optional;
    # when absent, start from the h-field ground state (a valid one-hot).
    current = hamiltonian.spins
    if current is None:
        q0 = h.argmax(dim=-1).to(torch.uint8)
        current = potts_onehot_spins(q0, k=K)
    current = current.clone()
    out_spins, out_codes, out_energies = [], [], []
    for _ in range(n_samples):
        for _ in range(steps):
            logits = 2.0 * h / max(temperature, 1e-9)      # [P, K]
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)           # [P, K]
            new_active = torch.multinomial(probs, 1, generator=g).squeeze(-1)  # [P]
            # Rebuild the one-hot spin rows: -1 everywhere, +1 at the new slot.
            next_rows = torch.full_like(current, -1.0)
            next_rows.scatter_(-1, new_active.unsqueeze(-1), 1.0)
            current = next_rows
        out_spins.append(current.clone())
        out_codes.append(current.argmax(dim=-1).reshape(nb, 4).to(torch.uint8))
        out_energies.append(float(hamiltonian.energy(current).item()))
    return {"spins": torch.stack(out_spins),
            "codes": torch.stack(out_codes),
            "energies": torch.tensor(out_energies)}


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# F1 carrier — adjoint su(3) Lie displacement kernel (SPEC-2026-08-28-F1-CARRIER)
# ---------------------------------------------------------------------------
# Zero-trainable, default-OFF (HENRI_F1_CARRIER=1). Per-block 8x8 real
# skew-symmetric generator family M_a in so(8), (M_a)_cb = -f_abc, built from
# the su(3) structure constants (chromodynamic_grounding Gell-Mann basis).
# R(theta) = exp(theta^a M_a) in SO(8); goal conditioning is ADDITIVE in Lie
# coordinates: theta = theta_base + lam * Pi_adj(w_task). The Hadamard coupling
# D_a <- D_a ⊙ proj_su3(W_task) is FALSIFIED (skew ⊙ skew is symmetric and
# exp diverges — measured orth error 9.1e10).
#
# Kernel contract (sealed spec §2.1): per-block 8x8 real matrix exponential via
# scaling-and-squaring Taylor (degree 6, unrolled, no list building), row
# stride multiplied (D27 rule), tl.maximum clamp on norms (D26 rule). Real-only
# path: no complex promotion. fp32 accumulation throughout.
#
# "Rodrigues exactitude" (audit doc §2.1 prose) is operationalized per the
# audit's own guardrails: unitarity < 1e-6 (guardrail 2) + fp64-reference
# agreement <= 1e-4 (spec §6.4). A generic adjoint SU(3) element has NO
# ordinary 3-D Rodrigues closed form (M_a^3 != -M_a), so the restricted
# single-generator plane is verified against the fp64 reference in the
# contract suite. Scale note: spec's s = ceil(log2(||A||)) is tightened by one
# guard bit to guarantee ||A/2^s|| <= 0.25 so the degree-6 Taylor error stays
# <= 1e-4 after squaring (verified vs fp64 reference; implementation detail of
# the sealed S&S contract, disclosed, not a gate or interface change).
#
# Interface (spec §10, immutable for F1):
#   F1LieDisplacementCarrier(num_blocks, device)
#     .compile_theta(w_task_adj, theta_base=None, lam=0.0) -> [num_blocks, 8]
#     .step(state_wave, theta) -> [num_blocks, 8]
#     .fit_adjoint(traj) -> [num_blocks, 8]   # C5 closed-form per-block theta
#   f1_expm_triton(theta, M=None) -> [num_blocks, 8, 8]   # Triton, CUDA-only
#   f1_expm_torch_ref(theta, M=None) -> [num_blocks, 8, 8]  # fp64 reference
# ---------------------------------------------------------------------------

F1_CARRIER_ENABLED = os.environ.get("HENRI_F1_CARRIER", "0") == "1"


class F1CarrierDisabledError(RuntimeError):
    """F1 carrier constructed with HENRI_F1_CARRIER unset/0 (default-OFF)."""


class F1KernelUnavailableError(RuntimeError):
    """Triton F1 kernel requested without a CUDA device (fail-closed)."""


def _f1_generators(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[8, 8, 8] real adjoint su(3) generators (M_a)_cb = -f_abc.

    f_abc from [lambda_a, lambda_b] = 2i f_abc lambda_c with
    f_abc = Tr(comm(a,b) @ lambda_c) / 4i (real-valued for Hermitian Gell-Mann
    generators). Deterministic: no RNG.
    """
    from chromodynamic_grounding import GELL_MANN_BASIS

    lam = GELL_MANN_BASIS.to(torch.complex128)          # [8, 3, 3]
    f = torch.zeros(8, 8, 8, dtype=torch.complex128)
    for a in range(8):
        for b in range(8):
            comm = lam[a] @ lam[b] - lam[b] @ lam[a]
            for c in range(8):
                f[a, b, c] = torch.trace(comm @ lam[c]) / (4j)
    # f is real-valued (Tr of skew-Hermitian is imaginary; /4i -> real).
    M = -f.real                                        # [8, 8, 8]
    return M.to(dtype)


def _f1_theta_to_M(theta: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """[num_blocks, 8] theta -> [num_blocks, 8, 8] sum_a theta^a M_a."""
    return torch.einsum("ba,aij->bij", theta, M)


def f1_expm_torch_ref(theta: torch.Tensor,
                      M: torch.Tensor | None = None) -> torch.Tensor:
    """fp64 reference per-block matrix exponential (independent of Triton).

    theta: [num_blocks, 8] float32/float64. Returns [num_blocks, 8, 8] fp64.
    """
    if M is None:
        M = _f1_generators(dtype=torch.float64)
    theta64 = theta.to(torch.float64)
    return torch.linalg.matrix_exp(_f1_theta_to_M(theta64, M))


if _HAS_TRITON:

    @triton.jit
    def _f1_mm8(A, B):
        """[8, 8] x [8, 8] -> [8, 8] register-only matmul.

        C[i, j] = sum_k A[i, k] * B[k, j]. The earlier broadcast
        expand_dims(A, 1) * expand_dims(B, 0) contracted over B's ROW axis
        (giving A[i,k] * colsum_k(B), NOT a matmul) — caught by the remote
        CUDA suite (kernel vs fp64 ref err 6.35; torch sim of the same
        algorithm matched to 1e-6). PROBE1 passed only because the identity
        is symmetric and blind to transposition.
        """
        prod = tl.expand_dims(A, 2) * tl.expand_dims(B, 0)   # [8,8,1] * [1,8,8] = [i,k,j]
        return tl.sum(prod, axis=1)                          # [i, j]

    @triton.jit
    def _f1_expm_kernel(theta_ptr, M_ptr, R_ptr, nb):
        """Per-block 8x8 real matrix exponential (scaling-and-squaring Taylor).

        theta: [nb, 8] float32; M: [8, 8, 8] float32; R: [nb, 8, 8] float32.
        One program per block. A = sum_a theta[a] M_a; scale s so that
        ||A/2^s||_F <= 0.25 (spec ceil(log2 ||A||) + one guard bit); Taylor
        degree 6 (Horner); square s times (guarded static unroll, MAX_S = 16).
        Row-major strides: theta row stride 8, R row stride 64 (D27 rule).
        """
        offs = tl.program_id(0)
        if offs >= nb:
            return
        offs_i = tl.arange(0, 8)
        offs_j = tl.arange(0, 8)
        # ---- build A = sum_a th[a] * M_a (M stride 64 per generator) ----
        # Scalar loads inside static_range: Triton does NOT support indexing a
        # tensor by a constexpr scalar (`th[a]` -> "unsupported tensor index"),
        # so load each theta component as a 0-d scalar (same pattern as the
        # existing _su3_log_kernel scalar loads).
        A = tl.zeros([8, 8], dtype=tl.float32)
        for a in tl.static_range(8):
            th_a = tl.load(theta_ptr + offs * 8 + a)
            Ma = tl.load(M_ptr + a * 64 + offs_i[:, None] * 8 + offs_j[None, :])
            A = A + th_a * Ma
        # ---- scaling-and-squaring ----
        nrm = tl.sqrt(tl.maximum(tl.sum(A * A), 0.0))        # F-norm (D26 clamp)
        s_f = tl.ceil(tl.log2(tl.maximum(nrm, 1e-30)))
        # guard bit: guarantee ||A * 2^-s|| <= 0.25
        s_f = s_f + 1.0
        scale0 = tl.exp2(-s_f)
        s_f = s_f + tl.where(nrm * scale0 > 0.25, 1.0, 0.0)
        s_i = s_f.to(tl.int32)
        scale = tl.exp2(-s_f)
        As = A * scale
        # ---- Taylor degree 6 (Horner, unrolled) ----
        I8 = (offs_i[:, None] == offs_j[None, :]).to(tl.float32)
        T = I8
        P = I8
        for k in tl.static_range(1, 7):
            P = _f1_mm8(P, As) * (1.0 / k)
            T = T + P
        # ---- squaring (guarded static unroll; MAX_S = 16 covers ||A|| < 2^15) ----
        for i in tl.static_range(16):
            T = tl.where(i < s_i, _f1_mm8(T, T), T)
        # ---- store R [nb, 8, 8] (row stride 64) ----
        tl.store(R_ptr + offs * 64 + offs_i[:, None] * 8 + offs_j[None, :], T)


def f1_expm_triton(theta: torch.Tensor,
                   M: torch.Tensor | None = None) -> torch.Tensor:
    """[num_blocks, 8] -> [num_blocks, 8, 8] via the Triton kernel.

    CUDA-only; raises F1KernelUnavailableError without CUDA (fail-closed, so a
    CPU or torch fallback is never silently labelled TRITON).
    """
    if not _HAS_TRITON or not torch.cuda.is_available():
        raise F1KernelUnavailableError(
            "f1_expm_triton requires CUDA + Triton (fail-closed; "
            "BLOCKED_VACUOUS_CPU_PATH if used as a gate)")
    theta = theta.to(torch.float32).contiguous()
    if M is None:
        M = _f1_generators(dtype=torch.float32)
    M = M.to(theta.device).contiguous()
    nb = theta.shape[0]
    R = torch.empty((nb, 8, 8), dtype=torch.float32, device=theta.device)
    _f1_expm_kernel[(nb,)](theta, M, R, nb)
    return R


class F1LieDisplacementCarrier:
    """Zero-trainable adjoint su(3) displacement carrier (spec §10).

    Default-OFF: construction requires HENRI_F1_CARRIER=1 (checked live in
    __init__; the module constant is informational only). theta = 0 => exact
    identity => byte-identical output (spec invariant 4). No trainable
    parameters: theta is compiled closed-form (compile_theta / fit_adjoint).
    """

    def __init__(self, num_blocks: int, device: torch.device):
        if os.environ.get("HENRI_F1_CARRIER", "0") != "1":
            raise F1CarrierDisabledError(
                "HENRI_F1_CARRIER must be 1 to construct F1LieDisplacementCarrier")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = int(num_blocks)
        self.device = torch.device(device)
        self._M32 = _f1_generators(dtype=torch.float32)   # [8, 8, 8] cpu
        self._M64 = _f1_generators(dtype=torch.float64)   # [8, 8, 8] cpu
        self._impl = "TORCH_REF"

    @property
    def impl(self) -> str:
        """Implementation marker for the last step(): TRITON | TORCH_REF."""
        return self._impl

    def _check_theta(self, theta: torch.Tensor, name: str) -> torch.Tensor:
        if theta.shape != (self.num_blocks, 8):
            raise ValueError(
                f"{name} must be [{self.num_blocks}, 8], got {tuple(theta.shape)}")
        if theta.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must be float32/float64, got {theta.dtype}")
        if not bool(torch.isfinite(theta).all()):
            raise ValueError(f"{name} contains NaN/Inf")
        return theta.to(torch.float32)

    def compile_theta(self, w_task_adj: torch.Tensor | None = None,
                      theta_base: torch.Tensor | None = None,
                      lam: float = 0.0) -> torch.Tensor:
        """theta = theta_base + lam * w_task_adj (additive Lie coordinates).

        Returns [num_blocks, 8] float32 on self.device. Default (both None):
        zeros -> identity arm (byte-identical baseline).
        """
        out = torch.zeros(self.num_blocks, 8, device=self.device)
        if theta_base is not None:
            out = out + self._check_theta(theta_base, "theta_base").to(self.device)
        if w_task_adj is not None:
            out = out + lam * self._check_theta(w_task_adj, "w_task_adj").to(self.device)
        return out

    def step(self, state_wave: torch.Tensor,
             theta: torch.Tensor) -> torch.Tensor:
        """Psi' = exp(theta^a M_a) Psi per block. Returns [num_blocks, 8]."""
        if state_wave.shape != (self.num_blocks, 8):
            raise ValueError(
                f"state_wave must be [{self.num_blocks}, 8], "
                f"got {tuple(state_wave.shape)}")
        if state_wave.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"state_wave must be float32/float64, got {state_wave.dtype}")
        theta = self._check_theta(theta, "theta")
        psi = state_wave.to(torch.float32)
        # Identity arm: byte-identical baseline (spec invariant 4).
        if bool((theta == 0.0).all()):
            self._impl = "IDENTITY"
            return psi
        if (torch.cuda.is_available() and _HAS_TRITON
                and psi.device.type == "cuda"):
            R = f1_expm_triton(theta, self._M32.to(psi.device))
            self._impl = "TRITON"
        else:
            R = f1_expm_torch_ref(theta, self._M64).to(psi.device).float()
            self._impl = "TORCH_REF"
        return torch.einsum("bij,bj->bi", R, psi)

    def fit_adjoint(self, traj: torch.Tensor) -> torch.Tensor:
        """Closed-form per-block theta from observed transitions (C5).

        traj: [T, num_blocks, 8] float32/float64. Per block: Procrustes fit
        R = U V^T from W = Y^T X (SVD), matrix log via eigendecomposition,
        projection onto span{M_a} via the Gram matrix. Returns
        [num_blocks, 8] float32.
        """
        if traj.dim() != 3 or traj.shape[1] != self.num_blocks or traj.shape[2] != 8:
            raise ValueError(f"traj must be [T, {self.num_blocks}, 8], "
                             f"got {tuple(traj.shape)}")
        if traj.shape[0] < 3:
            raise ValueError("traj needs T >= 3 for a Procrustes fit")
        t = traj.to(torch.float64)                          # [T, nb, 8]
        X = t[:-1]                                          # [T-1, nb, 8]
        Y = t[1:]                                           # [T-1, nb, 8]
        W = torch.einsum("tnk,tni->nki", Y, X)              # [nb, 8, 8] = Y^T X
        U, _, Vh = torch.linalg.svd(W)                      # [nb, 8, 8]
        R = U @ Vh                                          # Procrustes rotor
        ev, V = torch.linalg.eig(R)                         # right eigenvectors
        L = (V @ torch.diag_embed(torch.log(ev)) @ V.inverse()).real
        M = self._M64
        G = torch.einsum("aij,bij->ab", M, M)               # [8, 8] Gram
        b = torch.einsum("aij,nij->na", M, L)               # [nb, 8]
        theta = torch.linalg.solve(G, b.t()).t()            # [nb, 8]
        return theta.float()

