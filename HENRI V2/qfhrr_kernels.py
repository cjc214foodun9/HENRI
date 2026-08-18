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
# Flat phase-ring codec (complex sidecar family, default-OFF)
# ---------------------------------------------------------------------------
# Phase 8.28 scaffold-compatible quantize/dequantize for FLAT complex wave
# vectors [..., D]. This is the "third representation family" sidecar
# (complex flat Z_256 rings). It is NOT wired into any action policy;
# consumers must opt in explicitly (henri-architecture sidecar boundary).

def quantize_phase_flat(wave: torch.Tensor, K: int = K_PHASE) -> torch.Tensor:
    """Complex [..., D] wave -> uint8/int32 Z_K phase ring (scaffold form).

    Complex input: theta = angle(wave). Real input is treated as raw phase
    angles in [-pi, pi) (scaffold convention; do NOT feed real Clifford
    [num_blocks, 8] waves here — use wave_to_phase_codes instead).
    """
    angles = torch.angle(wave) if wave.is_complex() else wave
    q = torch.floor((angles + math.pi) * (K / (2.0 * math.pi))).to(torch.int64) % K
    if wave.device.type == "cuda":
        return q.to(torch.uint8)
    return q.to(torch.int32)


def dequantize_phase_flat(q_ring: torch.Tensor, K: int = K_PHASE,
                          dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """Z_K ring -> continuous complex unit-modulus wave (scaffold form)."""
    angles = q_ring.to(torch.float32) * (2.0 * math.pi / K) - math.pi
    return torch.complex(torch.cos(angles), torch.sin(angles)).to(dtype)


# ---------------------------------------------------------------------------
# Batch swarm unbinding: Q [B, D] x C [M, D] -> Sim [B, M]
# ---------------------------------------------------------------------------
# Phase 8.28: fused Triton kernel (one program per (b, m) pair, register
# accumulation over D, no [B, M, D] intermediate) plus a memory-safe chunked
# torch fallback. Same INT32 LUT_SCALE convention as qfhrr_similarity
# (LUT int32 scaled by 127; host divides by LUT_SCALE * D). The mask
# `& 255` is valid only for K_PHASE == 256.

def _as_flat_codes(x: torch.Tensor) -> torch.Tensor:
    """Accept flat [D] or block [num_blocks, 4] / [B, num_blocks, 4] uint8
    codes; return contiguous [1, D] (dim-1) or [B, D] (dim>=2) rows.
    The trailing dims are fully flattened, so both code families share the
    same flat-D contract as the scalar similarity path."""
    return x.contiguous().view(x.shape[0], -1) if x.dim() >= 2 else x.contiguous().view(1, -1)


if _HAS_TRITON:

    @triton.jit
    def _qfhrr_batch_sim_kernel(
        Q_ptr, C_ptr, LUT_ptr, Out_ptr,
        stride_qb, stride_qd,
        stride_cm, stride_cd,
        stride_ob, stride_om,
        D: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Batch unbinding: one program per (b, m); register-accumulate over D.

        acc is INT32: max |acc| = LUT_SCALE * D = 127 * 131072 ~ 16.6M, well
        inside int32. Host divides by LUT_SCALE * D. No atomics, no
        [B, M, D] intermediate.
        """
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        q_ptr = Q_ptr + pid_b * stride_qb
        c_ptr = C_ptr + pid_m * stride_cm
        acc = tl.zeros((), dtype=tl.int32)
        for d_off in range(0, D, BLOCK_D):
            cols = d_off + tl.arange(0, BLOCK_D)
            mask = cols < D
            q_val = tl.load(q_ptr + cols * stride_qd, mask=mask, other=0).to(tl.int16)
            c_val = tl.load(c_ptr + cols * stride_cd, mask=mask, other=0).to(tl.int16)
            diff = (q_val - c_val) & 255
            cos_val = tl.load(LUT_ptr + diff, mask=mask, other=0)
            acc += tl.sum(cos_val, axis=0)
        tl.store(Out_ptr + pid_b * stride_ob + pid_m * stride_om, acc)

    def qfhrr_batch_similarity_triton(Q: torch.Tensor, C: torch.Tensor,
                                      lut: torch.Tensor,
                                      max_programs: int = 4096) -> torch.Tensor:
        """Fused Triton batch unbinding. Q: [B, D] uint8 cuda; C: [M, D]
        uint8 cuda; lut: [256] int32 cuda. Returns [B, M] float32 in ~[-1, 1].
        Grid is bounded by chunking M (one program per (b, m) pair)."""
        B, D = Q.shape
        M, Dc = C.shape
        assert D == Dc, f"Dimension mismatch: Q={D}, C={Dc}"
        Out = torch.empty((B, M), device=Q.device, dtype=torch.float32)
        BLOCK_D = 2048 if D >= 2048 else 512
        m_chunk = max(1, min(M, max_programs // max(1, B)))
        for m0 in range(0, M, m_chunk):
            m1 = min(m0 + m_chunk, M)
            _qfhrr_batch_sim_kernel[(B, m1 - m0)](
                Q, C[m0:m1], lut, Out[:, m0:m1],
                Q.stride(0), Q.stride(1),
                C.stride(0), C.stride(1),
                Out.stride(0), Out.stride(1),
                D=D, BLOCK_D=BLOCK_D,
            )
        return Out / (LUT_SCALE * D)


def _pytorch_batch_similarity_fallback(Q: torch.Tensor, C: torch.Tensor,
                                       lut: torch.Tensor) -> torch.Tensor:
    """Chunked pure-torch batch unbinding. Memory-safe for large B*M*D:
    intermediates bounded to ~64M elements (~256 MB int32)."""
    B, D = Q.shape
    M, _ = C.shape
    out = torch.empty((B, M), device=Q.device, dtype=torch.float32)
    max_elems = 64 * 1024 * 1024
    chunk_b = max(1, max_elems // max(1, M * D))
    q_int = Q.to(torch.int32)
    c_int = C.to(torch.int32)
    for b0 in range(0, B, chunk_b):
        b1 = min(b0 + chunk_b, B)
        diff = (q_int[b0:b1].unsqueeze(1) - c_int.unsqueeze(0)) & 255  # [bc, M, D]
        cos_vals = lut[diff.long()].to(torch.int64).sum(dim=-1)          # [bc, M]
        out[b0:b1] = cos_vals.to(torch.float32) / (LUT_SCALE * D)
    return out


def qfhrr_batch_similarity(
    Q: torch.Tensor,
    C: torch.Tensor,
    lut: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    max_programs: int = 4096,
) -> torch.Tensor:
    """Pairwise quantized phase-unbinding cosine similarity -> [B, M] ~ [-1, 1].

    Q: [B, D] or [D] uint8 rings (flat, or [num_blocks, 4] block codes).
    C: [M, D] or [D] uint8 rings (same families). D must match.
    Returns [B, M]; squeezes singleton dims (scaffold-compatible).
    """
    q_batched = Q.dim() >= 3 or (Q.dim() == 2 and Q.shape[1] != 4)
    c_batched = C.dim() >= 3 or (C.dim() == 2 and C.shape[1] != 4)
    Qf = _as_flat_codes(Q)
    Cf = _as_flat_codes(C)
    if Qf.shape[1] != Cf.shape[1]:
        raise ValueError(f"Dimension mismatch: Q={Qf.shape[1]}, C={Cf.shape[1]}")
    if lut is None:
        lut = build_cos_lut(Qf.device)
    if use_triton and _HAS_TRITON and Qf.is_cuda and Cf.is_cuda:
        try:
            res = qfhrr_batch_similarity_triton(Qf, Cf, lut, max_programs=max_programs)
        except Exception:
            res = _pytorch_batch_similarity_fallback(Qf, Cf, lut)
    else:
        res = _pytorch_batch_similarity_fallback(Qf, Cf, lut)
    if not q_batched and not c_batched:
        return res.squeeze(0).squeeze(0)
    if not q_batched:
        return res.squeeze(0)
    if not c_batched:
        return res.squeeze(1)
    return res


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.randn(64, 8) / torch.norm(torch.randn(64, 8), dim=-1, keepdim=True)
    q = wave_to_phase_codes(w).contiguous().view(-1)
    r = qfhrr_batch_similarity(q, q)
    assert abs(float(r) - 1.0) < 1e-4, f"self-similarity {float(r)}"
    Q = wave_to_phase_codes(torch.randn(2, 64, 8))
    C = wave_to_phase_codes(torch.randn(3, 64, 8))
    S = qfhrr_batch_similarity(Q, C)
    assert S.shape == (2, 3)
    print(f"qfhrr batch smoke OK: self={float(r):.6f}, batch={S.shape}")

