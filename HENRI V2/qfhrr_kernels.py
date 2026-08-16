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
        # ---- load 9 real + 9 imag ----
        mre = [tl.load(a_ptr + base + i * 3 + j, mask=mask, other=0.0)
               for i in tl.static_range(3) for j in tl.static_range(3)]
        mim = [tl.load(a_ptr + 9 + base + i * 3 + j, mask=mask, other=0.0)
               for i in tl.static_range(3) for j in tl.static_range(3)]
        # ---- trace tr = sum diag ----
        tr_r = mre[0] + mre[4] + mre[8]
        tr_i = mim[0] + mim[4] + mim[8]
        # ---- cubic x^3 + a x^2 + b x + c: a = -tr, b = conj(tr), c = -1 ----
        a_r = -tr_r
        a_i = -tr_i
        b_r = tr_r
        b_i = -tr_i
        c_r = tl.full([BLOCK_N], -1.0, dtype=tl.float32)
        c_i = tl.zeros([BLOCK_N], dtype=tl.float32)
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
        q_i = 2.0 * a3_i / 27.0 - ab_i / 3.0 + c_i
        # D = (q/2)^2 + (p/3)^3
        q2_r = q_r / 2.0
        q2_i = q_i / 2.0
        p3_r = p_r / 3.0
        p3_i = p_i / 3.0
        p3c_r = p3_r * p3_r * p3_r - 3.0 * p3_r * p3_i * p3_i
        p3c_i = 3.0 * p3_r * p3_r * p3_i - p3_i * p3_i * p3_i
        D_r = q2_r * q2_r - q2_i * q2_i + p3c_r
        D_i = 2.0 * q2_r * q2_i + p3c_i
        # sqrt(D)
        D_mag = tl.sqrt(D_r * D_r + D_i * D_i)
        sD_r = tl.sqrt((D_mag + D_r) / 2.0)
        sD_i = tl.where(D_i >= 0.0, 1.0, -1.0) * tl.sqrt((D_mag - D_r) / 2.0)
        # u^3 = -q/2 + sqrt(D); cube root (3 branches)
        u3_r = -q2_r + sD_r
        u3_i = -q2_i + sD_i
        u3_mag = tl.sqrt(u3_r * u3_r + u3_i * u3_i)
        u3_ang = tl.math.atan2(u3_i, u3_r)
        u_mag = tl.exp2(tl.log2(tl.maximum(u3_mag, 1e-30)) * (1.0 / 3.0))
        # ---- 3 roots via per-branch pairing v = -p/(3u) ----
        xs_re = []
        xs_im = []
        for k in tl.static_range(3):
            ang = (u3_ang + 2.0 * 3.141592653589793 * k) / 3.0
            u_r = u_mag * tl.cos(ang)
            u_i = u_mag * tl.sin(ang)
            den = 3.0 * (u_r * u_r + u_i * u_i)
            v_r = (-p_r * u_r - p_i * u_i) / den
            v_i = (p_r * u_i - p_i * u_r) / den
            y_r = u_r + v_r
            y_i = u_i + v_i
            xs_re.append(y_r - a_r / 3.0)
            xs_im.append(y_i - a_i / 3.0)
        # ---- Lagrange projectors: log U = sum_k i theta_k P_k ----
        log_re = [tl.zeros([BLOCK_N], dtype=tl.float32) for _ in tl.static_range(9)]
        log_im = [tl.zeros([BLOCK_N], dtype=tl.float32) for _ in tl.static_range(9)]
        for k in tl.static_range(3):
            th = tl.math.atan2(xs_im[k], xs_re[k])
            P_re = [tl.where(mask, 1.0, 0.0) if (i == j) else tl.zeros([BLOCK_N], dtype=tl.float32)
                    for i in tl.static_range(3) for j in tl.static_range(3)]
            P_im = [tl.zeros([BLOCK_N], dtype=tl.float32) for _ in tl.static_range(9)]
            for jj in tl.static_range(3):
                if jj == k:
                    continue
                lj_r = xs_re[jj]
                lj_i = xs_im[jj]
                # M = U - l_j I
                M_re = [(mre[i * 3 + j] - (lj_r if i == j else 0.0))
                        for i in tl.static_range(3) for j in tl.static_range(3)]
                M_im = [(mim[i * 3 + j] - (lj_i if i == j else 0.0))
                        for i in tl.static_range(3) for j in tl.static_range(3)]
                # P = P @ M (complex 3x3, unrolled)
                N_re = []
                N_im = []
                for i in tl.static_range(3):
                    for j in tl.static_range(3):
                        s_re = tl.zeros([BLOCK_N], dtype=tl.float32)
                        s_im = tl.zeros([BLOCK_N], dtype=tl.float32)
                        for kk in tl.static_range(3):
                            s_re += P_re[i * 3 + kk] * M_re[kk * 3 + j] - P_im[i * 3 + kk] * M_im[kk * 3 + j]
                            s_im += P_re[i * 3 + kk] * M_im[kk * 3 + j] + P_im[i * 3 + kk] * M_re[kk * 3 + j]
                        N_re.append(s_re)
                        N_im.append(s_im)
                P_re = N_re
                P_im = N_im
                # divide by (l_k - l_j) complex scalar
                d_r = xs_re[k] - lj_r
                d_i = xs_im[k] - lj_i
                d_mag = d_r * d_r + d_i * d_i
                for i in tl.static_range(3):
                    for j in tl.static_range(3):
                        idx = i * 3 + j
                        P_re[idx] = (P_re[idx] * d_r + P_im[idx] * d_i) / d_mag
                        P_im[idx] = (P_im[idx] * d_r - P_re[idx] * d_i) / d_mag
            # accumulate i*theta*P: re = -th*P_im, im = th*P_re
            for i in tl.static_range(3):
                for j in tl.static_range(3):
                    idx = i * 3 + j
                    log_re[idx] += -th * P_im[idx]
                    log_im[idx] += th * P_re[idx]
        for i in tl.static_range(3):
            for j in tl.static_range(3):
                idx = i * 3 + j
                tl.store(o_ptr + base + idx, log_re[idx], mask=mask)
                tl.store(o_ptr + 9 + base + idx, log_im[idx], mask=mask)

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
# Smoke
# ---------------------------------------------------------------------------

