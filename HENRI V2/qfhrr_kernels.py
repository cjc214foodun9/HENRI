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


def qfhrr_similarity(q_a: torch.Tensor, q_b: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Dispatch: Triton on CUDA when available, torch fallback otherwise."""
    if _HAS_TRITON and q_a.is_cuda:
        return qfhrr_similarity_triton(q_a, q_b, lut)
    return qfhrr_similarity_torch(q_a, q_b, lut)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nb = 8192 if device == "cuda" else 64
    lut = build_cos_lut(device)

    g = torch.Generator().manual_seed(1)
    w = torch.randn(nb, 8, generator=g).to(device)
    w = w / torch.norm(w, p=2, dim=-1, keepdim=True)
    err = quantization_roundtrip_error(w)
    print(f"device={device} blocks={nb} roundtrip_phase_err={err:.5f} rad (bound < 0.0123)")
    assert err < 0.0123

    qa = wave_to_phase_codes(w).view(-1)
    # Self-similarity must be 1.0
    s_self = qfhrr_similarity(qa, qa, lut)
    # Random-other similarity must be near 0
    w2 = torch.randn(nb, 8, generator=g).to(device)
    w2 = w2 / torch.norm(w2, p=2, dim=-1, keepdim=True)
    qb = wave_to_phase_codes(w2).view(-1)
    s_rand = qfhrr_similarity(qa, qb, lut)
    print(f"sim(self)={float(s_self):.4f}  sim(random)={float(s_rand):.4f}")
    assert abs(float(s_self) - 1.0) < 1e-3
    assert abs(float(s_rand)) < 0.1

    # Batch path: M=8 engrams
    store = torch.stack([wave_to_phase_codes(
        (lambda v: v / torch.norm(v, p=2, dim=-1, keepdim=True))(
            torch.randn(nb, 8, generator=g).to(device))).view(-1)
        for _ in range(8)])
    sims = qfhrr_similarity(qa, store, lut)
    print(f"batch sims [M=8]: {[round(float(s), 3) for s in sims]}")
    assert sims.shape == (8,)

    # Triton-vs-torch agreement on GPU
    if device == "cuda":
        s_t = qfhrr_similarity_torch(qa, store, lut)
        s_g = qfhrr_similarity_triton(qa, store, lut)
        max_dev = (s_t - s_g).abs().max().item()
        print(f"triton-vs-torch max dev: {max_dev:.2e}")
        assert max_dev < 1e-4
    print("qFHRR kernel smoke test PASSED")
