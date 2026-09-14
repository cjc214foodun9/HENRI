"""Group-structured (torus) ingress encoder + per-slot diagonal task operator.

WHY THIS MODULE EXISTS -- OBSERVED_GPU 2026-09-14
-------------------------------------------------
Source of record: experiments/verification/arc_encoder_reform_gate_v4_observed.json
(60 real ARC-AGI-1 training tasks, RTX PRO 6000 WS sm_120, torch 2.12.0+cu130,
triton 3.7.1).

The incumbent `O_VSA_IngressTokenizer.encode_spatial_grid` builds position
codes as a FRACTION of a random full-range phase:

    total_phase = theta_v + norm_x*spatial_theta_x + norm_y*spatial_theta_y
    norm_x = (2x/(W-1)) - 1

Measured consequences (ARM G kill gate, synthetic cyclic roll of 3 columns):

    encoder  operator            structured   noise    identity
    LIVE     ls_diag              0.9311     0.5758    0.8592   <- FAILS the gate
    TORUS    ls_diag              1.0000    -0.0051   -0.0399   <- PASSES

The LIVE `identity` of 0.8592 is the decisive diagnostic: its waves are so
dominated by a shared carrier that a grid and its own roll are nearly identical
BEFORE any operator is applied, and pure noise still scores 0.5758. The encoder
cannot separate structure from noise.

THE REFORM. Quantize the position frequencies to the canvas modulus S:

    wx[b,s] = 2*pi*k[b,s]/S,  k integer
    Psi_X   = sum_{x,y} phasor(v(X[x,y])) * exp(i*(x*wx + y*wy))

A cyclic grid roll by (dw, dh) is then an EXACT wave operator:

    enc(roll(X, dw, dh)) = M * enc(X),   M = exp(-i*(dw*wx + dh*wy))

Measured analytic error at S=32 (ARM P): 2.4e-05 .. 3.0e-05 for d in {1,3,5}.
Because the multiplier is INDEPENDENT of the grid contents and dimensions, a
grid-space transform IS a wave-space operator -- which is the property the
incumbent fractional code does not have, and which the whole VLA premise needs.

DELIBERATELY NOT IMPLEMENTED: the FFT/circulant operator.
    A separate hypothesis proposed W = mean(fft(Y)*conj(fft(X))). It is
    FALSIFIED for the production [num_blocks, 8] layout: it circulates the BLOCK
    axis, whereas a spatial transform lives in the per-(block, slot) phase.
    Measured 0.00 on both the synthetic roll and on 60 real ARC tasks, versus
    0.93-1.00 for the diagonal family. On a FLAT wave the same FFT form scores
    1.0000 -- that earlier number was a different representation and does not
    transfer. Do not re-propose it for this layout.

OPERATOR FAMILY. The correct family for [num_blocks, 8] is a per-slot DIAGONAL
operator with the least-squares optimum

    W* = sum_p conj(x_p)*y_p / sum_p |x_p|^2

which is NOT what `arc_task_functor.py` computes. That module uses
mean(conj(x)*y), which equals W* only when |x| is constant. Since the encoder
L2-normalises per block, |x| is strongly non-flat. Both are provided here;
`compile_task_operator_ls` is the ceiling and is the recommended consumer.

DEFAULT OFF. Nothing in this module runs unless HENRI_ENCODER_TORUS=1. The
tokenizer's default path is untouched and byte-identical.

HONEST LIMITS
-------------
1. The reform is validated as an OPERATOR ALGEBRA property (ARM P) and as a
   discrimination property (ARM G). It is NOT yet an external task win.
2. On 60 real ARC tasks the in-sample CEILING is 0.76-0.82 and the held-out
   score is 0.43-0.60. The family is sufficient (ceiling beats identity), so the
   remaining gap is FEW-SHOT FITTING, not operator capacity. That is the next
   carrier and it is NOT solved here.
3. Colour (value) codes are seeded-random by default; mode="TORUS_VAL" instead
   makes phasor(v) = (v+1)*w_v, so a colour remap is also an exact operator.
   Measured difference on real ARC held-out is small (0.4285 vs 0.4368) and is
   within the run's variance -- it is not established as an improvement.
4. S=32 and D=8192 blocks are the measured configuration. Other values are
   untested.
"""
from __future__ import annotations

import math
import os
import torch

DEFAULT_MODULUS = 32
DEFAULT_SEED = 20260914
BLOCK_SLOTS = 4          # complex slots per block; 8 real = 4 complex


def _phasor(angle: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.cos(angle), torch.sin(angle))


class TorusIngressEncoder:
    """Deterministic group-structured encoder over a Z_S x Z_S position canvas.

    Args:
        num_blocks: spatial block count (production 8192).
        vocab_size: value-code cardinality (>= max colour id + 1).
        modulus:      canvas modulus S for the quantized position frequency.
        mode:         "TORUS"     seeded-random value codebook
                      "TORUS_VAL" group value codebook, phasor(v)=(v+1)*w_v
        seed:         RNG seed. FIXED, because frozen priors and reproducible
                      operator provenance require identical bytes across
                      processes. (The incumbent tokenizer uses UNSEEDED
                      torch.rand/randn, so two instances disagree.)
    """

    def __init__(self, num_blocks: int = 8192, vocab_size: int = 256,
                 modulus: int = DEFAULT_MODULUS, mode: str = "TORUS_VAL",
                 seed: int = DEFAULT_SEED, device="cpu"):
        if mode not in ("TORUS", "TORUS_VAL"):
            raise ValueError(f"mode must be TORUS or TORUS_VAL, got {mode!r}")
        self.num_blocks = int(num_blocks)
        self.vocab_size = int(vocab_size)
        self.modulus = int(modulus)
        self.mode = mode
        self.seed = int(seed)
        self.device = torch.device(device)

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        if mode == "TORUS":
            self.value_phase = (torch.rand(self.vocab_size, self.num_blocks,
                                           BLOCK_SLOTS, generator=g)
                                * 2.0 * math.pi).to(self.device)
        else:
            w_v = (torch.rand(self.num_blocks, BLOCK_SLOTS, generator=g)
                   * 2.0 * math.pi).to(self.device)
            v = torch.arange(self.vocab_size, dtype=torch.float32, device=self.device) + 1.0
            self.value_phase = (v[:, None, None] * w_v[None]).contiguous()

        # Quantized position frequencies: k integer in [1, S), w = 2*pi*k/S.
        kx = torch.randint(1, self.modulus, (self.num_blocks, BLOCK_SLOTS),
                           generator=g).float().to(self.device)
        ky = torch.randint(1, self.modulus, (self.num_blocks, BLOCK_SLOTS),
                           generator=g).float().to(self.device)
        self.wx = 2.0 * math.pi * kx / float(self.modulus)
        self.wy = 2.0 * math.pi * ky / float(self.modulus)

    # ------------------------------------------------------------------ encode
    def encode(self, grid, chunk: int = 128) -> torch.Tensor:
        """Grid -> real [num_blocks, 8], per-block L2 unit norm."""
        rows = [list(map(int, r)) for r in grid]
        H = len(rows)
        W = len(rows[0]) if H else 1
        vals, xs, ys = [], [], []
        for y in range(H):
            row = rows[y]
            for x in range(W):
                vals.append(min(row[x], self.vocab_size - 1))
                xs.append(x)
                ys.append(y)
        if not vals:
            raise ValueError("empty grid")

        vi = torch.tensor(vals, dtype=torch.long, device=self.device)
        X = torch.tensor(xs, dtype=torch.float32, device=self.device)[:, None, None]
        Y = torch.tensor(ys, dtype=torch.float32, device=self.device)[:, None, None]

        acc = torch.zeros(self.num_blocks, BLOCK_SLOTS, dtype=torch.complex64,
                          device=self.device)
        for i in range(0, len(vals), chunk):
            ang = (self.value_phase[vi[i:i + chunk]]
                   + X[i:i + chunk] * self.wx[None]
                   + Y[i:i + chunk] * self.wy[None])
            acc = acc + _phasor(ang).sum(dim=0)
        return self._to_real(acc)

    def encode_canvas(self, grid, chunk: int = 128) -> torch.Tensor:
        """Encode a FULL-CANVAS grid: dimensions MUST equal (modulus, modulus).

        The exact roll-operator property is CONDITIONAL and this method exists to
        make the condition explicit and checkable:

            enc(roll(X, dw, dh)) == M * enc(X)   holds IFF the grid fills the
            canvas, i.e. H == W == modulus.

        WHY (this is the constraint my first wiring test tripped):
        A roll within a width-W grid is a Z_W action, not a Z_S action. Writing
        the two cases out:
            x >= d : element moves to x-d   -> multiplier  exp(-i*d*w)
            x <  d : element wraps to x-d+W -> multiplier  exp(+i*(W-d)*w)
        These agree only when exp(-i*d*w) == exp(+i*(W-d)*w), i.e. when
        w*W = 2*pi*n -- which holds exactly when W == S with w = 2*pi*k/S.
        Measured consequence at S=32 with a 12-wide grid: max error 1.16 (FAIL);
        with a 32-wide canvas: max error 2.4e-05 (PASS). Same encoder, same code.
        """
        rows = [list(map(int, r)) for r in grid]
        H = len(rows)
        W = len(rows[0]) if H else 0
        if H != self.modulus or W != self.modulus:
            raise ValueError(
                f"encode_canvas requires a full {self.modulus}x{self.modulus} grid, "
                f"got {H}x{W}. Rolling a smaller grid is a Z_{W} action and is NOT "
                f"an exact operator under a Z_{self.modulus} position code.")
        return self.encode(rows, chunk=chunk)

    @staticmethod
    def roll_canvas(grid, dw: int, dh: int = 0):
        """Cyclic roll of a canvas-sized grid (the action the property covers)."""
        out = [list(r) for r in grid]
        if dh:
            out = out[dh:] + out[:dh]
        return [r[dw:] + r[:dw] for r in out]

    @staticmethod
    def _to_real(z: torch.Tensor) -> torch.Tensor:
        out = torch.stack([z.real, z.imag], dim=-1).reshape(z.shape[0], 8)
        return out / (out.norm(p=2, dim=-1, keepdim=True) + 1e-9)

    @staticmethod
    def _to_complex(w: torch.Tensor) -> torch.Tensor:
        v = w.reshape(-1, BLOCK_SLOTS, 2)
        return torch.complex(v[..., 0], v[..., 1])

    # -------------------------------------------------- exact roll operator
    def roll_multiplier(self, dw: int, dh: int = 0) -> torch.Tensor:
        """EXACT wave operator M for a cyclic grid roll by (dw, dh).

        Property (measured err 2.4e-05 at S=32): encode(roll(X, dw, dh)) == M*encode(X).
        """
        a = -(float(dw) * self.wx + float(dh) * self.wy)
        return _phasor(a)

    def apply_roll(self, wave: torch.Tensor, dw: int, dh: int = 0) -> torch.Tensor:
        """Apply the roll operator to an encoded wave. No grid recomputation."""
        return self._to_real(self.roll_multiplier(dw, dh) * self._to_complex(wave))

    # ------------------------------------------------------ task operator
    @staticmethod
    def compile_task_operator_ls(pairs) -> torch.Tensor:
        """Least-squares per-slot DIAGONAL operator. Returns complex [num_blocks, 4].

        W* = sum_p conj(x_p)*y_p / sum_p |x_p|^2  -- the family ceiling.
        This is the correct operator for the [num_blocks, 8] layout. It recovers
        a cyclic roll at 1.0000 (measured) and reaches real-ARC in-sample
        ceilings of 0.76-0.82.
        """
        if not pairs:
            raise ValueError("no demonstration pairs")
        x0, _ = pairs[0]
        num = torch.zeros_like(TorusIngressEncoder._to_complex(x0))
        den = torch.zeros(num.shape, dtype=torch.float32, device=x0.device)
        for x, y in pairs:
            cx = TorusIngressEncoder._to_complex(x)
            num = num + torch.conj(cx) * TorusIngressEncoder._to_complex(y)
            den = den + cx.abs() ** 2
        return num / (den + 1e-9)

    @staticmethod
    def compile_task_operator_mean(pairs) -> torch.Tensor:
        """Legacy mean(conj(x)*y) form, kept for A/B against the LS form.

        Equals the LS optimum ONLY when |x| is constant. The encoder L2-normalises
        per block, so |x| is strongly non-flat and this form is biased by the
        block energy profile. Do not treat it as the ceiling.
        """
        acc = None
        for x, y in pairs:
            t = torch.conj(TorusIngressEncoder._to_complex(x)) * TorusIngressEncoder._to_complex(y)
            acc = t if acc is None else acc + t
        return acc / float(len(pairs))

    @staticmethod
    def predict(w_op: torch.Tensor, wave: torch.Tensor) -> torch.Tensor:
        return TorusIngressEncoder._to_real(w_op * TorusIngressEncoder._to_complex(wave))


def is_enabled() -> bool:
    """Default OFF. The tokenizer's legacy path is byte-identical while False."""
    return os.environ.get("HENRI_ENCODER_TORUS", "0") == "1"


def encode_spatial_grid_torus(tokenizer, grid, modulus: int = None, mode: str = None):
    """Bounded hook for O_VSA_IngressTokenizer.encode_spatial_grid.

    Builds (and caches) a TorusIngressEncoder sized from the host tokenizer so the
    output shape/device contract is unchanged: real [1, num_blocks, 8].
    """
    mod = int(modulus or os.environ.get("HENRI_ENCODER_TORUS_MODULUS", DEFAULT_MODULUS))
    md = mode or os.environ.get("HENRI_ENCODER_TORUS_MODE", "TORUS_VAL")
    enc = getattr(tokenizer, "_torus_encoder", None)
    need = dict(num_blocks=tokenizer.num_blocks, vocab_size=tokenizer.vocab_size,
                modulus=mod, mode=md, device=str(tokenizer.device))
    if enc is None or getattr(enc, "_need", None) != need:
        enc = TorusIngressEncoder(**need)
        enc._need = need
        tokenizer._torus_encoder = enc
    return enc.encode(grid).unsqueeze(0)
