#!/usr/bin/env python3
"""OBSERVED: EXACT-ROLL -- STRUCTURAL LAW or float32 ACCUMULATION FLOOR?

The wiring gate reports max|enc(roll) - M*enc| ~1.6e-04 and labels it NOT exact.
Two rival hypotheses, and mislabelling either way is the units-conflation defect
class already hit three times in this project:

  H1 STRUCTURAL : the multiplier is wrong; the residual has a floor that does
                  NOT shrink when precision increases.
  H2 NUMERICAL  : the residual is float32 accumulation only; it shrinks by
                  roughly the float32/float64 epsilon ratio (~1e9) when the
                  same arithmetic runs in float64.

Decisive test: the SAME quantity in complex64 (through the module's own
_to_complex / roll_multiplier / _to_real) and in complex128 (hand-rolled from
the integer frequency indices, so no float32 constant enters).

WHY complex128 must be built from INTEGER kx/ky
    The module stores wx = 2*pi*kx/S in float32. Any "float64" arm that reuses
    that float32 constant measures the constant, not the structure. Here the
    float64 arm rebuilds exp(i*(x*wx + y*wy)) from enc.kx/enc.ky cast to
    int64 -> float64.

ROLL CONVENTION (stated so the sign cannot be silently flipped)
    roll_canvas(rows, d, 0) does r[d:] + r[:d], i.e. the element at new column
    x holds old column x+d. Therefore
        enc(roll(X, d)) = exp(-i*d*wx) * enc(X)
    so the analytic multiplier is M = exp(-1j * d * wx).

MODULE FACT MEASURED HERE (corrects an earlier claim of mine)
    _to_real does torch.stack([z.real, z.imag], -1).reshape(z.shape[0], 8), so
    it requires z of shape [NB, BLOCK_SLOTS] with BLOCK_SLOTS == 4. That is 4
    COMPLEX slots expanding to 8 real components -- not "8 complex slots, 4
    kept". The reserved DC slot is 1 of 4 complex slots.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from o_vsa_torus_encoder import BLOCK_SLOTS, TorusIngressEncoder  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def enc128(rows, S, KX_i, KY_i):
    """complex128 encode, built from INTEGER kx/ky. [_NB, J]"""
    nb, j = KX_i.shape
    xs, ys, vals = [], [], []
    for y in range(len(rows)):
        for x in range(len(rows[y])):
            xs.append(x); ys.append(y); vals.append(min(int(rows[y][x]), j - 1))
    v = torch.tensor(vals, dtype=torch.long, device=DEV)
    ph = torch.zeros(len(vals), j, dtype=torch.complex128, device=DEV)
    ph[torch.arange(len(vals), device=DEV), v] = 1.0
    wx = (2.0 * math.pi * KX_i.double() / float(S)).to(DEV)      # [nb, j]
    wy = (2.0 * math.pi * KY_i.double() / float(S)).to(DEV)
    X = torch.tensor(xs, dtype=torch.float64, device=DEV)[:, None, None]  # [N,1,1]
    Y = torch.tensor(ys, dtype=torch.float64, device=DEV)[:, None, None]
    P = torch.exp(1j * (X * wx[None] + Y * wy[None]))            # [N,nb,j]
    return torch.einsum("pj,pij->ij", ph, P)


def main():
    out = {"device": DEV, "torch": torch.__version__,
           "BLOCK_SLOTS": int(BLOCK_SLOTS), "by_S": {}}
    for S in (16, 32, 64):
        enc = TorusIngressEncoder(num_blocks=8192, vocab_size=8, modulus=S,
                                  mode="TORUS_VAL", device=DEV)
        gen = torch.Generator().manual_seed(5)
        g = torch.randint(0, 8, (S, S), generator=gen).tolist()
        rows = [list(map(int, r)) for r in g]
        KX_i = enc.kx.detach().to("cpu").to(torch.int64)
        KY_i = enc.ky.detach().to("cpu").to(torch.int64)

        e64, e32_pre, e32_post, caps = {}, {}, {}, {}
        for d in (1, 3, 5):
            # --- complex64 through the module's own path
            z1 = enc._to_complex(enc.encode_canvas(enc.roll_canvas(rows, d, 0)))
            z2 = enc.roll_multiplier(d, 0) * enc._to_complex(enc.encode_canvas(rows))
            scale = float(z2.abs().max().item())
            caps[d] = scale
            e32_pre[str(d)] = float((z1 - z2).abs().max().item() / (scale + 1e-30))
            r1, r2 = enc._to_real(z1), enc._to_real(z2)
            e32_post[str(d)] = float((r1 - r2).abs().max().item())

            # --- complex128, analytic multiplier from integer kx/ky
            a = enc128(rows, S, KX_i, KY_i)
            b = enc128(enc.roll_canvas(rows, d, 0), S, KX_i, KY_i)
            M = torch.exp(-1j * float(d) * (2.0 * math.pi * KX_i.double() / S)).to(DEV)
            e64[str(d)] = float((M * a - b).abs().max().item()
                                / (b.abs().max().item() + 1e-30))

        uni = enc.encode_canvas([[0] * S for _ in range(S)])
        var = enc.encode_canvas(rows)
        surv = {}
        for slot in (0, BLOCK_SLOTS - 1):
            pr = torch.zeros(8192, BLOCK_SLOTS, dtype=torch.complex64, device=DEV)
            pr[:, slot] = 1.0 + 0.0j
            surv[int(slot)] = float(enc._to_real(pr).abs().max().item())

        ratio = {k: e32_pre[k] / max(e64[k], 1e-300) for k in e32_pre}
        out["by_S"][S] = {
            "err_c64_rel_pre_norm": e32_pre,
            "err_c64_post_norm": e32_post,
            "err_c128_rel": e64,
            "c64_over_c128_ratio": ratio,
            "op_magnitude": caps,
            "uniform_norm": float(uni.norm().item()),
            "varied_norm": float(var.norm().item()),
            "slot_survival": surv,
        }
        mx32 = max(e32_pre.values()); mx64 = max(e64.values())
        print(f"S={S:3d} c64_pre={mx32:.3e} c128={mx64:.3e} ratio={mx32/max(mx64,1e-300):.2e}"
              f"  uni={float(uni.norm().item()):.4f} varied={float(var.norm().item()):.4f}"
              f"  slot0={surv[0]:.3f} slot{BLOCK_SLOTS-1}={surv[BLOCK_SLOTS-1]:.3f}")

    all64 = [v for S in out["by_S"] for v in out["by_S"][S]["err_c128_rel"].values()]
    all32 = [v for S in out["by_S"] for v in out["by_S"][S]["err_c64_rel_pre_norm"].values()]
    r = [a / max(b, 1e-300) for a, b in zip(all32, all64)]
    med = lambda z: sorted(z)[len(z) // 2]
    out["verdict"] = {
        "median_c64": med(all32), "median_c128": med(all64), "median_ratio": med(r),
        "H2_numerical_supported": bool(max(all64) < 1e-9 and min(r) > 1e3),
        "H1_structural_supported": bool(max(all64) > 1e-9),
        "reading": ("c128 << c64 by ~1e9 and c128 is at float64 round-off => the "
                    "1.6e-04 residual is FLOAT32 ACCUMULATION, not a wrong "
                    "multiplier. The exact-roll property holds; what reported "
                    "'EXACT: False' was the gate's absolute threshold."
                    if max(all64) < 1e-9 else
                    "the residual survives float64 => STRUCTURAL; the multiplier "
                    "or the quantisation is wrong."),
    }
    print()
    print(json.dumps(out["verdict"], indent=1))
    dst = ("/workspace/phase10/HENRI V2/experiments/verification/"
           "arc_torus_exactness_floor_observed.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWROTE {dst} ({os.path.getsize(dst)} bytes)")
    print("### DONE")


if __name__ == "__main__":
    main()
