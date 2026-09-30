"""Where does the remaining encode_grid time go, and what does D change?

TWO QUESTIONS
=============
Q1 (substrate): the geometric skip fires for all 230 single-pixel components at
   16x16, yet dense_16x16 measured ~65.8 ms. Hypothesis H6: encode_grid's `else`
   branch made a SECOND compute_parity_contour call with the default
   want_exterior=True, which does not take the skip, so 230 full floods still
   ran. This probe measures each stage separately to confirm or refute H6.

Q2 (the gate's ambiguity): the spec says "D = 2048 for local unit tests and
   D = 65536 for Blackwell/GB202 target execution" (spec 4.1.1). All
   measurements so far are at D=65536. The superposition cost is linear in D, so
   the 15 us clause may be a different question at D=2048. Measure BOTH.

CPU only. Free.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from connected_component_segmenter import (  # noqa: E402
    ParityContourMask,
    ConnectedComponentSegmenter,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402

GRIDS = {
    "4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]],
    "8x8": [[(i * 3 + j) % 4 for j in range(8)] for i in range(8)],
    "16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
    "30x30": [[(i + j) % 10 for j in range(30)] for i in range(30)],
    "ring_7x7": [[1 if (i in (0, 6) or j in (0, 6)) else 0 for j in range(7)]
                 for i in range(7)],
}


def timed(fn, reps=5):
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6


print("=" * 82)
print("Q1: STAGE TIMING -- does the else-branch re-flood? (CPU, us)")
print("=" * 82)
print(f"  {'grid':9s} {'comps':>6s} {'enclosable':>11s} {'seg_false':>10s} "
      f"{'seg_true':>10s} {'2nd_true':>10s} {'2nd_false':>10s}")
for name, g in GRIDS.items():
    arr = np.array(g, dtype=int)
    H, W = arr.shape
    seg = ConnectedComponentSegmenter(background_color=0)
    comps = seg.segment_grid(arr, want_exterior=False)
    n_can = sum(1 for c in comps if ParityContourMask._can_enclose(list(c.pixels)))

    t_false = timed(lambda: seg.segment_grid(arr, want_exterior=False), reps=3)
    t_true = timed(lambda: seg.segment_grid(arr, want_exterior=True), reps=3)

    def call(want_ext):
        for c in comps:
            ParityContourMask.compute_parity_contour((H, W), list(c.pixels),
                                                     want_exterior=want_ext)

    t2t = timed(lambda: call(True), reps=3)
    t2f = timed(lambda: call(False), reps=3)
    print(f"  {name:9s} {len(comps):6d} {n_can:11d} {t_false:10.1f} "
          f"{t_true:10.1f} {t2t:10.1f} {t2f:10.1f}")

print()
print("=" * 82)
print("Q2: encode_grid TOTAL at D=2048 vs D=65536 (CPU, us)")
print("=" * 82)


def mk(D, **kw):
    return HENRIVisionEncoder(d_model=D, k_blocks=8192, device="cpu",
                              spatial_basis_kind="incommensurate",
                              bg_mask=True, **kw)


CFGS = {
    "baseline": dict(),
    "fast": dict(parity_fast=True),
    "scipy": dict(parity_scipy=True),
    "dedup": dict(parity_dedup=True),
}
for D in (2048, 65536):
    print(f"\n  --- D={D} ---")
    print(f"  {'grid':9s}" + "".join(f"{k:>11s}" for k in CFGS) + "   best/base")
    for name, g in GRIDS.items():
        row = f"  {name:9s}"
        base = None
        for label, kw in CFGS.items():
            e = mk(D, **kw)
            us = timed(lambda: e.encode_grid(g), reps=5)
            if base is None:
                base = us
            row += f"{us:11.1f}"
            best = us
        row += f"   {base/best:.2f}x"
        print(row)

print()
print("=" * 82)
print("IDENTITY: best config vs baseline at both D (max abs wave diff)")
print("=" * 82)
for D in (2048, 65536):
    worst = 0.0
    arg = ""
    for name, g in GRIDS.items():
        wa = mk(D).encode_grid(g)
        for label, kw in CFGS.items():
            if label == "baseline":
                continue
            wb = mk(D, **kw).encode_grid(g)
            d = float((wa - wb).abs().max())
            if d > worst:
                worst, arg = d, f"{name}/{label}"
    print(f"  D={D}: worst = {worst:.3e}  ({arg})")
print()
print("STAGE2_DONE")
