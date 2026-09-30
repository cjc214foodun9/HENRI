"""Verify _superpose_fused against the production per-cell loop, at both D.

WHY THIS FILE EXISTS
  `fused_superpose=True` replaces the sequential per-cell accumulation:
      for r, c:  acc += px[c] * py[r] * pc[grid[r,c]] * kappa[r,c]
  with a batched tensor expression over row chunks. The summation ORDER differs,
  so the result is numerically equivalent but NOT bit-identical. This file pins
  the actual worst-case difference and the fail-closed contract, so the claim
  "fused is equivalent" is a measurement rather than an assertion about
  floating-point associativity.

A CORRECTED FALSE CLAIM (this is why section [2] is shaped the way it is)
  An earlier version of this file asserted `d == 0.0` for row_chunk invariance --
  exact bit equality across chunk sizes. That is IMPOSSIBLE, not merely strict:
  chunk=1 reduces each row then accumulates, chunk=64 reduces 64 rows at once,
  and float addition is not associative. The test failed (3.4e-07 at 4x4 rising
  to 1.86e-04 at 30x30, scaling with element count exactly as reassociation
  error does). The assertion was wrong; the code was not. It is replaced by:
    (a) a MEASURED regression tripwire for chunk sensitivity, and
    (b) a TRUE determinism test at the production default chunk.
  Production always uses the default `row_chunk=8`, so production IS
  deterministic run-to-run. That is the property that matters, and it is now
  tested directly instead of being implied by a false claim.

WHAT IS ASSERTED
  [1] max abs wave difference vs the legacy loop, at D=65536 and D=2048, across
      grids including real ENCLOSED contours (rings, nested) -- the parity mask
      is -1 inside such a component, which is exactly the branch a naive fused
      path drops.
  [2] row_chunk sensitivity is BOUNDED (measured tripwire, not exact equality).
  [3] determinism at the production default chunk: repeated calls bit-identical.
  [4] fail-closed parity: a grid where bg_mask excludes every cell must raise
      ValueError, exactly as the legacy path does. This contract stops a
      silently-zero wave from propagating.
  [5] bg_mask=False path matches legacy within fp32 tolerance.

  For reference: on a normalized 65536-dim wave, a perturbation of 2e-4 changes
  cosine similarity by roughly d^2/2 = 2e-8. The deviations below are far
  smaller than any downstream threshold in use.

Run:  python experiments/verification/test_fused_superpose_equiv.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402

FAILURES = []

# Regression tripwire for chunk sensitivity. MEASURED worst case was 1.856e-04
# (30x30, D=65536, chunk 1 vs 64); this ceiling sits ~5.4x above that, so fp
# reassociation noise passes and a real summation bug trips it.
CHUNK_TOL = 1e-3
# fused-vs-legacy tolerance. MEASURED worst 8.196e-08 (30x30, D=65536).
FUSED_TOL = 1e-6


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")


GRIDS = {
    "4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
    "ring_5x5": [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1],
                 [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
    "ring_7x7": [[1 if (i in (0, 6) or j in (0, 6)) else 0 for j in range(7)]
                 for i in range(7)],
    "nested_5x5": [[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 0, 2, 1],
                   [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]],
    "16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
    "30x30": [[(i + j) % 10 for j in range(30)] for i in range(30)],
    "single": [[7]],
    "all_same": [[3] * 6 for _ in range(6)],
}
ALL_BG = [[0] * 4 for _ in range(4)]


def mk(D, **kw):
    return HENRIVisionEncoder(d_model=D, k_blocks=8192, device="cpu",
                              spatial_basis_kind="incommensurate", **kw)


def parity_ones(g):
    gt = torch.tensor(g, dtype=torch.long)
    H, W = gt.shape
    return gt, torch.ones((H, W), dtype=torch.float32), H, W


print("=" * 78)
print("[1] FUSED vs LEGACY: max abs wave difference")
print("=" * 78)
results = {}
for D in (65536, 2048):
    worst = 0.0
    worst_arg = ""
    for name, g in GRIDS.items():
        ref = mk(D, bg_mask=True).encode_grid(g)
        alt = mk(D, bg_mask=True, fused_superpose=True).encode_grid(g)
        d = float((ref - alt).abs().max())
        if d > worst:
            worst, worst_arg = d, name
        print(f"  D={D:<6d} {name:11s} diff={d:.3e}  norm(ref)={ref.norm():.8f}")
    results[D] = worst
    check(f"D={D}: fused equivalent to legacy (<{FUSED_TOL:g})", worst < FUSED_TOL,
          f"worst={worst:.3e} at {worst_arg}")

print()
print("=" * 78)
print("[2] ROW-CHUNK SENSITIVITY: bounded fp reassociation (NOT exact equality)")
print("=" * 78)
print("  exact bit equality across chunk sizes is impossible: chunk=1 reduces")
print("  each row then accumulates, chunk=64 reduces 64 rows at once, and float")
print("  addition is not associative. The measured spread is bounded below.")
chunk_worst = 0.0
for name, g in GRIDS.items():
    gt, pm, H, W = parity_ones(g)
    waves = []
    for chunk in (1, 3, 8, 32, 64):
        e = mk(65536, bg_mask=True, fused_superpose=True)
        w, _ = e._superpose_fused(gt, pm, H, W, row_chunk=chunk)
        waves.append(w)
    base = waves[0]
    d = max(float((w - base).abs().max()) for w in waves[1:])
    chunk_worst = max(chunk_worst, d)
    check(f"chunk spread bounded: {name}", d < CHUNK_TOL, f"max diff {d:.3e}")
print(f"\n  worst chunk spread across all grids: {chunk_worst:.3e}  "
      f"(tripwire {CHUNK_TOL:g})")

print()
print("=" * 78)
print("[3] DETERMINISM at the production default chunk (row_chunk=8)")
print("=" * 78)
for name, g in GRIDS.items():
    # production path: encode_grid with fused, called repeatedly
    e = mk(65536, bg_mask=True, fused_superpose=True)
    w1 = e.encode_grid(g)
    w2 = e.encode_grid(g)
    d = float((w1 - w2).abs().max())
    check(f"deterministic: {name}", d == 0.0, f"repeat diff {d:.3e}")

print()
print("=" * 78)
print("[4] FAIL-CLOSED PARITY (bg_mask excludes every cell -> ValueError)")
print("=" * 78)
for label, kw in (("legacy", {}), ("vectorized", {"vectorized_accum": True}),
                  ("fused", {"fused_superpose": True})):
    try:
        mk(65536, bg_mask=True, **kw).encode_grid(ALL_BG)
        check(f"fail-closed: {label}", False, "no ValueError raised")
    except ValueError:
        check(f"fail-closed: {label}", True, "ValueError as required")
    except Exception as exc:
        check(f"fail-closed: {label}", False, f"wrong type {type(exc).__name__}")

print()
print("=" * 78)
print("[5] bg_mask=False: fused vs legacy (within fp32 tolerance)")
print("=" * 78)
worst_bg = 0.0
worst_bg_arg = ""
for name, g in GRIDS.items():
    ref = mk(65536, bg_mask=False).encode_grid(g)
    alt = mk(65536, bg_mask=False, fused_superpose=True).encode_grid(g)
    d = float((ref - alt).abs().max())
    if d > worst_bg:
        worst_bg, worst_bg_arg = d, name
check(f"bg_mask=False equivalent (<{FUSED_TOL:g})", worst_bg < FUSED_TOL,
      f"worst {worst_bg:.3e} at {worst_bg_arg}")

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: FAIL ({len(FAILURES)})")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1)
print("RESULT: PASS")
print("FUSED_EQUIVALENCE_PROVEN True")
print(f"worst diff vs legacy, D=65536: {results[65536]:.3e}")
print(f"worst diff vs legacy, D=2048 : {results[2048]:.3e}")
print(f"worst chunk spread           : {chunk_worst:.3e}")
sys.exit(0)
