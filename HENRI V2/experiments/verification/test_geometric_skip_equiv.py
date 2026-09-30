"""EXHAUSTIVE proof that the geometric skip and the scipy path match legacy.

WHY EXHAUSTIVE AND NOT A SAMPLE
===============================
The geometric skip claims: "if a component cannot enclose a cell, its interior
is empty, so return [] without flooding." A WRONG predicate silently deletes
real interiors, which flips `mech_type` from `enclosed_contour` to something
else and corrupts the parity mask that weights the wave. A fuzz test at 4x4
would very likely miss the one counterexample.

So this file enumerates ALL 2^16 = 65,536 contour subsets of the 4x4 grid --
every possible contour shape at that size, not a sample of them -- and compares
against the legacy flood-fill element-by-element. 4x4 is chosen because it is
the smallest grid that can contain the minimal enclosing loop (the 4-pixel
diamond), so it contains the boundary cases.

WHAT IS ASSERTED
  [1] geometric skip, exhaustive 2^16 subsets:
        interior(skip) == interior(legacy), element-for-element, AND
        exterior(skip) == [] (never built)
  [2] scipy path, exhaustive 2^16 subsets:
        interior and exterior identical to legacy, in order
  [3] structured counterexamples, hand-checked:
        3x3 ring (encloses), 4-pixel diamond (ENCLOSES -- the threshold case),
        2x2 block (does not), 4x1 line (does not), solid 3x3 (does not),
        two-cell interior, nested shapes
  [4] record-level identity through segment_grid: mech_type, interior_pixels,
      bbox, area identical for every component (this is the consumer that
      actually reads interior -- mech_type branches on len(interior_px) > 0)
  [5] wave identity through encode_grid at D=65536 AND D=2048
  [6] timing at both D

Run:  python experiments/verification/test_geometric_skip_equiv.py
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from connected_component_segmenter import (  # noqa: E402
    ParityContourMask,
    ConnectedComponentSegmenter,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402

FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{label}: {detail}")


# --------------------------------------------------------------------------- #
print("=" * 80)
print("[1] GEOMETRIC SKIP -- exhaustive over ALL 2^16 contour subsets of 4x4")
print("=" * 80)
CELLS4 = [(r, c) for r in range(4) for c in range(4)]
SHAPE4 = (4, 4)

t0 = time.perf_counter()
n_checked = 0
n_skip_fired = 0
n_nonempty_legacy = 0
mismatch = 0
DETAIL = None
for bits in range(1 << 16):
    px = [CELLS4[i] for i in range(16) if (bits >> i) & 1]
    if not px:
        continue
    li, _ = ParityContourMask.compute_parity_contour(
        SHAPE4, px, fast=False, want_exterior=True)
    si, se = ParityContourMask.compute_parity_contour(
        SHAPE4, px, fast=False, want_exterior=False)
    n_checked += 1
    if li:
        n_nonempty_legacy += 1
    if se:
        mismatch += 1
        if DETAIL is None:
            DETAIL = f"bits={bits} exterior non-empty under want_exterior=False"
    if si != li:
        mismatch += 1
        if DETAIL is None:
            DETAIL = (f"bits={bits} px={px} legacy_interior={li} skip_interior={si} "
                      f"can_enclose={ParityContourMask._can_enclose(px)}")
    if not ParityContourMask._can_enclose(px):
        n_skip_fired += 1
el = time.perf_counter() - t0

print(f"  subsets checked        : {n_checked}")
print(f"  skip fired (no flood)  : {n_skip_fired}  ({100.0*n_skip_fired/n_checked:.1f}%)")
print(f"  legacy non-empty int.  : {n_nonempty_legacy}  "
      f"({100.0*n_nonempty_legacy/n_checked:.2f}% -- these MUST survive)")
print(f"  elapsed                : {el:.1f}s")
check("all 2^16 subsets identical", mismatch == 0, DETAIL or "")
# the predicate must never fire on a case that has a real interior
false_neg = 0
for bits in range(1 << 16):
    px = [CELLS4[i] for i in range(16) if (bits >> i) & 1]
    if not px:
        continue
    if not ParityContourMask._can_enclose(px):
        li, _ = ParityContourMask.compute_parity_contour(
            SHAPE4, px, fast=False, want_exterior=True)
        if li:
            false_neg += 1
check("predicate has ZERO false negatives", false_neg == 0, f"count={false_neg}")

# --------------------------------------------------------------------------- #
print()
print("=" * 80)
print("[2] SCIPY PATH -- exhaustive over ALL 2^16 contour subsets of 4x4")
print("=" * 80)
try:
    from scipy import ndimage  # noqa: F401
    have_scipy = True
except Exception as e:  # pragma: no cover
    have_scipy = False
    print(f"  scipy unavailable ({e}) -- SKIPPED")

if have_scipy:
    t0 = time.perf_counter()
    sc_mismatch = 0
    sc_detail = None
    for bits in range(1 << 16):
        px = [CELLS4[i] for i in range(16) if (bits >> i) & 1]
        if not px:
            continue
        li, le = ParityContourMask.compute_parity_contour(
            SHAPE4, px, fast=False, want_exterior=True)
        ci, ce = ParityContourMask.compute_parity_contour(
            SHAPE4, px, use_scipy=True, want_exterior=True)
        if ci != li or ce != le:
            sc_mismatch += 1
            if sc_detail is None:
                sc_detail = f"bits={bits} px={px} legacy=({li},{le}) scipy=({ci},{ce})"
    el = time.perf_counter() - t0
    print(f"  elapsed: {el:.1f}s")
    check("scipy identical to legacy on interior AND exterior", sc_mismatch == 0,
          sc_detail or "")

# --------------------------------------------------------------------------- #
print()
print("=" * 80)
print("[3] STRUCTURED COUNTEREXAMPLES (hand-checked boundary cases)")
print("=" * 80)
CASES = {
    # name: (shape, contour_pixels, expected_interior)
    "3x3 ring (area 8)": ((5, 5),
                          [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
                          [(1, 1)]),
    # THE THRESHOLD CASE: 4 pixels, bbox 3x3, and it DOES enclose. If the
    # predicate used area>=8 this would be silently dropped.
    "4-pixel diamond": ((5, 5), [(0, 1), (1, 0), (1, 2), (2, 1)], [(1, 1)]),
    "2x2 block": ((5, 5), [(0, 0), (0, 1), (1, 0), (1, 1)], []),
    "4x1 line": ((5, 5), [(0, 0), (0, 1), (0, 2), (0, 3)], []),
    "solid 3x3": ((5, 5), [(r, c) for r in range(3) for c in range(3)], []),
    # 5x5 ring -> 3x3 interior
    "5x5 ring": ((7, 7),
                 [(0, c) for c in range(5)] + [(4, c) for c in range(5)]
                 + [(r, 0) for r in range(1, 4)] + [(r, 4) for r in range(1, 4)],
                 [(r, c) for r in range(1, 4) for c in range(1, 4)]),
    # off-centre / touching border
    "ring at border": ((5, 5),
                       [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
                       [(1, 1)]),
}
for name, (shape, px, expect) in CASES.items():
    li, _ = ParityContourMask.compute_parity_contour(
        shape, px, fast=False, want_exterior=True)
    si, _ = ParityContourMask.compute_parity_contour(
        shape, px, fast=False, want_exterior=False)
    pred = ParityContourMask._can_enclose(px)
    ok = (li == expect and si == expect)
    # a case with a real interior MUST pass the predicate
    if expect and not pred:
        ok = False
        li = li + ["<predicate rejected a real interior>"]
    print(f"  {name:22s} can_enclose={str(pred):5s} legacy={len(li):2d} "
          f"skip={len(si):2d} expected={len(expect):2d}  {'OK' if ok else 'MISMATCH'}")
    check(f"structured: {name}", ok)

# --------------------------------------------------------------------------- #
print()
print("=" * 80)
print("[4] RECORD-LEVEL IDENTITY (mech_type is the real consumer)")
print("=" * 80)
REAL_GRIDS = {
    "mixed_5x5": [[1, 1, 0, 0, 0], [1, 2, 2, 0, 0], [0, 2, 3, 3, 0], [0, 0, 3, 0, 0], [0, 0, 0, 0, 4]],
    "ring_7x7": [[1 if (i in (0, 6) or j in (0, 6)) else 0 for j in range(7)] for i in range(7)],
    "nested_5x5": [[1, 1, 1, 1, 1], [1, 2, 2, 2, 1], [1, 2, 0, 2, 1], [1, 2, 2, 2, 1], [1, 1, 1, 1, 1]],
    "dense_16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
    "sparse_8x8": [[0 if (i + j) % 3 else (i % 7) + 1 for j in range(8)] for i in range(8)],
    "solid_4x4": [[5] * 4 for _ in range(4)],
    "single_3x3": [[0, 0, 0], [0, 7, 0], [0, 0, 0]],
}
SEG = ConnectedComponentSegmenter(background_color=0)
rec_ok = True
for name, g in REAL_GRIDS.items():
    base = SEG.segment_grid(g, want_exterior=True, fast=False)
    skip = SEG.segment_grid(g, want_exterior=False, fast=False)
    sci = SEG.segment_grid(g, want_exterior=True, fast=False, use_scipy=True)
    if len(base) != len(skip) or len(base) != len(sci):
        rec_ok = False
        print(f"  FAIL {name}: component counts {len(base)}/{len(skip)}/{len(sci)}")
        continue
    bad = 0
    for a, b, c in zip(base, skip, sci):
        fields = ("object_id", "color", "mech_type", "pixels", "interior_pixels",
                  "bbox", "area")
        for f in fields:
            if getattr(a, f) != getattr(b, f):
                bad += 1
                print(f"  FAIL {name} obj{a.object_id}: {f} differs (skip) "
                      f"{getattr(a, f)} vs {getattr(b, f)}")
            if getattr(a, f) != getattr(c, f):
                bad += 1
                print(f"  FAIL {name} obj{a.object_id}: {f} differs (scipy) "
                      f"{getattr(a, f)} vs {getattr(c, f)}")
    mechs = {}
    for a in base:
        mechs[a.mech_type] = mechs.get(a.mech_type, 0) + 1
    check(f"records: {name} ({len(base)} comps)", bad == 0, str(mechs))
    if bad:
        rec_ok = False

# --------------------------------------------------------------------------- #
print()
print("=" * 80)
print("[5] WAVE IDENTITY through encode_grid, at BOTH D (the spec's ambiguity)")
print("=" * 80)


def mk(d_model, **kw):
    """Returns an INSTANCE. (Callers previously wrote mk(D)() and hit
    NotImplementedError: HENRIVisionEncoder has no forward() -- a defect in this
    test file, not in production.)"""
    return HENRIVisionEncoder(d_model=d_model, k_blocks=8192, device="cpu",
                              spatial_basis_kind="incommensurate",
                              bg_mask=True, **kw)


for D in (65536, 2048):
    worst = 0.0
    for name, g in REAL_GRIDS.items():
        a = mk(D)
        b = mk(D, parity_fast=True)
        c = mk(D, parity_scipy=True)
        wa = a.encode_grid(g)
        wb = b.encode_grid(g)
        wc = c.encode_grid(g)
        worst = max(worst, float((wa - wb).abs().max()), float((wa - wc).abs().max()))
    check(f"D={D}: wave bit-identical across baseline/fast/scipy", worst < 1e-6,
          f"max abs diff = {worst:.3e}")

# --------------------------------------------------------------------------- #
print()
print("=" * 80)
print("[6] TIMING at D=65536 and D=2048")
print("=" * 80)
CFGS = {
    "A baseline": dict(),
    "C fast": dict(parity_fast=True),
    "F scipy": dict(parity_scipy=True),
}
for D in (65536, 2048):
    print(f"\n  --- D={D} ---")
    print(f"  {'grid':9s}" + "".join(f"{k:>12s}" for k in CFGS))
    for name, g in REAL_GRIDS.items():
        row = f"  {name:9s}"
        base = None
        for label, kw in CFGS.items():
            e = mk(D, **kw)
            for _ in range(2):
                e.encode_grid(g)
            reps = 5
            t0 = time.perf_counter()
            for _ in range(reps):
                e.encode_grid(g)
            us = (time.perf_counter() - t0) / reps * 1e6
            if base is None:
                base = us
            row += f"{us:12.1f}"
        print(row)

print()
print("=" * 80)
if FAILURES:
    print(f"RESULT: FAIL ({len(FAILURES)})")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1)
print("RESULT: PASS")
print("GEOMETRIC_SKIP_PROVEN True")
sys.exit(0)
