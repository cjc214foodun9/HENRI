"""WHY is the segmenter slow? Three candidate causes, measured on the real pattern.

MEASURED SO FAR
  1. Phase split (RTX 5090): segmentation = 97.8% of encode_grid at 16x16
     (109634.7 of 112061.3 us). The superposition is 0.3%.
  2. My launch-count hypothesis was FALSIFIED: 78x fewer launches bought 1.09x.
  3. My deque fix `_parity_contour_fast` is CORRECT (200/200 fuzz identical)
     but 0.70x -- SLOWER. So `pop(0)` was not the dominant cost either.

KEY FACT: encode_grid does `grid_clamped.cpu().numpy()` and runs segmentation in
Python/numpy on the CPU. The 56 ms at 16x16 is CPU time. So this is measurable
locally, free.

THREE CANDIDATE CAUSES
  C1 numpy SCALAR INDEXING inside the BFS loop: `padded[nr, nc]` costs ~150 ns
     each. 324 cells x 4 neighbours ~ 1300 scalar accesses per parity call, x230
     components ~ 300k accesses ~ 45 ms. This fits the measurement.
  C2 list.pop(0) in the BFS                       -- FALSIFIED by test [4] above.
  C3 Python-level extraction loops (2x256 iters)  -- partially addressed already.

VARIANTS TESTED
  legacy   : list.pop(0) + numpy scalar indexing          (production)
  deque    : collections.deque + numpy scalar indexing    (_parity_contour_fast)
  pylist   : list.pop() + PYTHON LIST-OF-LISTS            (tests C1)
  scipy    : scipy.ndimage.label, C-level flood fill      (reference floor)
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from connected_component_segmenter import (  # noqa: E402
    ConnectedComponentSegmenter,
    ParityContourMask,
)

NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def parity_legacy(rows, cols, contour):
    mask = np.zeros((rows, cols), dtype=int)
    for r, c in contour:
        if 0 <= r < rows and 0 <= c < cols:
            mask[r, c] = 1
    padded = np.zeros((rows + 2, cols + 2), dtype=int)
    padded[1:rows + 1, 1:cols + 1] = mask
    q = [(0, 0)]
    padded[0, 0] = 2
    while q:
        cr, cc = q.pop(0)
        for dr, dc in NB4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows + 2 and 0 <= nc < cols + 2 and padded[nr, nc] == 0:
                padded[nr, nc] = 2
                q.append((nr, nc))
    inter, ext = [], []
    for r in range(rows):
        for c in range(cols):
            v = padded[r + 1, c + 1]
            if v == 0:
                inter.append((r, c))
            elif v == 2 and (r, c) not in contour:
                ext.append((r, c))
    return inter, ext


def parity_pylist(rows, cols, contour):
    """C1 test: Python list-of-lists instead of a numpy array for `padded`."""
    H, W = rows + 2, cols + 2
    padded = [[0] * W for _ in range(H)]
    for r, c in contour:
        if 0 <= r < rows and 0 <= c < cols:
            padded[r + 1][c + 1] = 1
    cs = set(contour)
    stack = [(0, 0)]
    padded[0][0] = 2
    while stack:
        cr, cc = stack.pop()
        for dr, dc in NB4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and padded[nr][nc] == 0:
                padded[nr][nc] = 2
                stack.append((nr, nc))
    inter, ext = [], []
    for r in range(rows):
        row = padded[r + 1]
        for c in range(cols):
            v = row[c + 1]
            if v == 0:
                inter.append((r, c))
            elif v == 2 and (r, c) not in cs:
                ext.append((r, c))
    return inter, ext


def parity_scipy(rows, cols, contour):
    """C-level reference floor. 4-connectivity flood fill via ndimage.label."""
    from scipy import ndimage
    pad = np.zeros((rows + 2, cols + 2), dtype=bool)
    for r, c in contour:
        if 0 <= r < rows and 0 <= c < cols:
            pad[r + 1, c + 1] = True
    st = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    lab, _ = ndimage.label(~pad, structure=st)
    ext_label = lab[0, 0]
    inner = lab[1:rows + 1, 1:cols + 1]
    inter = [(int(r), int(c)) for r, c in
             np.argwhere((inner != 0) & (inner != ext_label))]
    ext = [(int(r), int(c)) for r, c in np.argwhere(inner == ext_label)]
    return inter, ext


def timeit(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e6


def main() -> int:
    print("=" * 78)
    print("SEGMENTER COST ATTRIBUTION (CPU -- correct, because encode_grid")
    print("segments on CPU after .cpu().numpy())")
    print("=" * 78)

    try:
        import scipy  # noqa: F401
        have_scipy = True
    except Exception:
        have_scipy = False
    print(f"  scipy available: {have_scipy}")

    for gname, g in (("mixed_5x5", [[1, 1, 0, 0, 0], [1, 2, 2, 0, 0],
                                     [0, 2, 3, 3, 0], [0, 0, 3, 0, 0],
                                     [0, 0, 0, 0, 4]]),
                     ("16x16", [[(i * 3 + j) % 10 for j in range(16)]
                                for i in range(16)]),
                     ("30x30", [[(i + j) % 10 for j in range(30)]
                                for i in range(30)])):
        rows, cols = len(g), len(g[0])
        comps = ConnectedComponentSegmenter(background_color=0).segment_grid(g)
        px_lists = [c.pixels for c in comps]
        total_px = sum(len(p) for p in px_lists)

        print(f"\n--- {gname}  ({rows}x{cols})  components={len(comps)}  "
              f"contour pixels total={total_px} ---")

        # correctness first: all variants must agree with legacy
        bad = []
        for ci, px in enumerate(px_lists):
            ref = parity_legacy(rows, cols, px)
            for nm, fn in (("pylist", parity_pylist),
                           ("scipy", parity_scipy if have_scipy else None)):
                if fn is None:
                    continue
                got = fn(rows, cols, px)
                if got != ref:
                    bad.append((ci, nm))
        print(f"  equivalence: {'ALL MATCH' if not bad else f'{len(bad)} MISMATCH {bad[:4]}'}")

        # timing over the REAL per-component pattern
        variants = [("legacy", lambda px: parity_legacy(rows, cols, px)),
                    ("deque ", lambda px: ParityContourMask._parity_contour_fast(rows, cols, px)),
                    ("pylist", lambda px: parity_pylist(rows, cols, px))]
        if have_scipy:
            variants.append(("scipy ", lambda px: parity_scipy(rows, cols, px)))

        reps = max(1, 20 // max(1, len(comps) // 10))
        base = None
        for nm, fn in variants:
            def run_all():
                for px in px_lists:
                    fn(px)
            us = timeit(run_all, reps)
            if base is None:
                base = us
            print(f"  {nm}  {us:10.1f} us / full segment   speedup {base/max(us,1e-9):6.2f}x")

        # isolate ONE parity call to see fixed overhead
        px0 = px_lists[0] if px_lists else [(0, 0)]
        for nm, fn in variants:
            us = timeit(lambda: fn(px0), 200)
            print(f"    one call ({len(px0):4d} px): {nm} {us:8.2f} us")

    print("\n" + "=" * 78)
    print("READING: if `pylist` >> `deque`, then C1 (numpy scalar indexing) is the")
    print("cost and the fix is to move the BFS off numpy scalars.")
    print("SEG_COST_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())