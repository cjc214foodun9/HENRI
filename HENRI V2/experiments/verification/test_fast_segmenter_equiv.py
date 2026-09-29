"""Prove _parity_contour_fast is ELEMENT-FOR-ELEMENT identical to the legacy path.

WHY THIS TEST EXISTS
====================
`compute_parity_contour` is 97.8% of encode_grid wall time at 16x16
(experiments/verification/phase_split_receipt.json: 109634.7 of 112061.3 us).
It contains two quadratic costs:

    (1) `queue.pop(0)` on a list -> O(n) per dequeue, so BFS is O(n^2)
    (2) `(r, c) not in contour_pixels` -> O(len) list scan inside a rows*cols loop

The fast path uses a deque and a set, and vectorises the extraction with
np.argwhere. A faster-but-different function would silently change the encoder's
output, so this asserts IDENTITY of both lists in order and content, not
similarity. Randomised fuzzing is included because the failure modes (BFS visit
order, duplicate contour entries, out-of-bounds entries) only appear on inputs
that hand-written cases miss.
"""
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

from connected_component_segmenter import (  # noqa: E402
    ConnectedComponentSegmenter,
    ParityContourMask,
)


def check(shape, pixels, label, fails):
    a_int, a_ext = ParityContourMask.compute_parity_contour(shape, pixels)
    b_int, b_ext = ParityContourMask._parity_contour_fast(shape[0], shape[1], pixels)
    ok = (a_int == b_int) and (a_ext == b_ext)
    if not ok:
        fails.append(label)
        print(f"  FAIL {label}: legacy(interior={len(a_int)},exterior={len(a_ext)}) "
              f"fast(interior={len(b_int)},exterior={len(b_ext)})")
        if a_int != b_int:
            for i, (x, y) in enumerate(zip(a_int, b_int)):
                if x != y:
                    print(f"        interior first diff at {i}: {x} vs {y}")
                    break
        if a_ext != b_ext:
            for i, (x, y) in enumerate(zip(a_ext, b_ext)):
                if x != y:
                    print(f"        exterior first diff at {i}: {x} vs {y}")
                    break
    return ok


def main() -> int:
    print("=" * 78)
    print("FAST SEGMENTER EQUIVALENCE (element-for-element, order included)")
    print("=" * 78)
    fails = []

    # ---- 1. hand-built shapes that exercise each mech branch ----------------
    print("\n[1] structured shapes")
    cases = {
        "empty_1x1": ((1, 1), []),
        "single_pixel": ((3, 3), [(1, 1)]),
        "horizontal_line": ((3, 5), [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]),
        "vertical_line": ((5, 3), [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]),
        "full_grid": ((3, 3), [(r, c) for r in range(3) for c in range(3)]),
        "closed_ring_5x5": ((5, 5), [(0, c) for c in range(5)] + [(4, c) for c in range(5)]
                            + [(r, 0) for r in range(5)] + [(r, 4) for r in range(5)]),
        "ring_touching_corner": ((4, 4), [(0, 0), (0, 1), (0, 2), (0, 3),
                                          (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3)]),
        "diagonal": ((4, 4), [(i, i) for i in range(4)]),
        "out_of_bounds": ((3, 3), [(0, 0), (5, 5), (-1, 2), (1, 1)]),
        "duplicate_entries": ((3, 3), [(1, 1), (1, 1), (1, 1), (0, 0)]),
        "empty_contour_big_grid": ((16, 16), []),
        "single_cell_16x16": ((16, 16), [(7, 8)]),
    }
    for label, (shape, px) in cases.items():
        ok = check(shape, px, label, fails)
        print(f"  {'PASS' if ok else 'FAIL'}  {label:24s} pixels={len(px)}")

    # ---- 2. segment_grid: whole-object records must match too ---------------
    print("\n[2] segment_grid on real grid patterns")
    grids = {
        "mixed_5x5": [[1, 1, 0, 0, 0], [1, 2, 2, 0, 0], [0, 2, 3, 3, 0],
                      [0, 0, 3, 0, 0], [0, 0, 0, 0, 4]],
        "ring_5x5": [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]],
        "checker_4x4": [[(i + j) % 2 for j in range(4)] for i in range(4)],
        "solid_4x4": [[2] * 4 for _ in range(4)],
    }
    for label, g in grids.items():
        seg = ConnectedComponentSegmenter(background_color=0)
        recs = seg.segment_grid(g)
        shapes = len(g), len(g[0])
        allok = True
        for rec in recs:
            allok &= check(shapes, rec.pixels, f"{label}/obj{rec.object_id}", fails)
        print(f"  {'PASS' if allok else 'FAIL'}  {label:24s} objects={len(recs)}")

    # ---- 3. randomised fuzz -------------------------------------------------
    print("\n[3] randomised fuzz (200 cases, random shapes/contours)")
    rng = random.Random(1234)
    fuzz_ok = 0
    for i in range(200):
        rows = rng.randint(1, 12)
        cols = rng.randint(1, 12)
        n = rng.randint(0, rows * cols)
        px = [(rng.randrange(rows), rng.randrange(cols)) for _ in range(n)]
        if check((rows, cols), px, f"fuzz{i}", fails):
            fuzz_ok += 1
    print(f"  {fuzz_ok}/200 identical")

    # ---- 4. timing sanity on CPU (speed, not correctness) -------------------
    print("\n[4] CPU timing (correctness is [1]-[3]; this is the speed claim)")
    import time
    g = [[(i * 7 + j * 3) % 5 for j in range(16)] for i in range(16)]
    seg = ConnectedComponentSegmenter(background_color=0)
    recs = seg.segment_grid(g)
    px = sum((r.pixels for r in recs), [])
    t0 = time.perf_counter()
    for _ in range(5):
        ParityContourMask.compute_parity_contour((16, 16), px)
    legacy_us = (time.perf_counter() - t0) / 5 * 1e6
    t0 = time.perf_counter()
    for _ in range(5):
        ParityContourMask._parity_contour_fast(16, 16, px)
    fast_us = (time.perf_counter() - t0) / 5 * 1e6
    print(f"  legacy {legacy_us:10.1f} us   fast {fast_us:10.1f} us   "
          f"speedup {legacy_us/max(fast_us,1e-9):6.2f}x   (contour px={len(px)})")

    print("\n" + "=" * 78)
    if fails:
        print(f"RESULT: FAIL ({len(fails)} mismatches)")
        for f in fails[:10]:
            print("   ", f)
        print("IDENTICAL False")
        return 1
    print("RESULT: PASS")
    print("IDENTICAL True")
    return 0


if __name__ == "__main__":
    sys.exit(main())