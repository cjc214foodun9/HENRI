"""Phase-split of encode_grid: WHERE do the 112 ms actually go?

MY HYPOTHESIS WAS WRONG, MEASURED
=================================
I claimed the Stage-1 latency failure was per-cell kernel-launch overhead.
`vectorized_accum=True` cut launches on a 16x16 grid from 3834.3 to 49.1 per call
(78x fewer) and wall time moved only 1.09x (122331 -> 112039 us). Dispatch is
therefore NOT the dominant cost. This probe splits the real path into phases to
find what is.

encode_grid phases (from henri_vision_encoder.py):
  P1 grid -> device tensor (long) + clamp
  P2 zeros accumulator allocation
  P3 grid.cpu().numpy()                    <- device->host sync
  P4 ConnectedComponentSegmenter.segment   <- numpy/Python, per grid
  P5 ParityContourMask per component       <- numpy/Python, per component
  P6 parity_mask -> device tensor          <- host->device
  P7 superposition (legacy or vectorized)
  P8 cat + L2 normalize

Reported per grid, with component count, so the scaling law is visible.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from connected_component_segmenter import (  # noqa: E402
    ConnectedComponentSegmenter,
    ParityContourMask,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402

D, K = 65536, 8192
GRIDS = {
    "4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]],
    "8x8": [[(i * 3 + j) % 4 for j in range(8)] for i in range(8)],
    "16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
    "30x30": [[(i + j) % 10 for j in range(30)] for i in range(30)],
}
REPS = 10


def t(fn, reps=REPS):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


def main() -> int:
    dev = "cuda"
    enc = HENRIVisionEncoder(d_model=D, k_blocks=K, device=dev,
                             spatial_basis_kind="incommensurate", bg_mask=True,
                             vectorized_accum=True)
    print("=" * 80)
    print("PHASE SPLIT of encode_grid (D=%d) on %s" % (D, torch.cuda.get_device_name(0)))
    print("=" * 80)
    out = {}
    hdr = (f"\n{'grid':6s} {'HxW':>6s} {'comp':>5s} "
           + "".join(f"{p:>9s}" for p in
                     ("P1", "P3", "P4seg", "P5par", "P6", "P7sup", "P8norm", "TOTAL")))
    print(hdr)
    for gname, g in GRIDS.items():
        H, W = len(g), len(g[0])
        gt = torch.tensor(g, dtype=torch.long, device=dev)

        # component count (for the scaling law)
        gnp = gt.cpu().numpy()
        comps = ConnectedComponentSegmenter(background_color=0).segment_grid(gnp)
        ncomp = len(comps)

        p1 = t(lambda: torch.clamp(gt, 0, 15))
        p3 = t(lambda: torch.clamp(gt, 0, 15).cpu().numpy())

        def p4fn():
            return ConnectedComponentSegmenter(background_color=0).segment_grid(
                torch.clamp(gt, 0, 15).cpu().numpy())
        p4 = t(p4fn)

        def p5fn():
            gc = torch.clamp(gt, 0, 15).cpu().numpy()
            seg = ConnectedComponentSegmenter(background_color=0)
            cs = seg.segment_grid(gc)
            grid = np.ones((H, W), dtype=np.float32)
            for c in cs:
                ip, _ = ParityContourMask.compute_parity_contour((H, W), c.pixels)
                for r_i, c_i in ip:
                    grid[r_i, c_i] = -1.0
            return grid
        p4_5 = t(p5fn)
        p5 = max(p4_5 - p4, 0.0)

        def p6fn():
            return torch.tensor(np.ones((H, W), dtype=np.float32), device=dev)
        p6 = t(p6fn)

        pm = torch.ones((H, W), dtype=torch.float32, device=dev)
        gc = torch.clamp(gt, 0, 15)
        p7 = t(lambda: enc._superpose_vectorized(gc, pm, H, W))

        def p8fn():
            a = torch.zeros(D // 2, dtype=torch.complex64, device=dev)
            w = torch.cat([a.real, a.imag], dim=-1)
            return F.normalize(w, p=2, dim=-1)
        p8 = t(p8fn)

        total = t(lambda: enc.encode_spatial_grid(g))
        rec = {"H": H, "W": W, "components": ncomp,
               "P1_grid_to_dev": round(p1, 2), "P3_dev_to_host": round(p3, 2),
               "P4_segment": round(p4, 2), "P5_parity_mask": round(p5, 2),
               "P6_host_to_dev": round(p6, 2), "P7_superpose": round(p7, 2),
               "P8_norm": round(p8, 2), "TOTAL_measured": round(total, 2),
               "P4_5_python_block": round(p4_5, 2)}
        out[gname] = rec
        print(f"{gname:6s} {H}x{W:<4d} {ncomp:5d} "
              + "".join(f"{v:9.1f}" for v in
                        (p1, p3, p4, p5, p6, p7, p8, total)))

    # ---- the scaling law: is the Python segmentation block the dominant term? --
    print("\n" + "=" * 80)
    print("ATTRIBUTION")
    print("=" * 80)
    for gname, r in out.items():
        tot = max(r["TOTAL_measured"], 1e-9)
        py = r["P4_5_python_block"]
        print(f"  {gname:6s} total {tot:10.1f} us | numpy+Python segmentation block "
              f"{py:10.1f} us = {100*py/tot:5.1f}% | superpose {r['P7_superpose']:8.1f} us "
              f"= {100*r['P7_superpose']/tot:4.1f}%")

    # ---- the gate question with a MEASURED bandwidth floor --------------------
    print("\n" + "=" * 80)
    print("IS THE SPEC'S 15 us GATE REACHABLE AT D=65536?")
    print("=" * 80)
    for n_cells, label in ((16, "4x4"), (256, "16x16")):
        terms = torch.randn(n_cells, D // 2, dtype=torch.complex64, device="cuda")
        for _ in range(3):
            terms.sum(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            terms.sum(0)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / 50 * 1e6
        gb = n_cells * (D // 2) * 8 / (us * 1e-6) / 1e9
        print(f"  {label:6s} one reduction pass over [{n_cells},{D//2}] complex64: "
              f"{us:8.2f} us  ({gb:7.1f} GB/s)")
    print("  NOTE: this is a MEASURED floor for touching the term tensor once.")
    print("        D=65536 complex64 = 0.52 MB; any honest implementation must")
    print("        materialise and reduce it.")

    with open("/tmp/phase_split.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nWROTE /tmp/phase_split.json")
    print("PHASE_SPLIT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())