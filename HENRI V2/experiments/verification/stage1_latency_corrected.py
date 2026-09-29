"""Corrected latency profile: separate REAL GPU kernel time from Python gaps.

MY INSTRUMENT DEFECT (caught by reading my own output)
======================================================
lat_profile.py classified every row "GPU-BOUND" using
    verdict = "GPU-BOUND" if gpu_us > 0.5 * wall_us
where gpu_us came from cuda Events recorded AROUND THE WHOLE LOOP. Elapsed-time
events measure wall clock on the GPU timeline, INCLUDING gaps where the GPU is idle
waiting for Python. So gpu_us ~= wall_us is guaranteed by construction for any
grid, and the verdict column carries no information. It is retracted.

This replaces it with torch.profiler, which reports summed KERNEL durations. Then:
    gap_us = wall_us - kernel_us   ->  Python/launch overhead
Measured, not inferred.
"""
import json
import os
import sys
import time

os.environ.setdefault("HENRI_UNIFIED_VLA", "1")
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402


def census(grid):
    """Non-background pixel count -- tests the per-pixel-cost hypothesis."""
    flat = [v for row in grid for v in row]
    nz = sum(1 for v in flat if v != 0)
    return len(flat), nz


def main() -> int:
    print("=" * 76)
    print("CORRECTED LATENCY DECOMPOSITION (torch.profiler kernel time)")
    print("=" * 76)
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))

    tok = HENRIVisionEncoder(d_model=65536, k_blocks=8192, device="cuda",
                             spatial_basis_kind="incommensurate", bg_mask=True)

    grids = {
        "4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        "16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
        "30x30": [[(i + j) % 10 for j in range(30)] for i in range(30)],
    }

    from torch.profiler import ProfilerActivity, profile

    out = {}
    print(f"\n{'grid':7s} {'cells':>6s} {'nonzero':>8s} {'wall_us':>11s} "
          f"{'kernel_us':>11s} {'gap_us':>10s} {'gap%':>6s} {'kernels':>8s}")
    for name, g in grids.items():
        cells, nz = census(g)
        for _ in range(5):
            tok.encode_spatial_grid(g)
        torch.cuda.synchronize()
        REPS = 20
        t0 = time.perf_counter()
        for _ in range(REPS):
            tok.encode_spatial_grid(g)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / REPS * 1e6

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(REPS):
                tok.encode_spatial_grid(g)
            torch.cuda.synchronize()
        ka = prof.key_averages()
        kern_us = sum(getattr(e, "self_device_time_total", 0.0) for e in ka) / REPS
        n_kern = sum(getattr(e, "count", 0) for e in ka) / REPS
        gap = wall - kern_us
        pct = 100.0 * gap / wall if wall else 0.0
        verdict = ("PYTHON/LAUNCH-BOUND" if pct > 50 else
                   "GPU-ARITHMETIC-BOUND" if pct < 20 else "mixed")
        print(f"{name:7s} {cells:6d} {nz:8d} {wall:11.2f} {kern_us:11.2f} "
              f"{gap:10.2f} {pct:5.1f}% {n_kern:8.1f}  {verdict}")
        out[name] = {"cells": cells, "nonzero": nz, "wall_us": round(wall, 2),
                     "kernel_us": round(kern_us, 2), "gap_us": round(gap, 2),
                     "gap_pct": round(pct, 1), "kernels_per_call": round(n_kern, 1),
                     "verdict": verdict,
                     "top_kernels": [{"k": e.key[:52],
                                      "dev_us": round(getattr(e, "self_device_time_total", 0.0) / REPS, 2)}
                                     for e in ka][:6]}

    print("\n" + "=" * 76)
    print("TOP KERNELS / OPS (4x4) -- what actually costs the time")
    print("=" * 76)
    for row in out["4x4"]["top_kernels"]:
        print(f"  {row['dev_us']:9.2f} us  {row['k']}")

    print("\n" + "=" * 76)
    print("SPEC GATES (with the corrected numbers)")
    print("=" * 76)
    w4 = out["4x4"]["wall_us"]
    print(f"  spec clause (c): step latency <= 15 us")
    print(f"    measured 4x4  wall {w4:10.2f} us  -> ratio to gate {w4/15:7.1f}x")
    print(f"    spec 4.1 also states Tier-1 cadence = 20 kHz = 50 us period")
    print(f"    measured implies cadence {(1e6/w4)/1000:8.2f} kHz at 4x4")
    cores = [n for n in ("16x16", "30x30") if n in out]
    for n in cores:
        print(f"    measured {n} wall {out[n]['wall_us']:10.2f} us "
              f"-> {(1e6/out[n]['wall_us'])/1000:7.3f} kHz")
    print(f"  GATE_15us_4x4 = {w4 <= 15}")

    with open("/tmp/lat_corrected.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nWROTE /tmp/lat_corrected.json")
    print("LAT_CORRECTED_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())