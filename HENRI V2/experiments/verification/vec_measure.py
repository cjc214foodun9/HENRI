"""Measure the vectorized accumulator on the 5090: launches, wall, kernel, speedup.

Also settles the question the spec's 15 us gate raises: the legacy 4x4 path showed
wall 498.37 us with kernel floor 103.75 us. Even at ZERO Python overhead the gate
is unreachable, so this measures BOTH:
  (a) the speedup the vectorized path actually achieves, and
  (b) the pure-kernel bandwidth floor, to state the ceiling honestly.

A memory-bandwidth floor is included because it is the physical limit: the
superposition must at minimum touch its accumulators and read the per-cell terms.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402

D, K = 65536, 8192
GRIDS = {
    "4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]],
    "16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
    "30x30": [[(i + j) % 10 for j in range(30)] for i in range(30)],
}


def profile_call(fn, grid, reps=20):
    for _ in range(5):
        fn(grid)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(grid)
    cpu = (time.perf_counter() - t0) / reps * 1e6
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / reps * 1e6
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn(grid)
        torch.cuda.synchronize()
    ka = prof.key_averages()
    kern = sum(getattr(e, "self_device_time_total", 0.0) for e in ka) / reps
    nk = sum(getattr(e, "count", 0) for e in ka) / reps
    return round(cpu, 2), round(wall, 2), round(kern, 2), round(nk, 1)


def main() -> int:
    print("=" * 78)
    print("VECTORIZED ACCUMULATOR: measured on", torch.cuda.get_device_name(0))
    print("=" * 78)
    out = {}
    print(f"\n{'grid':6s} {'path':11s} {'cpu_us':>10s} {'wall_us':>11s} "
          f"{'kernel_us':>10s} {'launch/call':>12s} {'speedup':>9s}")
    for gname, g in GRIDS.items():
        rec = {}
        for label, vec in (("legacy", False), ("vectorized", True)):
            e = HENRIVisionEncoder(d_model=D, k_blocks=K, device="cuda",
                                   spatial_basis_kind="incommensurate",
                                   bg_mask=True, vectorized_accum=vec)
            try:
                cpu, wall, kern, nk = profile_call(e.encode_spatial_grid, g)
            except Exception as exc:                          # noqa: BLE001
                print(f"{gname:6s} {label:11s}  ERROR {type(exc).__name__}: "
                      f"{str(exc)[:44]}")
                rec[label] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
                continue
            rec[label] = {"cpu_us": cpu, "wall_us": wall, "kernel_us": kern,
                          "launches_per_call": nk}
            sp = ""
            if "legacy" in rec and "wall_us" in rec.get("legacy", {}):
                sp = f"{rec['legacy']['wall_us']/max(wall,1e-9):7.2f}x"
            print(f"{gname:6s} {label:11s} {cpu:10.2f} {wall:11.2f} "
                  f"{kern:10.2f} {nk:12.1f} {sp:>9s}")
        out[gname] = rec

    # ---- the physical ceiling: is the 15 us gate reachable at all? ----------
    print("\n" + "=" * 78)
    print("THE CEILING QUESTION: can <= 15 us be reached on this design?")
    print("=" * 78)
    g = torch.randn(16, 16, D, dtype=torch.complex64, device="cuda")
    # measure achievable elementwise throughput on this GPU
    for _ in range(3):
        s = g.sum(dim=(0, 1))
    torch.cuda.synchronize()
    rep = 50
    t0 = time.perf_counter()
    for _ in range(rep):
        s = g.sum(dim=(0, 1))
    torch.cuda.synchronize()
    reduce_us = (time.perf_counter() - t0) / rep * 1e6
    elems = 16 * 16 * (D // 2)
    bytes_moved = elems * 8 * 2          # complex64 read + write-ish
    bw = bytes_moved / (reduce_us * 1e-6) / 1e9
    print(f"  16x16 sum over [16,16,{D//2}] complex64:")
    print(f"    measured {reduce_us:.2f} us   elements {elems:,}")
    print(f"    implied bandwidth {bw:.1f} GB/s")
    print(f"  -> a single pass over the terms alone costs {reduce_us:.2f} us,")
    print(f"     so a 15 us budget for the FULL encode is "
          f"{'PLAUSIBLE' if reduce_us < 15 else 'NOT PLAUSIBLE'} at this D.")
    print(f"     D={D} = {D*4/1e6:.2f} MB per complex64 wave; "
          f"{len(GRIDS['4x4'])*len(GRIDS['4x4'][0])} cells at 4x4.")

    with open("/tmp/vec_measure.json", "w") as fh:
        json.dump({"grids": out, "sum_pass_us": round(reduce_us, 2),
                   "implied_gb_s": round(bw, 1)}, fh, indent=1)
    print("\nWROTE /tmp/vec_measure.json")
    print("VEC_MEASURE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())