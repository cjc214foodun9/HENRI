"""Localize the 652.62 us 1-step path: Python/dispatch overhead vs GPU arithmetic.

WHY THIS EXISTS
===============
Stage 1's exit gate has three clauses (spec 4.1). Measured on the RTX 5090:
  (a) smoke passage          PASS   UNIFIED_VLA_CUDA_SMOKE_PASS
  (b) ||Psi|| = 1 +- 1e-5    PASS   norm 1.0 exactly
  (c) step latency <= 15 us  FAIL   652.62 us perceive / 2840.76 us act

Before reporting (c) as a blocker, split it. If wall ~= CPU submit time, the cost is
Python/dispatch/segmentation overhead and the gate is reachable by caching or CUDA
graphs. If wall >> CPU submit, the GPU arithmetic itself is the cost and the gate
requires new kernels (the spec's Tier 0). Measured, not guessed.
"""
import os
import sys
import time

os.environ.setdefault("HENRI_UNIFIED_VLA", "1")
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402


def make(bg_mask: bool) -> "HENRIVisionEncoder":
    return HENRIVisionEncoder(d_model=65536, k_blocks=8192, device="cuda",
                              spatial_basis_kind="incommensurate", bg_mask=bg_mask)


def split_us(fn, reps: int = 100, warm: int = 10):
    """Return (cpu_submit_us, wall_synced_us, gpu_timeline_us)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t_cpu = (time.perf_counter() - t0) / reps * 1e6        # CPU hands back control
    torch.cuda.synchronize()
    t_wall = (time.perf_counter() - t0) / reps * 1e6       # GPU drained
    # pure GPU timeline via events
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    t_gpu = s.elapsed_time(e) / reps * 1000.0              # ms -> us
    return round(t_cpu, 2), round(t_wall, 2), round(t_gpu, 2)


def main() -> int:
    print("=" * 74)
    print("LATENCY DECOMPOSITION -- where do the 652.62 us go?")
    print("=" * 74)
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    print()

    grids = {
        "4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        "16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
        "30x30": [[(i + j) % 10 for j in range(30)] for i in range(30)],
    }

    print(f"{'config':12s} {'grid':7s} {'cpu_submit_us':>13s} {'wall_us':>9s} "
          f"{'gpu_us':>8s}  verdict")
    results = {}
    for tag, bg in (("bg=True", True), ("bg=False", False)):
        tok = make(bg)
        for gname, g in grids.items():
            try:
                c, w, gp = split_us(lambda t=tok, gg=g: t.encode_spatial_grid(gg))
            except Exception as exc:                       # noqa: BLE001
                print(f"{tag:12s} {gname:7s}  ERROR {type(exc).__name__}: {str(exc)[:40]}")
                continue
            if gp > 0.5 * w:
                verdict = "GPU-BOUND"
            elif c > 0.5 * w:
                verdict = "CPU/PYTHON-BOUND"
            else:
                verdict = "mixed"
            print(f"{tag:12s} {gname:7s} {c:13.2f} {w:9.2f} {gp:8.2f}  {verdict}")
            results[f"{tag}:{gname}"] = {"cpu_us": c, "wall_us": w, "gpu_us": gp,
                                         "verdict": verdict}
        print()

    # ---- is the cost PER-CALL or first-call (i.e. cacheable)? ----------------
    print("=" * 74)
    print("CACHEABILITY: 1st call vs steady state (bg=True, 16x16)")
    print("=" * 74)
    tok = make(True)
    g = grids["16x16"]
    t0 = time.perf_counter()
    tok.encode_spatial_grid(g)
    torch.cuda.synchronize()
    first = (time.perf_counter() - t0) * 1e6
    steady = []
    for _ in range(30):
        t0 = time.perf_counter()
        tok.encode_spatial_grid(g)
        torch.cuda.synchronize()
        steady.append((time.perf_counter() - t0) * 1e6)
    steady.sort()
    print(f"  first_call_us      {first:10.2f}")
    print(f"  steady_min_us      {steady[0]:10.2f}")
    print(f"  steady_median_us   {steady[len(steady)//2]:10.2f}")
    print(f"  first/steady ratio {first/max(steady[0],1e-9):10.2f}")

    # ---- does the encoder expose cacheable basis tables? --------------------
    print()
    print("=" * 74)
    print("ENCODER INTERNALS (candidate cache points)")
    print("=" * 74)
    for attr in ("d_model", "k_blocks", "device", "spatial_basis_kind", "bg_mask"):
        print(f"  {attr:22s} {getattr(tok, attr, '<absent>')}")
    tbl = [a for a in dir(tok) if any(k in a.lower() for k in
           ("basis", "freq", "cache", "table", "mask", "coord", "segment"))]
    print(f"  cache-ish attrs ({len(tbl)}): {', '.join(tbl[:14])}")
    seg = [a for a in dir(tok) if "segment" in a.lower() or "mask" in a.lower()]
    print(f"  segmentation attrs: {seg}")

    import json
    out = os.environ.get("LAT_OUT", "/tmp/lat_profile.json")
    with open(out, "w") as fh:
        json.dump({"results": results, "first_call_us": first,
                   "steady_min_us": steady[0], "steady_median_us":
                   steady[len(steady)//2]}, fh, indent=1)
    print(f"\nWROTE {out}")
    print("LAT_PROFILE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())