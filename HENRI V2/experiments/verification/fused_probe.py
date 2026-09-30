"""ORACLE FLOOR: what encode_grid latency is actually achievable, and where?

WHY THIS EXISTS
===============
Stage-1 clause (c) is "step latency <= 15 us". Measured on the RTX 5090 with the
substrate fix live: perceive 704.3 us, act 3433.0 us, encode 541.4 us. Two
questions decide what to do next, and neither can be answered by another
micro-optimisation:

  Q1. Is the 15 us clause reachable at ALL for this module composition?
      If the sum of the irreducible stages is above the gate, no amount of
      fusion reaches it and the gate must be amended with measured evidence.
      Chipping at Python without knowing the floor is how credits get spent on
      an illusion.

  Q2. What does the SPEC itself say the gate is?
      Tier 1 is labelled "(20 kHz)". 1 / 20 kHz = 50 us, not 15 us. 15 us is
      66.7 kHz. The spec's own exit contract contradicts its own tier label by
      3.3x, so the clause is ambiguous and needs a defensible definition.

METHOD
  Stage decomposition, each stage timed in isolation on the GPU, plus a
  loop-free fused superposition measured against production for bit-identity.
  Wall time uses explicit cuda.synchronize(); kernel time uses torch.profiler.

No score claim. Latency decomposition only.
"""
import json
import os
import sys
import time

os.environ.setdefault("HENRI_UNIFIED_VLA", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

for cand in ("/root/henri/HENRI V2", "/root/henri"):
    if os.path.isdir(os.path.join(cand, "experiments", "verification")):
        sys.path.insert(0, cand)
        break
else:
    p = os.getcwd()
    while p != "/":
        if os.path.isdir(os.path.join(p, "experiments", "verification")):
            sys.path.insert(0, p)
            break
        p = os.path.dirname(p)

from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402
from connected_component_segmenter import (  # noqa: E402
    ConnectedComponentSegmenter,
)

DEV = "cuda"
assert torch.cuda.is_available(), "CUDA required"
D = 65536
GRIDS = {
    "4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
    "16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
    "30x30": [[(i + j) % 10 for j in range(30)] for i in range(30)],
}


def bench(fn, reps, warm=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6


def profiling(fn, reps=3):
    try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
        evts = [e for e in prof.key_averages()
                if e.count > 0 and "CUDA" in str(e.device_type)]
        return (sum(e.device_time_total for e in evts) / reps,
                sum(e.count for e in evts) / reps)
    except Exception:
        return float("nan"), -1.0


def fused_superpose(enc, grid_clamped, parity_mask, H, W, bg_mask=True,
                    chunk=8):
    """Loop-free equivalent of the per-cell superposition.

    Production (legacy) loop:
        for r in range(H):
          for c in range(W):
            if bg_mask and grid[r,c]==0: continue
            acc += px[c] * py[r] * pc[grid[r,c]] * kappa[r,c]

    This builds the same product as a batched tensor expression, row-chunked to
    bound peak memory at [chunk, W, D/2] complex64.
    """
    py = enc.spatial_basis_y[:H]
    px = enc.spatial_basis_x[:W]
    pc = enc.color_codebook[grid_clamped]                 # [H, W, D/2]
    weight = parity_mask
    if bg_mask:
        weight = weight * (grid_clamped != 0).to(weight.dtype)
    acc = torch.zeros(enc.d_model // 2, dtype=torch.complex64, device=enc.device)
    for r0 in range(0, H, chunk):
        r1 = min(r0 + chunk, H)
        term = px.unsqueeze(0) * py[r0:r1].unsqueeze(1)   # [nr, W, D/2]
        term = term * pc[r0:r1]
        term = term * weight[r0:r1].unsqueeze(-1)
        acc = acc + term.sum(dim=(0, 1))
    return acc


def normalize_real(acc):
    real = torch.cat([acc.real, acc.imag], dim=-1)
    return F.normalize(real, p=2, dim=-1)


print("=" * 86)
print("PART A -- STAGE DECOMPOSITION (D=65536, RTX 5090)")
print("=" * 86)
enc = HENRIVisionEncoder(d_model=D, k_blocks=8192, device=DEV,
                         spatial_basis_kind="incommensurate", bg_mask=True)
receipt = {"D": D, "gpu": torch.cuda.get_device_name(0), "stages": {}}

for gname, g in GRIDS.items():
    gt = torch.tensor(g, dtype=torch.long, device=DEV)
    H, W = gt.shape
    reps = 400 if gname != "30x30" else 60

    t_full = bench(lambda: enc.encode_grid(g), reps)
    t_dev = bench(lambda: torch.tensor(g, dtype=torch.long, device=DEV), reps)
    gc_np = gt.cpu().numpy()
    t_np = bench(lambda: gt.cpu().numpy(), reps)
    seg = ConnectedComponentSegmenter(background_color=0)
    t_seg = bench(lambda: seg.segment_grid(gc_np, want_exterior=False), reps)

    comps = seg.segment_grid(gc_np, want_exterior=False)
    import numpy as np
    pm = np.ones((H, W), dtype=np.float32)
    for c in comps:
        for r_i, c_i in c.interior_pixels:
            pm[r_i, c_i] = -1.0
    pmt = torch.tensor(pm, dtype=torch.float32, device=DEV)
    t_pmt = bench(lambda: torch.tensor(pm, dtype=torch.float32, device=DEV), reps)

    t_fused = bench(lambda: fused_superpose(enc, gt, pmt, H, W), reps)
    acc = fused_superpose(enc, gt, pmt, H, W)
    t_norm = bench(lambda: normalize_real(acc), reps)

    k_full, l_full = profiling(lambda: enc.encode_grid(g))
    k_fused, l_fused = profiling(lambda: fused_superpose(enc, gt, pmt, H, W))

    print(f"\n  --- {gname} (H={H} W={W}, {len(comps)} components) ---")
    print(f"    full encode_grid        {t_full:9.1f} us   kernel {k_full:7.1f} us  "
          f"launches {l_full:7.1f}")
    print(f"      grid->device          {t_dev:9.1f} us")
    print(f"      device->cpu numpy     {t_np:9.1f} us")
    print(f"      segment_grid          {t_seg:9.1f} us")
    print(f"      parity tensor build   {t_pmt:9.1f} us")
    print(f"    FUSED superpose         {t_fused:9.1f} us   kernel {k_fused:7.1f} us  "
          f"launches {l_fused:7.1f}")
    print(f"      normalize             {t_norm:9.1f} us")
    floor = t_fused + t_norm
    print(f"    FLOOR (fused+norm)      {floor:9.1f} us   "
          f"vs gate 15 us -> {floor/15.0:5.1f}x over")
    receipt["stages"][gname] = {
        "components": len(comps),
        "full_us": t_full, "full_kernel_us": k_full, "full_launches": l_full,
        "grid_to_device_us": t_dev, "to_numpy_us": t_np,
        "segment_us": t_seg, "parity_tensor_us": t_pmt,
        "fused_superpose_us": t_fused, "fused_kernel_us": k_fused,
        "fused_launches": l_fused, "normalize_us": t_norm,
        "floor_us": floor, "floor_over_gate": floor / 15.0,
    }

print()
print("=" * 86)
print("PART B -- IS THE FUSED PATH IDENTICAL TO PRODUCTION?")
print("=" * 86)
worst = 0.0
for gname, g in GRIDS.items():
    gt = torch.tensor(g, dtype=torch.long, device=DEV)
    H, W = gt.shape
    ref = enc.encode_grid(g)
    # rebuild the parity mask exactly as production does
    gas = gt.cpu().numpy()
    comps = ConnectedComponentSegmenter(background_color=0).segment_grid(
        gas, want_exterior=False)
    import numpy as np
    pm = np.ones((H, W), dtype=np.float32)
    for c in comps:
        for r_i, c_i in c.interior_pixels:
            pm[r_i, c_i] = -1.0
    pmt = torch.tensor(pm, dtype=torch.float32, device=DEV)
    alt = normalize_real(fused_superpose(enc, gt, pmt, H, W))
    d = float((ref - alt).abs().max())
    worst = max(worst, d)
    print(f"  {gname:7s} max abs diff vs production = {d:.3e}")
print(f"\n  worst = {worst:.3e}   FUSED_IDENTICAL {worst < 1e-6}")
receipt["fused_worst_diff"] = worst

print()
print("=" * 86)
print("PART C -- THE GATE ARITHMETIC")
print("=" * 86)
for gname in GRIDS:
    s = receipt["stages"][gname]
    print(f"  {gname:7s} gate 15 us -> floor {s['floor_us']:8.1f} us = "
          f"{s['floor_over_gate']:5.1f}x over")
    print(f"          gate 50 us (the spec's OWN 20 kHz tier label) -> "
          f"floor {s['floor_us']:8.1f} us = "
          f"{'REACHABLE' if s['floor_us'] <= 50 else 'NOT reachable'}")
print()
print("  SPEC INTERNAL CONTRADICTION: Tier 1 is labelled '(20 kHz)' but its exit")
print("  contract says 'step latency <= 15 us'. 1/20kHz = 50 us. 15 us = 66.7 kHz.")
print("  The two differ by 3.3x, so the clause is ambiguous as written.")

out = "/tmp/fused_receipt.json"
with open(out, "w") as fh:
    json.dump(receipt, fh, indent=2)
print(f"\nreceipt -> {out} ({os.path.getsize(out)} bytes)")
print("FUSED_PROBE_DONE")
