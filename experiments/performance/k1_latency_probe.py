"""K1 probe: inner-loop latency invariance at D=65536 (CUDA, RTX 5090).

Pre-registered gate (design doc a2886ec):
  ACCEPT: mean t_step <= 50 us AND p99 <= 2x mean AND no positive regression
          slope over steps 500-1000 (p >= 0.05).
  REJECT: any bound fails -> harness demoted to measured Hz.

Runs N=1000 inner steps (WaveJEPA predict + dual-channel Sagnac veto).
Scheduling rule: must run ALONE on the GPU (no concurrent production runs).

Usage: python k1_latency_probe.py --steps 1000 --d 65536 --blocks 8192 --rank 16
Output: JSON telemetry to stdout + file.
"""

import argparse
import json
import time

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--d", type=int, default=65536)
    ap.add_argument("--blocks", type=int, default=8192)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", type=str, default="/tmp/k1_latency.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "K1 requires CUDA"
    torch.cuda.init()
    dev = "cuda"

    from henri_dual_speed_harness import HENRIDualSpeedHarness

    h = HENRIDualSpeedHarness(
        d_model=args.d,
        num_blocks=args.blocks,
        r_rank=args.rank,
        device=dev,
        checkpoint_policy="auto",  # auto: skip incompatible checkpoint at probe time
        zone_c_required=False,
    )

    # Warmup
    w = torch.randn(args.blocks, 8, device=dev)
    w = torch.nn.functional.normalize(w, p=2, dim=-1)
    for _ in range(10):
        h.inner_step(w, w, axiom_wave=w)
    torch.cuda.synchronize()

    lat = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        h.inner_step(w, w, axiom_wave=w)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e6)  # us

    lat = np.asarray(lat)
    mean_us = float(lat.mean())
    p99_us = float(np.percentile(lat, 99))
    slope = float(np.polyfit(np.arange(500, args.steps), lat[500:], 1)[0])
    slope_p = float(np.corrcoef(np.arange(500, args.steps), lat[500:])[0, 1])

    accept = mean_us <= 50.0 and p99_us <= 2.0 * mean_us and (slope_p >= -0.05)
    telemetry = {
        "probe": "K1",
        "steps": args.steps,
        "d": args.d,
        "blocks": args.blocks,
        "rank": args.rank,
        "device": dev,
        "mean_us": mean_us,
        "p99_us": p99_us,
        "slope_us_per_step": float(slope),
        "slope_corr": float(slope_p),
        "accept": bool(accept),
        "cuda_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
    }
    print(json.dumps(telemetry, indent=2))
    with open(args.out, "w") as f:
        json.dump(telemetry, f, indent=2)
    return 0 if accept else 1


if __name__ == "__main__":
    raise SystemExit(main())
