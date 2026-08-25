"""C11 child: run the fused kernel over identical inputs, emit state+telemetry.

Spawned twice by deltamem_verify.py from two fresh CUDA processes.
Inputs are generated from a FIXED seed inside this process (identical bytes
across processes). Output: {state_hash, telemetry, n_steps}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deltamem_triton_fused import FusedTritonDeltaKernel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    kern = FusedTritonDeltaKernel(d=4096, r=8, v_seed=20260824)
    g = torch.Generator(device="cuda").manual_seed(424242)
    telemetry = []
    for _ in range(args.steps):
        k = torch.randn(4096, generator=g, device="cuda")
        v = torch.randn(4096, generator=g, device="cuda")
        e = kern.step_once(k, v)
        telemetry.append(float(e.norm().item()))
    torch.cuda.synchronize()
    out = {"state_hash": kern.state_hash(),
           "telemetry": [round(x, 9) for x in telemetry],
           "n_steps": kern.step, "veto": kern.veto_count}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("child done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
