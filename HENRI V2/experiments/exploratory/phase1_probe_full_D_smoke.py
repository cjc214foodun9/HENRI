#!/usr/bin/env python
"""Prove the D=65536 memory fix at REAL full D, on CPU, without the full H3 sweep.

The full H3 harness at D=65536 is dominated by repeated CPU svdvals on
[<=512, 65536] complex and does not finish inside a 420 s foreground budget.
That is a timing limit, not a memory failure.  This smoke isolates the memory
contract: it drives the probe at D=65536 and asserts

  1. every retained tensor is 2-D with trailing dim D (never D x D),
  2. total retained state is < 0.5 GiB,
  3. the spectrum is finite, descending, and non-negative,
  4. peak process RSS stays far below the D x D requirement of 32 GiB.

Exit 0 on success.  Any failure is a real defect.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_V2 = _HERE.parents[2]
sys.path.insert(0, str(_V2))

from henri_zone_a_backbone import PreSnapCovarianceProbe  # noqa: E402

D = int(os.environ.get("HENRI_ZA_DIM", "65536"))
K = 8
N = 16          # small batch: this smoke tests MEMORY, not statistics
CALLS = 3


def rss_mib() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        return float("nan")


def main() -> int:
    t0 = time.perf_counter()
    probe = PreSnapCovarianceProbe(dim=D, k=K, ema=0.3)
    g = torch.Generator().manual_seed(20261002)
    specs = []
    for i in range(CALLS):
        h = torch.randn(N, D, generator=g).to(torch.complex64)
        specs.append(probe.observe(h))
    dt = time.perf_counter() - t0

    shapes = probe.state_shapes()
    retained = sum(int(s[0]) * int(s[1]) for s in shapes) * 8   # complex64
    dense = D * D * 8

    checks = {
        "state_is_2d_trailing_D": all(len(s) == 2 and s[1] == D for s in shapes),
        "no_tensor_is_DxD": all(not (s[0] == D and s[1] == D) for s in shapes),
        "retained_under_512MiB": retained <= 512 * 1024**2,
        "retained_far_below_dense": retained * 100 < dense,
        "spec_finite": bool(torch.isfinite(specs[-1]).all()),
        "spec_nonneg": bool((specs[-1] >= -1e-6).all()),
        "spec_descending": bool((specs[-1][:-1] >= specs[-1][1:] - 1e-9).all()),
        "rss_under_8GiB": rss_mib() < 8 * 1024,
    }
    out = {
        "experiment": "probe_full_D_memory_smoke",
        "D": D,
        "calls": CALLS,
        "batch": N,
        "state_shapes": shapes,
        "retained_bytes": retained,
        "dense_bytes_refused": dense,
        "seconds": round(dt, 3),
        "rss_mib": round(rss_mib(), 1),
        "spectrum_top3": [float(x) for x in specs[-1][:3]],
        "checks": checks,
        "verdict": "PROBE_FULL_D_MEMORY_OK" if all(checks.values()) else "PROBE_FULL_D_MEMORY_FAIL",
    }
    print(json.dumps(out, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
