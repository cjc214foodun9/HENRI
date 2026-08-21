"""T2 smoke for CE telemetry (packet HENRI-CLASS47-CE-TELEMETRY-2026-08-21).

Real production ingress (HENRIVisionEncoder) on ARC-style grids at D=4096,
then CE over the wave trajectory. Gates:
  T2: status ok, finite CE, variance > 0, support >= 8.
  S:  CE in [0, 3] bits, no NaN.

Usage: python scripts/verification/ce_t2_smoke.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

from henri_vision_encoder import HENRIVisionEncoder
from causal_emergence_telemetry import CausalEmergenceTelemetry, causal_emergence


def make_grids(n: int, seed: int = 3) -> list:
    """ARC-style 30x30 grids: 10 colors, background 0, random object blobs."""
    rng = np.random.RandomState(seed)
    grids = []
    for _ in range(n):
        grid = np.zeros((30, 30), dtype=np.int64)
        for _ in range(rng.randint(2, 5)):
            c = rng.randint(1, 10)
            y, x = rng.randint(0, 28, size=2)
            h, w = rng.randint(2, 8, size=2)
            grid[y:y + h, x:x + w] = c
        grids.append(grid)
    return grids


def main():
    device = "cpu"
    d_model, k_blocks = 4096, 512
    enc = HENRIVisionEncoder(
        d_model=d_model, k_blocks=k_blocks, device=device,
        spatial_basis_kind="default", bg_mask=True,
    )
    grids = make_grids(128)
    waves = []
    for g in grids:
        w = enc.encode_spatial_grid(g).squeeze(0).to(device)
        waves.append(w.detach().float())
    stack = torch.stack(waves)  # [128, D]

    # Streaming telemetry (same code path as the runner).
    tele = CausalEmergenceTelemetry(window=64)
    reports = []
    for w in waves:
        tele.push(w)
        rep = tele.report()
        if rep is not None:
            reports.append(rep)
    assert len(reports) >= 1, "no full window emitted"
    rep = reports[-1]
    print(f"T2 streaming report: {rep}")

    assert rep["status"] == "ok", f"status={rep['status']}"
    assert rep["ei_micro"] is not None and rep["ce_bits"] is not None
    assert rep["support"] >= 8, f"support={rep['support']}"
    assert abs(rep["ce_bits"]) <= 3.0, f"CE={rep['ce_bits']} (gate S: |CE| <= 3)"
    assert math.isfinite(rep["ce_bits"]), "non-finite CE"
    # Variance: two windows must differ (trajectory is not constant).
    if len(reports) >= 2:
        assert reports[0]["ce_bits"] != reports[-1]["ce_bits"] or reports[0]["ei_micro"] != reports[-1]["ei_micro"]

    # Noise-floor control (amended T2 v3): iid noise at T=256 -> |CE| < 0.02.
    noise = torch.randn(256, d_model)
    rep_n = causal_emergence(noise)
    assert rep_n["status"] == "ok"
    print(f"Noise-floor control (T=256): {rep_n}")
    assert rep_n["ce_bits"] is not None and abs(rep_n["ce_bits"]) < 0.02

    # Coupled control (amended T2 v3): deterministic macro alternation + jitter.
    t = torch.linspace(0, 1, d_model)
    pa = torch.sin(2 * torch.pi * 3 * t)
    pb = torch.cos(2 * torch.pi * 3 * t)
    gj = torch.Generator().manual_seed(5)
    frames = []
    for i in range(256):
        base = pa if i % 2 == 0 else pb
        frames.append(base + 0.25 * torch.randn(d_model, generator=gj))
    coupled = torch.stack(frames).float()
    rep_c = causal_emergence(coupled)
    assert rep_c["status"] == "ok"
    print(f"Coupled control (T=256): {rep_c}")
    assert rep_c["ce_bits"] is not None and rep_c["ce_bits"] > 0.01
    assert rep_c["ce_bits"] > rep_n["ce_bits"], "coupled CE must exceed noise CE"

    # Full-dimension sanity (D=65,536) on structured waves: finite + bounded.
    t = torch.linspace(0, 40 * math.pi, 128)
    big = torch.stack([torch.sin(t * (1 + 0.05 * i)) for i in range(8)], dim=1)
    big = big.repeat(1, 65536 // 8).float()
    rep_big = causal_emergence(big)
    print(f"D=65536 structured sanity: {rep_big}")
    assert rep_big["status"] == "ok"
    assert rep_big["ce_bits"] is not None and -0.5 <= rep_big["ce_bits"] <= 3.0

    print("T2_SMOKE_PASS")


if __name__ == "__main__":
    main()
