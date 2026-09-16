"""G4 calibration: measure the basal syncytium's r(t) law instead of assuming it.

The first default (K=2.45, 32 relaxation steps, dt=0.01) FAILED the r >= 0.93
gate. This script measures WHY, at 8192 channels, and derives the operating
point from the measurement.

Questions this answers with numbers:
  Q1  How does r grow with relaxation horizon at fixed K?
  Q2  Is the required horizon a function of K (and how)?
  Q3  Does the subcritical control (heterogeneous natural frequencies, K=2.45)
      actually stay below the gate, so the gate discriminates?
  Q4  Does the evanescent decay length change the answer?
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

# The repo root is two levels up; the production modules are flat at the root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from basal_boundary_engine import (
    SPEC_NUM_TILES,
    SPEC_R_GATE,
    EvanescentKuramotoSyncytium,
)

STEPS = (16, 32, 64, 128, 256, 512, 1024)
KS = (1.0, 2.45, 4.0, 8.0, 16.0, 32.0)
DECAYS = (2.0, 8.0, 32.0)


def measure_r(num_channels, K, decay, steps, hetero_scale, dt, seed):
    syn = EvanescentKuramotoSyncytium(
        num_channels=num_channels, coupling_K=K, decay_length=decay, dt=dt,
        natural_frequency_scale=hetero_scale, noise_temperature=0.0, seed=seed,
    )
    syn.random_phases(seed=seed)
    out = syn.relax(steps)
    return float(out["r"]), float(out["r_local_mean"]), bool(out["finite"])


def main() -> None:
    t0 = time.time()
    N = SPEC_NUM_TILES
    rep = {"num_channels": N, "r_gate": SPEC_R_GATE, "dt": 0.01, "sections": {}}

    # Q1/Q2: r vs (K, steps), zero natural frequencies (the mandate reading).
    grid = {}
    for K in KS:
        row = {}
        for s in STEPS:
            r, rl, fin = measure_r(N, K, 8.0, s, 0.0, 0.01, 0)
            row[s] = {"r": r, "r_local_mean": rl, "finite": fin}
        grid[K] = row
        hit = next((s for s in STEPS if row[s]["r"] >= SPEC_R_GATE), None)
        print(f"K={K:6.2f}  r(steps): " + "  ".join(
            f"{s}:{row[s]['r']:.4f}" for s in STEPS
        ) + f"   first_pass_steps={hit}")
    rep["sections"]["r_vs_K_vs_steps"] = {str(k): {str(s): v for s, v in r.items()}
                                          for k, r in grid.items()}

    # Q3: subcritical control, heterogeneous natural frequencies.
    ctrl = {}
    for K in (1.0, 2.45, 3.0, 4.0, 6.0, 8.0, 16.0):
        r, rl, fin = measure_r(N, K, 8.0, 1024, 1.0, 0.01, 1234)
        ctrl[K] = {"r": r, "r_local_mean": rl, "finite": fin,
                   "clears_gate": bool(r >= SPEC_R_GATE)}
        print(f"control K={K:6.2f}  r={r:.4f}  clears_gate={ctrl[K]['clears_gate']}")
    rep["sections"]["heterogeneous_control_r_at_1024_steps"] = {
        str(k): v for k, v in ctrl.items()
    }

    # Q4: decay length sensitivity at the configured K.
    dec = {}
    for d in DECAYS:
        r, rl, fin = measure_r(N, 2.45, d, 256, 0.0, 0.01, 0)
        dec[d] = {"r": r, "r_local_mean": rl}
        print(f"decay={d:5.1f}  r={r:.4f}  r_local_mean={rl:.4f}")
    rep["sections"]["decay_length_at_K_2p45_256_steps"] = {
        str(k): v for k, v in dec.items()
    }

    # Q5: time-step sensitivity at the configured K and 256 steps.
    dts = {}
    for dt in (0.005, 0.01, 0.02, 0.05):
        r, rl, fin = measure_r(N, 2.45, 8.0, 256, 0.0, dt, 0)
        dts[dt] = {"r": r, "t_total": dt * 256}
        print(f"dt={dt:6.3f} t={dt*256:.2f}  r={r:.4f}")
    rep["sections"]["dt_sensitivity_K2p45_256_steps"] = {
        str(k): v for k, v in dts.items()
    }

    rep["elapsed_s"] = round(time.time() - t0, 2)
    out = os.path.join(_HERE, "basal_syncytium_calibration.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nreceipt: {out}   elapsed {rep['elapsed_s']}s")


if __name__ == "__main__":
    main()
