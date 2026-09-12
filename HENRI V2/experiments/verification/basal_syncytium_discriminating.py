"""G4 discriminating diagnostic: is low global r an implementation defect or a
genuine property of the evanescent-leakage coupling?

The K/step sweep showed global r capping near 0.20 while the LOCAL order
parameter reached 0.83. That pattern is either:
  (a) a bug in the coupled-field convolution / force sign, or
  (b) a real property: a row-normalized local kernel produces LOCAL
      synchrony (chimera-like clusters) without GLOBAL phase locking.

DISCRIMINATING TEST
  Run the same integrator with the kernel width swept from local (decay=2) to
  effectively all-to-all (decay=100000). The all-to-all limit has a known
  analytic answer: with zero natural frequencies, the incoherent state is
  unstable for K > K_c = 4 and the stable state is global sync, r -> 1.

  If the all-to-all limit does NOT reach r -> 1, the integrator is wrong.
  If it DOES, then the local-kernel result is a real property of the mechanism.

Also measures whether r keeps growing with a much longer horizon.
"""
from __future__ import annotations

import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from basal_boundary_engine import EvanescentKuramotoSyncytium

N = 8192


def track(K, decay, steps, every, dt=0.01, seed=0, hetero=0.0):
    syn = EvanescentKuramotoSyncytium(
        num_channels=N, coupling_K=K, decay_length=decay, dt=dt,
        natural_frequency_scale=hetero, noise_temperature=0.0, seed=seed,
    )
    syn.random_phases(seed=seed)
    traj = []
    for done in range(0, steps, every):
        out = syn.relax(every)
        traj.append({"t": round((done + every) * dt, 3),
                     "r": round(float(out["r"]), 5),
                     "r_local": round(float(out["r_local_mean"]), 5)})
    return traj


def main():
    rep = {}

    print("=== A. ALL-TO-ALL LIMIT (control: must reach r -> 1 for K > K_c=4) ===")
    for K in (2.0, 8.0, 32.0):
        tr = track(K, 100000.0, 2000, 250)
        rep[f"alltoall_K{K}"] = tr
        print(f"  K={K:5.1f} decay=1e5 : " +
              "  ".join(f"t={d['t']}:r={d['r']:.3f}" for d in tr))

    print("\n=== B. LOCAL KERNEL WIDTH SWEEP (K=32, 2000 steps) ===")
    for decay in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0):
        tr = track(32.0, decay, 2000, 250)
        rep[f"local_decay{decay}_K32"] = tr
        print(f"  decay={decay:7.1f}: " +
              "  ".join(f"{d['r']:.3f}" for d in tr) +
              f"   r_local_final={tr[-1]['r_local']:.3f}")

    print("\n=== C. LONG HORIZON at the spec K=2.45 (20000 steps) ===")
    tr = track(2.45, 8.0, 20000, 2000)
    rep["long_K2.45"] = tr
    print("  " + "  ".join(f"t={d['t']}:r={d['r']:.3f}" for d in tr))

    out = os.path.join(_HERE, "basal_syncytium_discriminating.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
