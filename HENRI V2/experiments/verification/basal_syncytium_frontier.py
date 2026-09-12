"""G4 frontier: the minimum evanescent leakage length for global phase locking.

ESTABLISHED BY THE PREVIOUS RUN (`basal_syncytium_discriminating.json`):
  - The integrator is CORRECT. In the all-to-all limit (decay -> infinity) the
    global order parameter reaches r = 1.000 for K = 2, 8, 32, exactly as
    Kuramoto mean-field theory requires for zero natural frequencies.
  - With a LOCAL kernel the system reaches r_local -> 0.996 at every setting
    (each channel's own neighbourhood is perfectly locked) while the GLOBAL r
    plateaus near 0.20 and then DECAYS (0.203 at t=40 -> 0.096 at t=200).
    Dense local order, no global order.

That combination - perfect local order, absent global order, decaying with time -
is the signature of a 1-D locally-coupled oscillator lattice, which is at the
lower critical dimension for global phase order. So the mandate's r >= 0.93 is
NOT reachable with a purely evanescent (exponentially local) kernel.

This script measures WHERE the boundary sits, in a scale-free form: the leakage
length as a FRACTION of the ring. That number is what an engineer needs, because
an absolute leakage length in channels does not transfer between a 256-channel
and an 8192-channel syncytium.
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

from basal_boundary_engine import SPEC_R_GATE, EvanescentKuramotoSyncytium

N = 8192


def r_at(K, decay, steps, dt=0.01, seed=0, hetero=0.0):
    syn = EvanescentKuramotoSyncytium(
        num_channels=N, coupling_K=K, decay_length=decay, dt=dt,
        natural_frequency_scale=hetero, noise_temperature=0.0, seed=seed,
    )
    syn.random_phases(seed=seed)
    out = syn.relax(steps)
    return float(out["r"]), float(out["r_local_mean"]), bool(out["finite"])


def main():
    rep = {"num_channels": N, "r_gate": SPEC_R_GATE, "dt": 0.01}
    print(f"N={N}  gate r >= {SPEC_R_GATE}\n")

    print("=== A. r vs leakage fraction, K=32, 2000 steps ===")
    fracs = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024]
    rows_a = []
    for lam in fracs:
        r, rl, fin = r_at(32.0, float(lam), 2000)
        rows_a.append({"decay": lam, "fraction": lam / N, "r": r,
                       "r_local": rl, "clears": bool(r >= SPEC_R_GATE)})
        print(f"  decay={lam:5d} ({lam/N:7.5f} of ring)  r={r:.4f}  "
              f"r_local={rl:.4f}  clears={rows_a[-1]['clears']}")
    rep["A_r_vs_leakage_K32_2000steps"] = rows_a

    print("\n=== B. Does the ordered state HOLD, or decay? (K=32, decay=512) ===")
    syn = EvanescentKuramotoSyncytium(
        num_channels=N, coupling_K=32.0, decay_length=512.0, dt=0.01,
        natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
    )
    syn.random_phases(seed=0)
    hold = []
    for k in range(12):
        out = syn.relax(500)
        hold.append({"t": round((k + 1) * 500 * 0.01, 1),
                     "r": round(float(out["r"]), 5)})
    rep["B_stability_decay512_K32"] = hold
    print("  " + "  ".join(f"t={h['t']}:r={h['r']:.4f}" for h in hold))

    print("\n=== C. Stability at the SPEC K=2.45, decay=512 ===")
    syn = EvanescentKuramotoSyncytium(
        num_channels=N, coupling_K=2.45, decay_length=512.0, dt=0.01,
        natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
    )
    syn.random_phases(seed=0)
    hold2 = []
    for k in range(12):
        out = syn.relax(500)
        hold2.append({"t": round((k + 1) * 500 * 0.01, 1),
                      "r": round(float(out["r"]), 5)})
    rep["C_stability_decay512_K2p45"] = hold2
    print("  " + "  ".join(f"t={h['t']}:r={h['r']:.4f}" for h in hold2))

    print("\n=== D. Minimum leakage at each K (horizon 4000 steps) ===")
    lam_grid = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    frontier = []
    for K in (2.45, 4.0, 8.0, 16.0, 32.0):
        hit = None
        for lam in lam_grid:
            r, rl, fin = r_at(K, float(lam), 4000)
            if r >= SPEC_R_GATE:
                hit = {"decay": lam, "fraction": lam / N, "r": r}
                break
        frontier.append({"K": K, "min_decay": None if hit is None else hit["decay"],
                         "min_fraction": None if hit is None else hit["fraction"],
                         "r_at_min": None if hit is None else hit["r"]})
        print(f"  K={K:5.2f}  min_decay={frontier[-1]['min_decay']}  "
              f"fraction={frontier[-1]['min_fraction']}  r={frontier[-1]['r_at_min']}")
    rep["D_frontier"] = frontier

    out = os.path.join(_HERE, "basal_syncytium_frontier.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
