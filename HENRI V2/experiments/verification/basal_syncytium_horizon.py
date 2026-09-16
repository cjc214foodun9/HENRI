"""G4 relaxation-horizon probe at the MEASURED working point.

Working point: K = 2.45 (spec default), leakage = 504 channels = 0.0615 of the
8192-channel ring (the measured minimum fraction 0.0205 x margin 3.0).

The question this answers: how many relaxation steps does the basal syncytium
need to reach r >= 0.93, from (a) the REAL data path (phases seeded from an
ingress wave) and (b) a cold uniform-random start (worst case)?

The answer sets the shutter budget. A single 20 kHz slot carries
`ticks_per_slot` relaxation steps, so the horizon has a physical meaning:
steps_to_lock / ticks_per_slot = number of isolation slots to first lock.
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

from basal_boundary_engine import (
    SPEC_NUM_TILES,
    SPEC_R_GATE,
    EvanescentKuramotoSyncytium,
    clifford_bivector_tiles,
    recommended_leakage_length,
)

N = SPEC_NUM_TILES
K = 2.45
LEAK = recommended_leakage_length(N, margin=3.0)
DT = 0.01


def trajectory(seed_from_wave: bool, total: int = 6000, chunk: int = 250):
    syn = EvanescentKuramotoSyncytium(
        num_channels=N, coupling_K=K, decay_length=LEAK, dt=DT,
        natural_frequency_scale=0.0, noise_temperature=0.0, seed=0,
    )
    if seed_from_wave:
        g = torch.Generator().manual_seed(7)
        wave = torch.randn(N * 8, generator=g)
        syn.seed_phases_from_wave(wave)
        start = "wave_seeded"
    else:
        syn.random_phases(seed=0)
        start = "cold_random"

    traj = []
    steps_to_lock = None
    for done in range(0, total, chunk):
        out = syn.relax(chunk)
        s = done + chunk
        r = float(out["r"])
        traj.append({"steps": s, "t": round(s * DT, 2), "r": round(r, 5),
                     "r_local": round(float(out["r_local_mean"]), 5)})
        if steps_to_lock is None and r >= SPEC_R_GATE:
            steps_to_lock = s
    return start, traj, steps_to_lock


def main():
    rep = {"N": N, "K": K, "leakage": LEAK,
           "leakage_fraction_of_ring": LEAK / N,
           "r_gate": SPEC_R_GATE, "dt": DT,
           "slot_seconds": 1.0 / 20000.0}

    results = {}
    for seed_from_wave in (True, False):
        start, traj, stl = trajectory(seed_from_wave)
        results[start] = {"trajectory": traj, "steps_to_lock": stl}
        print(f"=== {start} ===")
        print("  " + "  ".join(f"{d['steps']}:{d['r']:.4f}" for d in traj))
        print(f"  steps_to_lock={stl}")

    rep["results"] = results
    rep["initial_r"] = {
        k: v["trajectory"][0]["r"] for k, v in results.items()
    }

    # The real data path: phases seeded from the ingress wave. Decide the
    # shutter budget from the measured need.
    wave_stl = results["wave_seeded"]["steps_to_lock"]
    rep["derived"] = {
        "wave_seeded_steps_to_lock": wave_stl,
        "cold_random_steps_to_lock": results["cold_random"]["steps_to_lock"],
        "seconds_to_lock_wave_seeded": None if wave_stl is None else wave_stl * DT,
        "slots_required_at_32_ticks": None if wave_stl is None else wave_stl / 32,
    }

    out = os.path.join(_HERE, "basal_syncytium_horizon.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nreceipt: {out}")
    print(json.dumps(rep["derived"], indent=2))


if __name__ == "__main__":
    main()
