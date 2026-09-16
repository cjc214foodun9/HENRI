#!/usr/bin/env python3
"""Zone A swarm coordinate arithmetic + Zone C orchestrator budget (Phase 10.4).

Two directive claims are pure arithmetic and can be settled exactly, with no model:

CLAIM A (Zone A, 64 bytes/agent):
    "Each agent is defined strictly by a 16-element coordinate vector c_p in R^k over a
     frozen universal basis stored in L2 cache."
    16 float32 = 64 bytes EXACTLY. So the claim is TRUE iff k == 16. The directive ALSO
    says k <= 4..8 in one place and U_k in R^{D x 4} in another. Those are inconsistent
    with 64 bytes: k=4 -> 16 B, k=8 -> 32 B. This script states the arithmetic so the
    inconsistency is visible instead of hidden behind a round number.

CLAIM B (Zone C meta-orchestrator, "< 1 KB"):
    Counted from the ACTUAL controller logic the directive specifies (4 monitored
    channels -> 3 actuators), which is a table of thresholds plus three scalars. The
    count below is an honest floor: data only, no code, no pointers.

CLAIM C (gamma):
    "inverts gamma = M*S/D from 1e-4 to 42.4". Checked against the live dimensions
    (M=3 demos, S=8 real slots per block, D=32768 complex flat). Both endpoints are
    recomputed, and the UWSH endpoint is recomputed under its OWN definition
    gamma = (M * N_active) / k, which requires N_active -- a quantity the directive
    never defines. The script reports which N_active would be needed.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "uwsh_zone_arithmetic_observed.json")

NB_, SL_, S_ = 8192, 4, 32
N_FLAT = NB_ * SL_                 # 32768 complex flat slots
D_REAL = 2 * N_FLAT                # 65536 real params if treated as one operator
M_DEMOS = 3


def main():
    out = {"schema": "henri.uwsh.zone-arithmetic.v1",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "evidence_class": "DERIVED",
           "source": {"directive_sha256":
                      "1ea527bd87c352839ee2aba0b88080b1aed9033c4268b5681cda6313e7a7b2ef"},
           "live_dims": {"num_blocks": NB_, "block_slots_complex": SL_, "modulus_S": S_,
                         "N_flat_complex": N_FLAT, "D_real_if_dense": D_REAL,
                         "M_demos": M_DEMOS}}

    print("=" * 78)
    print("CLAIM A -- Zone A: '16-element coordinate vector' == '64 bytes per agent'?")
    print("=" * 78)
    rows = []
    for k in (4, 8, 16, 32):
        b = k * 4
        rows.append({"k": k, "bytes_float32": b, "is_64_bytes": b == 64})
        print(f"  k={k:3d}  float32 -> {b:4d} bytes   {'<- matches 64 B claim' if b == 64 else ''}")
    k4 = next(r for r in rows if r["k"] == 4)
    k8 = next(r for r in rows if r["k"] == 8)
    k16 = next(r for r in rows if r["k"] == 16)
    out["claim_A_zone_a_bytes"] = {
        "table": rows,
        "64_bytes_requires_k": 16,
        "directive_k_variants": {"k<=4..8": [k4["bytes_float32"], k8["bytes_float32"]],
                                 "U_k in R^{D x 4}": k4["bytes_float32"],
                                 "16-element": k16["bytes_float32"]},
        "consistent": bool(k16["is_64_bytes"] and not (k4["is_64_bytes"] or k8["is_64_bytes"])),
    }
    print(f"\n  64 B is EXACT for k=16. The same document also states k<=4..8 and")
    print(f"  U_k in R^(D x 4), which give {k4['bytes_float32']} B and {k8['bytes_float32']} B.")
    print(f"  => the 64-byte figure and the k<=4..8 figure are MUTUALLY INCONSISTENT as stated.")
    print(f"  => swarm memory is 64 B/agent ONLY at k=16; the BASIS itself is additional")
    print(f"     (see basis footprint below).")

    basis = {"k": 16, "dtype": "float32", "elements": 2 * N_FLAT * 16,
             "bytes": 2 * N_FLAT * 16 * 4}
    basis["MiB"] = basis["bytes"] / 2**20
    basis["L2_resident_at"] = "k=16 -> 4.0 MiB"
    out["basis_footprint"] = basis
    print(f"\n  Basis U for k=16: shape [2N, k] = [{2*N_FLAT}, 16] float32")
    print(f"    bytes = {basis['bytes']:,} = {basis['MiB']:.1f} MiB")
    print(f"    The directive says '~2.5 MB' for U in shared L2/L3. Measured: "
          f"{basis['MiB']:.1f} MiB at k=16, {2*N_FLAT*8*4/2**20:.1f} MiB at k=8, "
          f"{2*N_FLAT*4*4/2**20:.1f} MiB at k=4.")
    print(f"    => '~2.5 MB' corresponds to k~10, NOT to the k=16 needed for 64 B/agent.")

    print()
    print("=" * 78)
    print("CLAIM B -- Zone C meta-orchestrator '< 1 KB'")
    print("=" * 78)
    # monitored channels x (threshold float32 + scale float32 + enabled uint8)
    channels = ["sagnac_stress", "topo_charge_drift", "stiefel_orthogonality",
                "free_energy_dissipation"]
    per_ch = 2 * 4 + 1
    actuators = ["sagnac_veto_sensitivity", "langevin_microheater_dissipation",
                 "staticity_threshold"]
    per_act = 4
    state = 4 * 4                      # ema/hysteresis per channel
    total = len(channels) * per_ch + len(actuators) * per_act + state
    out["claim_B_zone_c_budget"] = {
        "monitored_channels": channels, "bytes_per_channel": per_ch,
        "actuators": actuators, "bytes_per_actuator": per_act,
        "controller_state_bytes": state, "total_bytes_floor": total,
        "under_1KB": bool(total < 1024)}
    for c in channels:
        print(f"  channel {c:26s} {per_ch:3d} B  (threshold+scale f32 + enable u8)")
    for a in actuators:
        print(f"  actuator {a:26s} {per_act:3d} B  (f32)")
    print(f"  controller state (ema/hysteresis){state:4d} B")
    print(f"  TOTAL DATA FLOOR: {total} B  -> '< 1 KB' is {'TRUE' if total < 1024 else 'FALSE'}")
    print(f"  NOTE: this is DATA ONLY. Code, pointers and the TimescaleDB client are")
    print(f"  excluded; the claim is true as a payload budget, not as a process footprint.")

    print()
    print("=" * 78)
    print("CLAIM C -- gamma arithmetic")
    print("=" * 78)
    g_inc = M_DEMOS * 8 / D_REAL       # the directive's own 'gamma = M*S/D'
    g_slots = M_DEMOS / 1.0            # per-slot view: 3 obs per complex slot
    print(f"  directive 'gamma = M*S/D' = {M_DEMOS}*8/{D_REAL} = {g_inc:.3e}  (matches 1e-4)")
    print(f"  SAME formula on the live flat dim  = {M_DEMOS}*8/{N_FLAT} = "
          f"{M_DEMOS*8/N_FLAT:.3e}")
    print(f"  per-slot view (what diag_ls actually solves): {g_slots:.1f} obs per complex "
          f"slot -> already >= 1, NOT 1e-4")
    print(f"  => the 1e-4 figure treats W as ONE dense {D_REAL}-parameter operator. The")
    print(f"     incumbent diag_ls is per-slot, so its own gamma is {g_slots:.1f}.")
    print(f"     The stated baseline therefore does not describe the incumbent.")
    gamma_rows = []
    for k in (1, 2, 4, 8, 16):
        for N_act, lbl in ((16, "H*W=16"), (128, "BLOCK_SPAN"), (1024, "S^2"), (N_FLAT, "N")):
            g = M_DEMOS * N_act / k
            gamma_rows.append({"k": k, "N_active": N_act, "N_active_label": lbl,
                               "gamma": g})
    near = sorted(gamma_rows, key=lambda r: abs(r["gamma"] - 42.4))[:4]
    print(f"\n  gamma = (M * N_active)/k needs N_active, which the directive NEVER DEFINES.")
    print(f"  candidates closest to 42.4:")
    for r in near:
        print(f"    k={r['k']:2d} N_active={r['N_active']:5d} ({r['N_active_label']:10s}) "
              f"-> gamma={r['gamma']:.2f}")
    out["claim_C_gamma"] = {
        "directive_formula": "gamma = M*S/D",
        "directive_baseline_value": g_inc,
        "value_on_live_flat_dim": M_DEMOS * 8 / N_FLAT,
        "per_slot_obs_per_complex_slot": g_slots,
        "baseline_describes_incumbent": False,
        "note": ("1e-4 treats W as one dense %d-parameter operator; diag_ls is per-slot, "
                 "so its own gamma is %.1f. Endpoints are not comparable."
                 % (D_REAL, g_slots)),
        "uwsh_formula": "gamma = (M * N_active) / k",
        "N_active_undefined_in_directive": True,
        "candidates_near_42_4": near,
    }

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
