"""probe_ndl_ratchet.py -- measure the Nested Daydream Loop RATCHET, with a control.

THESIS ITEM 6. The wiring is verified and byte-identical on the legacy path
(test_ndl_replay_interleave.py, 5/5). The RATCHET itself is unmeasured. Measured: grep for
"ratchet" in daydream.py returns 0 hits -- the metric does not exist. This probe defines it.

WHAT A RATCHET MEANS HERE, OPERATIONALLY
    Offline consolidation replays wake episodes interleaved with synthetic D_u programs.
    "Ratchet" = the CUMULATIVE stock of correctly-verified (program, input) pairs grows
    across consolidation cycles and does not decay. A ratchet is monotone by definition, so
    a metric that merely fluctuates is not one.

THE METRIC, AND WHY IT CAN FAIL
    Every replayed pair is (spec, trace) with spec = "apply <PROG> to <INPUT>". The program
    is deterministic, so the trace is CHECKABLE by re-execution:
        verified(spec, trace) := run_program(PROG, INPUT) == trace
    A cycle's contribution is the set of distinct (PROG, INPUT) pairs that verify. The
    ratchet is the running union of those sets. This can fail: if ingestion stores
    mismatched pairs, nothing verifies and the union stops growing.

THE CONTROL THAT MUST NOT RATCHET
    ARM S (shuffled): identical replay stream, but the (spec, trace) pairing is permuted.
    A shuffled trace usually fails re-execution, so ARM S must accumulate FAR LESS. Without
    this control the number would grow for a trivial reason -- the sample count -- and the
    measurement would be vacuous.

MY OWN TRAP, GUARDED HERE
    A shuffle is a NO-OP wherever the permuted traces are identical. If every trace in the
    window were the same string, shuffling would change nothing and the control would pass
    by vacuity. The probe therefore measures and reports the DISTINGUISHABLE FRACTION and
    refuses a verdict when it is too low.

ARM N (no-wake): with an empty store, replay_batch must equal stage1_fictive exactly. This
is re-asserted here because it is the load-bearing invariant that keeps every prior receipt
byte-reproducible.

SCOPE, STATED PLAINLY
    Store-level and deterministic. No model is trained, so this is NOT a capability claim.
    It measures whether consolidation ACCUMULATES VERIFIED STRUCTURE. The model-level
    ratchet remains open and needs a trained readout. Meanwhile daydream.py's stage2/stage4
    need a live system, so the engine is constructed with system=None, tok=None; the paths
    exercised here (ingest_wake, replay_batch, wake_items, stage1_fictive) do not use them.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
V = HERE.parents[1]
HC = V / "henri_core"
for p in (str(V), str(HC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from daydream import DaydreamConfig, DaydreamEngine            # noqa: E402
from m4_generative import INPUTS, run_program                  # noqa: E402

REPO = V.parent
EV = REPO / "design" / "zone_a" / "evidence"
EV.mkdir(parents=True, exist_ok=True)

# ---- pre-registered configuration, fixed BEFORE the run -------------------------------
CYCLES = 6
WAKE_PER_CYCLE = 24
REPLAY_PER_CYCLE = 40
MAX_LEN = 4                       # one step beyond the M4 training distribution (<= 3)
SEED = 20261011
MIN_DISTINGUISHABLE = 0.60        # the control is vacuous below this fraction
RATCHET_MARGIN = 2.0              # real final must exceed shuffle final by this factor

R = {
    "schema": "henri.ndl.ratchet.observed.v1",
    "evidence_class": "OBSERVED",
    "what": "Nested Daydream Loop ratchet measurement",
    "preregistered": {
        "cycles": CYCLES, "wake_per_cycle": WAKE_PER_CYCLE,
        "replay_per_cycle": REPLAY_PER_CYCLE, "max_len": MAX_LEN, "seed": SEED,
        "min_distinguishable_fraction": MIN_DISTINGUISHABLE,
        "ratchet_margin_real_over_shuffle": RATCHET_MARGIN,
    },
    "metric": ("cumulative count of DISTINCT (PROGRAM, INPUT) pairs whose stored trace "
               "re-verifies under run_program; monotone union across cycles"),
    "non_claims": [
        "store-level and deterministic; NO model is trained, so this is not a capability claim",
        "the model-level ratchet (readout competence across cycles) remains open",
        "engine built with system=None; stage2/stage4 are not exercised by this probe",
    ],
}


def parse_spec(spec: str):
    """spec = 'apply <PROG> to <INPUT>'. Return (PROG, INPUT) or None."""
    try:
        _, prog, _, inp = spec.split()
        return prog, inp
    except Exception:                                          # noqa: BLE001
        return None


def verify_stream(stream):
    """Return (verified_pairs_set, n_pairs, n_verified, n_parse_fail)."""
    seen, nv, nbad = set(), 0, 0
    for spec, trace in stream:
        pi = parse_spec(spec)
        if pi is None:
            nbad += 1
            continue
        prog, inp = pi
        try:
            ok = (run_program(prog, inp) == trace)
        except Exception:                                      # noqa: BLE001
            ok = False
        if ok:
            seen.add((prog, inp))
            nv += 1
    return seen, len(stream), nv, nbad


def run_arm(name: str, shuffled: bool):
    cfg = DaydreamConfig(seed=SEED, wake_ratio=0.5)
    eng = DaydreamEngine(None, None, cfg)
    gen = torch.Generator().manual_seed(SEED + (7 if shuffled else 0))

    cumulative: set = set()
    rows, distinguishable = [], []

    for cyc in range(CYCLES):
        # ---- ingest this cycle's wake episodes (the real experience) ----
        wake = eng.stage1_fictive(WAKE_PER_CYCLE, gen, max_len=MAX_LEN)
        for spec, trace in wake:
            eng.ingest_wake(spec, trace, alignment=0.0, info_gain=0.0)

        # ---- replay, interleaved with synthetic D_u programs ----
        replay = eng.replay_batch(REPLAY_PER_CYCLE, gen, max_len=MAX_LEN)

        if shuffled:
            # THE CONTROL: permute the traces, breaking the (spec, trace) pairing.
            specs = [s for s, _ in replay]
            traces = [t for _, t in replay]
            order = torch.randperm(len(traces), generator=gen).tolist()
            stream = [(specs[i], traces[order[i]]) for i in range(len(specs))]
            # how many positions actually changed? a no-op shuffle is a vacuous control
            changed = sum(1 for i in range(len(traces)) if order[i] != i)
            distinguishable.append(changed / max(len(traces), 1))
        else:
            stream = replay

        got, n_pairs, n_verified, n_parse_fail = verify_stream(stream)
        before = len(cumulative)
        cumulative |= got
        rows.append({
            "cycle": cyc,
            "wake_episodes_ingested": len(wake),
            "wake_stored": len(eng.store.wake_items()),
            "replay_pairs": n_pairs,
            "verified_this_cycle": n_verified,
            "verified_fraction": n_verified / max(n_pairs, 1),
            "distinct_verified_this_cycle": len(got),
            "cumulative_distinct_verified": len(cumulative),
            "cumulative_growth": len(cumulative) - before,
            "parse_failures": n_parse_fail,
        })

    return {
        "arm": name,
        "rows": rows,
        "final_cumulative": len(cumulative),
        "monotone": all(rows[i]["cumulative_distinct_verified"]
                        <= rows[i + 1]["cumulative_distinct_verified"]
                        for i in range(len(rows) - 1)),
        "mean_verified_fraction": (sum(r["verified_fraction"] for r in rows) / len(rows)),
        "min_distinguishable_fraction": (min(distinguishable) if distinguishable else 1.0),
    }


print("=" * 78)
print("ARM N -- no-wake invariant (empty store must equal stage1_fictive)")
print("=" * 78)
cfgN = DaydreamConfig(seed=SEED)
engN = DaydreamEngine(None, None, cfgN)
gN = torch.Generator().manual_seed(999)
gN2 = torch.Generator().manual_seed(999)
empty_replay = engN.replay_batch(20, gN, max_len=MAX_LEN)
fictive = engN.stage1_fictive(20, gN2, max_len=MAX_LEN)
byte_identical = (empty_replay == fictive)
R["arm_N_no_wake"] = {
    "byte_identical_to_stage1_fictive": bool(byte_identical),
    "wake_stored": len(engN.store.wake_items()),
    "n": len(empty_replay),
}
print(f"  byte_identical={byte_identical}  wake_stored={len(engN.store.wake_items())}  "
      f"n={len(empty_replay)}")
assert byte_identical, "the no-wake invariant must hold; prior receipts depend on it"

print()
print("=" * 78)
print("ARM R -- real wake ingest, interleaved replay")
print("=" * 78)
armR = run_arm("R_real", shuffled=False)
for r in armR["rows"]:
    print(f"  cyc {r['cycle']}  stored {r['wake_stored']:3d}  replay {r['replay_pairs']:3d}  "
          f"verified {r['verified_fraction']:.3f}  distinct+ {r['distinct_verified_this_cycle']:3d}  "
          f"CUM {r['cumulative_distinct_verified']:4d}  (+{r['cumulative_growth']})")
print(f"  final cumulative = {armR['final_cumulative']}  monotone={armR['monotone']}  "
      f"mean verified {armR['mean_verified_fraction']:.4f}")

print()
print("=" * 78)
print("ARM S -- SHUFFLED pairing control (must NOT ratchet like ARM R)")
print("=" * 78)
armS = run_arm("S_shuffled", shuffled=True)
for r in armS["rows"]:
    print(f"  cyc {r['cycle']}  stored {r['wake_stored']:3d}  replay {r['replay_pairs']:3d}  "
          f"verified {r['verified_fraction']:.3f}  distinct+ {r['distinct_verified_this_cycle']:3d}  "
          f"CUM {r['cumulative_distinct_verified']:4d}  (+{r['cumulative_growth']})")
print(f"  final cumulative = {armS['final_cumulative']}  monotone={armS['monotone']}  "
      f"mean verified {armS['mean_verified_fraction']:.4f}  "
      f"min distinguishable {armS['min_distinguishable_fraction']:.4f}")

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)

real_final, shuf_final = armR["final_cumulative"], armS["final_cumulative"]
margin = (real_final / shuf_final) if shuf_final > 0 else float("inf")
distinguish_ok = armS["min_distinguishable_fraction"] >= MIN_DISTINGUISHABLE

checks = {
    "no_wake_invariant_holds": byte_identical,
    "real_accumulates_monotonically": armR["monotone"],
    "real_grows_beyond_cycle_0": armR["final_cumulative"] > armR["rows"][0]["cumulative_distinct_verified"],
    "control_is_not_a_noop": bool(distinguish_ok),
    "real_beats_control_by_margin": bool(margin >= RATCHET_MARGIN),
    "control_does_not_accumulate": bool(shuf_final < real_final),
}
R["checks"] = checks
R["margin_real_over_shuffle"] = float(margin) if margin != float("inf") else None
R["arm_R"] = armR
R["arm_S"] = armS
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"  margin real/shuffle = {margin:.3f}  (pre-registered >= {RATCHET_MARGIN})")

if not distinguish_ok:
    R["verdict"] = "RATCHET_INCONCLUSIVE_CONTROL_VACUOUS"
elif all(checks.values()):
    R["verdict"] = "RATCHET_CONFIRMED_AT_STORE_LEVEL"
else:
    R["verdict"] = "RATCHET_NOT_CONFIRMED_AT_STORE_LEVEL"
R["residual"] = ("The model-level ratchet (readout competence across cycles) is NOT "
                 "measured. A store-level ratchet is necessary but not sufficient: it shows "
                 "consolidation accumulates verifiable structure, not that the model uses it.")
print(f"\n  VERDICT: {R['verdict']}")
print(f"  residual: {R['residual']}")

out = EV / "henri_ndl_ratchet_observed.json"
out.write_text(json.dumps(R, indent=2) + "\n", encoding="utf-8")
print(f"\n  receipt -> {out.relative_to(REPO)}")
