"""Re-characterize G-U4 with the ingress PINNED (the D128 remedy).

WHY
    The sweep proved G-U4 was a random draw: range 0.150874 over 8 construction
    seeds, 2 pass / 6 fail against the unchanged 0.95 bound. Root cause D128:
    CliffordVLASlotEncoder.token_emb (nn.Embedding) and slot_router (nn.Linear)
    take their init from the GLOBAL torch RNG, and the ingress is FROZEN under
    the zero-D_c contract, so routing is a draw that training never corrects.

WHAT THIS MEASURES, bounds unchanged
    Q1 REPRODUCIBILITY  same pinned seed, measured twice -> identical value?
    Q2 THE NUMBER       the pinned G-U4 value, positional OFF and ON
    Q3 THE VERDICT      does the pinned value clear 0.95?

PRE-REGISTERED READING (before the run)
    If the pinned value reproduces exactly, G-U4 becomes a fixed measurement and
    the 0.95 bound is judgeable for the first time.
    If it clears 0.95, the earlier FAILs were draws below the bound.
    If it fails, the shortfall is real at this seed, consistent with the
    cross-scale diagnostic (routing contributes ~0.002; the ceiling is the
    readout feature).
    Either way the measurement becomes honest. No bound moves.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.gates import gate_u4_retention                  # noqa: E402
from henri_core.m4_generative import build_corpus, build_system  # noqa: E402

PIN = 20261004          # the system's own default seed
BOUND = 0.95


def measure(positional: bool, n_texts: int) -> dict:
    corpus = build_corpus(max_len=3, holdout_len=3)
    # D130: pin the WHOLE construction, not just the ingress. See build_system.
    system, tok = build_system(corpus, positional=positional,
                               ingress_seed=PIN, pin_seed=PIN)
    r = gate_u4_retention(system, tok, n_texts=n_texts)
    return {"positional": bool(positional),
            "value": round(float(r["value"]), 6),
            "status": r["status"],
            "positive_control": round(float(r["positive_control"]), 6),
            "negative_control": round(float(r["control_value"]), 6)}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-texts", type=int, default=1024)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    off_a = measure(False, args.n_texts)
    off_b = measure(False, args.n_texts)
    on_a = measure(True, args.n_texts)

    out = {
        "schema": "henri.gate_u4.pinned_characterization.v1",
        "ingress_seed": PIN, "bound": BOUND, "bounds_moved": False,
        "off_run1": off_a, "off_run2": off_b, "on_run1": on_a,
        "reproduces": abs(off_a["value"] - off_b["value"]) < 1e-9,
        "positional_effect": round(on_a["value"] - off_a["value"], 6),
        "defect": "D128: frozen random token_emb/slot_router made G-U4 a draw",
    }

    print("G-U4 PINNED CHARACTERIZATION (ingress_seed=%d, bound %.2f)"
          % (PIN, BOUND))
    print("=" * 78)
    print(f"  OFF run1 : {off_a['value']:.6f}  ({off_a['status']})")
    print(f"  OFF run2 : {off_b['value']:.6f}  ({off_b['status']})")
    print(f"  ON  run1 : {on_a['value']:.6f}  ({on_a['status']})")
    print(f"  reproduces exactly : {out['reproduces']}")
    print(f"  positional effect  : {out['positional_effect']:+.6f}")
    print(f"  positive control   : {off_a['positive_control']:.6f}")
    print(f"  negative control   : {off_a['negative_control']:.6f}")
    print("=" * 78)
    v = off_a["value"]
    print(f"  VERDICT at pinned seed: {'PASS' if v >= BOUND else 'FAIL'}"
          f"  ({v:.6f} vs {BOUND})")
    if v >= BOUND:
        print("  Earlier FAILs were draws below the bound. The value is honest")
        print("  only as PASS-AT-THIS-SEED; the seed dependence remains the defect.")
    else:
        print("  The shortfall is REAL at this seed, not a draw artifact.")
        print("  Consistent with the cross-scale diagnostic: routing contributes")
        print("  ~0.002; the ceiling is the readout feature.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
