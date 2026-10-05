"""Is G-U4 a REPRODUCIBLE measurement? Sweep the construction seed.

WHY
    Four G-U4 values for the same metric and bound now exist:
        0.906043  committed gates_v2_receipt.json
        0.847784  fresh cli.py gates run B
        0.937131  isolation, seed 1234
        0.944001  isolation, first unseeded run
    Spread ~0.096. D128 says why: CliffordVLASlotEncoder.token_emb is an
    nn.Embedding and slot_router is an nn.Linear. Both take their init from the
    GLOBAL torch RNG. The ingress is FROZEN by the zero-D_c contract, so the
    learned routing is random per construction and never corrected by training.
    G-U4 therefore reads a DRAW, not a fixed property.

WHAT THIS MEASURES
    G-U4 on a fresh, untrained system for a set of pinned global seeds.
    Reported: every value, the min, the max, the range, the mean, and how many
    draws clear the document's 0.95 bound.

    If the range crosses 0.95, the gate cannot decide: the verdict depends on
    construction luck. That is a harness defect, not a model property.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.gates import gate_u4_retention                 # noqa: E402
from henri_core.m4_generative import build_corpus, build_system  # noqa: E402


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-texts", type=int, default=1024)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    corpus = build_corpus(max_len=3, holdout_len=3)
    rows = []
    for i in range(args.seeds):
        seed = 1000 + i
        torch.manual_seed(seed)          # D128: pin so `seed` is the only variable
        system, tok = build_system(corpus)
        r = gate_u4_retention(system, tok, n_texts=args.n_texts)
        rows.append({"seed": seed, "value": round(float(r["value"]), 6),
                     "status": r["status"],
                     "positive_control": round(float(r["positive_control"]), 6),
                     "negative_control": round(float(r["control_value"]), 6)})
        print(f"  seed {seed}  G-U4={r['value']:.6f}  {r['status']:<5}"
              f"  pos_ctl={r['positive_control']:.6f}"
              f"  neg_ctl={r['control_value']:.6f}")

    vals = [r["value"] for r in rows]
    bound = 0.95
    out = {
        "schema": "henri.gate_u4.reproducibility.v1",
        "metric": "heldout_slot_profile_r2",
        "bound": bound, "bounds_moved": False,
        "n_draws": len(vals),
        "values": vals, "seeds": [r["seed"] for r in rows],
        "min": round(min(vals), 6), "max": round(max(vals), 6),
        "range": round(max(vals) - min(vals), 6),
        "mean": round(statistics.fmean(vals), 6),
        "std": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
        "n_pass": sum(1 for v in vals if v >= bound),
        "n_fail": sum(1 for v in vals if v < bound),
        "estimator_sane_every_draw": all(r["positive_control"] > 0.99 for r in rows),
        "controls_collapse_every_draw": all(abs(r["negative_control"]) < 1e-6
                                            for r in rows),
        "verdict": ("REPRODUCIBLE" if (max(vals) - min(vals)) < 0.01
                    else "NOT_REPRODUCIBLE"),
        "defect": "D128: frozen randomly-initialised ingress makes G-U4 a random "
                  "draw; the reported value depends on construction RNG state.",
        "rows": rows,
    }

    print("=" * 78)
    print(f"G-U4 reproducibility over {len(vals)} pinned seeds, bound {bound}")
    print(f"  min {out['min']:.6f}  max {out['max']:.6f}  range {out['range']:.6f}")
    print(f"  mean {out['mean']:.6f}  std {out['std']:.6f}")
    print(f"  pass {out['n_pass']}  fail {out['n_fail']}")
    print(f"  estimator sane on every draw : {out['estimator_sane_every_draw']}")
    print(f"  negative control 0 on every draw: {out['controls_collapse_every_draw']}")
    print(f"  VERDICT: {out['verdict']}")
    if out["range"] >= (bound - min(vals)) and max(vals) >= bound:
        print("  The draw range CROSSES the bound: the gate cannot decide.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
