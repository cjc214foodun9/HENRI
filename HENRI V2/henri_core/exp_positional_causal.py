"""CAUSAL TEST: does positional algebra move M4-G1 and G-U4?

D127 established that the Zone A codec discarded token order:
cos('ab','ba') was exactly 1.000000. The fix rotates token t's phasor by
t * pos_omega. The unit diagnostic then showed order encoded (cos -> 0.000000),
identity kept (1.000000), set overlap kept (0.500000).

This script asks the causal question. Both gates are re-measured with the
encoder OFF (the committed behaviour) and ON (the fix), against the UNCHANGED
bounds. No threshold moves.

    M4-G1  heldout_token_accuracy, bound = max(unigram floor, no-info arm) + 0.05
    G-U4   decoder information retention, bound = 0.95 (the document's value)

Pre-registered interpretation, written before the run:
    order IS the binding constraint   -> both arms move toward their bounds
    order is NOT the constraint       -> neither moves, and the constraint sits
                                         elsewhere (capacity, readout, or target)
Either outcome is decisive. A null result retires the codec hypothesis.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                       # noqa: E402
from henri_core.gates import gate_u4_retention                # noqa: E402
from henri_core.m4_generative import (M4Config, build_corpus,  # noqa: E402
                                      build_system, run_gates)


def run_arm(positional: bool, steps: int, n_texts: int) -> dict:
    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, positional=positional)
    t0 = time.time()
    m4 = run_gates(system, tok, corpus, M4Config(steps=steps))
    u4 = gate_u4_retention(system, tok, n_texts=n_texts)
    return {"elapsed_s": round(time.time() - t0, 1),
            "m4": m4, "u4": u4}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-texts", type=int, default=1024)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = {"off": run_arm(False, args.steps, args.n_texts),
           "on": run_arm(True, args.steps, args.n_texts)}

    def gate(d, gid):
        return next((g for g in d["m4"]["gates"] if g["id"] == gid), None)

    m_off, m_on = gate(cfg["off"], "M4-G1"), gate(cfg["on"], "M4-G1")
    u_off, u_on = cfg["off"]["u4"], cfg["on"]["u4"]

    out = {
        "schema": "henri.positional.causal_test.v1",
        "defect": "D127 Zone A codec discarded token order; cos('ab','ba')=1.0",
        "bounds_moved": False,
        "config": {"steps": args.steps, "n_texts": args.n_texts},
        "M4_G1": {
            "metric": m_off["metric"], "bound": m_off["bound"],
            "off": m_off["value"], "on": m_on["value"],
            "delta": round(m_on["value"] - m_off["value"], 6),
            "status_off": m_off["status"], "status_on": m_on["status"],
            "positive_control": m_on.get("positive_control_value"),
            "no_info_control": m_on.get("control_value"),
        },
        "G_U4": {
            "metric": u_off.get("metric"), "bound": u_off.get("bound"),
            "off": u_off.get("value"), "on": u_on.get("value"),
            "delta": round(u_on.get("value", 0) - u_off.get("value", 0), 6),
            "status_off": u_off.get("status"), "status_on": u_on.get("status"),
            "positive_control": u_on.get("positive_control"),
            "negative_control": u_on.get("control_value"),
        },
        "timing": {"off_s": cfg["off"]["elapsed_s"], "on_s": cfg["on"]["elapsed_s"]},
    }

    print("D127 CAUSAL TEST -- both gates, bounds unchanged")
    print("=" * 78)
    print(f"M4-G1 {out['M4_G1']['metric']}  bound={out['M4_G1']['bound']}")
    print(f"   OFF {out['M4_G1']['off']:.6f} ({out['M4_G1']['status_off']})"
          f"  ->  ON {out['M4_G1']['on']:.6f} ({out['M4_G1']['status_on']})"
          f"   delta={out['M4_G1']['delta']:+.6f}")
    print(f"   pos_ctl={out['M4_G1']['positive_control']} "
          f"no_info_ctl={out['M4_G1']['no_info_control']}")
    print(f"G-U4 {out['G_U4']['metric']}  bound={out['G_U4']['bound']}")
    print(f"   OFF {out['G_U4']['off']:.6f} ({out['G_U4']['status_off']})"
          f"  ->  ON {out['G_U4']['on']:.6f} ({out['G_U4']['status_on']})"
          f"   delta={out['G_U4']['delta']:+.6f}")
    print(f"   pos_ctl={out['G_U4']['positive_control']} "
          f"neg_ctl={out['G_U4']['negative_control']}")
    print("=" * 78)
    m_moved = abs(out["M4_G1"]["delta"]) > 0.01
    u_moved = abs(out["G_U4"]["delta"]) > 0.01
    print(f"order IS the binding constraint (either gate moved >0.01): "
          f"{'YES' if (m_moved or u_moved) else 'NO'}")
    print(f"  M4-G1 moved: {m_moved}   G-U4 moved: {u_moved}")
    if not (m_moved or u_moved):
        print("  NULL RESULT: the codec-order hypothesis is retired on this config.")
    print(f"timing: {out['timing']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
