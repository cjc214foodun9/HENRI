"""Isolate G-U4 from the gate-order confound.

OBSERVED
    exp_positional_causal.py measures M4-G1 first, then G-U4 on the SAME system.
    Its G-U4 OFF arm read 0.9446. The gates battery (cli.py gates) read 0.9060
    for the same metric and bound. One of two things is true:
      (1) run_gates trains system.decoder in place, so the later G-U4 sees a
          trained readout and scores higher; or
      (2) the difference is noise.

    A confounded number is not evidence. Measure G-U4 on a FRESH system BEFORE
    any training, and again after, for both positional arms.

WHAT THIS REPORTS
    u4_clean_off / u4_clean_on   G-U4 on an untouched system
    u4_post_off  / u4_post_on    G-U4 after run_gates trained the readout
    The gap quantifies the confound. Only the clean numbers are comparable with
    the gates battery.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.gates import gate_u4_retention                 # noqa: E402
from henri_core.m4_generative import (M4Config, build_corpus,   # noqa: E402
                                      build_system, run_gates)


def arm(positional: bool, steps: int, n_texts: int,
        seed: int = 1234) -> dict:
    # D128 (self-caught): token_emb and slot_router are nn.Embedding / nn.Linear
    # and take their init from the GLOBAL torch RNG. The ingress is FROZEN, so
    # the slot routing is random per construction. The first version built the
    # OFF and ON systems at different global RNG states, so the two arms differed
    # by more than the positional flag. Observed symptom: G-U4 read 0.9440 here
    # against 0.9060 in the committed battery, for the same metric and bound.
    # Pin the global seed before construction so `positional` is the ONLY
    # difference between the arms.
    torch.manual_seed(seed)
    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, positional=positional)
    t0 = time.time()
    clean = gate_u4_retention(system, tok, n_texts=n_texts)
    m4 = run_gates(system, tok, corpus, M4Config(steps=steps))
    post = gate_u4_retention(system, tok, n_texts=n_texts)
    g1 = next((g for g in m4["gates"] if g["id"] == "M4-G1"), None)
    return {"positional": positional, "elapsed_s": round(time.time() - t0, 1),
            "u4_clean": clean, "u4_post": post,
            "m4_g1": g1["value"], "m4_g1_status": g1["status"],
            "m4_g1_bound": g1["bound"]}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n-texts", type=int, default=1024)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    off = arm(False, args.steps, args.n_texts)
    on = arm(True, args.steps, args.n_texts)

    c_off, c_on = off["u4_clean"], on["u4_clean"]
    p_off, p_on = off["u4_post"], on["u4_post"]

    def g(x, k, d=None):
        return x.get(k, d)

    out = {
        "schema": "henri.positional.isolation.v1",
        "defect": "D127 codec discarded token order (cos('ab','ba')=1.0)",
        "bounds_moved": False,
        "G_U4_clean": {  # comparable with the gates battery
            "metric": g(c_off, "metric"), "bound": g(c_off, "bound"),
            "off": g(c_off, "value"), "on": g(c_on, "value"),
            "delta": round(g(c_on, "value", 0) - g(c_off, "value", 0), 6),
            "status_off": g(c_off, "status"), "status_on": g(c_on, "status"),
            "pos_ctl_off": g(c_off, "positive_control"),
            "neg_ctl_off": g(c_off, "control_value"),
        },
        "G_U4_post_training": {  # confounded by run_gates
            "off": g(p_off, "value"), "on": g(p_on, "value"),
            "delta": round(g(p_on, "value", 0) - g(p_off, "value", 0), 6),
        },
        "confound": {
            "off_gap": round(g(p_off, "value", 0) - g(c_off, "value", 0), 6),
            "on_gap": round(g(p_on, "value", 0) - g(c_on, "value", 0), 6),
            "note": "post - clean. Nonzero means run_gates trains the readout in "
                    "place and only the clean numbers compare with the battery.",
        },
        "M4_G1": {
            "bound": off["m4_g1_bound"],
            "off": off["m4_g1"], "on": on["m4_g1"],
            "delta": round(on["m4_g1"] - off["m4_g1"], 6),
            "status_off": off["m4_g1_status"], "status_on": on["m4_g1_status"],
        },
    }

    print("D127 ISOLATION -- G-U4 on a fresh system vs after training")
    print("=" * 78)
    print(f"G-U4 clean (battery-comparable) bound={out['G_U4_clean']['bound']}")
    print(f"   OFF {out['G_U4_clean']['off']:.6f} ({out['G_U4_clean']['status_off']})"
          f"  ->  ON {out['G_U4_clean']['on']:.6f} ({out['G_U4_clean']['status_on']})"
          f"   delta={out['G_U4_clean']['delta']:+.6f}")
    print(f"   pos_ctl={out['G_U4_clean']['pos_ctl_off']} "
          f"neg_ctl={out['G_U4_clean']['neg_ctl_off']}")
    print(f"G-U4 after training (confounded)")
    print(f"   OFF {out['G_U4_post_training']['off']:.6f}"
          f"  ->  ON {out['G_U4_post_training']['on']:.6f}"
          f"   delta={out['G_U4_post_training']['delta']:+.6f}")
    print(f"CONFOUND (post - clean): off={out['confound']['off_gap']:+.6f} "
          f"on={out['confound']['on_gap']:+.6f}")
    print(f"M4-G1 bound={out['M4_G1']['bound']}")
    print(f"   OFF {out['M4_G1']['off']:.6f}  ->  ON {out['M4_G1']['on']:.6f}"
          f"   delta={out['M4_G1']['delta']:+.6f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
