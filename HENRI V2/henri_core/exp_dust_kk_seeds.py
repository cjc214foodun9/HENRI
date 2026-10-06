"""Seed replication of G-DUST-1 at K=4096 and K=16384.

Why: the single-seed run (73fcac35...) PASSED, but G-DUST-1 is STOCHASTIC. The
rigor rule is explicit: sweep >= 8 pinned construction seeds before trusting a
single value, and report NOT_REPRODUCIBLE if the range crosses the bound.

This driver imports the frozen estimator and the frozen input builder UNCHANGED.
It changes only the seed, and defaults to 8 seeds at K=4096 (the crossover
region, where the verdict is most fragile) and 4 seeds at K=16384 (4x cost).

Bound stays 0.85. No tuning. Seeds are disclosed.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

import torch

sys.path.insert(0, ".")
from henri_core.dust_zo import DustZOConfig, matcher_loss_fn
from henri_core.exp_dust_g1 import autograd_descent, cosine, run_arm
from henri_core.exp_dust_kk import build_frozen_inputs

SEEDS = [20261005, 20261006, 20261007, 20261008,
         20261009, 20261010, 20261011, 20261012]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--d-in", type=int, default=32)
    ap.add_argument("--d-out", type=int, default=256)
    ap.add_argument("--classes", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--k-wide", type=int, default=4096)
    ap.add_argument("--k-deep", type=int, default=16384)
    ap.add_argument("--n-wide", type=int, default=8)
    ap.add_argument("--n-deep", type=int, default=4)
    ap.add_argument("--bound", type=float, default=0.85)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    arms = [(args.k_wide, SEEDS[:args.n_wide]), (args.k_deep, SEEDS[:args.n_deep])]
    rows = []
    t0 = time.time()
    for K, seeds in arms:
        for s in seeds:
            X, labels, W, readout = build_frozen_inputs(
                args.n, args.t, args.d_in, args.d_out, args.classes, s)
            ref = autograd_descent(W, X, matcher_loss_fn(readout), labels)
            t1 = time.time()
            est, _, _ = run_arm(W[:, :args.d_in], X, labels, readout,
                                DustZOConfig(K=K, sigma=args.sigma, gamma=args.gamma, seed=s), "real")
            c = cosine(est, ref)
            rows.append({"K": K, "seed": s, "cos": c, "sec": round(time.time() - t1, 1)})
            print(f"K={K} seed={s} cos={c:.6f}", file=sys.stderr, flush=True)

    summary = {}
    for K in (args.k_wide, args.k_deep):
        vals = [r["cos"] for r in rows if r["K"] == K]
        summary[str(K)] = {
            "n": len(vals), "min": min(vals), "max": max(vals),
            "mean": statistics.fmean(vals), "median": statistics.median(vals),
            "frac_ge_bound": sum(1 for v in vals if v >= args.bound) / len(vals),
            "range_crosses_bound": min(vals) < args.bound <= max(vals),
            "all_pass": all(v >= args.bound for v in vals),
        }

    verdict = "PASS_REPLICATED"
    for K in (args.k_wide, args.k_deep):
        s = summary[str(K)]
        if s["range_crosses_bound"]:
            verdict = "NOT_REPRODUCIBLE"
        elif not s["all_pass"] and verdict != "NOT_REPRODUCIBLE":
            verdict = "PARTIAL"

    receipt = {
        "schema": "henri.gdust1.seeds.receipt.v1",
        "pin": 20261005,
        "question": "is the G-DUST-1 recovery at K in {4096,16384} reproducible across pinned seeds?",
        "estimator_source": "henri_core.dust_zo.node_perturbation_descent (IMPORTED UNCHANGED)",
        "base_receipt": "henri_gdust1_kk_receipt.json sha256 73fcac354af175b433ebe3ebbadd61cb9cb3ce84efd8a2a361750a4ca39095e9",
        "bound": args.bound,
        "snapshot": {"n": args.n, "t": args.t, "d_in": args.d_in, "d_out": args.d_out,
                     "classes": args.classes, "sigma": args.sigma, "gamma": args.gamma},
        "seeds": SEEDS,
        "rows": rows,
        "summary": summary,
        "verdict": verdict,
        "disclosure": {"device": "cpu", "torch": torch.__version__,
                       "threads": torch.get_num_threads(), "host": platform.node(),
                       "scope": "single linear layer + fixed readout"},
        "seconds": round(time.time() - t0, 1),
    }
    out = json.dumps(receipt, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
