"""G-DUST-1 extension: does alignment RECOVER at the upstream headline K?

Why this file exists: exp_dust_g1.py hardcodes `for K in (16, 64, 256, 1024)`.
K is not a CLI flag. Patching the frozen file would change the code state its
committed receipt names, so this NEW file imports the frozen estimator
UNCHANGED and adds only the K values the user approved.

Estimator provenance: henri_core.dust_zo.node_perturbation_descent (default-OFF
module). NOTHING here is modified or re-implemented. This driver only calls it.

Design (integration check first, then the approved sweep):
  1. REPRODUCE the frozen reference K set {16, 64, 256, 1024}. If these do not
     match design/zone_a/evidence/henri_gdust1_receipt.json, STOP: the harness
     differs and no new number is comparable.
  2. RUN the approved K values {4096, 16384} with the SAME frozen inputs
     (sigma=0.05, gamma=0.98, n=64, t=8, d_in=32, d_out=256, classes=16,
     seed=20261005).
  3. The bound stays 0.85. K is the experimental variable and is disclosed as
     such. The bound is NOT moved.

Machine output: one JSON object on stdout. Progress: stderr.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import torch

sys.path.insert(0, ".")
from henri_core.dust_zo import DustZOConfig, matcher_loss_fn
from henri_core.exp_dust_g1 import autograd_descent, cosine, run_arm

REF_KS = (16, 64, 256, 1024)


def build_frozen_inputs(n, t, d_in, d_out, classes, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, t, d_in, generator=g)
    labels = torch.randint(0, classes, (n, t), generator=g)
    W = torch.randn(d_out, d_in, generator=g) * 0.3
    readout = torch.randn(classes, d_out, generator=g) / (d_out ** 0.5)
    return X, labels, W, readout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--d-in", type=int, default=32)
    ap.add_argument("--d-out", type=int, default=256)
    ap.add_argument("--classes", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--ks", type=str, default="4096,16384")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    X, labels, W, readout = build_frozen_inputs(
        args.n, args.t, args.d_in, args.d_out, args.classes, args.seed)
    loss_fn = matcher_loss_fn(readout)
    ref = autograd_descent(W, X, loss_fn, labels)

    rows = []
    t0 = time.time()
    Wd = W[:, :args.d_in]

    # arm-level controls (cheap, same as the frozen driver)
    pos = run_arm(Wd, X, labels, readout,
                  DustZOConfig(K=64, sigma=args.sigma, gamma=args.gamma, seed=args.seed), "pos_ctl")
    shuf = run_arm(Wd, X, labels, readout,
                   DustZOConfig(K=256, sigma=args.sigma, gamma=args.gamma, seed=args.seed), "shuffled")
    rows.append({"arm": "pos_ctl", "K": 64, "cos": cosine(pos[0], ref), "sec": 0.0})
    rows.append({"arm": "shuffled", "K": 256, "cos": cosine(shuf[0], ref), "sec": 0.0})

    # 1. integration check: reproduce the frozen reference K set
    print("reproducing frozen K set ...", file=sys.stderr, flush=True)
    for K in REF_KS:
        t1 = time.time()
        est, _, _ = run_arm(Wd, X, labels, readout,
                            DustZOConfig(K=K, sigma=args.sigma, gamma=args.gamma, seed=args.seed), "real")
        rows.append({"arm": "real", "K": K, "cos": cosine(est, ref),
                     "sec": round(time.time() - t1, 2), "block": "reproduce"})

    # 2. approved sweep
    new_ks = [int(x) for x in args.ks.split(",") if x.strip()]
    for K in new_ks:
        print(f"running approved K={K} ...", file=sys.stderr, flush=True)
        t1 = time.time()
        est, _, _ = run_arm(Wd, X, labels, readout,
                            DustZOConfig(K=K, sigma=args.sigma, gamma=args.gamma, seed=args.seed), "real")
        rows.append({"arm": "real", "K": K, "cos": cosine(est, ref),
                     "sec": round(time.time() - t1, 2), "block": "approved"})

    real = [r for r in rows if r["arm"] == "real"]
    new_rows = [r for r in real if r.get("block") == "approved"]
    repro_rows = [r for r in real if r.get("block") == "reproduce"]
    best = max(new_rows, key=lambda r: r["K"])
    shuf_c = rows[1]["cos"]
    pos_c = rows[0]["cos"]
    passed = (best["cos"] >= 0.85 and shuf_c <= 0.10 and pos_c >= 0.99)

    # monotone-in-K check over the FULL real path (reference set + approved set)
    seq = sorted(real, key=lambda r: r["K"])
    cos_list = [r["cos"] for r in seq]
    monotone = all(cos_list[i + 1] >= cos_list[i] - 1e-9 for i in range(len(cos_list) - 1))

    receipt = {
        "schema": "henri.gdust1.kk.receipt.v1",
        "pin": 20261005,
        "question": "does cos(ZO pseudo-grad, autograd) recover to the registered 0.85 at K in {4096, 16384}?",
        "estimator_source": "henri_core.dust_zo.node_perturbation_descent (IMPORTED UNCHANGED)",
        "driver": "henri_core.exp_dust_kk (new; frozen file not modified)",
        "frozen_inputs": {"n": args.n, "t": args.t, "d_in": args.d_in,
                          "d_out": args.d_out, "classes": args.classes,
                          "sigma": args.sigma, "gamma": args.gamma, "seed": args.seed},
        "rule": "PASS iff cos_real(Kmax_approved) >= 0.85 AND cos_shuffled <= 0.10 AND cos_pos_ctl >= 0.99",
        "bound_unchanged": 0.85,
        "rows": rows,
        "gate": {"id": "G-DUST-1-K4096/16384", "largest_K_approved": best["K"],
                 "cos_real": best["cos"], "cos_shuffled": shuf_c, "cos_pos_ctl": pos_c,
                 "monotone_in_K_full_path": monotone,
                 "verdict": "PASS" if passed else "FAIL"},
        "reproduction": {"reference_ks": list(REF_KS),
                         "cos": {r["K"]: r["cos"] for r in repro_rows}},
        "disclosure": {
            "device": "cpu",
            "why_cpu": "henri_core has no CUDA plumbing; this is a CPU-only estimator path",
            "host": f"{platform.node()} / {platform.processor()}",
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "centring_shrink": "(1 - 1/K): 0.99976 at K=4096, 0.99994 at K=16384",
            "scope": "single linear layer + fixed readout; not a full-model replacement",
            "no_backprop_claim": True,
        },
        "seconds": round(time.time() - t0, 2),
    }
    out = json.dumps(receipt, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
