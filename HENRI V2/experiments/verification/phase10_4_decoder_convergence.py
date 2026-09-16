#!/usr/bin/env python3
"""Decoder: is the >9x9 error an INFORMATION limit or an OPTIMIZATION budget?

The follow-up measured exact recovery at <=9x9 but cell accuracy 0.9375 (8x8) and 0.9688
(16x16), with mean_block_cos only ~0.91 -- i.e. the per-block cosine objective had NOT
converged (exact cases sit at ~0.999). Those two explanations make very different claims:

    H_opt  : the decoder converges with more steps / better conditioning  -> fixable
    H_info : the stored wave genuinely cannot determine the grid           -> a real bound

This distinguishes them by sweeping optimization effort at a FIXED grid, and by reporting
the achieved per-block cosine alongside cell accuracy. If cell accuracy tracks the cosine
toward 1.0 as effort rises, the earlier number was my budget, not the encoder's ceiling.
Also tests an exact algebraic variant (no gradient descent) as an independent path.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

V2 = r"C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
sys.path.insert(0, V2)
import torus_encoder_adjoint as TEA                      # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "phase10_4_decoder_convergence_observed.json")


def main():
    t0 = time.time()
    os.environ["HENRI_ENCODER_TORUS"] = "1"
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
    tok = O_VSA_IngressTokenizer(num_blocks=8192, vocab_size=256, device="cpu")
    tok.encode_spatial_grid([[0, 1], [2, 3]])
    enc = tok._torus_encoder

    out = {"schema": "henri.phase10_4.decoder-convergence.v1",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "evidence_class": "OBSERVED", "torch": torch.__version__,
           "question": "is >9x9 decode error an optimization-budget effect or an "
                       "information limit?"}

    rng = np.random.default_rng(20260914)
    print("=" * 76)
    print("Decode at FIXED grid, sweeping optimization effort")
    print("=" * 76)
    rows = []
    for S, V in ((8, 4), (16, 4)):
        grid = rng.integers(0, V, size=(S, S)).astype(np.int64)
        gt = torch.as_tensor(grid, dtype=torch.long)
        w = enc.encode(grid.tolist())
        print(f"\n  --- {S}x{S} V={V} ({V*S*S} unknowns vs 65536 equations) ---")
        for steps, lr in ((400, 0.1), (1500, 0.1), (4000, 0.1), (4000, 0.05)):
            t1 = time.time()
            r = TEA.decode_wave_to_grid(enc, w, S, S, V, steps=steps, lr=lr)
            rec = r["grid"]
            acc = float((rec == gt).float().mean())
            rows.append({"S": S, "V": V, "steps": steps, "lr": lr,
                         "exact": bool(torch.equal(rec, gt)), "cell_accuracy": acc,
                         "mean_block_cos": r["mean_block_cos"],
                         "min_block_cos": r["min_block_cos"],
                         "secs": round(time.time() - t1, 2)})
            print(f"    steps={steps:5d} lr={lr:<5} exact={rows[-1]['exact']!s:5s} "
                  f"cell_acc={acc:.4f} cos={r['mean_block_cos']:.6f} "
                  f"min_cos={r['min_block_cos']:.5f} ({rows[-1]['secs']}s)")
    out["effort_sweep"] = rows

    # ---- independent path: EXACT algebraic solve on the measured accumulator ----
    print()
    print("=" * 76)
    print("Exact algebraic path (no gradient descent): solve for the known accumulator")
    print("=" * 76)
    alg = []
    for S, V in ((8, 4), (16, 4)):
        grid = rng.integers(0, V, size=(S, S)).astype(np.int64)
        A = TEA.dense_operator(enc, S, S, V)
        n = TEA.indicator_from_grid(grid, V)
        acc = A @ n.to(torch.complex64)
        n_rec = torch.linalg.lstsq(A, acc).solution
        grec = n_rec.real.reshape(V, S, S).argmax(0)
        gt = torch.as_tensor(grid, dtype=torch.long)
        acc_ok = float((grec == gt).float().mean())
        err = float((n_rec - n.to(torch.complex64)).abs().max())
        alg.append({"S": S, "V": V, "unknowns": V * S * S,
                    "grid_exact": bool(torch.equal(grec, gt)),
                    "cell_accuracy": acc_ok, "max_indicator_err": err})
        print(f"  {S}x{S} V={V}: unknowns={V*S*S:5d} grid_exact="
              f"{alg[-1]['grid_exact']} cell_acc={acc_ok:.4f} "
              f"max|dn|={err:.2e}")
    out["exact_algebraic"] = alg

    # ---- the decisive comparison ----------------------------------------------
    conv = [r for r in rows if not r["exact"]]
    out["summary"] = {
        "n_effort_runs": len(rows),
        "n_exact": sum(r["exact"] for r in rows),
        "exact_at_highest_effort": {
            f"{r['S']}x{r['S']}": next((x["exact"] for x in reversed(rows)
                                        if x["S"] == r["S"]), None)
            for r in rows},
        "cosines_by_size": {f"{S}x{S}": [r["mean_block_cos"] for r in rows if r["S"] == S]
                            for S in (8, 16)},
        "algebraic_exact_all": bool(all(a["grid_exact"] for a in alg)),
    }
    if conv:
        out["summary"]["worst_nonconverged_cos"] = min(r["mean_block_cos"] for r in conv)
    out["elapsed_secs"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n  exact runs: {out['summary']['n_exact']}/{len(rows)}")
    print(f"  algebraic path exact for all sizes: {out['summary']['algebraic_exact_all']}")
    print(f"=== elapsed {out['elapsed_secs']}s -> {OUT}")


if __name__ == "__main__":
    main()
