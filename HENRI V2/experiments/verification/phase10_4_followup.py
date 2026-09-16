#!/usr/bin/env python3
"""Phase 10.4 follow-up: required rank (directive 3) and decoder scale (directive 4).

TWO QUESTIONS THE FIRST SWEEP LEFT OPEN

1. REQUIRED RANK.
   The canonical run found the ORACLE in-sample ceiling still rising at k=16:
       k=1 0.0995, k=2 0.1268, k=4 0.2619, k=8 0.3519, k=16 0.5118
   against diag_ls 0.7536. A monotone rise with no plateau means the RANK is binding, so
   "ACCEPT: false" is true but incomplete. The useful number is where the ceiling recovers.

   STRUCTURAL LIMIT, stated because it bounds the answer: the oracle basis is the SVD of a
   [60, 2N] matrix of eval tasks' own W*, so its rank is at most 60, and 59 after
   centering. No k above that is constructible this way. That is a limit on the
   EXPERIMENT, not a property of ARC: a higher-rank oracle would need more reference
   tasks. Results are reported with the achievable ceiling stated explicitly.

   The oracle is an UPPER BOUND (it fits on the very tasks being scored), so if even the
   upper bound fails to recover the incumbent ceiling, no transferred subspace can.

2. DECODER SCALE.
   The adjoint recovered all 8 test grids EXACTLY (cell accuracy 1.0000), but every case
   was <= 9x9. A grid is a (H, W, V) family with V*H*W unknowns against 2*NB*SL = 65536
   equations, so the spec is only credible beyond toy sizes. Re-measures at 8x8 .. 32x32.

Writes experiments/verification/phase10_4_followup_observed.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

V2 = r"C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
sys.path.insert(0, V2)
import torus_encoder_adjoint as TEA                      # noqa: E402

ARC_ROOT = os.environ.get("ARC_CORPUS", "C:/Users/chan/henri_data/ARC-AGI/data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "phase10_4_followup_observed.json")

BASELINE_HELD = 0.42150651891715823
BASELINE_CEIL = 0.753553180727694
RANK_KS = [int(x) for x in os.environ.get("FOLLOWUP_KS",
                                          "16,24,32,40,48,56").split(",")]
DECODE_SIZES = [int(x) for x in os.environ.get("FOLLOWUP_SIZES", "8,16,24,32").split(",")]
LAM = 1e-3


def load_tasks(root, split, skip=0, n=60):
    d = os.path.join(root, split)
    out = []
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        try:
            t = json.load(open(os.path.join(d, f), encoding="utf-8"))
        except Exception:
            continue
        if skip > 0:
            skip -= 1
            continue
        out.append((f, t))
        if len(out) >= n:
            break
    return out


def main():
    t0 = time.time()
    os.environ["HENRI_ENCODER_TORUS"] = "1"
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
    tok = O_VSA_IngressTokenizer(num_blocks=8192, vocab_size=256, device="cpu")
    tok.encode_spatial_grid([[0, 1], [2, 3]])
    enc = tok._torus_encoder
    co, to_real = enc._to_complex, enc._to_real
    NB_, SL_ = int(enc.num_blocks), 4
    N = NB_ * SL_

    cos = lambda a, b: float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())
    realify = lambda c: to_real(c.reshape(NB_, SL_))
    flat_c = lambda g: co(enc.encode(g.tolist())).reshape(-1)
    flat_wave = lambda g: enc.encode(g.tolist()).flatten()
    sm = lambda c: torch.cat([c.real, c.imag])

    out = {"schema": "henri.phase10_4.followup.v1",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "evidence_class": "OBSERVED", "device_kind": "cpu",
           "torch": torch.__version__, "n_blocks": NB_, "D_flat_complex": N,
           "baseline": {"diag_ls_held": BASELINE_HELD, "diag_ls_ceiling": BASELINE_CEIL}}

    # ---------------- part 1: required rank ---------------------------------
    print("=" * 76)
    print("PART 1 -- REQUIRED RANK: does the ORACLE recover the incumbent ceiling?")
    print("=" * 76)
    tasks = load_tasks(ARC_ROOT, "training", 0, 60)
    recs = []
    for tid, t in tasks:
        try:
            dm = [(np.asarray(p["input"], dtype=np.int64),
                   np.asarray(p["output"], dtype=np.int64)) for p in t["train"][:3]]
            te = t["test"][0]
            Xc = torch.stack([flat_c(a) for a, _ in dm])
            Yc = torch.stack([flat_c(b) for _, b in dm])
            Wd = (Xc.conj() * Yc).sum(0) / ((Xc.abs() ** 2).sum(0) + 1e-9)
            recs.append(dict(tid=tid, Xc=Xc, Yc=Yc, Wd=Wd,
                             xtc=flat_c(np.asarray(te["input"], dtype=np.int64)),
                             yt_r=flat_wave(np.asarray(te["output"], dtype=np.int64)),
                             ytr=[flat_wave(b) for _, b in dm]))
        except Exception:
            pass
    print(f"  eval recs: {len(recs)}")

    Mev = torch.stack([sm(r["Wd"]) for r in recs])
    Mev = Mev - Mev.mean(0)
    _, Sv, Vte = torch.linalg.svd(Mev, full_matrices=False)
    ev = (Sv ** 2) / (Sv ** 2).sum()
    max_k = int((Sv > 1e-9 * Sv[0]).sum())
    print(f"  oracle basis rank available: {max_k} "
          f"(SVD of [{len(recs)}, {2*N}] -> rank <= {min(len(recs), 2*N)})")
    print(f"  eval-W* variance: top1={ev[0]:.4f} top8={ev[:8].sum():.4f} "
          f"top16={ev[:16].sum():.4f} top32={ev[:32].sum():.4f}")
    out["oracle_rank_available"] = max_k
    out["eval_w_variance"] = {"top1": float(ev[0]), "top8": float(ev[:8].sum()),
                              "top16": float(ev[:16].sum()), "top32": float(ev[:32].sum()),
                              "top56": float(ev[:56].sum())}

    ureal = torch.stack([r["Wd"].abs() for r in recs])
    out["oracle_w_abs"] = {"median": float(ureal.median()),
                           "p99": float(ureal.flatten().quantile(0.99)),
                           "max": float(ureal.max())}
    print(f"  |W*| per-slot: median={out['oracle_w_abs']['median']:.3e} "
          f"p99={out['oracle_w_abs']['p99']:.3e} max={out['oracle_w_abs']['max']:.3e}")

    sweep = {}
    for k in RANK_KS:
        if k > max_k:
            print(f"  k={k:3d} SKIPPED: exceeds available oracle rank {max_k}")
            sweep[f"{k}"] = {"skipped": f"rank {max_k} < k"}
            continue
        tk = time.time()
        U = Vte[:k].t().contiguous()
        Ur, Ui = U[:N, :], U[N:, :]
        hs, cs = [], []
        for r in recs:
            Xc, Yc = r["Xc"], r["Yc"]
            G = torch.zeros(k, k)
            rhs = torch.zeros(k)
            for m in range(Xc.shape[0]):
                xr, xi = Xc[m].real, Xc[m].imag
                Am = torch.cat([xr[:, None] * Ur - xi[:, None] * Ui,
                                xi[:, None] * Ur + xr[:, None] * Ui], dim=0)
                G += Am.t() @ Am
                rhs += Am.t() @ torch.cat([Yc[m].real, Yc[m].imag])
            G += LAM * torch.eye(k)
            c = torch.linalg.solve(G, rhs)
            w = U @ c
            wc = torch.complex(w[:N], w[N:])
            hs.append(cos(realify(wc * r["xtc"]), r["yt_r"]))
            cs.append(float(np.mean([cos(realify(wc * xc), yr)
                                     for xc, yr in zip(Xc, r["ytr"])])))
        sweep[f"{k}"] = {"held_out_mean": float(np.mean(hs)),
                         "in_sample_ceiling_mean": float(np.mean(cs)),
                         "gap": float(np.mean(cs) - np.mean(hs)),
                         "secs": round(time.time() - tk, 1)}
        print(f"  k={k:3d}  held={sweep[f'{k}']['held_out_mean']:+.4f} "
              f"ceiling={sweep[f'{k}']['in_sample_ceiling_mean']:+.4f} "
              f"gap={sweep[f'{k}']['gap']:.4f}  ({sweep[f'{k}']['secs']}s)")
        del U, Ur, Ui
    out["oracle_rank_sweep"] = sweep

    recovered = [int(k) for k in RANK_KS
                 if str(k) in sweep and "in_sample_ceiling_mean" in sweep[str(k)]
                 and sweep[str(k)]["in_sample_ceiling_mean"] >= 0.95 * BASELINE_CEIL]
    best_ceil = max((v["in_sample_ceiling_mean"] for v in sweep.values()
                     if "in_sample_ceiling_mean" in v), default=float("nan"))
    out["required_rank"] = {
        "incumbent_ceiling": BASELINE_CEIL,
        "best_oracle_ceiling_achieved": best_ceil,
        "ks_reaching_95pct_of_incumbent_ceiling": recovered,
        "first_k_at_95pct": min(recovered) if recovered else None,
        "max_constructible_rank": max_k,
        "recovered_within_available_rank": bool(recovered),
        "note": ("oracle is an UPPER BOUND (fits on the scored tasks). If it cannot "
                 "recover the ceiling, no transferred subspace can. Rank ceiling here is "
                 "bounded by the 60 reference tasks, not by ARC."),
    }
    print(f"\n  best oracle ceiling: {best_ceil:+.4f} vs incumbent {BASELINE_CEIL:+.4f}")
    print(f"  ks within 95% of incumbent ceiling: {recovered or 'NONE'}")

    # ---------------- part 2: decoder scale ---------------------------------
    print()
    print("=" * 76)
    print("PART 2 -- DECODER SCALE: does exact recovery hold beyond 9x9?")
    print("=" * 76)
    dec = []
    rng = np.random.default_rng(20260914)
    for S in DECODE_SIZES:
        V = 4
        grid = rng.integers(0, V, size=(S, S)).astype(np.int64)
        w = enc.encode(grid.tolist())
        t1 = time.time()
        r = TEA.decode_wave_to_grid(enc, w, S, S, V, steps=400, lr=0.1)
        rec = r["grid"]
        gt = torch.as_tensor(grid, dtype=torch.long)
        row = {"S": S, "V": V, "unknowns": V * S * S, "equations": 2 * N,
               "exact": bool(torch.equal(rec, gt)),
               "cell_accuracy": float((rec == gt).float().mean()),
               "mean_block_cos": r["mean_block_cos"], "min_block_cos": r["min_block_cos"],
               "secs": round(time.time() - t1, 2)}
        dec.append(row)
        print(f"  {S}x{S} V={V}: unknowns={row['unknowns']:5d} eqs={row['equations']} "
              f"exact={row['exact']} cell_acc={row['cell_accuracy']:.4f} "
              f"cos={row['mean_block_cos']:.5f} ({row['secs']}s)")
        del grid
    out["decoder_scale"] = dec
    out["decoder_summary"] = {
        "n_sizes": len(dec), "n_exact": sum(d["exact"] for d in dec),
        "all_exact": bool(all(d["exact"] for d in dec)),
        "min_block_cos": min(d["min_block_cos"] for d in dec)}

    out["elapsed_secs"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n=== elapsed {out['elapsed_secs']}s -> {OUT}")


if __name__ == "__main__":
    main()
