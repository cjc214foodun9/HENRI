"""RESOLVE THE CONTRADICTION + TEST THE RAW-SIMILARITY ROUTE.

TWO QUESTIONS, measured, multiple pins.

Q1 (mechanism hygiene). With |z_k| CONSTANT, argmin L must equal argmax Re<.>
    EXACTLY. My prior probe reported argmin_eq_argmax_frac varying 0.758..0.960,
    which a monotone rescale cannot produce. Decide: float32 tie-breaking, or a
    real effect? Compare elementwise.

Q2 (is there a positive?). The blueprint disparages the raw wave similarity
    S = Re<psi_q, psi_k> as "uncalibrated". Measured on the FULL bank:
        raw similarity   top1 0.0000  cov5 0.3558
        ridge ranker     top1 0.0317  cov5 0.1142
        row-uniform rnd  top1 0.0534  cov5 0.2277
    Raw similarity has the BEST coverage and the WORST top-1. Hypothesis: its
    top-5 are duplicates of ONE trace, so dedup converts coverage into top-1.
    If so, the dedup bank restores the head WITHOUT any trained readout.

PRE-REGISTERED
  Q1 bar : argmax(Re<.>) == argmin(1 - Re<.>) must hold at frac 1.000000.
           Any deviation is a float artifact and is reported as such.
  Q2 bar : on the DEDUP bank, raw-similarity top1 > ridge top1 on the same bank.
  metric : every number states its candidate set (full bank vs dedup bank).
  pins   : 20261010, 20262244, 20271009.

NO TRAINING. Analysis only. CPU, D=4096.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import torch
import torch.nn.functional as F

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

import henri_core.exp_attributable_veto as vt                              # noqa: E402

sel = vt.sel


def run_pin(pin):
    inputs = sel.make_inputs(sel.N_INPUTS)
    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN,
                              inputs=inputs)
    system, tok = sel.build_system(corpus, pin_seed=pin)
    tr, ho = corpus.train_idx, corpus.heldout_idx
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace

    gen = sel.WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ztr = F.normalize(torch.nan_to_num(sel.flat(gen.wave(
            [corpus.specs[i] for i in tr]))), dim=-1)
        Zho = F.normalize(torch.nan_to_num(sel.flat(gen.wave(
            [corpus.specs[i] for i in ho]))), dim=-1)

    nq, n = len(ho_tgt), len(tr_tgt)
    Sm = Zho @ Ztr.T
    if torch.is_complex(Sm):
        Sm = Sm.real
    R = {"pin": str(pin), "n_train": n, "n_held": nq}

    def gold(ix_list):
        """labels[q] = set of cand indices whose trace equals held-out q's trace"""
        tset = {}
        for c, j in enumerate(ix_list):
            tset.setdefault(tuple(tr_tgt[j]), []).append(c)
        return [tset.get(tuple(ho_tgt[q]), []) for q in range(nq)]

    def Ymat(tg):
        Y = torch.zeros(len(tg), M * Vv)
        for i, ids in enumerate(tg):
            for p, t in enumerate(ids[:M]):
                Y[i, p * Vv + int(t)] = 1.0
        return Y

    def evaluate(S, ix_list, tag):
        """S [nq, K] higher=better. ix_list[c] -> global train index."""
        K = S.shape[1]
        gl = gold(ix_list)
        order = torch.argsort(-S, dim=1)
        out = {}
        for k in (1, 5, 20):
            hits = sum(1 for q in range(nq)
                       if any(int(order[q, c]) in gl[q] for c in range(min(k, K))))
            out[f"top{k}"] = round(hits / nq, 4)
        out["mean_tie_at_max"] = round(float(
            (S.max(1).values.unsqueeze(1) == S).sum(1).float().mean()), 3)
        out["frac_untied"] = round(float(
            (S.max(1).values.unsqueeze(1) == S).sum(1).eq(1).float().mean()), 4)
        # how many DISTINCT traces in the top-5
        d5 = []
        for q in range(nq):
            d5.append(len({tuple(tr_tgt[ix_list[int(c)]])
                           for c in order[q, :min(5, K)]}))
        out["mean_distinct_top5"] = round(sum(d5) / len(d5), 3)
        return out

    # ---- Q1: elementwise tie check on the constant-length claim ----------
    Lt = torch.tensor([float(len(corpus.traces[i].encode("utf-8")) * 8)
                       for i in tr])
    Delta = 1.0 - Sm
    R["Q1_trace_bits_constant"] = bool(Lt.std().item() == 0.0)
    R["Q1_argmax_eq_argmin_frac"] = round(float(
        (Sm.argmax(1) == Delta.argmin(1)).float().mean()), 6)
    L2 = Lt.unsqueeze(0) + Delta / 0.038316 * (1.0 / 0.6931471805599453)
    R["Q1_const_len_arm_eq_argmax_frac"] = round(float(
        (L2.argmin(1) == Sm.argmax(1)).float().mean()), 6)
    R["Q1_delta_min"] = round(float(Delta.min()), 6)
    R["Q1_delta_max"] = round(float(Delta.max()), 6)

    # ---- Q2: raw similarity on FULL bank ---------------------------------
    R["full_raw_sim"] = evaluate(Sm, list(range(n)), "full_raw")
    R["full_row_uniform_estimated"] = None
    g = torch.Generator().manual_seed(12345)
    rr = {1: [], 5: [], 20: []}
    for _ in range(10):
        jj = torch.randint(0, n, (nq,), generator=g)
        gl = gold(list(range(n)))
        for k in (1, 5, 20):
            jk = torch.randint(0, n, (nq, k), generator=g)
            rr[k].append(sum(1 for q in range(nq)
                             if any(int(jk[q, c]) in gl[q] for c in range(k)))
                         / nq)
    R["full_row_uniform"] = {f"top{k}": round(sum(v) / len(v), 4)
                             for k, v in rr.items()}

    # ---- ridge ranker on the FULL bank (same rows) -----------------------
    Yall = Ymat(tr_tgt)
    sc = sel.ridge_scores(Ztr, Yall, Zho).reshape(nq, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc, Yall.reshape(n, M, Vv))
    R["full_ridge"] = evaluate(C, list(range(n)), "full_ridge")

    # ---- Q2: DEDUP bank --------------------------------------------------
    rep = {}
    for j, t in enumerate(tr_tgt):
        rep.setdefault(tuple(t), j)
    ded = sorted(rep.values())
    nd = len(ded)
    R["dedup_candidates"] = nd
    Sd = Sm[:, ded]
    Rd_ = evaluate(Sd, ded, "dedup_raw")
    R["dedup_raw_sim"] = Rd_
    Yded = Ymat([tr_tgt[j] for j in ded])
    scd = sel.ridge_scores(Ztr[ded], Yded, Zho).reshape(nq, M, Vv)
    Cd = torch.einsum("qmv,jmv->qj", scd, Yded.reshape(nd, M, Vv))
    R["dedup_ridge"] = evaluate(Cd, ded, "dedup_ridge")
    g2 = torch.Generator().manual_seed(999)
    rr2 = {1: [], 5: [], 20: []}
    gl_d = gold(ded)
    for _ in range(10):
        for k in (1, 5, 20):
            jk = torch.randint(0, nd, (nq, k), generator=g2)
            rr2[k].append(sum(1 for q in range(nq)
                              if any(int(jk[q, c]) in gl_d[q] for c in range(k)))
                          / nq)
    R["dedup_row_uniform"] = {f"top{k}": round(sum(v) / len(v), 4)
                              for k, v in rr2.items()}

    R["verdict_pin"] = {
        "raw_sim_beats_ridge_on_dedup_top1": bool(
            Rd_["top1"] > R["dedup_ridge"]["top1"]),
        "raw_sim_beats_ridge_on_dedup_top5": bool(
            Rd_["top5"] > R["dedup_ridge"]["top5"]),
        "dedup_converts_coverage_to_top1": bool(
            Rd_["top1"] > R["full_raw_sim"]["top1"]),
    }
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", default="20261010,20262244,20271009")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pins = [int(p) for p in a.pins.split(",") if p.strip()]
    res = {}
    for p in pins:
        res[str(p)] = run_pin(p)
        print(f"[pin {p}] done", flush=True)
    agg = {
        "schema": "henri.raw_similarity_route.receipt.v1",
        "results": res,
    }
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
