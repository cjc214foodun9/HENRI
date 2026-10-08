"""MDL LENGTH PROBE + PLATEAU FALSIFIER -- two pure-analysis tests, one setup.

PART A: can the two-part MDL objective discriminate at all?   (the cheap kill)
PART B: the plateau falsifier (dedup bank, drop top-4, random on the SAME set).

GROUNDING (paper bytes, 181,849 chars,
C:/Users/chan/AppData/Local/hermes/cache/web/arxiv.org-4e74096038.md)
  IN PAPER    : two-part universal codes (S4); adaptive-GMM variational objective
                (S5.1); optimizer-failure-from-random-init (S6.1, C.2).
  IN PAPER    : Kraft's inequality -- 4 hits, ALL prefix-code theory
                (lines 923, 925, 952, 1117).
  NOT IN PAPER: Sagnac, Hopfield, HENRI, "candidate ranker" -- 0 hits. The S2
                mapping table is HENRI-side design, not paper-derived.
  BLUEPRINT DEFECT: Table 4 quoted as "Random 288.0 53.9 100%" is the 1e-3 SWEEP
                row (bytes 2794). Untuned random init is bytes 2791: KL 2.87 /
                NLL 2640 / acc 54%. Also bytes 2785: MLP init is "manually chosen
                weights ... inspired by how the ALTA compiler generates MLP
                layers" -- NOT an ALTA compile.

CANDIDATE SCORE RECIPE (taken from exp_bank_structure_dedup.py:99-103, my own
verified code -- NOT re-derived):
    sc = ridge_scores(Ztr[ix], Ymat(traces[ix]), Zho).reshape(nq, M, Vv)
    C  = einsum("qmv,jmv->qj", sc, Ymat(traces[ix]))      # [nq, n] higher=better

PRE-REGISTERED BARS
  A-kill : argmin L == argmax C on >99% of queries      -> MDL is a no-op
  A-kill : corr(|z|, precision) <= 0                     -> length penalty hurts
  B-bar  : dedup rank-4-excluded arm > row-uniform random on the SAME dedup set
  refs   : ranker top1 0.0317 | row-uniform random top1 0.0522 | analytic chance@5 0.2353
  metric : every number states its candidate set (full bank vs dedup bank)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

import henri_core.exp_attributable_veto as vt                              # noqa: E402

sel = vt.sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default="20261010")
    ap.add_argument("--taus", default="0.038316,0.1,0.5,1,4,16,64,256")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pin = int(a.pin) if a.pin.isdigit() else a.pin
    taus = [float(x) for x in a.taus.split(",") if x]

    inputs = sel.make_inputs(sel.N_INPUTS)
    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN,
                              inputs=inputs)
    system, tok = sel.build_system(corpus, pin_seed=pin)
    tr = corpus.train_idx
    ho = corpus.heldout_idx
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho]
    tr_prog = [corpus.programs[i] for i in tr]
    ho_prog = [corpus.programs[i] for i in ho]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace

    gen = sel.WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ztr = F.normalize(torch.nan_to_num(sel.flat(gen.wave(
            [corpus.specs[i] for i in tr]))), dim=-1)
        Zho = F.normalize(torch.nan_to_num(sel.flat(gen.wave(
            [corpus.specs[i] for i in ho]))), dim=-1)

    nq, n = len(ho_tgt), len(tr_tgt)
    R: dict = {"pin": str(pin), "n_train_rows": n, "n_held_rows": nq,
               "candidate_set_full_bank": n}

    def Ymat(tg):
        Y = torch.zeros(len(tg), M * Vv)
        for i, ids in enumerate(tg):
            for p, t in enumerate(ids[:M]):
                Y[i, p * Vv + int(t)] = 1.0
        return Y

    def top1_of(ix_pick):
        return round(sum(1 for q in range(nq)
                         if tuple(tr_tgt[int(ix_pick[q])]) == tuple(ho_tgt[q]))
                     / nq, 4)

    def cov5_of(C, ix_map):
        tk = torch.topk(C, min(5, C.shape[1]), dim=1).indices
        return round(sum(1 for q in range(nq)
                         if any(tuple(tr_tgt[int(ix_map[int(j)])]) == tuple(ho_tgt[q])
                                for j in tk[q])) / nq, 4)

    # ---- full-bank candidate scores (verified recipe) --------------------
    Yall = Ymat(tr_tgt)
    sc = sel.ridge_scores(Ztr, Yall, Zho).reshape(nq, M, Vv)
    Yr = Yall.reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc, Yr)                  # [nq, n]
    ix_map = list(range(n))

    R["A0_C_min"] = round(float(C.min()), 4)
    R["A0_C_max"] = round(float(C.max()), 4)
    R["A0_ranker_top1"] = top1_of(C.argmax(1).tolist())
    R["A0_ranker_cov5"] = cov5_of(C, ix_map)

    # ---------- lengths ---------------------------------------------------
    trace_bits = [len(corpus.traces[i].encode("utf-8")) * 8 for i in tr]
    prog_bits = [len(p.encode("utf-8")) * 8 for p in tr_prog]
    Lt = torch.tensor([float(b) for b in trace_bits])
    Lp = torch.tensor([float(b) for b in prog_bits])
    R["A1_trace_bits_distinct"] = sorted({int(b) for b in trace_bits})
    R["A1_prog_bits_distinct"] = sorted({int(b) for b in prog_bits})
    R["A1_trace_bits_constant"] = (Lt.std().item() == 0.0)
    R["A1_prog_bits_constant"] = (Lp.std().item() == 0.0)

    # precision = P(candidate correct | query) averaged over queries
    corr_ok = torch.zeros(nq, n, dtype=torch.float32)
    ho_set = {tuple(t) for t in ho_tgt}
    for q in range(nq):
        hq = tuple(ho_tgt[q])
        for j in range(n):
            if tuple(tr_tgt[j]) == hq:
                corr_ok[q, j] = 1.0
    prec = corr_ok.mean(0)

    def corr_with(x):
        if float(x.std()) == 0.0 or float(prec.std()) == 0.0:
            return None
        return round(float(torch.corrcoef(torch.stack([x, prec]))[0, 1]), 4)

    R["A2_corr_tracebits_precision"] = corr_with(Lt)
    R["A2_corr_progbits_precision"] = corr_with(Lp)
    R["A2_prec_mean"] = round(float(prec.mean()), 6)
    R["A2_queries_with_gold_in_bank"] = int(
        sum(1 for t in ho_tgt if tuple(t) in {tuple(x) for x in tr_tgt}))

    # Kraft over the candidate set actually present (distinct traces)
    uniq_bits = sorted({len(corpus.traces[i].encode("utf-8")) * 8
                        for i in tr})
    kraft = sum(2.0 ** (-b) for b in uniq_bits)
    R["A3_kraft_sum_distinct_traces"] = kraft
    R["A3_kraft_holds"] = bool(kraft <= 1.0 + 1e-12)
    R["A3_kraft_note"] = ("reported, NOT asserted: N candidate traces are not "
                          "one prefix code")

    # ---------- MDL sweep (full bank) -------------------------------------
    D = (C.max() - C)                                        # higher divergence = worse
    sweep = {}
    for tau in taus:
        data_bits = (D / tau) * (1.0 / math.log(2.0))
        for name, Lv in (("trace", Lt), ("prog", Lp)):
            L = Lv.unsqueeze(0) + data_bits
            ag = L.argmin(1)
            sweep[f"tau={tau:g}|len={name}"] = {
                "top1": top1_of(ag.tolist()),
                "cov5": cov5_of(-L, ix_map),
                "argmin_eq_argmax_frac": round(
                    float((ag == C.argmax(1)).float().mean()), 6),
                "mean_data_bits": round(float(data_bits.mean()), 4),
                "mean_len_bits": round(float(Lv.mean()), 4),
                "data_over_len": round(float(data_bits.mean())
                                       / max(1e-9, float(Lv.mean())), 4),
            }
    R["A5_mdl_sweep"] = sweep

    # ---------- row-uniform random control (full bank) --------------------
    g = torch.Generator().manual_seed(12345)
    draws = [top1_of(torch.randint(0, n, (nq,), generator=g).tolist())
             for _ in range(5)]
    R["A6_row_uniform_random_top1"] = round(sum(draws) / len(draws), 4)

    # ================= PART B: PLATEAU FALSIFIER =========================
    # dedup candidate bank: one representative per distinct trace
    rep = {}
    for j, t in enumerate(tr_tgt):
        rep.setdefault(tuple(t), j)
    ded = sorted(rep.values())
    nd = len(ded)
    Zded = Ztr[ded]
    Yded = Ymat([tr_tgt[j] for j in ded])
    scd = sel.ridge_scores(Zded, Yded, Zho).reshape(nq, M, Vv)
    Yrd = Yded.reshape(nd, M, Vv)
    Cd = torch.einsum("qmv,jmv->qj", scd, Yrd)               # [nq, nd]

    B = {"B0_dedup_candidates": nd}
    B["B1_ranker_top1"] = round(sum(
        1 for q in range(nq)
        if tuple(tr_tgt[ded[int(Cd[q].argmax())]]) == tuple(ho_tgt[q])) / nq, 4)

    def topk_trace_hit(Cm, k):
        tk = torch.topk(Cm, min(k, Cm.shape[1]), dim=1).indices
        return round(sum(1 for q in range(nq)
                         if any(tuple(tr_tgt[ded[int(j)]]) == tuple(ho_tgt[q])
                                for j in tk[q])) / nq, 4)

    B["B2_cov1"] = topk_trace_hit(Cd, 1)
    B["B3_cov5"] = topk_trace_hit(Cd, 5)
    B["B4_cov10"] = topk_trace_hit(Cd, 10)
    B["B5_cov20"] = topk_trace_hit(Cd, 20)

    # rank-4-excluded arm: argmax over ranks >= 4 (0-indexed)
    exc = torch.topk(Cd, min(nd, 20), dim=1).indices
    pick_excl = [int(exc[q, 4 + int(Cd[q, exc[q, 4:]].argmax())]) for q in range(nq)]
    B["B6_rank4_excluded_top1"] = round(sum(
        1 for q in range(nq)
        if tuple(tr_tgt[ded[pick_excl[q]]]) == tuple(ho_tgt[q])) / nq, 4)

    # row-uniform random over the SAME dedup set
    gr = torch.Generator().manual_seed(999)
    rd = []
    for _ in range(5):
        jj = torch.randint(0, nd, (nq,), generator=gr)
        rd.append(round(sum(1 for q in range(nq)
                            if tuple(tr_tgt[ded[int(jj[q])]]) == tuple(ho_tgt[q]))
                        / nq, 4))
    B["B7_row_uniform_random_top1_same_dedup"] = round(sum(rd) / len(rd), 4)
    B["B8_analytic_chance_top1_dedup"] = round(
        sum(1 for t in ho_tgt if tuple(t) in {tuple(x) for x in tr_tgt})
        / nq / nd, 6)

    # untied fractions
    B["B9_frac_untied_full"] = round(float(
        (C.max(1).values.unsqueeze(1) == C).sum(1).eq(1).float().mean()), 6)
    B["B10_frac_untied_dedup"] = round(float(
        (Cd.max(1).values.unsqueeze(1) == Cd).sum(1).eq(1).float().mean()), 6)

    R["B_plateau"] = B

    # ---------- verdicts as DATA (no hardcoded labels) --------------------
    a_kill_len = bool(any(v["argmin_eq_argmax_frac"] > 0.99
                          for v in sweep.values()))
    ct = R["A2_corr_tracebits_precision"]
    cp = R["A2_corr_progbits_precision"]
    R["verdict_A_mdl"] = {
        "no_op_when_argmin_equals_argmax": a_kill_len,
        "trace_len_constant": R["A1_trace_bits_constant"],
        "corr_trace_len_precision": ct,
        "corr_prog_len_precision": cp,
        "length_penalty_helps": bool((ct is not None and ct > 0)
                                     or (cp is not None and cp > 0)),
    }
    R["verdict_B_plateau"] = {
        "dedup_top1": B["B1_ranker_top1"],
        "rank4_excluded_top1": B["B6_rank4_excluded_top1"],
        "random_same_set": B["B7_row_uniform_random_top1_same_dedup"],
        "arm_beats_random": bool(B["B6_rank4_excluded_top1"]
                                 > B["B7_row_uniform_random_top1_same_dedup"]),
    }

    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
