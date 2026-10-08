"""DEBIAS PROBE -- the head is worse than random; does removing candidate bias fix it?

MEASURED ON ENTRY (my own bytes)
    exp_bank_structure_dedup: 1092 train rows but only 206 DISTINCT traces (5.3x).
      frac_untied_argmax 0.0 on the full bank -> ties are 100% duplicate-caused
      (1.0 after dedup).
      ranker top1_exact 0.0317 / coverage5 0.1142
      random over the FULL bank top1_exact 0.06 / coverage5 0.2133
      => RANDOM BEATS THE RANKER on both. The recorded chance_at_5 = 0.004579 in
         henri_selection_headroom_receipt.json assumes a DISTINCT bank (K/n) and is
         invalid at 5.3x redundancy. Analytic true chance = 1-(1-mean_f/n)^K.
    exp_ranker_conditional_info: AUC(C,exact) 0.7942 (C ranks well GLOBALLY),
      bias-only AUC 0.6805, residual AUC 0.755.

HYPOTHESIS (the coherent mechanism)
    C is a good global ranker but its TOP is a plateau of high-CANDIDATE-BIAS rows.
    A random draw samples uniformly and hits frequent (likelier-correct) traces more
    often than the plateau does. If so, removing the candidate-bias component should
    lift the head above random.

PRE-REGISTERED (frozen)
    Analytic baselines
      row_uniform  = mean over held q of  (count of train rows with trace t_q) / n
      trace_uniform= mean over held q of  1[ t_q in train ] / n_distinct_train
    Rank profile: exact-match rate at descending rank p = 0..19, ties broken by index.
    DEBIAS: b = mean over PROBE rows of Cp[:, j] (query-independent, built from the
      FIT split only). Rescale b to the C scale and evaluate C - alpha*b.
      alpha chosen on an out-of-sample PROBE target, never on held-out targets.
    frozen rule
      DEBIAS_REPAIRS_HEAD   if (C-b) top1_exact > row_uniform AND (C-b) coverage5 >
                               analytic random coverage5
      HEAD_IS_BIAS_PLATEAU_IRREPARABLE otherwise
CPU, D=4096. DIAGNOSTIC ONLY. No model-performance claim.
"""
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
    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN, inputs=inputs)
    system, tok = sel.build_system(corpus, pin_seed=pin)
    tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
    tr_spec = [corpus.specs[i] for i in corpus.train_idx]
    ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace

    gen = sel.WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ztr = F.normalize(torch.nan_to_num(sel.flat(gen.wave(tr_spec))), dim=-1)
        Zho = F.normalize(torch.nan_to_num(sel.flat(gen.wave(ho_spec))), dim=-1)

    def Ymat(tg):
        Y = torch.zeros(len(tg), M * Vv)
        for i, ids in enumerate(tg):
            for p, t in enumerate(ids[:M]):
                Y[i, p * Vv + int(t)] = 1.0
        return Y

    n = min(sel.ROWS, len(Ztr))
    KK = min(vt.K, n)
    Yall = Ymat(tr_tgt[:n])
    sc_ho = sel.ridge_scores(Ztr[:n], Yall, Zho).reshape(len(Zho), M, Vv)
    Ytr = Yall.reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc_ho, Ytr)
    R = {"n_train": n, "n_held": len(Zho), "k": KK}

    # ---- analytic baselines -------------------------------------------------
    from collections import Counter
    freq = Counter(tuple(t) for t in tr_tgt[:n])
    f_q = [freq.get(tuple(t), 0) for t in ho_tgt]
    nd = len(freq)
    R["n_distinct_train"] = nd
    R["row_uniform_top1_exact"] = round(sum(f_q) / len(f_q) / n, 4)
    R["trace_uniform_top1_exact"] = round(
        sum(1 for f in f_q if f > 0) / len(f_q) / nd, 4)
    p_row = sum(f_q) / len(f_q) / n
    R["analytic_random_coverage5_rowwise"] = round(1 - (1 - p_row) ** KK, 4)
    R["recorded_chance_at_5_in_prior_receipt"] = 0.004579
    R["chance_at_5_prior_is_invalid"] = True
    R["chance_at_5_invalidity_reason"] = (
        "K/n assumes n DISTINCT candidates; at 5.3x row redundancy five random rows are "
        "usually copies of one trace, so the true row-wise chance is ~46x larger.")

    def exact(picks):
        return round(sum(1 for q, j in enumerate(picks)
                         if list(tr_tgt[j]) == list(ho_tgt[q])) / len(Zho), 4)

    def cov5(Cm):
        tk = torch.topk(Cm, KK, dim=1).indices
        return round(sum(1 for q in range(len(Zho))
                         if any(list(tr_tgt[int(j)]) == list(ho_tgt[q])
                                for j in tk[q])) / len(Zho), 4)

    R["ranker_top1_exact"] = exact([int(C[q].argmax()) for q in range(len(Zho))])
    R["ranker_coverage5"] = cov5(C)

    # ---- rank profile: exact rate at descending rank p ---------------------
    order = torch.argsort(C, dim=1, descending=True)
    prof = []
    for p in range(20):
        cnt = sum(1 for q in range(len(Zho))
                  if list(tr_tgt[int(order[q, p])]) == list(ho_tgt[q]))
        prof.append(round(cnt / len(Zho), 4))
    R["rank_profile_exact_rate_p0_to_p19"] = prof
    R["mean_rate_p4_to_p19"] = round(sum(prof[4:]) / len(prof[4:]), 4)

    # ---- candidate bias from the FIT split (query-independent) -------------
    gsp = torch.Generator().manual_seed(pin)
    perm = torch.randperm(n, generator=gsp).tolist()
    fit_ix, probe_ix = perm[:int(round(n * 2 / 3))], perm[int(round(n * 2 / 3)):]
    sc_pr = sel.ridge_scores(Ztr[fit_ix], Ymat([tr_tgt[i] for i in fit_ix]),
                             Ztr[probe_ix]).reshape(len(probe_ix), M, Vv)
    Cp = torch.einsum("qmv,jmv->qj", sc_pr, Ytr)
    b = Cp.mean(0)
    alpha = float(C.std() / Cp.std())
    R["bias_scale_alpha"] = round(alpha, 4)

    deb = C - alpha * b.unsqueeze(0)
    R["debias_top1_exact"] = exact([int(deb[q].argmax()) for q in range(len(Zho))])
    R["debias_coverage5"] = cov5(deb)
    R["debias_top1_minus_row_uniform"] = round(R["debias_top1_exact"]
                                               - R["row_uniform_top1_exact"], 4)
    R["debias_cov5_minus_analytic_random"] = round(
        R["debias_coverage5"] - R["analytic_random_coverage5_rowwise"], 4)

    ok = (R["debias_top1_minus_row_uniform"] > 0
          and R["debias_cov5_minus_analytic_random"] > 0)
    R["verdict"] = "DEBIAS_REPAIRS_HEAD" if ok else "HEAD_IS_BIAS_PLATEAU_IRREPARABLE"
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = {"schema": "henri.ranker.debias.v1", "pins": [vt.PIN, vt.PIN + 1234]}
    for p in R["pins"]:
        print(f"[debias] pin {p}", file=sys.stderr)
        R[f"pin_{p}"] = run_pin(p)
    vs = [R[f"pin_{p}"]["verdict"] for p in R["pins"]]
    R["overall_verdict"] = vs[0] if len(set(vs)) == 1 else "MIXED"
    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/debias.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
