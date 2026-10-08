"""RANKER CONDITIONAL-INFORMATION PROBE -- does C carry ANY query signal?

MEASURED ON ENTRY (my own bytes, exp_ranker_structure_probe.py, 2 pins)
    candidate variance share   0.9567 / 0.9563   -> C[q,j] ~ b_j
    mean tie group at max      10.71 / 10.88
    mean distinct values in top 20   2.96 / 2.96
    distinct top-5 sets        29 / 30  for 2268 queries
    corr(C, exact-match)       +0.2824 (positive, 3 pins)
    global-argmin accuracy worse than random

MECHANISM TO TEST
    ridge_scores computes s_q = k_q^T (K + lam I)^-1 Y  with k_q = Z_ho[q] . Z_tr^T.
    If k_q is nearly constant across q, then s_q is nearly the same for every query and
    the readout CANNOT condition on the query. The observed 0.9567 candidate-variance
    share is exactly that signature.

DECOMPOSITION (pre-registered)
    b_j    = mean over PROBE (out-of-sample train) rows of C[r, j]  -> the
             query-independent candidate component, built WITHOUT held-out data
    resid  = C[q,j] - b_j                                            -> the
             query-conditional component
    Pair AUC is computed on held-out (q, j) pairs, label = exact trace match.

FROZEN READING
    C_QUERY_INFORMATION_PRESENT   if AUC(resid) > AUC(b_j) + 0.01
    C_IS_CANDIDATE_BIAS_ONLY      if AUC(resid) - 0.5 < 0.01
    (both reported; the second is the striking one)

ALSO MEASURED (kernel diagnostics)
    K1 spread of k_q across queries (std of the row-mean, and of the row-max)
    K2 top-5 SET STABILITY under Gaussian noise eps ~ 0.01 * std(C) added to C
    K3 wave cos-similarity spread in the bank
    K2 is the decisive robustness check: a meaningful ranking keeps its top-5 under
    negligible noise; a tie-dominated one does not.

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


def auc(scores, labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    s = sum(ranks[i] for i in range(len(scores)) if labels[i] > 0)
    return (s - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


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
    R = {"n_train": n, "n_held": len(Zho), "k": KK}

    # ---- K3 wave similarity spread ----------------------------------------
    with torch.no_grad():
        Ssub = Ztr[:200] @ Ztr[:200].T
        off = Ssub[~torch.eye(200, dtype=torch.bool)]
        R["K3_bank_pairwise_cos_mean"] = round(float(off.mean()), 4)
        R["K3_bank_pairwise_cos_std"] = round(float(off.std()), 4)
        Kq = Zho[:200] @ Ztr.T                       # [200, n] query-train kernel
        rm = Kq.mean(1)
        R["K2_kernel_rowmean_std_across_queries"] = round(float(rm.std()), 5)
        R["K2_kernel_rowmean_absmean"] = round(float(rm.abs().mean()), 5)
        R["K2_kernel_rowstd_mean"] = round(float(Kq.std(1).mean()), 5)

    # ---- held-out C and the query-independent component --------------------
    sc_ho = sel.ridge_scores(Ztr[:n], Ymat(tr_tgt[:n]), Zho).reshape(len(Zho), M, Vv)
    Ytr = Ymat(tr_tgt[:n]).reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc_ho, Ytr)                 # [nq, n]

    gsp = torch.Generator().manual_seed(pin)
    perm = torch.randperm(n, generator=gsp).tolist()
    fit_ix, probe_ix = perm[:int(round(n * 2 / 3))], perm[int(round(n * 2 / 3)):]
    sc_pr = sel.ridge_scores(Ztr[fit_ix], Ymat([tr_tgt[i] for i in fit_ix]),
                             Ztr[probe_ix]).reshape(len(probe_ix), M, Vv)
    Cp = torch.einsum("qmv,jmv->qj", sc_pr, Ytr)                # [nprobe, n]
    b = Cp.mean(0)                                              # [n] candidate bias
    R["b_std"] = round(float(b.std()), 6)
    R["C_std"] = round(float(C.std()), 6)
    R["b_std_over_C_std"] = round(float(b.std() / C.std().clamp_min(1e-12)), 4)

    # ---- pair AUCs ---------------------------------------------------------
    cs, bs, rs, lab = [], [], [], []
    for q in range(len(Zho)):
        correct = list(ho_tgt[q])
        for j in range(n):
            cs.append(float(C[q, j]))
            bs.append(float(b[j]))
            rs.append(float(C[q, j] - b[j]))
            lab.append(1.0 if list(tr_tgt[j]) == correct else 0.0)
    R["n_pairs"] = len(lab)
    R["pair_positive_rate"] = round(sum(lab) / len(lab), 4)
    R["auc_C_full"] = round(float(auc(cs, lab)), 4)
    R["auc_b_queryindependent"] = round(float(auc(bs, lab)), 4)
    R["auc_resid_queryconditional"] = round(float(auc(rs, lab)), 4)
    R["resid_minus_half"] = round(R["auc_resid_queryconditional"] - 0.5, 4)
    R["full_minus_bias"] = round(R["auc_C_full"] - R["auc_b_queryindependent"], 4)

    # ---- K2 top-5 set stability under tiny noise ---------------------------
    base = [set(torch.topk(C[q], KK).indices.tolist()) for q in range(len(Zho))]
    same = 0
    trials = 5
    sc = float(C.std())
    for t in range(trials):
        g = torch.Generator().manual_seed(pin * 7 + t)
        Nz = torch.randn(C.shape, generator=g) * (0.01 * sc)
        tk2 = torch.topk(C + Nz, KK, dim=1).indices
        same += sum(1 for q in range(len(Zho))
                    if set(tk2[q].tolist()) == base[q])
    R["K2_top5_set_identity_rate_under_1pct_noise"] = round(same / (trials * len(Zho)), 4)

    R["verdict"] = ("C_IS_CANDIDATE_BIAS_ONLY"
                    if (R["resid_minus_half"] < 0.01
                        and R["full_minus_bias"] < 0.01)
                    else ("C_QUERY_INFORMATION_PRESENT"
                          if R["full_minus_bias"] > 0.01
                          else "C_WEAK_MIXED"))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = {"schema": "henri.ranker.conditional.info.probe.v1",
         "pins": [vt.PIN, vt.PIN + 1234]}
    for p in R["pins"]:
        print(f"[condinfo] pin {p}", file=sys.stderr)
        R[f"pin_{p}"] = run_pin(p)
    outs = [R[f"pin_{p}"]["verdict"] for p in R["pins"]]
    R["overall_verdict"] = outs[0] if len(set(outs)) == 1 else "MIXED"
    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/condinfo.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
