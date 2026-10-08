"""ATTRIBUTABLE VETO -- a NON-similarity verifier over executor-labeled rejections.

PRE-REGISTERED before this run:
    design/zone_a/evidence/henri_attributable_veto_preregistration.json

WHY THIS EXISTS
    The old verifier consumed d = <z_query, z_candidate> -- the SAME inner product the
    ridge ranker already used to build the ranking. A monotone function of the ranking
    statistic cannot re-rank. Measured: real 0.0402 vs top1 0.0401 vs shuffled 0.0401,
    while a RANDOM pick scored 0.0579. That is an INFORMATION IDENTITY defect (F3),
    not a capacity defect.

    This verifier consumes only signals the ranker does NOT have:
      A  position-resolved bottleneck structure (the ranker's score is a SUM; a sum
         hides one catastrophic position)
      B  candidate-SET contrast (the ranker scores each candidate independently)
      C  leave-one-out wave influence and set-internal structure (set-relative, so not
         a monotone function of any single candidate's similarity)

HONEST TRAIN/TEST SPLIT
    Training features need a query-side ridge output. Ridge fit on train and read on
    train is memorisation, so the train rows are split: FIT (2/3) supplies the ridge,
    PROBE (1/3) supplies out-of-sample query rows. Held-out evaluation uses the FULL
    train fit, identical to the committed baseline ranker. No held-out label is used
    to train anything.

LESSON APPLIED THIS TURN
    Duplication drift made a phase_gain flag unreachable earlier today. So every
    harness helper and split constant is IMPORTED from the committed executor script.
    One implementation, one source of truth.

NOT WIRED TO THE SAGNAC PORT. Blueprint Directive 3 stays BLOCKED: the ingress does
not beat legacy and the veto has no measured attribution yet.
CPU, D=4096. DIAGNOSTIC ONLY. No model-performance claim.
"""
import argparse
import io
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

import henri_core.exp_executor_verified_selection as sel                     # noqa: E402

PIN = sel.PIN
MAX_LEN = sel.MAX_LEN
N_INPUTS = sel.N_INPUTS
ROWS = sel.ROWS
K = sel.K
MRC = 0.005          # pre-registered margin (V_A, V_B)
AUC_BAR = 0.55       # pre-registered (V_C)
IDENT_R = 0.995      # pre-registered (V_E)

FEATS = [
    "min_token_score", "argmin_pos_norm", "frac_below_half",
    "pred_len_mismatch", "first_disagree_norm", "score_std",
    "mean_agree_others", "rank_by_agree_norm", "is_medoid", "max_agree_others",
    "loo_centroid_delta", "set_internal_mean_cos",
]


def auc(scores, labels):
    """Rank AUC with tie handling. Returns float('nan') if one class is absent."""
    n_pos = int(sum(labels))
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
    s_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] > 0)
    return (s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def feats_for(sq, tj, j, zq, set_idx, set_traces, bank):
    """12 features. sq [M,Vv] pre-mixing. j = candidate's GLOBAL train index.

    `bank` is the full normalised wave bank Ztr. The first draft named it Zc and I
    then "fixed" a non-bug: every caller already passes the bank, so bank[j] indexes
    GLOBALLY and bank[list(set_idx)] indexes the same tensor. Naming it `bank` removes
    the ambiguity that produced a NameError.
    """
    L = len(tj)
    Mdim = sq.shape[0]
    idxs = torch.tensor(tj, dtype=torch.long)
    sc = sq[torch.arange(L), idxs]                      # [L]
    pred_ids = sq.argmax(-1).tolist()
    pred_len = int(sq.abs().sum(-1).argmax()) + 1
    first = next((i for i in range(L) if pred_ids[i] != tj[i]), L)
    out = [
        float(sc.min()),
        float(sc.argmin()) / float(Mdim),
        float((sc < 0.5).sum()) / float(L),
        float(abs(pred_len - L)) / float(Mdim),
        float(first) / float(Mdim),
        float(sc.std(unbiased=False)),
    ]
    ags = []
    for k2, to in zip(set_idx, set_traces):
        if k2 == j:
            continue
        ags.append(sum(1 for a, b in zip(tj, to) if a == b)
                   / float(max(1, min(L, len(to)))))
    mean_ag = sum(ags) / max(1, len(ags))
    out += [
        float(mean_ag),
        float(sum(1 for x in ags if x > mean_ag)) / float(max(1, len(ags))),
        1.0 if (ags and mean_ag >= max(ags) - 1e-9) else 0.0,
        float(max(ags)) if ags else 0.0,
    ]
    zc = bank[list(set_idx)]                            # [K, dim] set members
    cen_all = F.normalize(zc.mean(0, keepdim=True), dim=-1)
    keep = [i for i in range(len(set_idx)) if set_idx[i] != j]
    if keep:
        cen_wo = F.normalize(zc[keep].mean(0, keepdim=True), dim=-1)
        out.append(float((zq @ cen_all.T) - (zq @ cen_wo.T)))
    else:
        out.append(0.0)
    others = [i for i in set_idx if i != j]
    out.append(float((bank[j] @ bank[others].T).mean()) if others else 0.0)
    return out


def fit(X, y):
    return torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)


def pick(w, X):
    return int(torch.tensor([float(x @ w) for x in X]).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    R = {"schema": "henri.attributable.veto.v1", "pin": PIN, "k": K, "features": FEATS,
         "prereg": "design/zone_a/evidence/henri_attributable_veto_preregistration.json"}

    inputs = sel.make_inputs(N_INPUTS)
    corpus = sel.build_corpus(max_len=MAX_LEN, holdout_len=MAX_LEN, inputs=inputs)
    system, tok = sel.build_system(corpus, pin_seed=PIN)
    tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
    tr_spec = [corpus.specs[i] for i in corpus.train_idx]
    ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace
    R["split"] = {"n_train": len(tr_tgt), "n_held": len(ho_tgt)}
    print(f"[veto] train={len(tr_tgt)} held={len(ho_tgt)}", file=sys.stderr)

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

    # ---- estimator control (harness sanity) --------------------------------
    g = torch.Generator().manual_seed(7)
    nrc = min(128, len(Ztr))
    Zrc = F.normalize(torch.randn(nrc, Ztr.shape[1], generator=g), dim=-1)
    ctl = sel.tokacc(
        sel.ridge_scores(Zrc, Ymat(tr_tgt[:nrc]), Zrc).reshape(-1, M, Vv).argmax(-1),
        tr_tgt[:nrc])
    R["estimator_control"] = ctl

    n = min(ROWS, len(Ztr))
    Yall = Ymat(tr_tgt[:n])

    # ---- BASELINE ranker: full-train fit, held-out scoring -----------------
    sc_ho = sel.ridge_scores(Ztr[:n], Yall, Zho).reshape(len(Zho), M, Vv)
    Ytr = Yall.reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc_ho, Ytr)                # [nq, n]
    top1 = [tr_tgt[j] for j in C.argmax(1).tolist()]
    topk = [torch.topk(C[q], min(K, n)).indices.tolist() for q in range(len(Zho))]
    R["top1_baseline"] = sel.tokacc(top1, ho_tgt)
    R["unigram_floor"] = sel.unigram_floor(tr_tgt, ho_tgt)

    def same(c, w):
        return list(c) == list(w)

    orc = sum(1 for q in range(len(Zho))
              if any(same(tr_tgt[j], ho_tgt[q]) for j in topk[q]))
    R["oracle_rerank_top5"] = round(orc / len(Zho), 4)
    R["oracle_gain_over_top1"] = round(R["oracle_rerank_top5"] - R["top1_baseline"], 4)

    # ---- the labeled-rejection budget (the veto's label source) ------------
    n_avail = len(Zho) * min(K, n)
    n_corr = sum(1 for q in range(len(Zho)) for j in topk[q]
                 if same(tr_tgt[j], ho_tgt[q]))
    R["n_pair_labels_available"] = int(n_avail)
    R["n_pair_labels_correct"] = int(n_corr)
    R["pair_label_rate"] = round(n_corr / max(1, n_avail), 4)

    # ---- honest train split for verifier features --------------------------
    gsp = torch.Generator().manual_seed(PIN)
    perm = torch.randperm(n, generator=gsp).tolist()
    n_fit = int(round(n * 2 / 3))
    fit_ix, probe_ix = perm[:n_fit], perm[n_fit:]
    R["n_fit"], R["n_probe"] = len(fit_ix), len(probe_ix)
    sc_pr = sel.ridge_scores(Ztr[fit_ix], Ymat([tr_tgt[i] for i in fit_ix]),
                             Ztr[probe_ix]).reshape(len(probe_ix), M, Vv)

    gv = torch.Generator().manual_seed(PIN)
    Xtr, ytr = [], []
    for r, i in enumerate(probe_ix[:600]):
        cand = torch.randperm(n, generator=gv)[:8].tolist()
        cand[0] = i
        order = torch.randperm(len(cand), generator=gv).tolist()
        for c in order:
            j = cand[c]
            Xtr.append(feats_for(sc_pr[r], tr_tgt[j], j, Ztr[i], cand, 
                                 [tr_tgt[x] for x in cand], Ztr))
            ytr.append(1.0 if same(tr_tgt[j], tr_tgt[i]) else 0.0)
    Xtr = torch.tensor(Xtr, dtype=torch.float64)
    ytr = torch.tensor(ytr, dtype=torch.float64)
    R["n_train_pairs"] = int(len(ytr))
    R["train_pair_rate"] = round(float(ytr.mean()), 4)

    # ---- V_E information-identity check -----------------------------------
    ids = torch.tensor([feats_for(sc_ho[q], tr_tgt[j], j, Zho[q], topk[q],
                                  [tr_tgt[x] for x in topk[q]], Ztr)
                        for q in range(0, len(Zho), 5) for j in topk[q]],
                       dtype=torch.float64)
    c_sel = torch.tensor([float(C[q, j]) for q in range(0, len(Zho), 5)
                          for j in topk[q]], dtype=torch.float64)
    corr = {}
    for f, name in enumerate(FEATS):
        x = ids[:, f]
        sd = x.std(unbiased=False)
        corr[name] = round(float((((x - x.mean()) * (c_sel - c_sel.mean())).mean()
                                  / (sd * c_sel.std(unbiased=False) + 1e-12)).item()), 4)
    R["corr_feature_vs_ranker_statistic"] = corr
    R["max_abs_identity_corr"] = round(max(abs(v) for v in corr.values()), 4)

    # ---- REAL verifier ----------------------------------------------------
    w = fit(Xtr, ytr)
    R["verifier_weights"] = [round(float(x), 4) for x in w]

    def rerank(wv, cols=None):
        """cols=None -> all features. LOO passes the 11-column subset it fitted.

        SELF-CAUGHT: the first draft fitted the leave-one-out verifier on 11 columns
        (Xtr[:, keep]) but called rerank, which always rebuilds 12 -> RuntimeError
        'expected tensor [12] and src [11]'. The subset must be applied to the
        evaluation matrix too, or the comparison is not like-for-like.
        """
        pred = []
        for q in range(len(Zho)):
            Xq = torch.tensor([feats_for(sc_ho[q], tr_tgt[j], j, Zho[q], topk[q],
                                         [tr_tgt[x] for x in topk[q]], Ztr)
                               for j in topk[q]], dtype=torch.float64)
            if cols is not None:
                Xq = Xq[:, cols]
            pred.append(tr_tgt[topk[q][pick(wv, Xq)]])
        return sel.tokacc(pred, ho_tgt)

    R["real_verifier_top5"] = rerank(w)

    # shuffled-label control
    yp = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(11))]
    R["shuffled_verifier_top5"] = rerank(fit(Xtr, yp))

    # random control
    rg = torch.Generator().manual_seed(PIN)
    R["random_verifier_top5"] = sel.tokacc(
        [tr_tgt[topk[q][int(torch.randint(len(topk[q]), (1,), generator=rg))]]
         for q in range(len(Zho))], ho_tgt)

    # OLD DEFECT reproduction: plain inner product only
    gd = torch.Generator().manual_seed(PIN)
    Xo, yo = [], []
    for i in range(min(600, n)):
        cand = torch.randperm(n, generator=gd)[:8].tolist()
        cand[0] = i
        for j in cand:
            d = float((Ztr[i] * Ztr[j]).sum())
            Xo.append([d, abs(d), d * d, 1.0])
            yo.append(1.0 if same(tr_tgt[j], tr_tgt[i]) else 0.0)
    wo = fit(torch.tensor(Xo, dtype=torch.float64),
             torch.tensor(yo, dtype=torch.float64))

    def rerank_old(wv):
        pred = []
        for q in range(len(Zho)):
            Xq = []
            for j in topk[q]:
                d = float((Zho[q] * Ztr[j]).sum())
                Xq.append([d, abs(d), d * d, 1.0])
            pred.append(tr_tgt[topk[q][pick(wv, torch.tensor(Xq, dtype=torch.float64))]])
        return sel.tokacc(pred, ho_tgt)

    R["old_defect_verifier_top5"] = rerank_old(wo)

    # ---- attribution ------------------------------------------------------
    attr = {}
    for f, name in enumerate(FEATS):
        attr[name] = {
            "weight": round(float(w[f]), 4),
            "standalone_auc_train": round(auc(Xtr[:, f].tolist(), ytr.tolist()), 4),
        }
    R["attribution_standalone"] = attr

    loo = {}
    for f, name in enumerate(FEATS):
        keep = [k2 for k2 in range(len(FEATS)) if k2 != f]
        loo[name] = round(rerank(fit(Xtr[:, keep], ytr), cols=keep)
                          - R["real_verifier_top5"], 4)
    R["attribution_leave_one_feature_out"] = loo

    # held-out pair AUC at the fitted score
    hs, hl = [], []
    for q in range(len(Zho)):
        for j in topk[q]:
            Xq = torch.tensor([feats_for(sc_ho[q], tr_tgt[j], j, Zho[q], topk[q],
                                         [tr_tgt[x] for x in topk[q]], Ztr)],
                              dtype=torch.float64)
            hs.append(float(Xq[0] @ w))
            hl.append(1.0 if same(tr_tgt[j], ho_tgt[q]) else 0.0)
    R["heldout_pair_auc"] = round(float(auc(hs, hl)), 4)
    R["heldout_pair_n"] = len(hl)

    # veto reject rate at the train-optimal threshold
    thr = float((Xtr @ w).median())
    R["veto_threshold_train_median"] = round(thr, 4)
    R["veto_reject_rate_heldout"] = round(
        sum(1 for s in hs if s < thr) / max(1, len(hs)), 4)

    # ---- frozen reading rule ---------------------------------------------
    c = {
        "V_A_real_gt_random": round(R["real_verifier_top5"] - R["random_verifier_top5"], 4),
        "V_B_real_gt_shuffled": round(R["real_verifier_top5"] - R["shuffled_verifier_top5"], 4),
        "V_C_pair_auc": R["heldout_pair_auc"],
        "V_D_reject_rate": R["veto_reject_rate_heldout"],
        "V_E_max_identity_corr": R["max_abs_identity_corr"],
    }
    R["checks"] = c
    vA = c["V_A_real_gt_random"] > MRC
    vB = c["V_B_real_gt_shuffled"] > MRC
    vC = c["V_C_pair_auc"] > AUC_BAR
    vD = 0.02 < c["V_D_reject_rate"] < 0.98
    vE = c["V_E_max_identity_corr"] < IDENT_R
    R["pass"] = {"V_A": vA, "V_B": vB, "V_C": vC, "V_D": vD, "V_E": vE}
    if ctl < 0.99:
        R["verdict"] = "HARNESS_BROKEN_NO_INTERPRETATION"
    elif all((vA, vB, vC, vD, vE)):
        R["verdict"] = "VETO_ATTRIBUTABLE_ON_THIS_TASK"
    elif abs(c["V_B_real_gt_shuffled"]) <= MRC:
        R["verdict"] = "VETO_NOT_ATTRIBUTABLE_ORACLE_ONLY"
    else:
        R["verdict"] = "VETO_NOT_ATTRIBUTABLE"
    R["elapsed_s"] = round(time.time() - t0, 1)

    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/veto.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
