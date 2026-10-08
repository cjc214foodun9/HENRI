"""VETO FIT SWEEP -- is the failure the FIT or the FEATURES?

PRE-REGISTERED: design/zone_a/evidence/henri_veto_fit_sweep_preregistration.json

WHY THIS EXISTS
    The veto run reproduced four historical quantities EXACTLY, so its anomaly is real:
      real verifier 0.0416   random 0.0579   shuffled 0.1072   oracle 0.1142
    The label-SHUFFLED control scored 2.6x the real arm and near the oracle ceiling.
    A control that beats its treatment means the fit is the defect, not the features.
    Supporting evidence: the same features score held-out pair AUC 0.6217 over 11,340
    labeled pairs and are NOT information-identical to the ranker (max corr 0.515).
    Separately, the unregularised lstsq fit is trained on pairs whose positive rate is
    0.1693 and whose positive is ALWAYS the query's own row; the test pair rate is
    0.0504 with no self row available. That distribution mismatch invites overfit.

WHAT IT DOES
    Standardises the features with TRAIN statistics, sweeps a ridge lambda grid with the
    shuffled control refitted at the SAME lambda, selects lambda on a VAL half of the
    held-out queries, and reports on the TEST half. Selecting on TEST is the defect
    retracted earlier this turn, so it is not repeated.

DEPENDENCY DISCIPLINE
    `feats_for` is IMPORTED from exp_attributable_veto, never re-declared. Duplication
    drift made a flag unreachable earlier today; the complex feature builder has ONE
    implementation.

NOT WIRED TO THE SAGNAC PORT. CPU, D=4096. DIAGNOSTIC ONLY.
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

import henri_core.exp_attributable_veto as vt                              # noqa: E402

sel = vt.sel
PIN = vt.PIN
LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
MRC = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    R = {"schema": "henri.veto.fit_sweep.v1", "pin": PIN, "features": vt.FEATS,
         "lams": LAMS, "margin": MRC,
         "prereg": "design/zone_a/evidence/henri_veto_fit_sweep_preregistration.json"}

    inputs = sel.make_inputs(sel.N_INPUTS)
    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN, inputs=inputs)
    system, tok = sel.build_system(corpus, pin_seed=PIN)
    tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
    tr_spec = [corpus.specs[i] for i in corpus.train_idx]
    ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace
    print(f"[sweep] train={len(tr_tgt)} held={len(ho_tgt)}", file=sys.stderr)

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
    sc_ho = sel.ridge_scores(Ztr[:n], Ymat(tr_tgt[:n]), Zho).reshape(len(Zho), M, Vv)
    Ytr = Ymat(tr_tgt[:n]).reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc_ho, Ytr)
    topk = [torch.topk(C[q], min(vt.K, n)).indices.tolist() for q in range(len(Zho))]
    R["top1_baseline"] = sel.tokacc([tr_tgt[j] for j in C.argmax(1).tolist()], ho_tgt)

    # ---- training pairs: identical construction to the veto run --------------
    gsp = torch.Generator().manual_seed(PIN)
    perm = torch.randperm(n, generator=gsp).tolist()
    n_fit = int(round(n * 2 / 3))
    fit_ix, probe_ix = perm[:n_fit], perm[n_fit:]
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
            Xtr.append(vt.feats_for(sc_pr[r], tr_tgt[j], j, Ztr[i], cand,
                                    [tr_tgt[x] for x in cand], Ztr))
            ytr.append(1.0 if list(tr_tgt[j]) == list(tr_tgt[i]) else 0.0)
    Xtr = torch.tensor(Xtr, dtype=torch.float64)
    ytr = torch.tensor(ytr, dtype=torch.float64)
    R["n_train_pairs"] = int(len(ytr))
    R["train_pair_rate"] = round(float(ytr.mean()), 4)
    R["test_pair_rate"] = round(
        sum(1 for q in range(len(Zho)) for j in topk[q]
            if list(tr_tgt[j]) == list(ho_tgt[q])) / max(1, len(Zho) * min(vt.K, n)), 4)

    # ---- held-out per-query feature sets (built once) -----------------------
    Xh = []
    for q in range(len(Zho)):
        Xh.append(torch.tensor(
            [vt.feats_for(sc_ho[q], tr_tgt[j], j, Zho[q], topk[q],
                          [tr_tgt[x] for x in topk[q]], Ztr) for j in topk[q]],
            dtype=torch.float64))

    # ---- standardise with TRAIN statistics ---------------------------------
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-9)
    XtrS = (Xtr - mu) / sd

    def zq_(q):
        return (Xh[q] - mu) / sd

    # ---- VAL / TEST split over held-out queries ----------------------------
    gq = torch.Generator().manual_seed(PIN + 1)
    qp = torch.randperm(len(Zho), generator=gq).tolist()
    half = len(qp) // 2
    val_ix, test_ix = qp[:half], qp[half:]
    R["n_val"], R["n_test"] = len(val_ix), len(test_ix)

    def ridge_w(X, y, lam):
        A = X.T @ X + lam * torch.eye(X.shape[1], dtype=torch.float64)
        return torch.linalg.solve(A, X.T @ y)

    def acc(queries, w):
        hits = tots = 0
        for q in queries:
            j = topk[q][int((zq_(q) @ w).argmax())]
            for x, y in zip(tr_tgt[j], ho_tgt[q]):
                tots += 1
                hits += int(x == y)
        return round(hits / max(1, tots), 4)

    rg = torch.Generator().manual_seed(PIN)

    def acc_rand(queries):
        hits = tots = 0
        for q in queries:
            j = topk[q][int(torch.randint(len(topk[q]), (1,), generator=rg))]
            for x, y in zip(tr_tgt[j], ho_tgt[q]):
                tots += 1
                hits += int(x == y)
        return round(hits / max(1, tots), 4)

    val_real, test_real, test_shuf = {}, {}, {}
    for lam in LAMS:
        w = ridge_w(XtrS, ytr, lam)
        val_real[lam] = acc(val_ix, w)
        test_real[lam] = acc(test_ix, w)
        yp = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(11))]
        test_shuf[lam] = acc(test_ix, ridge_w(XtrS, yp, lam))
    # unregularised reference arm
    w0 = torch.linalg.lstsq(XtrS, ytr.unsqueeze(-1)).solution.squeeze(-1)
    R["lstsq"] = {"val": acc(val_ix, w0), "test": acc(test_ix, w0)}

    lam_star = max(val_real, key=lambda l: val_real[l])
    spread = round(max(val_real.values()) - min(val_real.values()), 4)
    R["val_real_by_lam"] = {str(l): val_real[l] for l in LAMS}
    R["test_real_by_lam"] = {str(l): test_real[l] for l in LAMS}
    R["test_shuffled_by_lam"] = {str(l): test_shuf[l] for l in LAMS}
    R["lam_star_by_val"] = lam_star
    R["selector_val_spread"] = spread
    R["selector_discriminates"] = bool(spread > MRC)
    R["real_test_at_lam_star"] = test_real[lam_star]
    R["shuffled_test_at_lam_star"] = test_shuf[lam_star]
    R["random_test"] = acc_rand(test_ix)
    R["checks"] = {
        "real_minus_random": round(R["real_test_at_lam_star"] - R["random_test"], 4),
        "real_minus_shuffled": round(R["real_test_at_lam_star"]
                                     - R["shuffled_test_at_lam_star"], 4),
    }

    # ---- chosen-index distribution at lam* (slot-bias check) --------------
    w = ridge_w(XtrS, ytr, lam_star)
    ws = ridge_w(XtrS, ytr[torch.randperm(len(ytr),
                                          generator=torch.Generator().manual_seed(11))],
                 lam_star)
    for tag, ww in (("real", w), ("shuffled", ws)):
        dist = [0] * min(vt.K, n)
        for q in test_ix:
            dist[int((zq_(q) @ ww).argmax())] += 1
        R[f"chosen_index_dist_{tag}"] = dist
    rd = [0] * min(vt.K, n)
    r2 = torch.Generator().manual_seed(PIN)
    for q in test_ix:
        rd[int(torch.randint(len(topk[q]), (1,), generator=r2))] += 1
    R["chosen_index_dist_random"] = rd

    # ---- per-feature attribution (direction selected on VAL) --------------
    attr = {}
    for f, name in enumerate(vt.FEATS):
        best = None
        for sign, tag in ((1.0, "max"), (-1.0, "min")):
            wv = torch.zeros(len(vt.FEATS), dtype=torch.float64)
            wv[f] = sign
            v = acc(val_ix, wv)
            if best is None or v > best[1]:
                best = (tag, v)
        wv = torch.zeros(len(vt.FEATS), dtype=torch.float64)
        wv[f] = 1.0 if best[0] == "max" else -1.0
        attr[name] = {"direction_by_val": best[0], "val": best[1],
                      "test": acc(test_ix, wv)}
    R["per_feature_attribution"] = attr

    ok = (R["checks"]["real_minus_random"] > MRC
          and R["checks"]["real_minus_shuffled"] > MRC)
    if not R["selector_discriminates"]:
        R["verdict"] = "NO_INFORMATION_SELECTOR_DEGENERATE"
    elif ok:
        R["verdict"] = "VETO_ATTRIBUTABLE_WITH_REGULARISED_FIT"
    else:
        R["verdict"] = "VETO_FEATURES_INFORMATIVE_FIT_STILL_FAILS"
    R["elapsed_s"] = round(time.time() - t0, 1)

    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/veto_sweep.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
