"""VETO DISTRIBUTION-SHIFT FIX -- mirror the test candidate distribution.

PRE-REGISTERED here, before the run.

MEASURED ON ENTRY (my own bytes, pin 20261010)
    real fit 0.0435 with chosen_index_dist_real [736,152,76,40,130] -> 64.9% of picks on
    candidate slot 0 = the ranker's top-1. The fit DRIFTS BACK TO THE RANKER (F1).
    A single odd-one-out feature scored 0.1128 vs oracle 0.1142 (F2).
    Train pair rate 0.1693 vs test pair rate 0.0504.

ROOT-CAUSE HYPOTHESIS (to falsify)
    Training sets contain the query's own row, so the fit learns SELF-DETECTION, a task
    absent at test. It then collapses onto the ranker's direction.

CHANGE (one mechanism, one experiment)
    Training candidate sets MIRROR the test distribution: candidates = the ranker's top-K
    for the query, computed on the FIT split; positive = a DISTINCT row with the same
    trace; the query's own row is EXCLUDED. No other change.

FROZEN READING RULE
    T1 anti-drift : real fitted arm's slot-0 pick share < 0.50 (the old fit was 0.649)
    T2 real beats random  : real_test > random_test + 0.005
    T3 real beats shuffled: real_test > shuffled_test + 0.005
    T4 selector alive     : val spread > 0.005
    T5 single-feature arm confirmed: the odd-one-out feature reproduces on an
       INDEPENDENT pin (PIN+1234) with the same direction and clears random + 0.005.
    PASS  = T1 and T2 and T3 and T4
    verdict_pass = VETO_ATTRIBUTABLE_AFTER_DISTRIBUTION_MATCH
    verdict_fail = VETO_FIT_STILL_COLLAPSES_TO_RANKER  (falsifies the hypothesis)

lambda selected on VAL, reported on TEST. Shuffled refitted at the SAME lambda.
Deps IMPORTED from exp_attributable_veto (one implementation of feats_for).
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
ODD_ONE_OUT = vt.FEATS.index("mean_agree_others")


def build_pin(pin: int, out: dict) -> dict:
    """Fit the distribution-matched veto at one pin. Returns a result dict."""
    inputs = sel.make_inputs(sel.N_INPUTS)
    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN,
                              inputs=inputs)
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
    gsp = torch.Generator().manual_seed(pin)
    perm = torch.randperm(n, generator=gsp).tolist()
    n_fit = int(round(n * 2 / 3))
    fit_ix, probe_ix = perm[:n_fit], perm[n_fit:]

    # ---- TEST-side scoring uses the FULL-train fit (identical to the ranker) ----
    Yall = Ymat(tr_tgt[:n])
    sc_ho = sel.ridge_scores(Ztr[:n], Yall, Zho).reshape(len(Zho), M, Vv)
    Ytr = Yall.reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc_ho, Ytr)
    topk = [torch.topk(C[q], min(vt.K, n)).indices.tolist() for q in range(len(Zho))]

    # ---- TRAIN pairs MIRROR the test distribution ---------------------------
    # candidates = the ranker's top-K for a PROBE query, NO self row, positive =
    # a DISTINCT row with the same trace.
    sc_pr = sel.ridge_scores(Ztr[fit_ix], Ymat([tr_tgt[i] for i in fit_ix]),
                             Ztr[probe_ix]).reshape(len(probe_ix), M, Vv)
    Ypr = Ymat([tr_tgt[i] for i in probe_ix]).reshape(len(probe_ix), M, Vv)
    Cp = torch.einsum("qmv,jmv->qj", sc_pr, Ytr)
    # exclude the query's own row from its candidate set
    for r, i in enumerate(probe_ix):
        Cp[r, i] = float("-inf")
    Xtr, ytr = [], []
    for r, i in enumerate(probe_ix):
        cand = torch.topk(Cp[r], min(vt.K, n)).indices.tolist()
        for j in cand:
            Xtr.append(vt.feats_for(sc_pr[r], tr_tgt[j], j, Ztr[i], cand,
                                    [tr_tgt[x] for x in cand], Ztr))
            ytr.append(1.0 if list(tr_tgt[j]) == list(tr_tgt[i]) else 0.0)
    Xtr = torch.tensor(Xtr, dtype=torch.float64)
    ytr = torch.tensor(ytr, dtype=torch.float64)
    out["n_train_pairs"] = int(len(ytr))
    out["train_pair_rate"] = round(float(ytr.mean()), 4)

    Xh = [torch.tensor([vt.feats_for(sc_ho[q], tr_tgt[j], j, Zho[q], topk[q],
                                     [tr_tgt[x] for x in topk[q]], Ztr)
                        for j in topk[q]], dtype=torch.float64)
          for q in range(len(Zho))]

    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-9)
    XtrS = (Xtr - mu) / sd
    zq_ = lambda q: (Xh[q] - mu) / sd                                    # noqa: E731

    gq = torch.Generator().manual_seed(pin + 1)
    qp = torch.randperm(len(Zho), generator=gq).tolist()
    val_ix, test_ix = qp[:len(qp) // 2], qp[len(qp) // 2:]

    def ridge_w(X, y, lam):
        A = X.T @ X + lam * torch.eye(X.shape[1], dtype=torch.float64)
        return torch.linalg.solve(A, X.T @ y)

    def acc(queries, w):
        h = t = 0
        for q in queries:
            j = topk[q][int((zq_(q) @ w).argmax())]
            for x, y in zip(tr_tgt[j], ho_tgt[q]):
                t += 1
                h += int(x == y)
        return round(h / max(1, t), 4)

    rg = torch.Generator().manual_seed(pin)
    rh = rt = 0
    for q in test_ix:
        j = topk[q][int(torch.randint(len(topk[q]), (1,), generator=rg))]
        for x, y in zip(tr_tgt[j], ho_tgt[q]):
            rt += 1
            rh += int(x == y)
    out["random_test"] = round(rh / max(1, rt), 4)

    val_real, test_real, test_shuf = {}, {}, {}
    for lam in LAMS:
        w = ridge_w(XtrS, ytr, lam)
        val_real[lam] = acc(val_ix, w)
        test_real[lam] = acc(test_ix, w)
        yp = ytr[torch.randperm(len(ytr), generator=torch.Generator().manual_seed(11))]
        test_shuf[lam] = acc(test_ix, ridge_w(XtrS, yp, lam))
    lam_star = max(val_real, key=lambda l: val_real[l])
    out["val_real_by_lam"] = {str(l): val_real[l] for l in LAMS}
    out["test_real_by_lam"] = {str(l): test_real[l] for l in LAMS}
    out["test_shuffled_by_lam"] = {str(l): test_shuf[l] for l in LAMS}
    out["lam_star_by_val"] = lam_star
    out["selector_val_spread"] = round(max(val_real.values())
                                       - min(val_real.values()), 4)
    out["real_test_at_lam_star"] = test_real[lam_star]
    out["shuffled_test_at_lam_star"] = test_shuf[lam_star]
    out["real_minus_random"] = round(out["real_test_at_lam_star"]
                                     - out["random_test"], 4)
    out["real_minus_shuffled"] = round(out["real_test_at_lam_star"]
                                       - out["shuffled_test_at_lam_star"], 4)

    # T1 anti-drift: slot-0 pick share
    w = ridge_w(XtrS, ytr, lam_star)
    dist = [0] * min(vt.K, n)
    for q in test_ix:
        dist[int((zq_(q) @ w).argmax())] += 1
    out["chosen_index_dist_real"] = dist
    out["slot0_share_real"] = round(dist[0] / max(1, sum(dist)), 4)

    # T5 single-feature odd-one-out, direction by VAL, reported on TEST
    best = None
    for sgn, tag in ((1.0, "max"), (-1.0, "min")):
        wv = torch.zeros(len(vt.FEATS), dtype=torch.float64)
        wv[ODD_ONE_OUT] = sgn
        v = acc(val_ix, wv)
        if best is None or v > best[2]:
            best = (tag, sgn, v)
    wv = torch.zeros(len(vt.FEATS), dtype=torch.float64)
    wv[ODD_ONE_OUT] = best[1]
    out["odd_one_out_direction"] = best[0]
    out["odd_one_out_val"] = best[2]
    out["odd_one_out_test"] = acc(test_ix, wv)
    out["odd_one_out_minus_random"] = round(out["odd_one_out_test"]
                                            - out["random_test"], 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    R = {"schema": "henri.veto.distshift.fix.v1", "pin": PIN,
         "prereg": "docstring of HENRI V2/henri_core/exp_veto_distshift_fix.py"}

    print(f"[fix] pin {PIN}", file=sys.stderr)
    R["primary"] = build_pin(PIN, {})
    print(f"[fix] replicate pin {PIN + 1234}", file=sys.stderr)
    R["replicate"] = build_pin(PIN + 1234, {})

    p = R["primary"]
    R["checks"] = {
        "T1_slot0_share_lt_0.50": p["slot0_share_real"] < 0.50,
        "T2_real_gt_random": p["real_minus_random"] > MRC,
        "T3_real_gt_shuffled": p["real_minus_shuffled"] > MRC,
        "T4_selector_alive": p["selector_val_spread"] > MRC,
        "T5_odd_one_out_replicates": bool(
            R["replicate"]["odd_one_out_direction"] == p["odd_one_out_direction"]
            and R["replicate"]["odd_one_out_minus_random"] > MRC),
    }
    core = (R["checks"]["T1_slot0_share_lt_0.50"] and R["checks"]["T2_real_gt_random"]
            and R["checks"]["T3_real_gt_shuffled"] and R["checks"]["T4_selector_alive"])
    R["verdict"] = ("VETO_ATTRIBUTABLE_AFTER_DISTRIBUTION_MATCH" if core
                    else "VETO_FIT_STILL_COLLAPSES_TO_RANKER")
    R["elapsed_s"] = round(time.time() - t0, 1)

    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/veto_fix.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
