"""RANKER INVERSION PROBE -- is the ridge ranker ANTI-correlated with correctness?

MEASURED ON ENTRY (my own bytes, exp_veto_slot_prior_control.py, pins 20261010/20262244)
    per-slot positive rate, held-out top-5:  slot0 0.0317  slot1 0.0317  slot2 0.0317
                                             slot3 0.0441  slot4 0.1129
    always-pick-slot accuracy:               slot0 0.0421 ... slot4 0.1128
    coverage@5 = 0.1142, oracle = 0.1142, top1(argmax) = 0.0401
    The positive rate FALLS from slot 4 to slot 0. Slot 4 alone carries 0.1129 of a
    0.1142 ceiling. That is the signature of an INVERTED ranker, not a weak one.

WHY THIS MATTERS
    M4-G1 is recorded as failed: top-1 0.0401 < unigram floor 0.0792, verdict "not
    passable". If the ranker is inverted, the readout is not WEAK -- it is ANTI-ALIGNED,
    and flipping one comparison is the repair. That reframes a long-standing BLOCKED.

PRE-REGISTERED (frozen before this run)
  Definitions
    C[q,j]  = sum_m s_q[m, t_j[m]]   (the committed ranker statistic)
    desc    = argmax_j C[q,j]        (the committed top-1)
    asc5    = argmin over the top-5  (= slot 4)
    ascG    = argmin over ALL train candidates
  Measurements
    M1 per-slot positive rate in the held-out top-5
    M2 Pearson corr(C, exact-match label) over all (q,j) held-out pairs
    M3 accuracy of desc, asc5, ascG, random
    M4 coverage@5 descending and ascending (bottom-5)
    M5 mean percentile rank of the correct candidate among all n candidates
    M6 TRAIN-SIDE control: the same correlation on out-of-sample PROBE rows inside
       train. If the inversion also appears there it is a property of the READOUT,
       not of the held-out split.
  SELF-VERIFYING CHECK (must pass or nothing is read)
    topk(descending)[q][0] must equal argmax(C[q]) and topk[q][4] must equal argmin
    over the top-5. If this fails the topk usage or the ordering is wrong and the whole
    probe is a measurement-channel defect.
  FROZEN VERDICT
    RANKER_INVERTED            corr < -0.02 AND asc5 > desc + 0.02 at ALL pins
    RANKER_WEAK_NOT_INVERTED   otherwise
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


def pearson(x, y):
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    xx, yy = x - x.mean(), y - y.mean()
    den = (xx.pow(2).sum().sqrt() * yy.pow(2).sum().sqrt())
    return float((xx * yy).sum() / den) if float(den) > 0 else 0.0


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
    Yall = Ymat(tr_tgt[:n])
    sc_ho = sel.ridge_scores(Ztr[:n], Yall, Zho).reshape(len(Zho), M, Vv)
    Ytr = Yall.reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc_ho, Ytr)             # [nq, n]
    KK = min(vt.K, n)
    R = {"n_train": n, "n_held": len(Zho), "k": KK}

    # ---- SELF-VERIFYING ORDERING CHECK ------------------------------------
    # SELF-CAUGHT (first draft): this check compared INDEX IDENTITY and used
    #   C[q].topk(KK).indices.min()  <- min over INDEX VALUES, not over scores.
    # Both are wrong. argmax and topk may break exact ties differently when
    # duplicate traces give identical scores, and min-of-index-values is
    # meaningless. The check must compare VALUES and be tie-robust.
    tk = torch.topk(C, KK, dim=1).indices
    eps = 1e-5
    R["nan_rows_in_C"] = int(torch.isnan(C).any(dim=1).sum())
    R["rows_with_tie_at_max"] = int(
        ((C == C.max(dim=1, keepdim=True).values).sum(dim=1) > 1).sum())
    ok0 = all(abs(float(C[q, tk[q, 0]]) - float(C[q].max())) <= eps
              for q in range(len(Zho)))
    ok4 = all(abs(float(C[q, tk[q, KK - 1]])
                  - float(C[q].topk(KK).values.min())) <= eps
              for q in range(len(Zho)))
    R["order_check_topk0_is_max_value"] = bool(ok0)
    R["order_check_last_is_min_of_topk_by_value"] = bool(ok4)
    if not (ok0 and ok4):
        R["verdict"] = "MEASUREMENT_CHANNEL_DEFECT"
        return R

    def tok_acc(picks):
        h = t = 0
        for q, j in enumerate(picks):
            for x, y in zip(tr_tgt[j], ho_tgt[q]):
                t += 1
                h += int(x == y)
        return round(h / max(1, t), 4)

    def exact(picks):
        return round(sum(1 for q, j in enumerate(picks)
                         if list(tr_tgt[j]) == list(ho_tgt[q])) / len(Zho), 4)

    # ---- M3 the four rules -------------------------------------------------
    desc = [int(C[q].argmax()) for q in range(len(Zho))]
    asc5 = [int(tk[q, KK - 1]) for q in range(len(Zho))]
    ascG = [int(C[q].argmin()) for q in range(len(Zho))]
    g = torch.Generator().manual_seed(pin)
    rnd = [int(torch.randint(n, (1,), generator=g)) for _ in range(len(Zho))]
    R["M3_accuracy"] = {
        "desc_argmax_top1": {"tok": tok_acc(desc), "exact": exact(desc)},
        "asc_top5_slot4": {"tok": tok_acc(asc5), "exact": exact(asc5)},
        "asc_global_argmin": {"tok": tok_acc(ascG), "exact": exact(ascG)},
        "random": {"tok": tok_acc(rnd), "exact": exact(rnd)},
    }

    # ---- M1 per-slot positive rate ----------------------------------------
    R["M1_per_slot_positive_rate"] = {}
    for s in range(KK):
        hits = sum(1 for q in range(len(Zho))
                   if list(tr_tgt[int(tk[q, s])]) == list(ho_tgt[q]))
        R["M1_per_slot_positive_rate"][f"slot{s}"] = round(hits / len(Zho), 4)

    # ---- M4 coverage@5 both directions ------------------------------------
    def cov(picks_per_query):
        return round(sum(1 for q in range(len(Zho))
                         if any(list(tr_tgt[j]) == list(ho_tgt[q])
                                for j in picks_per_query[q])) / len(Zho), 4)
    R["M4_coverage5_desc"] = cov([tk[q].tolist() for q in range(len(Zho))])
    bot = torch.topk(C, KK, dim=1, largest=False).indices
    R["M4_coverage5_asc_bottom5"] = cov([bot[q].tolist() for q in range(len(Zho))])

    # ---- M2 correlation C vs labels over all held-out pairs ---------------
    cs, lab, agree = [], [], []
    for q in range(len(Zho)):
        for j in range(n):
            cs.append(float(C[q, j]))
            lab.append(1.0 if list(tr_tgt[j]) == list(ho_tgt[q]) else 0.0)
            a = sum(1 for x, y in zip(tr_tgt[j], ho_tgt[q]) if x == y)
            agree.append(a / max(1, max(len(tr_tgt[j]), len(ho_tgt[q]))))
    R["M2_corr_C_vs_exactmatch"] = round(pearson(cs, lab), 4)
    R["M2_corr_C_vs_token_agreement"] = round(pearson(cs, agree), 4)

    # ---- M5 mean percentile rank of the correct candidate -----------------
    pcts = []
    for q in range(len(Zho)):
        matches = [j for j in range(n) if list(tr_tgt[j]) == list(ho_tgt[q])]
        if not matches:
            continue
        j = matches[0]
        pcts.append(float((C[q] > C[q, j]).sum()) / n)
    R["M5_n_queries_with_correct_in_bank"] = len(pcts)
    R["M5_mean_percentile_rank_of_correct"] = (round(sum(pcts) / len(pcts), 4)
                                               if pcts else None)

    # ---- M6 TRAIN-SIDE control (out-of-sample within train) --------------
    gsp = torch.Generator().manual_seed(pin)
    perm = torch.randperm(n, generator=gsp).tolist()
    fit_ix, probe_ix = perm[:int(round(n * 2 / 3))], perm[int(round(n * 2 / 3)):]
    sc_pr = sel.ridge_scores(Ztr[fit_ix], Ymat([tr_tgt[i] for i in fit_ix]),
                             Ztr[probe_ix]).reshape(len(probe_ix), M, Vv)
    Cp = torch.einsum("qmv,jmv->qj", sc_pr, Ytr)
    cs2, lab2 = [], []
    for r, i in enumerate(probe_ix):
        for j in range(n):
            cs2.append(float(Cp[r, j]))
            lab2.append(1.0 if list(tr_tgt[j]) == list(tr_tgt[i]) else 0.0)
    R["M6_corr_train_probe_outofsample"] = round(pearson(cs2, lab2), 4)
    R["M6_n_train_probe_pairs"] = len(lab2)

    a = R["M3_accuracy"]
    R["inversion_gain_tok"] = round(a["asc_top5_slot4"]["tok"]
                                    - a["desc_argmax_top1"]["tok"], 4)
    R["global_argmin_gain_tok"] = round(a["asc_global_argmin"]["tok"]
                                        - a["desc_argmax_top1"]["tok"], 4)
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = {"schema": "henri.ranker.inversion.probe.v1",
         "pins": [vt.PIN, vt.PIN + 1234, vt.PIN + 9999]}
    for p in R["pins"]:
        print(f"[inversion] pin {p}", file=sys.stderr)
        R[f"pin_{p}"] = run_pin(p)
    ok = []
    for p in R["pins"]:
        r = R[f"pin_{p}"]
        ok.append(r.get("M2_corr_C_vs_exactmatch", 0) < -0.02
                  and r.get("inversion_gain_tok", 0) > 0.02)
    R["RANKER_INVERTED_at_all_pins"] = bool(all(ok))
    R["verdict"] = ("RANKER_INVERTED" if all(ok) else "RANKER_WEAK_NOT_INVERTED")
    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/inversion.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
