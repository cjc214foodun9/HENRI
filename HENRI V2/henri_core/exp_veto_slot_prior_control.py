"""SLOT-PRIOR CONTROL: is the veto win CONTENT or a constant slot preference?

MEASURED ON ENTRY (my own bytes, pin 20261010, exp_veto_distshift_fix.py)
    real fitted arm 0.1097   shuffled 0.0564   random 0.0579   oracle 0.1142
    chosen_index_dist_real [119, 81, 36, 367, 531] -> slots 3 and 4 take 79% of picks.
    slot0_share fell 0.649 -> 0.105, so the fit no longer drifts to the ranker.

THE QUESTION
    The arm now concentrates on the LOW-ranked slots. If slot 4 happens to carry a
    higher positive rate in the held-out top-5, then "always pick slot 4" may score as
    well as the feature-based arm, and the +0.05 would be a SLOT PRIOR, not content.

THE DISCRIMINATING CONTROL (pre-registered here)
    S1 per-slot positive rate in the held-out top-5.
    S2 "always pick slot k" accuracy, for every k.
    S3 SLOT-PRIOR-ONLY arm: sample a slot from the REAL arm's OWN slot distribution,
       uniformly and content-independently, and score it. This holds the slot marginal
       fixed and removes all content information.
    S4 content arms (odd-one-out, min_token_score) re-reported on the same folds.

READING (frozen)
    CONTENT if real_arm - slot_prior_only > 0.01 AND real_arm - best_always_slot > 0.01
    SLOT_ARTIFACT otherwise.
    If SLOT_ARTIFACT, the veto is NOT attributable and the +0.05 is withdrawn.

WHY THIS IS NOT A POST-HOC GATE FLIP
    The failed T4 in the fix run measures lambda-insensitivity, NOT collapse. Its
    fail-label is misnamed; that is reported separately as a defect. THIS probe tests a
    different, NEWLY PRE-REGISTERED question about the positive result. It does not
    revise the frozen rule of the previous run.

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
import henri_core.exp_veto_distshift_fix as fx                             # noqa: E402

sel = vt.sel
PIN = vt.PIN
LAMS = fx.LAMS
MRC_STRICT = 0.01
ODD = vt.FEATS.index("mean_agree_others")
MINTOK = vt.FEATS.index("min_token_score")


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
    C = torch.einsum("qmv,jmv->qj", sc_ho, Ytr)
    KK = min(vt.K, n)
    topk = [torch.topk(C[q], KK).indices.tolist() for q in range(len(Zho))]

    gq = torch.Generator().manual_seed(pin + 1)
    qp = torch.randperm(len(Zho), generator=gq).tolist()
    test_ix = qp[len(qp) // 2:]

    def hit(q, j):
        h = t = 0
        for x, y in zip(tr_tgt[j], ho_tgt[q]):
            t += 1
            h += int(x == y)
        return h, t

    # ---- S1 per-slot positive rate on the TEST fold -------------------------
    pos = [0] * KK
    tot = [0] * KK
    for q in test_ix:
        for s in range(KK):
            j = topk[q][s]
            h, t = hit(q, j)
            pos[s] += 1 if (h == t and t > 0) else 0
            tot[s] += 1
    R = {"n_test": len(test_ix),
         "per_slot_positive_rate": {f"slot{s}": round(pos[s] / max(1, tot[s]), 4)
                                    for s in range(KK)}}

    # ---- S2 always-pick-slot-k accuracy -------------------------------------
    always = {}
    for s in range(KK):
        h = t = 0
        for q in test_ix:
            hh, tt = hit(q, topk[q][s])
            h += hh
            t += tt
        always[f"slot{s}"] = round(h / max(1, t), 4)
    R["always_pick_slot_accuracy"] = always
    R["best_always_slot"] = max(always, key=lambda k: always[k])
    R["best_always_slot_acc"] = always[R["best_always_slot"]]

    # ---- rebuild the distribution-matched fit (same code path as the fix) ----
    gsp = torch.Generator().manual_seed(pin)
    perm = torch.randperm(n, generator=gsp).tolist()
    fit_ix, probe_ix = perm[:int(round(n * 2 / 3))], perm[int(round(n * 2 / 3)):]
    sc_pr = sel.ridge_scores(Ztr[fit_ix], Ymat([tr_tgt[i] for i in fit_ix]),
                             Ztr[probe_ix]).reshape(len(probe_ix), M, Vv)
    Cp = torch.einsum("qmv,jmv->qj", sc_pr, Ytr)
    for r, i in enumerate(probe_ix):
        Cp[r, i] = float("-inf")
    Xtr, ytr = [], []
    for r, i in enumerate(probe_ix):
        cand = torch.topk(Cp[r], KK).indices.tolist()
        for j in cand:
            Xtr.append(vt.feats_for(sc_pr[r], tr_tgt[j], j, Ztr[i], cand,
                                    [tr_tgt[x] for x in cand], Ztr))
            ytr.append(1.0 if list(tr_tgt[j]) == list(tr_tgt[i]) else 0.0)
    Xtr = torch.tensor(Xtr, dtype=torch.float64)
    ytr = torch.tensor(ytr, dtype=torch.float64)
    Xh = [torch.tensor([vt.feats_for(sc_ho[q], tr_tgt[j], j, Zho[q], topk[q],
                                     [tr_tgt[x] for x in topk[q]], Ztr)
                        for j in topk[q]], dtype=torch.float64) for q in range(len(Zho))]
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-9)
    XtrS = (Xtr - mu) / sd
    zq_ = lambda q: (Xh[q] - mu) / sd                                    # noqa: E731

    def acc(queries, w):
        h = t = 0
        for q in queries:
            hh, tt = hit(q, topk[q][int((zq_(q) @ w).argmax())])
            h += hh
            t += tt
        return round(h / max(1, t), 4)

    lam = 1e-3
    A = XtrS.T @ XtrS + lam * torch.eye(XtrS.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(A, XtrS.T @ ytr)
    R["real_arm_acc"] = acc(test_ix, w)

    # the arm's own slot marginal on the test fold
    dist = [0] * KK
    for q in test_ix:
        dist[int((zq_(q) @ w).argmax())] += 1
    R["arm_slot_marginal"] = [round(x / max(1, sum(dist)), 4) for x in dist]

    # ---- S3 SLOT-PRIOR-ONLY: same slot marginal, no content ------------------
    trials = []
    for seed in (1, 2, 3, 4, 5):
        g = torch.Generator().manual_seed(pin + seed)
        h = t = 0
        for q in test_ix:
            s = int(torch.multinomial(torch.tensor(dist, dtype=torch.float32), 1,
                                      generator=g))
            hh, tt = hit(q, topk[q][s])
            h += hh
            t += tt
        trials.append(round(h / max(1, t), 4))
    R["slot_prior_only_acc"] = trials
    R["slot_prior_only_mean"] = round(sum(trials) / len(trials), 4)

    # ---- S4 content arms -----------------------------------------------------
    def single(idx, sign):
        wv = torch.zeros(len(vt.FEATS), dtype=torch.float64)
        wv[idx] = sign
        return acc(test_ix, wv)

    R["odd_one_out_acc"] = {"min": single(ODD, -1.0), "max": single(ODD, 1.0)}
    R["min_token_score_acc"] = {"min": single(MINTOK, -1.0), "max": single(MINTOK, 1.0)}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = {"schema": "henri.veto.slot_prior_control.v1", "pin": PIN}
    for tag, p in (("primary", PIN), ("replicate", PIN + 1234)):
        print(f"[slotctl] {tag} pin {p}", file=sys.stderr)
        r = run_pin(p)
        r["real_minus_slot_prior"] = round(r["real_arm_acc"]
                                           - r["slot_prior_only_mean"], 4)
        r["real_minus_best_always_slot"] = round(r["real_arm_acc"]
                                                 - r["best_always_slot_acc"], 4)
        r["is_content"] = bool(r["real_minus_slot_prior"] > MRC_STRICT
                               and r["real_minus_best_always_slot"] > MRC_STRICT)
        r["verdict"] = ("CONTENT_SIGNAL_CONFIRMED" if r["is_content"]
                        else "SLOT_ARTIFACT_WITHDRAW_THE_GAIN")
        R[tag] = r
    R["overall"] = ("CONTENT_SIGNAL_CONFIRMED" if all(
        R[t]["is_content"] for t in ("primary", "replicate"))
        else "SLOT_ARTIFACT_WITHDRAW_THE_GAIN")
    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/slotctl.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
