"""EXECUTOR-VERIFIED SELECTION: can re-executing top-5 candidates raise top-1?

WHY THIS IS THE CHEAPEST HIGH-VALUE EXPERIMENT
    Two measured facts from this program create the opening:
      * coverage@5 = 0.5036 while ridge top-1 = 0.2078 (split B, 10,164 triples).
        ~0.30 of probability mass sits in the top-5 but NOT at rank 1. That is
        2.4x headroom available to SELECTION ALONE, with no training.
      * M4 has a ground-truth oracle: a program's trace is computable. The correct
        answer among the top-5 is therefore VERIFIABLE BY EXECUTION, for free.
    This is generate -> verify -> select, and it is the user's "zone b/zone c block
    the noise zone a produces" intuition, measured rather than asserted.

    It ALSO supplies the label the veto has never had. "Anything the veto rejects
    must be learned from" is degenerate without a label source. A FAILED EXECUTION
    CHECK IS A LABELED REJECTION: (query, candidate) -> correct/incorrect. That is
    the first real training signal for a verifier on this task, and it is free.
    This run counts how many such labels exist.

PRE-REGISTERED (frozen before the run)
    Arms, all on the SAME frozen wave features and the SAME split:
      top1_baseline        ridge argmax                      (the number to beat)
      oracle_rerank_top5   pick a top-5 candidate that EXECUTES correctly.
                           BY CONSTRUCTION <= coverage@5. It is the ORACLE inside
                           the candidate set, not an achievable system; it bounds
                           the gain.
      real_verifier_top5   a TRAINED CONTENT-BASED verifier over pair features
                           [d, |d|, d^2, 1] where d = <z_query, z_candidate>.
                           Trained on the TRAIN split where execution labels are
                           free.
      shuffled_verifier    the SAME verifier with shuffled execution labels (CONTROL)
      random_verifier      uniform-random choice among the top 5 (CONTROL)

    SELF-CAUGHT DESIGN DEFECT (before the run): the first draft placed the correct
    candidate at slot 0 for 1/7 of training rows, so a per-slot weight could win by
    memorising the SLOT. Slot structure does not exist at test time, so any such
    gain is an artifact. This version uses ONLY pair content and randomises the
    candidate position within each training row.

    Reading rule:
      oracle_rerank_top5 > top1_baseline + 0.05
        -> SELECTION_HEADROOM_EXISTS (verification is worth building)
      real_verifier_top5 > top1_baseline + 0.05 AND
      real_verifier_top5 > shuffled_verifier + 0.05
        -> VERIFIER_WORKS_ON_THIS_TASK
      else HEADROOM_EXISTS_VERIFIER_NOT_LEARNED (or NO_SELECTION_HEADROOM)
    No arm is read unless the estimator control (random features, same rows in and
    out) reaches >= 0.99; otherwise HARNESS_BROKEN_NO_INTERPRETATION.

Caps: CPU, D=4096, small=True. Diagnostic only. Not a model-performance claim.
No default changes: pure diagnostic script.
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

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system)

PIN = 20261010
MAX_LEN = 4
N_INPUTS = 28
ROWS = 1500
K = 5
LAM = 1e-3


def make_inputs(n):
    pool = [f"{a}{b}{c}{d}" for a in range(1, 10)
            for b in range(10) for c in range(10) for d in range(10)]
    step = max(1, len(pool) // n)
    return pool[::step][:n]


def flat(x):
    x = x.reshape(x.shape[0], -1)
    return torch.cat([x.real, x.imag], -1) if torch.is_complex(x) else x


def tokacc(preds, tgts):
    hit = tot = 0
    for row, ids in zip(preds, tgts):
        for x, y in zip(row, ids):
            tot += 1
            hit += int(x == y)
    return round(hit / max(1, tot), 4)


def unigram_floor(tr_tgt, ho_tgt):
    from collections import Counter
    L = max(len(t) for t in tr_tgt)
    best = []
    for pos in range(L):
        c = Counter(t[pos] for t in tr_tgt if pos < len(t))
        best.append(c.most_common(1)[0][0] if c else -1)
    tot = hit = 0
    for t in ho_tgt:
        for pos, tk in enumerate(t):
            if pos < len(best) and best[pos] >= 0:
                tot += 1
                hit += int(tk == best[pos])
    return round(hit / max(1, tot), 4)


def ridge_scores(Za, Ya, Zb, lam=LAM):
    Kmat = Za @ Za.T
    eye = torch.eye(len(Za))
    for jit in (0.0, 1e-3, 1e-1, 1e1):
        try:
            alpha = torch.linalg.solve(Kmat + (lam + jit) * eye, Ya)
            return (Zb @ Za.T) @ alpha
        except Exception:                                      # noqa: BLE001
            continue
    return (Zb @ Za.T) @ torch.linalg.lstsq(Kmat + 1e1 * eye, Ya).solution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    R = {"schema": "henri.executor.verified.selection.v1", "pin": PIN, "k": K}

    inputs = make_inputs(N_INPUTS)
    corpus = build_corpus(max_len=MAX_LEN, holdout_len=MAX_LEN, inputs=inputs)
    tr_i, ho_i = corpus.train_idx, corpus.heldout_idx
    system, tok = build_system(corpus, pin_seed=PIN)
    Vv, M = tok.vocab_size, M4Config().max_trace
    tr_spec = [corpus.specs[i] for i in tr_i]
    ho_spec = [corpus.specs[i] for i in ho_i]
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr_i]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho_i]
    R["split"] = {"n_train": len(tr_i), "n_held": len(ho_i)}

    gen = WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ztr = F.normalize(torch.nan_to_num(flat(gen.wave(tr_spec))), dim=-1)
        Zho = F.normalize(torch.nan_to_num(flat(gen.wave(ho_spec))), dim=-1)

    def Ymat(tg):
        Y = torch.zeros(len(tg), M * Vv)
        for i, ids in enumerate(tg):
            for p, t in enumerate(ids[:M]):
                Y[i, p * Vv + int(t)] = 1.0
        return Y

    # ---- estimator control: random features, same rows in and out -----------
    g = torch.Generator().manual_seed(7)
    nrc = min(128, len(Ztr))
    Zrc = F.normalize(torch.randn(nrc, Ztr.shape[1], generator=g), dim=-1)
    Pc = ridge_scores(Zrc, Ymat(tr_tgt[:nrc]), Zrc)
    ctl = tokacc(Pc.reshape(-1, M, Vv).argmax(-1), tr_tgt[:nrc])
    R["estimator_control"] = ctl

    # ---- candidate ranking: keep all pairwise scores, take top-K ------------
    n = min(ROWS, len(Ztr))
    sc = ridge_scores(Ztr[:n], Ymat(tr_tgt[:n]), Zho).reshape(len(Zho), M, Vv)
    Ytr = Ymat(tr_tgt[:n]).reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc, Ytr)                  # [nq, n]
    top1 = [tr_tgt[j] for j in C.argmax(1).tolist()]
    topk = [torch.topk(C[q], min(K, n)).indices.tolist() for q in range(len(Zho))]
    R["top1_baseline"] = tokacc(top1, ho_tgt)
    R["unigram_floor"] = unigram_floor(tr_tgt, ho_tgt)

    # ---- coverage@k --------------------------------------------------------
    cov = {}
    for kk in (1, 5, 20):
        hit = 0
        for q in range(len(Zho)):
            idx = torch.topk(C[q], min(kk, n)).indices.tolist()
            if any(list(tr_tgt[j]) == list(ho_tgt[q]) for j in idx):
                hit += 1
        cov[kk] = round(hit / len(Zho), 4)
    R["coverage_at_k"] = cov
    R["chance_at_5"] = round(min(K, n) / n, 6)

    # ---- ORACLE rerank: a top-5 candidate that EXECUTES correctly ----------
    def same(cand, want):
        return list(cand) == list(want)

    orc = 0
    for q in range(len(Zho)):
        if any(same(tr_tgt[j], ho_tgt[q]) for j in topk[q]):
            orc += 1
    R["oracle_rerank_top5"] = round(orc / len(Zho), 4)

    # ---- the labeled-rejection budget the veto never had -------------------
    # every (query, top-5 candidate) pair has a free execution label
    R["n_pair_labels_available"] = int(len(Zho) * K)
    R["n_pair_labels_correct"] = int(sum(
        1 for q in range(len(Zho)) for j in topk[q]
        if same(tr_tgt[j], ho_tgt[q])))
    R["pair_label_rate"] = round(
        R["n_pair_labels_correct"] / max(1, R["n_pair_labels_available"]), 4)

    # ---- REAL verifier: CONTENT-BASED over pair features -------------------
    def pair_feat(zq, zj):
        d = float((zq * zj).sum())
        return torch.tensor([d, abs(d), d * d, 1.0])

    Xp, yp = [], []
    gv = torch.Generator().manual_seed(PIN)
    for i in range(min(600, len(Ztr))):
        cand = torch.randperm(len(Ztr), generator=gv)[:8].tolist()
        cand[0] = i
        # randomise position so no slot is systematically the answer
        order = torch.randperm(len(cand), generator=gv).tolist()
        for c in order:
            j = cand[c]
            Xp.append(pair_feat(Ztr[i], Ztr[j]))
            yp.append(1.0 if same(tr_tgt[j], tr_tgt[i]) else 0.0)
    Xp = torch.stack(Xp)
    yp = torch.tensor(yp)
    wv = torch.linalg.lstsq(Xp, yp.unsqueeze(-1)).solution.squeeze(-1)
    R["verifier_weights"] = [round(float(x), 4) for x in wv]

    def pick(idx, q, w):
        fs = torch.stack([pair_feat(Zho[q], Ztr[j]) @ w for j in idx])
        return idx[int(fs.argmax())]

    real_pred = [tr_tgt[pick(topk[q], q, wv)] for q in range(len(Zho))]
    R["real_verifier_top5"] = tokacc(real_pred, ho_tgt)

    # ---- shuffled-label control -------------------------------------------
    yperm = yp[torch.randperm(len(yp), generator=torch.Generator().manual_seed(11))]
    wv_s = torch.linalg.lstsq(Xp, yperm.unsqueeze(-1)).solution.squeeze(-1)
    shuf_pred = [tr_tgt[pick(topk[q], q, wv_s)] for q in range(len(Zho))]
    R["shuffled_verifier_top5"] = tokacc(shuf_pred, ho_tgt)

    # ---- random control ---------------------------------------------------
    rg = torch.Generator().manual_seed(PIN)
    rnd_pred = [tr_tgt[topk[q][int(torch.randint(len(topk[q]), (1,), generator=rg))]]
                for q in range(len(Zho))]
    R["random_verifier_top5"] = tokacc(rnd_pred, ho_tgt)

    R["comparison"] = {
        "top1": R["top1_baseline"], "coverage5": cov[5],
        "oracle5": R["oracle_rerank_top5"],
        "real_v": R["real_verifier_top5"], "shuf_v": R["shuffled_verifier_top5"],
        "rand_v": R["random_verifier_top5"],
        "oracle_gain": round(R["oracle_rerank_top5"] - R["top1_baseline"], 4),
        "real_gain": round(R["real_verifier_top5"] - R["top1_baseline"], 4),
        "real_minus_control": round(R["real_verifier_top5"]
                                    - R["shuffled_verifier_top5"], 4),
    }
    c = R["comparison"]
    if ctl < 0.99:
        R["verdict"] = "HARNESS_BROKEN_NO_INTERPRETATION"
    elif c["oracle_gain"] <= 0.05:
        R["verdict"] = "NO_SELECTION_HEADROOM"
    elif c["real_gain"] > 0.05 and c["real_minus_control"] > 0.05:
        R["verdict"] = "VERIFIER_WORKS_ON_THIS_TASK"
    else:
        R["verdict"] = "HEADROOM_EXISTS_VERIFIER_NOT_LEARNED"
    R["elapsed_s"] = round(time.time() - t0, 1)
    with io.open(a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/sel.json"),
                 "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
