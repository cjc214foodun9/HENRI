"""M4-G1 REMEDY: constrained readout, lambda selected on a validation split.

PRE-REGISTERED before the run (frozen here).
  Question: is the production readout the binding defect on M4-G1?
  Established (exp_m4_readout_capacity, pinned 20261010):
      production 414M readout held_tok = 0.0
      ridge (lambda=10) on the SAME frozen wave held_tok = 0.1607
      ridge shuffled-label control = 0.0089  (18x separation)
      unigram floor = 0.1171, MLP control = 0.1250
  Bar = max(floor, best_arm_control) + 0.05. On this split that is 0.175.

  Rule:
    * lambda is selected on a HELD-INTERNAL validation split (16 of 48 train),
      never on the 108-spec held-out test set. One selection, no test peeking.
    * PASS requires ridge_held > bar AND ridge_held > ridge_control.
    * Coverage@k is reported as the FOCUS ceiling: the best any reranker can do.
  Caps: CPU, D=4096 diagnostic. Not a model-performance claim.
"""
import io
import json
import os
import sys
import time

import torch

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system,
                                      unigram_floor)

PIN = 20261010
LAMS = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
OUT = os.path.join(os.environ.get("LOCALAPPDATA", "."), "Temp", "m4_remedy.json")


def flat(x):
    x = x.reshape(x.shape[0], -1)
    return torch.cat([x.real, x.imag], -1) if torch.is_complex(x) else x


def tokacc(preds, tgts):
    hit = tot = 0
    for row, ids in zip(preds, tgts):
        for a, b in zip(row, ids):
            tot += 1
            hit += int(a == b)
    return hit / max(1, tot)


def emf(preds, tgts):
    h = 0
    for row, want in zip(preds, tgts):
        n = min(len(want), len(row))
        h += int(list(row[:n]) == list(want[:n]))
    return h / max(1, len(tgts))


def main():
    t0 = time.time()
    R = {"pin": PIN, "lam_grid": LAMS}
    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, pin_seed=PIN)
    tr, ho = corpus.train_idx, corpus.heldout_idx
    tr_spec = [corpus.specs[i] for i in tr]
    ho_spec = [corpus.specs[i] for i in ho]
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho]
    Vv, M = tok.vocab_size, M4Config().max_trace

    gen = WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ftr = flat(gen.wave(tr_spec)).float()
        Fho = flat(gen.wave(ho_spec)).float()

    # standardization uses FIT rows only
    g = torch.Generator().manual_seed(11)
    perm = torch.randperm(len(tr_spec), generator=g).tolist()
    val_i, fit_i = perm[:16], perm[16:]
    mu = Ftr[fit_i].mean(0, keepdim=True)
    sd = Ftr[fit_i].std(0, keepdim=True).clamp_min(1e-6)
    Ztr, Zho = (Ftr - mu) / sd, (Fho - mu) / sd

    def Ymat(tgts):
        Y = torch.zeros(len(tgts), M * Vv)
        for i, ids in enumerate(tgts):
            for p, t in enumerate(ids[:M]):
                Y[i, p * Vv + int(t)] = 1.0
        return Y

    def ridge(Xa, Ya, Xb, lam):
        K = Xa @ Xa.T
        alpha = torch.linalg.solve(K + lam * torch.eye(len(Xa)), Ya)
        return (Xb @ Xa.T) @ alpha

    # ---- lambda selection on the INTERNAL validation split -----------------
    sel = {}
    for lam in LAMS:
        P = ridge(Ztr[fit_i], Ymat([tr_tgt[i] for i in fit_i]),
                  Ztr[val_i], lam).reshape(len(val_i), M, Vv)
        pred = P.argmax(-1)
        sel[str(lam)] = round(tokacc(pred, [tr_tgt[i] for i in val_i]), 4)
    best_lam = float(max(sel, key=lambda k: sel[k]))
    R["lambda_selection_on_val"] = sel
    R["best_lambda"] = best_lam

    # ---- retrain on ALL train rows, evaluate on held-out --------------------
    Ytr = Ymat(tr_tgt)
    Praw = ridge(Ztr, Ytr, Zho, best_lam).reshape(len(ho_spec), M, Vv)
    Pr = Praw.argmax(-1)
    g2 = torch.Generator().manual_seed(7)
    tr_shuf = [tr_tgt[i] for i in torch.randperm(len(tr_tgt), generator=g2).tolist()]
    Pc = ridge(Ztr, Ymat(tr_shuf), Zho, best_lam).reshape(len(ho_spec), M, Vv)
    Pc = Pc.argmax(-1)

    floor = unigram_floor(tok, tr_tgt, ho_tgt)
    R["ridge_remedy"] = {
        "held_tok": round(tokacc(Pr, ho_tgt), 4),
        "held_em": round(emf(Pr, ho_tgt), 4),
        "control_held_tok": round(tokacc(Pc, ho_tgt), 4),
        "floor": round(floor, 4),
    }

    # ---- FOCUS ceiling: coverage@k of the best ranking ----------------------
    cov, chance = {}, {}
    for k in (1, 3, 5, 10, 20, 50, 100):
        kk = min(k, Vv)
        topk = Praw.topk(kk, dim=-1).indices
        hit = tot = 0
        for i, ids in enumerate(ho_tgt):
            for p, t in enumerate(ids[:M]):
                tot += 1
                hit += int(bool((topk[i, p] == int(t)).any()))
        cov[k] = round(hit / max(1, tot), 4)
        chance[k] = round(kk / Vv, 4)
    R["focus_coverage_at_k"] = cov
    R["focus_chance_at_k"] = chance
    R["focus_lift_at_5"] = round(cov[5] / chance[5], 2)

    # ---- production baseline for the same run ------------------------------
    big = WaveTextGenerator(system, tok, train_body=True)
    big.fit(tr_spec, tr_tgt, M4Config(steps=400))
    bp = big.predict_ids(ho_spec, M4Config())
    R["production_baseline"] = {"held_tok": round(tokacc(bp, ho_tgt), 4)}

    bar = round(max(floor, R["ridge_remedy"]["control_held_tok"]) + 0.05, 4)
    R["bar"] = bar
    R["verdict"] = {
        "readout_is_the_defect": bool(
            R["ridge_remedy"]["held_tok"] > R["production_baseline"]["held_tok"]),
        "representation_carries_signal": bool(
            R["ridge_remedy"]["held_tok"] > R["ridge_remedy"]["control_held_tok"]),
        "m4g1_passable_constrained_head": bool(
            R["ridge_remedy"]["held_tok"] > bar),
        "focus_ceiling_at_5": cov[5],
    }
    R["elapsed_s"] = round(time.time() - t0, 1)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
