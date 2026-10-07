"""RANK-LIMIT A/B: is the measured rank ceiling ENCODER-bound or CORPUS-bound?

WHY THIS IS THE DECISIVE NEXT MEASUREMENT
    Two independent measurements located the same defect class from two angles:
      * THIS PROGRAM (exp_m4_scale, 10,164 triples): the flat ingress wave spans
        only effective rank 49 across 512 specs; a linear model on those features
        reaches 0.331 trained AND scored on its own rows, while the SAME estimator
        on random orthogonal features reaches 1.000. Feature map is rank-limited.
      * docs/SPEC_B_pooling_width_v1.md (user upload, measured): the WAVE separates
        specs (|cos| 0.1489 vs random 0.011) but the POOLED FEATURE collapses them
        (|cos| 0.9553) because HopfieldCrossPooling uses d_k = 4 -> rank <= 4 per
        macro-token. VERDICT: L2_POOLING_DESTROYS_SIGNAL.
    These are DIFFERENT STAGES (pre-pooling wave vs post-pooling feature) and both
    are measured defects. This test asks which stage binds the RANK.

DEFAULT-OFF ONLY. No default path changes. All three encoder variants already exist
in zone_a and are OFF by default, so this A/B needs no new approval.

PRE-REGISTERED (frozen before the run)
    For each encoder arm, on the SAME corpus and split:
      eff_rank      effective rank at a 1e-6 eigenvalue cut
      pr            participation ratio (tr^2 / tr2)
      capacity      ridge trained AND scored on the same 800 held rows
      ridge_held    ridge trained on train rows, scored on held rows
      knn_held      nearest-neighbour over waves
      floor         unigram floor
    CONTROL: estimator on random orthogonal features must reach >= 0.99, else the
    row is not interpreted.

    READING RULE:
      some arm raises eff_rank well above the baseline AND raises ridge_held
        -> RANK_LIMIT_IS_ENCODER_BOUND  (the fix is the ingress)
      no arm raises eff_rank
        -> RANK_LIMIT_IS_CORPUS_BOUND   (no encoder change helps; the task's
           program algebra is the limit)
    A result that raises rank but NOT accuracy is RANK_UP_ACCURACY_FLAT and is
    reported as such: rank is necessary, not sufficient.

Caps: CPU, D=4096, small=True. Diagnostic only. Not a model-performance claim.
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
DIM = 4096
RIDGE_ROWS = 1200
ORACLE_ROWS = 800
LAM = 1e-3

ARMS = [
    ("baseline_positional_off", dict(positional=False, pos_multifreq=False)),
    ("positional_single_freq", dict(positional=True, pos_multifreq=False)),
    ("positional_multifreq_dft", dict(positional=False, pos_multifreq=True)),
]


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
        for a, b in zip(row, ids):
            tot += 1
            hit += int(a == b)
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


def ridge(Xa, Ya, Xb, lam=LAM, cap=RIDGE_ROWS):
    Xa = torch.nan_to_num(Xa, nan=0.0, posinf=0.0, neginf=0.0)
    Xb = torch.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)
    Ya = torch.nan_to_num(Ya, nan=0.0, posinf=0.0, neginf=0.0)
    if cap is not None and len(Xa) > cap:
        g = torch.Generator().manual_seed(PIN)
        s = torch.randperm(len(Xa), generator=g)[:cap]
        Xa, Ya = Xa[s], Ya[s]
    K = Xa @ Xa.T
    eye = torch.eye(len(Xa))
    for jit in (0.0, 1e-3, 1e-1, 1e1):
        try:
            alpha = torch.linalg.solve(K + (lam + jit) * eye, Ya)
            return (Xb @ Xa.T) @ alpha
        except Exception:                                      # noqa: BLE001
            continue
    return (Xb @ Xa.T) @ torch.linalg.lstsq(K + 1e1 * eye, Ya).solution


def rank_stats(Z, nb=512):
    nb = min(nb, len(Z))
    K = Z[:nb] @ Z[:nb].T
    ev = torch.linalg.eigvalsh(K.double())
    tot = float(ev.sum())
    sq = float((ev ** 2).sum())
    return {
        "rows": nb,
        "eff_rank_1e6": int((ev > 1e-6).sum()),
        "participation_ratio": round(tot * tot / max(sq, 1e-30), 2),
        "top_eigval": round(float(ev.max()), 4),
        "median_eigval": round(float(ev.median()), 8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    R = {"schema": "henri.rank.limit.ab.v1", "pin": PIN, "dim": DIM,
         "max_len": MAX_LEN, "n_inputs": N_INPUTS, "arms": {}}

    inputs = make_inputs(N_INPUTS)
    corpus = build_corpus(max_len=MAX_LEN, holdout_len=MAX_LEN, inputs=inputs)
    tr_i, ho_i = corpus.train_idx, corpus.heldout_idx
    R["split"] = {"n_train": len(tr_i), "n_held": len(ho_i),
                  "train_lens": sorted({len(corpus.programs[i]) for i in tr_i}),
                  "held_lens": sorted({len(corpus.programs[i]) for i in ho_i})}
    tr_spec = [corpus.specs[i] for i in tr_i]
    ho_spec = [corpus.specs[i] for i in ho_i]

    for name, kw in ARMS:
        system, tok = build_system(corpus, pin_seed=PIN, **kw)
        Vv, M = tok.vocab_size, M4Config().max_trace
        tr_tgt = [tok.encode(corpus.traces[i]) for i in tr_i]
        ho_tgt = [tok.encode(corpus.traces[i]) for i in ho_i]
        gen = WaveTextGenerator(system, tok, train_body=True)
        with torch.no_grad():
            Ztr = F.normalize(torch.nan_to_num(flat(gen.wave(tr_spec))), dim=-1)
            Zho = F.normalize(torch.nan_to_num(flat(gen.wave(ho_spec))), dim=-1)
        row = {"encoder": kw, "feature_dim": int(Ztr.shape[1])}
        row.update(rank_stats(Ztr))

        def Ymat(tg):
            Y = torch.zeros(len(tg), M * Vv)
            for i, ids in enumerate(tg):
                for p, t in enumerate(ids[:M]):
                    Y[i, p * Vv + int(t)] = 1.0
            return Y

        # estimator control: random orthogonal features, same rows in/out
        g = torch.Generator().manual_seed(7)
        nrc = min(128, len(Ztr))
        Zrc = F.normalize(torch.randn(nrc, Ztr.shape[1], generator=g), dim=-1)
        Pc = ridge(Zrc, Ymat(tr_tgt[:nrc]), Zrc, cap=None).reshape(-1, M, Vv)
        row["estimator_control"] = tokacc(Pc.argmax(-1), tr_tgt[:nrc])

        # capacity: same rows in and out (uncapped)
        no = min(ORACLE_ROWS, len(Zho))
        Po = ridge(Zho[:no], Ymat(ho_tgt[:no]), Zho[:no], cap=None)
        row["capacity_same_rows"] = tokacc(Po.reshape(-1, M, Vv).argmax(-1),
                                           ho_tgt[:no])
        # the actual arms
        Pr = ridge(Ztr, Ymat(tr_tgt), Zho).reshape(-1, M, Vv).argmax(-1)
        row["ridge_held"] = tokacc(Pr, ho_tgt)
        gp = torch.Generator().manual_seed(PIN)
        tr_shuf = [tr_tgt[i] for i in
                   torch.randperm(len(tr_tgt), generator=gp).tolist()]
        Pd = ridge(Ztr, Ymat(tr_shuf), Zho).reshape(-1, M, Vv).argmax(-1)
        row["ridge_own_control"] = tokacc(Pd, ho_tgt)
        nn = (F.normalize(Zho, dim=-1) @ F.normalize(Ztr, dim=-1).T).argmax(1)
        row["knn_held"] = tokacc([tr_tgt[j] for j in nn.tolist()], ho_tgt)
        row["unigram_floor"] = unigram_floor(tr_tgt, ho_tgt)
        row["bar"] = round(max(row["unigram_floor"], row["ridge_own_control"]) + 0.05, 4)
        row["passes_bar_ridge"] = bool(row["ridge_held"] > row["bar"])
        row["passes_bar_knn"] = bool(row["knn_held"] > row["bar"])
        R["arms"][name] = row
        print(f"{name}: {json.dumps(row)}", flush=True)

    base = R["arms"]["baseline_positional_off"]
    best = max(R["arms"].values(), key=lambda r: r["eff_rank_1e6"])
    R["comparison"] = {
        "baseline_eff_rank": base["eff_rank_1e6"],
        "best_eff_rank": best["eff_rank_1e6"],
        "rank_rise": best["eff_rank_1e6"] - base["eff_rank_1e6"],
        "baseline_ridge_held": base["ridge_held"],
        "best_arm_ridge_held": max(r["ridge_held"] for r in R["arms"].values()),
        "all_estimator_controls_pass": all(
            r["estimator_control"] >= 0.99 for r in R["arms"].values()),
    }
    c = R["comparison"]
    if not c["all_estimator_controls_pass"]:
        R["verdict"] = "HARNESS_BROKEN_NO_INTERPRETATION"
    elif c["rank_rise"] >= 10:
        R["verdict"] = ("RANK_LIMIT_IS_ENCODER_BOUND"
                        if c["best_arm_ridge_held"] > c["baseline_ridge_held"] + 0.01
                        else "RANK_UP_ACCURACY_FLAT")
    else:
        R["verdict"] = "RANK_LIMIT_IS_CORPUS_BOUND"
    R["elapsed_s"] = round(time.time() - t0, 1)
    with io.open(a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/rank.json"),
                 "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in R.items() if k != "arms"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
