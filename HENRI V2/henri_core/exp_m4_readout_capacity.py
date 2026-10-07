"""M4 READOUT-CAPACITY EXPERIMENT (pre-registered before the run).

AUDIT THAT MOTIVATES THIS
    M4-G1 fails. Measured on the pinned build:
      production readout (HenriDec450M, train_body=True, 400 steps):
        train token acc 1.0 / exact-match 1.0
        held-out token acc 0.0 / exact-match 0.0   (floor 0.1171)
      not constant: 107 distinct predictions for 108 held specs
      not a lookup: prediction == nearest train target only 0.0278
      the WAVE carries weak signal: held target == nearest-wave train target
        0.1667 vs shuffled control 0.0185  (9x lift)
      recall: correct token in top-1 0.0 | top-20 0.3839 | top-100 0.5179
        (chance 20/353 = 0.0566, 100/353 = 0.2833)
    So the wave is informative but weak, and the 414M-param readout memorizes
    48 training examples and lands at 0.0 held-out.

PRE-REGISTERED DECISION RULE (frozen before the run)
    bar = max(unigram_floor, shuffled-arm held acc) + 0.05
    CONSTRAINED arm beats bar with its OWN shuffled-label control intact
        -> the readout was the blocker; a constrained head is the fix.
    CONSTRAINED arm does not beat its shuffled-label control
        -> the frozen wave carries insufficient information; Subsystem 1
           (learnable ingress) is a prerequisite. No readout fix can pass.

ARMS (same pinned build, same 48/108 split, frozen wave in all arms)
    big     production WaveTextGenerator, 400 steps            [baseline]
    ridge   dual ridge on the standardized flattened wave, fixed lambda
    mlp     small MLP, weight decay
    ctrl_*  each arm retrained on SHUFFLED (spec, target) pairs
    knn     nearest-wave train target                          [6th control]

CAPS
    CPU, D=4096 diagnostic only. No model-performance claim.
"""
import io
import json
import os
import sys
import time

import torch
import torch.nn as nn

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system,
                                      unigram_floor)

PIN = 20261010
LAM = 10.0
OUT = os.path.join(os.environ.get("LOCALAPPDATA", "."), "Temp",
                   "m4_readout_capacity.json")


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


def em(preds, tgts):
    h = 0
    for row, want in zip(preds, tgts):
        n = min(len(want), len(row))
        h += int(list(row[:n]) == list(want[:n]))
    return h / max(1, len(tgts))


def main():
    t0 = time.time()
    R = {"pin": PIN, "lam": LAM}
    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, pin_seed=PIN)
    tr, ho = corpus.train_idx, corpus.heldout_idx
    tr_spec = [corpus.specs[i] for i in tr]
    ho_spec = [corpus.specs[i] for i in ho]
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho]
    Vv = tok.vocab_size
    M = M4Config().max_trace
    R["split"] = {"n_train": len(tr), "n_held": len(ho), "vocab": Vv, "M": M}

    gen = WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Xtr = flat(gen.wave(tr_spec)).float()
        Xho = flat(gen.wave(ho_spec)).float()
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd
    Xho = (Xho - mu) / sd
    R["feature_dim"] = int(Xtr.shape[1])

    def Ymat(tgts):
        Y = torch.zeros(len(tgts), M * Vv)
        for i, ids in enumerate(tgts):
            for p, t in enumerate(ids[:M]):
                Y[i, p * Vv + int(t)] = 1.0
        return Y

    def ridge(Xa, Ya, Xb):
        K = Xa @ Xa.T
        A = K + LAM * torch.eye(len(Xa), dtype=Xa.dtype)
        alpha = torch.linalg.solve(A, Ya)
        P = (Xb @ Xa.T @ alpha).reshape(len(Xb), M, Vv)
        return P

    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(len(tr_tgt), generator=g)
    tr_shuf = [tr_tgt[i] for i in perm]

    # ---- arm big: production readout ---------------------------------------
    big = WaveTextGenerator(system, tok, train_body=True)
    big.fit(tr_spec, tr_tgt, M4Config(steps=400))
    bp = big.predict_ids(ho_spec, M4Config())
    R["big"] = {"held_tok": round(tokacc(bp, ho_tgt), 4),
                "held_em": round(em(bp, ho_tgt), 4)}

    # ---- arm ridge + its shuffled control ----------------------------------
    Pr = ridge(Xtr, Ymat(tr_tgt), Xho).argmax(-1)
    Pc = ridge(Xtr, Ymat(tr_shuf), Xho).argmax(-1)
    R["ridge"] = {"held_tok": round(tokacc(Pr, ho_tgt), 4),
                  "held_em": round(em(Pr, ho_tgt), 4),
                  "ctrl_held_tok": round(tokacc(Pc, ho_tgt), 4)}

    # coverage@k of the ridge ranking (recall of the correct token)
    Praw = ridge(Xtr, Ymat(tr_tgt), Xho)
    cov = {}
    for k in (1, 3, 5, 10, 20, 50, 100):
        kk = min(k, Vv)
        topk = Praw.topk(kk, dim=-1).indices
        hit = tot = 0
        for i, ids in enumerate(ho_tgt):
            for p, t in enumerate(ids[:M]):
                tot += 1
                hit += int(bool((topk[i, p] == int(t)).any()))
        cov[k] = round(hit / max(1, tot), 4)
    R["ridge_coverage_at_k"] = cov
    R["chance_at_k"] = {k: round(min(k, Vv) / Vv, 4) for k in cov}

    # ---- arm mlp + its shuffled control ------------------------------------
    def run_mlp(targets, steps=300, seed=0):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.GELU(),
                            nn.Linear(128, M * Vv))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        Y = torch.full((len(targets), M), -100, dtype=torch.long)
        for i, ids in enumerate(targets):
            ids = ids[:M]
            Y[i, :len(ids)] = torch.tensor(ids)
        for _ in range(steps):
            out = net(Xtr).reshape(len(Xtr), M, Vv)
            loss = torch.nn.functional.cross_entropy(
                out.transpose(1, 2), Y, ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            o = net(Xho).reshape(len(Xho), M, Vv)
        return o.argmax(-1)

    Pm = run_mlp(tr_tgt, seed=0)
    Pmc = run_mlp(tr_shuf, seed=1)
    R["mlp"] = {"held_tok": round(tokacc(Pm, ho_tgt), 4),
                "held_em": round(em(Pm, ho_tgt), 4),
                "ctrl_held_tok": round(tokacc(Pmc, ho_tgt), 4)}

    # ---- control: nearest-wave train target --------------------------------
    import torch.nn.functional as F
    a = F.normalize(Xho, dim=-1)
    b = F.normalize(Xtr, dim=-1)
    nn_idx = (a @ b.T).argmax(1)
    Pk = [tr_tgt[j] for j in nn_idx.tolist()]
    R["knn"] = {"held_tok": round(tokacc(Pk, ho_tgt), 4),
                "held_em": round(em(Pk, ho_tgt), 4)}

    floor = unigram_floor(tok, tr_tgt, ho_tgt)
    R["unigram_floor"] = round(floor, 4)
    best_ctrl = max(R["mlp"]["ctrl_held_tok"], R["ridge"]["ctrl_held_tok"])
    bar = round(max(floor, best_ctrl) + 0.05, 4)
    R["bar"] = bar
    for arm in ("big", "ridge", "mlp", "knn"):
        R[arm]["passes_bar"] = bool(R[arm]["held_tok"] > bar)
    R["verdict"] = {
        "ridge_beats_own_control": bool(
            R["ridge"]["held_tok"] > R["ridge"]["ctrl_held_tok"]),
        "mlp_beats_own_control": bool(
            R["mlp"]["held_tok"] > R["mlp"]["ctrl_held_tok"]),
        "any_constrained_arm_passes": bool(
            R["ridge"]["passes_bar"] or R["mlp"]["passes_bar"]),
    }
    R["elapsed_s"] = round(time.time() - t0, 1)

    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
