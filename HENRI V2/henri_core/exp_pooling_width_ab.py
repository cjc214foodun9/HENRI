"""POOLING-WIDTH A/B: does d_k raise the FEATURE rank and the HELD accuracy?

WHY THIS IS DECISIVE
    Measured this turn:
      * exp_m4_scale (10,164 triples): the flat ingress wave spans only effective
        rank 49 across 512 specs; a linear model on those features reaches 0.331
        trained AND scored on its own rows. Feature map is rank-limited.
      * exp_rank_limit_ab + flag_live probe: the three POSITIONAL encoder variants
        are GRAM-INVARIANT. Waves differ (max_abs_diff 1.63) but the Gram matrix is
        identical to 1e-7, so every similarity-based readout is blind to them.
        Positional phase CANNOT raise the rank. That rules out the D131 family.
      * docs/SPEC_B_pooling_width_v1.md (user-authoritative): the WAVE separates
        specs (|cos| 0.1489) but the POOLED FEATURE collapses them (|cos| 0.9553),
        because HopfieldCrossPooling uses d_k = 4 -> rank <= 4 per macro-token.
        VERDICT there: L2_POOLING_DESTROYS_SIGNAL.
    So the remaining candidate is the POOLING stage, and SPEC_B names the lever:
    dk_target. Test it directly.

PRE-REGISTERED (frozen before the run)
    Arms: dk_target = 0 (current M4 default) vs dk_target = 32 (SPEC_B proposed).
    Same corpus, same split, same pin, same estimator, same caps.
    Report per arm: n_mem, d_k, feature eff_rank, participation ratio,
      capacity (ridge trained AND scored on the same rows),
      ridge_held, knn_held, unigram_floor, estimator control.
    CONTROL: estimator on random orthogonal features must reach >= 0.99.

    READING RULE:
      dk32 raises eff_rank AND ridge_held > baseline + 0.01
        -> RANK_LIMIT_IS_POOLING_BOUND, SPEC_B mechanism CONFIRMED
      dk32 raises eff_rank but NOT accuracy
        -> RANK_UP_ACCURACY_FLAT (rank necessary, not sufficient)
      dk32 does not raise eff_rank
        -> POOLING_NOT_THE_RANK_LEVER (SPEC_B mechanism insufficient at this scale)
    No arm is interpreted unless its own estimator control passes.

Caps: CPU, D=4096, small=True, 4-digit -> 5-digit programs. Diagnostic only.
This is NOT a model-performance claim and it does NOT promote SPEC_B.
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
RIDGE_ROWS = 1200
ORACLE_ROWS = 800
LAM = 1e-3
ARMS = [("dk0_current_default", 0), ("dk32_specB_proposed", 32)]


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


def pooling_info(system):
    """Find the pooling module and report its n_mem / d_k."""
    out = {}
    for name, mod in system.decoder.named_modules():
        for attr in ("n_mem", "d_k", "d_model"):
            if hasattr(mod, attr):
                out.setdefault(name or "<root>", {})[attr] = int(getattr(mod, attr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    R = {"schema": "henri.pooling.width.ab.v1", "pin": PIN, "max_len": MAX_LEN,
         "n_inputs": N_INPUTS,
         # INVALIDATED BY DESIGN (self-caught). This compares two SEPARATELY BUILT
         # systems. dk_target changes the parameter count, so the whole-
         # construction RNG fork consumes a different stream and the INGRESS INIT
         # DIFFERS between arms (ridge_held 0.1862 vs 0.1780 on identical wave
         # features). It also ranked the RAW WAVE, which passes through no pooling.
         # Do NOT read the verdict below as evidence. Use exp_raw_vs_pooled.py,
         # which measures both stages INSIDE ONE system.
         "status": "CONFOUNDED_NOT_EVIDENCE",
         "superseded_by": "exp_raw_vs_pooled.py",
         "arms": {}}

    inputs = make_inputs(N_INPUTS)
    corpus = build_corpus(max_len=MAX_LEN, holdout_len=MAX_LEN, inputs=inputs)
    tr_i, ho_i = corpus.train_idx, corpus.heldout_idx
    R["split"] = {"n_train": len(tr_i), "n_held": len(ho_i),
                  "train_lens": sorted({len(corpus.programs[i]) for i in tr_i}),
                  "held_lens": sorted({len(corpus.programs[i]) for i in ho_i})}
    tr_spec = [corpus.specs[i] for i in tr_i]
    ho_spec = [corpus.specs[i] for i in ho_i]

    for name, dk in ARMS:
        system, tok = build_system(corpus, pin_seed=PIN, dk_target=dk)
        Vv, M = tok.vocab_size, M4Config().max_trace
        tr_tgt = [tok.encode(corpus.traces[i]) for i in tr_i]
        ho_tgt = [tok.encode(corpus.traces[i]) for i in ho_i]
        row = {"dk_target_arg": dk,
               "decoder_params": sum(p.numel() for p in system.decoder.parameters()),
               "system_total": sum(p.numel() for p in system.parameters()),
               "pooling": pooling_info(system)}

        gen = WaveTextGenerator(system, tok, train_body=True)
        with torch.no_grad():
            Ztr = F.normalize(torch.nan_to_num(flat(gen.wave(tr_spec))), dim=-1)
            Zho = F.normalize(torch.nan_to_num(flat(gen.wave(ho_spec))), dim=-1)
        row["feature_dim"] = int(Ztr.shape[1])

        nb = min(512, len(Ztr))
        ev = torch.linalg.eigvalsh((Ztr[:nb] @ Ztr[:nb].T).double())
        tot, sq = float(ev.sum()), float((ev ** 2).sum())
        row["eff_rank_1e6"] = int((ev > 1e-6).sum())
        row["participation_ratio"] = round(tot * tot / max(sq, 1e-30), 2)

        def Ymat(tg):
            Y = torch.zeros(len(tg), M * Vv)
            for i, ids in enumerate(tg):
                for p, t in enumerate(ids[:M]):
                    Y[i, p * Vv + int(t)] = 1.0
            return Y

        g = torch.Generator().manual_seed(7)
        nrc = min(128, len(Ztr))
        Zrc = F.normalize(torch.randn(nrc, Ztr.shape[1], generator=g), dim=-1)
        Pc = ridge(Zrc, Ymat(tr_tgt[:nrc]), Zrc, cap=None)
        row["estimator_control"] = tokacc(Pc.reshape(-1, M, Vv).argmax(-1),
                                          tr_tgt[:nrc])
        no = min(ORACLE_ROWS, len(Zho))
        Po = ridge(Zho[:no], Ymat(ho_tgt[:no]), Zho[:no], cap=None)
        row["capacity_same_rows"] = tokacc(Po.reshape(-1, M, Vv).argmax(-1),
                                           ho_tgt[:no])
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
        R["arms"][name] = row
        print(f"{name}: {json.dumps(row)}", flush=True)

    b = R["arms"]["dk0_current_default"]
    t = R["arms"]["dk32_specB_proposed"]
    R["comparison"] = {
        "eff_rank_baseline": b["eff_rank_1e6"],
        "eff_rank_dk32": t["eff_rank_1e6"],
        "eff_rank_rise": t["eff_rank_1e6"] - b["eff_rank_1e6"],
        "ridge_held_baseline": b["ridge_held"],
        "ridge_held_dk32": t["ridge_held"],
        "ridge_held_rise": round(t["ridge_held"] - b["ridge_held"], 4),
        "controls_pass": all(r["estimator_control"] >= 0.99
                             for r in R["arms"].values()),
        "envelope_check_full": ("SPEC_B states the d_k=32 point is 447,145,231 "
                                "(in the [400M,500M] envelope); small-config "
                                "params are reported per arm above"),
    }
    c = R["comparison"]
    if not c["controls_pass"]:
        R["verdict"] = "HARNESS_BROKEN_NO_INTERPRETATION"
    elif c["eff_rank_rise"] <= 0:
        R["verdict"] = "POOLING_NOT_THE_RANK_LEVER"
    elif c["ridge_held_rise"] > 0.01:
        R["verdict"] = "RANK_LIMIT_IS_POOLING_BOUND_SPECB_CONFIRMED"
    else:
        R["verdict"] = "RANK_UP_ACCURACY_FLAT"
    R["elapsed_s"] = round(time.time() - t0, 1)
    with io.open(a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/pool.json"),
                 "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in R.items() if k != "arms"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
