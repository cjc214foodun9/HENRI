"""TIE-ROBUST RESOLUTION -- the only honest statistic under a 24-way tie at max.

THE CONTRADICTION I MUST RESOLVE
  exp_mdl_sagnac_probe.py  : R1_raw_ip_top1 0.0000  R1_raw_ip_cov5 0.3558
  exp_raw_similarity_route : full_raw_sim.top1 0.1340 top5 0.4824
  ...but BOTH report EITHER raw similarity on the SAME full 1092-row bank with the
  SAME normalized waves. The ridge arm is byte-identical (0.0317 / 0.1142) in both.
  So the setup is equal; only the raw-sim EVALUATION differs: argmax+topk vs
  argsort. Under mean_tie_at_max = 24.27 and frac_untied = 0.0, the k-th element
  is an ARBITRARY member of the tie group. Both numbers are tie artifacts.

THE TIE-BREAK-FREE STATISTIC
  For query q: M_q = max_c S[q, c];  G_q = {c : S[q,c] == M_q}.
  stat = P( some c in G_q is gold for q )          <- independent of tie order
  control = P( a random set of size |G_q| contains gold ), size-matched per query.
  Also: precision of G_q (fraction of G_q that is gold) vs the base rate.

This decides whether raw similarity carries REAL selectable signal or whether its
apparent advantage is only the width of its tie plateau.

NO TRAINING. Analysis only. CPU, D=4096. Pins 20261010, 20262244, 20271009.
"""
from __future__ import annotations

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


def run_pin(pin, n_ctl=40):
    inputs = sel.make_inputs(sel.N_INPUTS)
    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN,
                              inputs=inputs)
    system, tok = sel.build_system(corpus, pin_seed=pin)
    tr, ho = corpus.train_idx, corpus.heldout_idx
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace

    gen = sel.WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ztr = F.normalize(torch.nan_to_num(sel.flat(gen.wave(
            [corpus.specs[i] for i in tr]))), dim=-1)
        Zho = F.normalize(torch.nan_to_num(sel.flat(gen.wave(
            [corpus.specs[i] for i in ho]))), dim=-1)

    nq, n = len(ho_tgt), len(tr_tgt)
    Sm = Zho @ Ztr.T
    if torch.is_complex(Sm):
        Sm = Sm.real

    Yall = torch.zeros(n, M * Vv)
    for i, ids in enumerate(tr_tgt):
        for p, t in enumerate(ids[:M]):
            Yall[i, p * Vv + int(t)] = 1.0
    sc = sel.ridge_scores(Ztr, Yall, Zho).reshape(nq, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc, Yall.reshape(n, M, Vv))

    # gold membership per query
    tmap = {}
    for j, t in enumerate(tr_tgt):
        tmap.setdefault(tuple(t), set()).add(j)
    gold_sets = [tmap.get(tuple(ho_tgt[q]), set()) for q in range(nq)]
    n_gold = torch.tensor([float(len(g)) for g in gold_sets])

    R = {"pin": str(pin), "n_train": n, "n_held": nq}

    # ---- 0. resolve the argmax/argsort discrepancy directly --------------
    am = Sm.argmax(1)
    as0 = torch.argsort(-Sm, dim=1)[:, 0]
    R["tie_argmax_eq_argsort_frac"] = round(
        float((am == as0).float().mean()), 6)
    R["tie_frac_untied_raw"] = round(float(
        (Sm.max(1).values.unsqueeze(1) == Sm).sum(1).eq(1).float().mean()), 6)

    def tie_robust(S, tag):
        Mx = S.max(1).values
        grp = (S == Mx.unsqueeze(1))                       # [nq, n] bool
        sz = grp.sum(1)
        gold_any, prec, ctl = [], [], []
        g = torch.Generator().manual_seed(20261010)
        for q in range(nq):
            gq = grp[q]
            gg = gold_sets[q]
            insz = gg & set(torch.nonzero(gq).flatten().tolist())
            gold_any.append(1.0 if insz else 0.0)
            prec.append(len(insz) / max(1, int(sz[q])))
            k = int(sz[q])
            hits = 0
            for _ in range(n_ctl):
                pick = torch.randperm(n, generator=g)[:k]
                if any(int(c) in gg for c in pick):
                    hits += 1
            ctl.append(hits / n_ctl)
        base = n_gold / n
        return {
            "mean_maxgroup_size": round(float(sz.float().mean()), 3),
            "P_gold_in_maxgroup": round(sum(gold_any) / nq, 4),
            "control_P_gold_in_random_samesize": round(sum(ctl) / nq, 4),
            "maxgroup_precision": round(sum(prec) / nq, 5),
            "base_rate_gold": round(float(base.mean()), 5),
            "precision_lift_over_base": round(
                (sum(prec) / nq) / max(1e-9, float(base.mean())), 3),
            "p_lift_over_control": round(
                (sum(gold_any) / nq) / max(1e-9, sum(ctl) / nq), 3),
        }

    R["raw_similarity"] = tie_robust(Sm, "raw")
    R["ridge_ranker"] = tie_robust(C, "ridge")

    # a random score field, same shape, as a third control
    g2 = torch.Generator().manual_seed(4242)
    R["random_field"] = tie_robust(
        torch.randn(nq, n, generator=g2),
        "rand")

    R["verdict"] = {
        "raw_beats_ridge_on_tie_robust": bool(
            R["raw_similarity"]["P_gold_in_maxgroup"]
            > R["ridge_ranker"]["P_gold_in_maxgroup"]),
        "raw_beats_size_matched_control": bool(
            R["raw_similarity"]["P_gold_in_maxgroup"]
            > R["raw_similarity"]["control_P_gold_in_random_samesize"]),
        "raw_maxgroup_is_selective": bool(
            R["raw_similarity"]["maxgroup_precision"]
            > R["raw_similarity"]["base_rate_gold"]),
    }
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", default="20261010,20262244,20271009")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    for p in [int(x) for x in a.pins.split(",") if x.strip()]:
        res[str(p)] = run_pin(p)
        print(f"[pin {p}] done", flush=True)
    agg = {"schema": "henri.tie_robust_resolution.receipt.v1", "results": res}
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
