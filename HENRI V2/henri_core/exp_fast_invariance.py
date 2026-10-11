"""FAST INVARIANCE -- index-correlation and permutation drift, cheap arms only.

WHY THIS EXISTS: my corrected invariance probe (exp_selector_invariance.py) recomputes
an O(nq*K) novelty row-mean once per permutation, in Python. Measured cost: >530 s
without finishing. That is my own inefficiency, not a finding. This probe keeps only
the cheap arms and answers the same question.

THE QUESTION
    Is the two-stage selector CONTENT or SLOT?
    r(column index of the pick, correct) ~ 0 -> content
    r strongly negative for argmax       -> slot-bound (picks index 0 of the tie)

ARMS
    argmax                 the first member of the in-group list
    U_in_group             uniform within the max-similarity group (stage-1 ablation)
    U_bank                 uniform over the bank
    F_freq_uniform         trace-frequency key, uniform tie-break (the candidate)
    F_freq_argmax          trace-frequency key, argmax tie-break (the defect)

Then 5 bank permutations for argmax / U_in_group / F_freq_uniform only.

PRE-REGISTERED
    K1  a content arm's |drift| must sit inside ~3 binomial stds
    K2  argmax's |r_colindex| must be large and negative if it is slot-bound
    K3  report every number against its bank identity

NO TRAINING. CPU, D=4096.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

import henri_core.exp_attributable_veto as vt                              # noqa: E402

sel = vt.sel


def run_pin(pin, n_perm=5, seed=20261010):
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
    Sm0 = Zho @ Ztr.T
    if torch.is_complex(Sm0):
        Sm0 = Sm0.real
    tt = [tuple(x) for x in tr_tgt]
    gold_glob = [{j for j in range(n) if tt[j] == tuple(ho_tgt[q])}
                 for q in range(nq)]
    freq = {}
    for j in range(n):
        freq[tt[j]] = freq.get(tt[j], 0) + 1

    def arms(S, ix_map):
        K = S.shape[1]
        gs = [{c for c in range(K) if ix_map[c] in gold_glob[q]}
              for q in range(nq)]
        Mx = S.max(1).values
        grp = (S == Mx.unsqueeze(1))
        mem = [torch.nonzero(grp[q]).flatten().tolist() for q in range(nq)]
        g1 = torch.Generator().manual_seed(seed + 1)
        g2 = torch.Generator().manual_seed(seed + 2)
        g3 = torch.Generator().manual_seed(seed + 3)
        g4 = torch.Generator().manual_seed(seed + 4)
        out = {}
        out["argmax"] = [mem[q][0] for q in range(nq)]
        out["U_in_group"] = [mem[q][int(torch.randint(0, len(mem[q]), (1,),
                                                      generator=g1))]
                             for q in range(nq)]
        out["U_bank"] = [int(torch.randint(0, K, (1,), generator=g2))
                         for _ in range(nq)]
        # freq key, uniform tie-break
        p = []
        for q in range(nq):
            mm = mem[q]
            v = torch.tensor([float(freq[tt[ix_map[c]]]) for c in mm])
            mx = v.max()
            tied = [mm[i] for i in range(len(mm)) if v[i] == mx]
            p.append(tied[0] if len(tied) == 1 else
                     tied[int(torch.randint(0, len(tied), (1,), generator=g3))])
        out["F_freq_uniform"] = p
        # freq key, argmax tie-break (the defect)
        p2 = []
        for q in range(nq):
            mm = mem[q]
            v = [float(freq[tt[ix_map[c]]]) for c in mm]
            p2.append(mm[max(range(len(mm)), key=lambda i: v[i])])
        out["F_freq_argmax"] = p2
        return out, gs

    def stats(pk, gs):
        acc = sum(1 for q in range(nq) if pk[q] in gs[q]) / nq
        xs = torch.tensor([float(x) for x in pk])
        ys = torch.tensor([1.0 if pk[q] in gs[q] else 0.0 for q in range(nq)])
        if float(xs.std()) > 0 and float(ys.std()) > 0:
            r = float(torch.corrcoef(torch.stack([xs, ys]))[0, 1])
        else:
            r = float("nan")
        return round(acc, 4), (None if math.isnan(r) else round(r, 4))

    binom = math.sqrt(0.125 * 0.875 / nq)
    R = {"pin": str(pin), "n_train": n, "n_held": nq,
         "binomial_std_at_p0.125": round(binom, 5)}

    # ---- FULL bank -------------------------------------------------------
    a_full, gs_full = arms(Sm0, list(range(n)))
    R["FULL_bank"] = {k: dict(zip(("acc", "r_colindex"), stats(v, gs_full)))
                      for k, v in a_full.items()}
    # permutations (cheap arms only)
    gp = torch.Generator().manual_seed(seed + 11)
    perm = {k: [] for k in ("argmax", "U_in_group", "F_freq_uniform")}
    for _ in range(n_perm):
        order = torch.randperm(n, generator=gp).tolist()
        a2, g2_ = arms(Sm0[:, order], order)
        for k in perm:
            perm[k].append(stats(a2[k], g2_)[0])
    for k, v in perm.items():
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
        R["FULL_bank"][k]["acc_perm_mean"] = round(m, 4)
        R["FULL_bank"][k]["acc_perm_std"] = round(sd, 5)
        R["FULL_bank"][k]["drift"] = round(
            R["FULL_bank"][k]["acc"] - m, 4)
        R["FULL_bank"][k]["drift_in_sigma"] = round(
            abs(R["FULL_bank"][k]["drift"]) / max(1e-9, binom), 2)

    # ---- DEDUP bank ------------------------------------------------------
    rep = {}
    for j in range(n):
        rep.setdefault(tt[j], j)
    ded = sorted(rep.values())
    R["dedup_candidates"] = len(ded)
    a_ded, gs_ded = arms(Sm0[:, ded], ded)
    R["DEDUP_bank"] = {k: dict(zip(("acc", "r_colindex"), stats(v, gs_ded)))
                       for k, v in a_ded.items()}

    f, d = R["FULL_bank"], R["DEDUP_bank"]
    R["verdict"] = {
        "full_argmax_is_slot_bound": bool(
            f["argmax"]["r_colindex"] is not None
            and f["argmax"]["r_colindex"] < -0.05),
        "full_F_freq_is_index_neutral": bool(
            abs(f["F_freq_uniform"]["r_colindex"] or 0.0) < 0.05),
        "full_F_freq_drift_sigma": f["F_freq_uniform"].get("drift_in_sigma"),
        "full_U_in_group_drift_sigma": f["U_in_group"].get("drift_in_sigma"),
        "argmax_drift_sigma": f["argmax"].get("drift_in_sigma"),
        "full_F_beats_U_in_group": bool(
            f["F_freq_uniform"]["acc"] > f["U_in_group"]["acc"]),
        "dedup_F_beats_U_in_group": bool(
            d["F_freq_uniform"]["acc"] > d["U_in_group"]["acc"]),
        "dedup_F_acc": d["F_freq_uniform"]["acc"],
        "dedup_U_acc": d["U_in_group"]["acc"],
        "dedup_argmax_r_colindex": d["argmax"]["r_colindex"],
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
    agg = {"schema": "henri.fast_invariance.receipt.v1", "results": res}
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
