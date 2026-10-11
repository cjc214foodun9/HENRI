"""DECISIVE PRE-CHECK for the biophysical roadmap Phase 1.

The document prescribes:
    S_calibrated = exp(Re<psi_q,psi_k>/tau)
and claims tau calibration "compresses tie-group cardinality (24 -> 1)" with
"unique argmax fraction = 1.0", Gate G-1 top-1 > 0.1500 on the dedup bank.

THREE CLAIMS TO TEST. The first is elementary; the rest decide the roadmap.

P1 MONOTONICITY. exp(s/tau) is strictly monotone in s for every tau > 0. A strictly
   monotone transform preserves order AND exact equality. So temperature CANNOT
   split an exact tie. Test on real bank scores: does argmax(exp(S/tau)) ever
   differ from argmax(S), and does the tie group ever shrink?

P2 CONTAINMENT CEILING. On the DEDUPLICATED bank, what is
   P(gold in max-score group), the group size, and therefore the CEILING on any
   within-group tie-breaker (= containment x 1/group_size under uniform pick)?
   If that ceiling is near 0.15, Gate G-1 is unreachable with this score.

P3 TIE EXACTNESS. Are the ties bitwise EXACT (identical float32) or merely
   quantized-near? Exact ties can only be broken by a new key. Near ties could in
   principle be perturbed -- say which one this is.

P4 CONFOUND. Containment vs program length (train programs < 4 ops, held-out >= 4).

NO TRAINING. Analysis only. CPU, D=4096. Pins 20261010, 20262244, 20271009.
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


def run_pin(pin):
    inputs = sel.make_inputs(sel.N_INPUTS)
    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN,
                              inputs=inputs)
    system, tok = sel.build_system(corpus, pin_seed=pin)
    tr, ho = corpus.train_idx, corpus.heldout_idx
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho]
    tr_prog = [corpus.programs[i] for i in tr]
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
    R = {"pin": str(pin), "n_train": n, "n_held": nq, "full_bank": n}

    # ---- P1: monotonicity. argmax invariant, ties unsplittable -----------
    base_am = Sm.argmax(1)
    p1 = {}
    for tau in (0.038316, 0.5, 1.0, 4.0, 100.0):
        S = torch.exp(Sm / tau)
        p1[f"tau={tau:g}"] = {
            "argmax_identical": bool((S.argmax(1) == base_am).all()),
            "mean_maxgroup_size": round(float(
                (S.max(1).values.unsqueeze(1) == S).sum(1).float().mean()), 6),
        }
    # also softmax, which the document also writes
    for tau in (0.038316, 1.0):
        P = torch.softmax(Sm / tau, dim=1)
        p1[f"softmax_tau={tau:g}"] = {
            "argmax_identical": bool((P.argmax(1) == base_am).all()),
            "mean_maxgroup_size": round(float(
                (P.max(1).values.unsqueeze(1) == P).sum(1).float().mean()), 6),
        }
    R["P1_monotone"] = p1
    R["P1_reference_mean_maxgroup"] = round(float(
        (Sm.max(1).values.unsqueeze(1) == Sm).sum(1).float().mean()), 6)
    R["P1_verdict"] = ("MONOTONE_TIE_UNSPLITTABLE"
                       if all(v["argmax_identical"] for v in p1.values())
                       else "UNEXPECTED_ARGMAX_CHANGE")

    # ---- gold membership ------------------------------------------------
    tmap = {}
    for j, t in enumerate(tr_tgt):
        tmap.setdefault(tuple(t), set()).add(j)
    gold_sets = [tmap.get(tuple(ho_tgt[q]), set()) for q in range(nq)]

    def group_stats(S, ix_map, tag):
        Mx = S.max(1).values
        grp = (S == Mx.unsqueeze(1))
        sz = grp.sum(1)
        cont, inprec, gsize, ngold_tot = 0, 0.0, [], 0
        for q in range(nq):
            members = set(torch.nonzero(grp[q]).flatten().tolist())
            gg = gold_sets[q]
            insz = gg & {ix_map[int(c)] for c in members}
            if insz:
                cont += 1
            inprec += len(insz) / max(1, len(members))
            gsize.append(len(members))
            ngold_tot += len(insz)
        msz = sum(gsize) / len(gsize)
        contain = cont / nq
        return {
            "mean_group_size": round(msz, 4),
            "P_gold_in_maxgroup": round(contain, 4),
            "mean_within_group_gold_precision": round(inprec / nq, 5),
            "within_group_gold_count": ngold_tot,
            "CEILING_if_uniform_within_group": round(contain / max(1e-9, msz), 4),
            "chance_at_that_group_size": round(contain / max(1e-9, msz), 4),
        }

    # dedup bank
    rep = {}
    for j, t in enumerate(tr_tgt):
        rep.setdefault(tuple(t), j)
    ded = sorted(rep.values())
    nd = len(ded)
    R["dedup_candidates"] = nd
    R["dedup_raw_similarity"] = group_stats(Sm[:, ded], ded, "dedup_raw")

    # full bank for reference
    R["full_raw_similarity"] = group_stats(Sm, list(range(n)), "full_raw")

    # ---- P3: exact vs near ties -----------------------------------------
    Mx = Sm.max(1).values
    exact = (Sm.max(1).values.unsqueeze(1) == Sm).sum(1).float()
    # bitwise: compare against a tiny epsilon-perturbed copy
    eps = 1e-6
    near = (Sm >= (Mx - eps).unsqueeze(1)).sum(1).float()
    R["P3_exact_tie_group_mean"] = round(float(exact.mean()), 4)
    R["P3_eps1e-6_group_mean"] = round(float(near.mean()), 4)
    R["P3_ties_are_exact"] = bool(float(near.mean()) <= float(exact.mean()) + 1e-6)
    R["P3_frac_rows_exact_gt1"] = round(float((exact > 1).float().mean()), 4)

    # ---- P4: confound with program length -------------------------------
    plen = {j: len(p) for j, p in zip(range(n), tr_prog)}
    by_len = {}
    for ln in sorted(set(plen.values())):
        ix = [j for j in range(n) if plen[j] == ln]
        # fraction of candidates at this length that match some gold
        gm = sum(1 for j in ix if tuple(tr_tgt[j]) in
                 {tuple(t) for t in ho_tgt}) / len(ix)
        by_len[str(ln)] = {"n": len(ix), "frac_gold_match": round(gm, 4)}
    R["P4_gold_match_by_prog_len"] = by_len
    R["P4_train_prog_len_max"] = max(int(v) for v in plen.values())
    R["P4_held_prog_len_min"] = min(len(p) for p in
                                    [corpus.programs[i] for i in ho])

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
    agg = {"schema": "henri.roadmap_phase1_precheck.receipt.v1", "results": res}
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
