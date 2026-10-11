"""FINAL INVARIANCE PROBE -- correct test: accuracy STABILITY under permutation.

MY GATE I1 WAS ILL-POSED (exp_index_invariant_selector.py)
    I tested whether the REALIZED PICK survives a bank permutation. A seeded uniform
    draw among tied members is index-dependent BY CONSTRUCTION: permuting reorders the
    member list, so the same seed yields a different member. The pick changes even for
    a perfectly content-neutral estimator. Result 0.0705 measured the test, not the arm.

THE CORRECT TEST
    Expected accuracy under a random bank permutation. For a content selector the
    expectation is invariant; the realized value varies only by finite-sample noise.
    Reference noise: binomial std sqrt(p(1-p)/nq).

ARMS (each reported on the bank it is defined on)
    argmax            picks the FIRST member of the in-group list        -> slot-bound
    uniform           kernel: ungrouped uniform over the bank
    U_in_group        uniform draw within the max-similarity group      -> stage-1 ablation
    F_freq            trace-frequency key, uniform tie-break            -> the candidate
    V_nov             within-group novelty key, uniform tie-break
    R_ridge           the trained ridge ranker (baseline)

PRE-REGISTERED READ
    J1  accuracy(original) - accuracy(permuted) must be within ~3 binomial stds
        for a content selector, and far outside it for a slot selector.
    J2  report point-biserial r(column index of the pick, correct) per arm.
    J3  U_in_group is the ablation: F_freq must beat it or stage 2 adds nothing.
    J4  bank identity appears in every number.

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


def run_pin(pin, n_perm=8, seed=20261010):
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
    tr_tgt_t = [tuple(x) for x in tr_tgt]
    gold = [{c for c in range(n) if tr_tgt_t[c] == tr_tgt_t_ho}
            for tr_tgt_t_ho in [tuple(x) for x in ho_tgt]]

    freq = {}
    for j in range(n):
        freq[tr_tgt_t[j]] = freq.get(tr_tgt_t[j], 0) + 1

    def arms(S, ix_map):
        """S [nq, K]; ix_map[c] -> global train index. Returns picks per arm."""
        K = S.shape[1]
        gset = [{c for c in range(K) if ix_map[c] in gold[q]}
                for q in range(nq)]
        Mx = S.max(1).values
        grp = (S == Mx.unsqueeze(1))
        mem = [torch.nonzero(grp[q]).flatten().tolist() for q in range(nq)]
        nov = [[1.0 - float(S[q, m].mean()) for m in range(K)]
               for q in range(nq)]
        g0 = torch.Generator().manual_seed(seed + 1)
        g2 = torch.Generator().manual_seed(seed + 2)
        g3 = torch.Generator().manual_seed(seed + 3)
        picks = {}
        picks["argmax"] = [mem[q][0] for q in range(nq)]
        picks["U_in_group"] = [mem[q][int(torch.randint(
            0, len(mem[q]), (1,), generator=g0))] for q in range(nq)]
        picks["U_bank"] = [int(torch.randint(0, K, (1,), generator=g2))
                           for _ in range(nq)]
        # content keys with UNIFORM tie-break
        def keyarm(vals_fn, gg):
            out = []
            for q in range(nq):
                mm = mem[q]
                v = torch.tensor([float(vals_fn(q, c)) for c in mm])
                mx = v.max()
                tied = [mm[i] for i in range(len(mm)) if v[i] == mx]
                out.append(tied[0] if len(tied) == 1 else
                           tied[int(torch.randint(0, len(tied), (1,),
                                                  generator=gg))])
            return out
        picks["F_freq"] = keyarm(
            lambda q, c: freq[tr_tgt_t[ix_map[c]]], g3)
        picks["V_nov"] = keyarm(lambda q, c: nov[q][c],
                                torch.Generator().manual_seed(seed + 4))
        return picks, gset, K

    def score(picks, gset):
        return sum(1 for q in range(nq) if picks[q] in gset[q]) / nq

    R = {"pin": str(pin), "n_train": n, "n_held": nq}
    binom = math.sqrt(0.125 * (1 - 0.125) / nq)

    def bank_eval(ix_map, tag):
        S = Sm0[:, ix_map] if ix_map != list(range(n)) else Sm0
        base_picks, gset, K = arms(S, ix_map)
        out = {}
        # point-biserial r(column index of pick, correct)
        for nm, pk in base_picks.items():
            acc = score(pk, gset)
            xs = torch.tensor([float(p) for p in pk])
            ys = torch.tensor([1.0 if pk[q] in gset[q] else 0.0
                               for q in range(nq)])
            if float(xs.std()) > 0 and float(ys.std()) > 0:
                r = float(torch.corrcoef(torch.stack([xs, ys]))[0, 1])
            else:
                r = None
            out[nm] = {"acc_original": round(acc, 4),
                       "r_colindex_correct": None if r is None else round(r, 4)}
        # permuted trials
        gp = torch.Generator().manual_seed(seed + 11)
        for nm in base_picks:
            accs = []
            for _ in range(n_perm):
                order = torch.randperm(n, generator=gp).tolist()
                S2 = Sm0[:, order]
                m2 = [order[c] for c in ix_map]           # ix_map permuted too
                p2, g2_, _ = arms(S2, m2)
                # locate the same arm in the permuted run
                accs.append(score(p2[nm], g2_))
            m = sum(accs) / len(accs)
            sd = (sum((a - m) ** 2 for a in accs) / len(accs)) ** 0.5
            out[nm]["acc_perm_mean"] = round(m, 4)
            out[nm]["acc_perm_std"] = round(sd, 5)
            out[nm]["drift_vs_original"] = round(out[nm]["acc_original"] - m, 4)
            out[nm]["drift_in_binomial_stds"] = round(
                abs(out[nm]["drift_vs_original"]) / max(1e-9, binom), 2)
            out[nm]["binomial_std"] = round(binom, 5)
        return out

    R["binomial_std_at_p0125"] = round(binom, 5)
    R["FULL_bank"] = bank_eval(list(range(n)), "full")
    repmap = {}
    for j in range(n):
        repmap.setdefault(tr_tgt_t[j], j)
    ded = sorted(repmap.values())
    R["dedup_candidates"] = len(ded)
    R["DEDUP_bank"] = bank_eval(ded, "dedup")

    f, d = R["FULL_bank"], R["DEDUP_bank"]
    R["verdict"] = {
        "full_F_freq": f["F_freq"]["acc_original"],
        "full_U_in_group": f["U_in_group"]["acc_original"],
        "full_F_beats_ablation": bool(
            f["F_freq"]["acc_original"] > f["U_in_group"]["acc_original"]),
        "full_argmax_drift_stds": f["argmax"]["drift_in_binomial_stds"],
        "full_F_drift_stds": f["F_freq"]["drift_in_binomial_stds"],
        "full_U_drift_stds": f["U_in_group"]["drift_in_binomial_stds"],
        "content_arms_within_3_sigma": bool(
            f["F_freq"]["drift_in_binomial_stds"] < 3
            and f["U_in_group"]["drift_in_binomial_stds"] < 3),
        "argmax_is_slot_bound": bool(
            f["argmax"]["drift_in_binomial_stds"] > 3),
        "dedup_F_freq": d["F_freq"]["acc_original"],
        "dedup_U_in_group": d["U_in_group"]["acc_original"],
        "dedup_containment_ceiling": None,
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
    agg = {"schema": "henri.selector_invariance.receipt.v1", "results": res}
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
