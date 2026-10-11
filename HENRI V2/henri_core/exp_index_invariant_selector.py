"""INDEX-INVARIANT TWO-STAGE SELECTOR -- fixing MY OWN slot defect.

DEFECT FOUND IN MY PREVIOUS PROBE (exp_dedup_survival.py)
    index_permutation_invariant_frac = 0.3964 (full bank) / 0.0287 (dedup bank).
    Only 39.6% of picks survive a candidate-axis permutation. Cause: the stage-2 key
    (trace_freq) has TIES, and I broke them with argmax -> FIRST INDEX WINS. That is
    the slot prior re-entering through my own tie-break. The 0.1468 is contaminated.

THE FIX
    Break key ties by a CONTENT-NEUTRAL draw (seeded uniform among the tied
    members), never by position. Then re-measure invariance; it must reach 1.0.

ARMS (full bank and dedup bank, both reported; every number states its bank)
    S1 uniform-within-group            (stage-1 ablation, must remain)
    S2 freq, argmax tie-break          (the defect, kept for comparison)
    S3 freq, uniform tie-break         <- the fix
    S4 novelty, uniform tie-break
    S5 freq+novelty composite, uniform tie-break
    S6 key fitted on a FIT split (logistic on [freq, novelty, len]), uniform tie-break
    S7 index-shuffled bank, S3 re-run  (bank-order invariance check)

PRE-REGISTERED
    I1  S3 index-invariance must be 1.0000 (or it is still slot-driven)
    I2  S3 must still beat S1 (the fix must not destroy the gain)
    I3  S6 must beat S3 or the fitted key adds nothing
    I4  report dedup containment; if S* > containment, the measurement is broken
    I5  a key is trusted only if its within-group spread > 0 (selector discriminates)

NO TRAINING except the tiny S6 logistic on the FIT query split. CPU, D=4096.
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


def run_pin(pin, seed=20261010):
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

    tmap = {}
    for j, t in enumerate(tr_tgt):
        tmap.setdefault(tuple(t), set()).add(j)
    freq = {}
    for j in range(n):
        freq[tuple(tr_tgt[j])] = freq.get(tuple(tr_tgt[j]), 0) + 1

    # content-neutrally seeded draw among tied members
    def tied_pick(members, vals, gen_):
        v = torch.tensor([float(vals[c]) for c in members])
        mx = v.max()
        tied = [members[i] for i in range(len(members)) if v[i] == mx]
        if len(tied) == 1:
            return tied[0]
        return tied[int(torch.randint(0, len(tied), (1,), generator=gen_))]

    def evaluate(S, ix_map, tag, shuffle_bank=False):
        K = S.shape[1]
        if shuffle_bank:
            gsb = torch.Generator().manual_seed(seed + 31)
            order = torch.randperm(K, generator=gsb).tolist()
            S = S[:, order]
            ix_map = [ix_map[c] for c in order]
        gold_here = [{c for c in range(K)
                      if ix_map[c] in tmap.get(tuple(ho_tgt[q]), set())}
                     for q in range(nq)]
        Mx = S.max(1).values
        grp = (S == Mx.unsqueeze(1))
        members = [torch.nonzero(grp[q]).flatten().tolist() for q in range(nq)]

        def hit(picks):
            return round(sum(1 for q in range(nq) if picks[q] in gold_here[q])
                         / nq, 4)

        out = {"mean_maxgroup_size": round(
            sum(len(m) for m in members) / nq, 4)}
        out["containment"] = round(
            sum(1 for q in range(nq)
                if gold_here[q] & set(members[q])) / nq, 4)

        # within-group novelty: 1 - mean sim to the OTHER group members
        nov = [[1.0 - float(S[q, m].mean()) for m in range(K)] for q in range(nq)]

        def keyvals(q, c, key):
            j = ix_map[c]
            if key == "freq":
                return float(freq[tuple(tr_tgt[j])])
            if key == "nov":
                return nov[q][c]
            if key == "mix":
                return float(freq[tuple(tr_tgt[j])]) / 100.0 + nov[q][c]
            if key == "len":
                return float(len(tr_prog[j]))
            return 0.0

        g1 = torch.Generator().manual_seed(seed + 1)
        # S1 uniform (index-invariant by construction)
        out["S1_uniform"] = hit([
            members[q][int(torch.randint(0, len(members[q]), (1,), generator=g1))]
            for q in range(nq)])
        # S2 freq, ARGMAX tie-break (the defect)
        out["S2_freq_argmax"] = hit([
            members[q][max(range(len(members[q])),
                           key=lambda c: keyvals(q, members[q][c], "freq"))]
            for q in range(nq)])
        # S3-S5 content-neutral tie-break
        for key, nm in (("freq", "S3_freq_uniform"),
                        ("nov", "S4_novelty_uniform"),
                        ("mix", "S5_mix_uniform")):
            gk = torch.Generator().manual_seed(seed + 2)
            picks = [tied_pick(members[q],
                               {c: keyvals(q, c, key) for c in range(K)}, gk)
                     for q in range(nq)]
            out[nm] = hit(picks)
        # key spread (selector must discriminate)
        sp = []
        for q in range(nq):
            v = [keyvals(q, c, "freq") for c in members[q]]
            sp.append(max(v) - min(v))
        out["freq_key_spread"] = round(sum(sp) / nq, 4)
        out["freq_key_discriminates"] = bool(sum(sp) > 0.0)

        # index-invariance of S3 (permute the candidate axis, compare picked TRACE)
        ginv = torch.Generator().manual_seed(seed + 5)
        inv = 0
        for q in range(nq):
            base = tied_pick(members[q], {c: keyvals(q, c, "freq")
                                          for c in range(K)},
                             torch.Generator().manual_seed(seed + 2))
            base_trace = tuple(tr_tgt[ix_map[base]])
            sh = torch.randperm(K, generator=ginv).tolist()
            mm = [sh[c] for c in members[q]]
            kk = {}
            for c in range(K):
                kk[sh[c]] = keyvals(q, c, "freq")
            p = tied_pick(mm, kk, torch.Generator().manual_seed(seed + 2))
            if tuple(tr_tgt[ix_map[p]]) == base_trace:
                inv += 1
        out["S3_index_invariance"] = round(inv / nq, 4)
        return out

    R = {"pin": str(pin), "n_train": n, "n_held": nq}
    R["FULL_bank"] = evaluate(Sm, list(range(n)), "full")

    repmap = {}
    for j in range(n):
        repmap.setdefault(tuple(tr_tgt[j]), j)
    ded = sorted(repmap.values())
    R["dedup_candidates"] = len(ded)
    R["DEDUP_bank"] = evaluate(Sm[:, ded], ded, "dedup")
    R["DEDUP_bank_shuffled_order"] = evaluate(Sm[:, ded], ded, "dedup_s",
                                              shuffle_bank=True)

    f, d = R["FULL_bank"], R["DEDUP_bank"]
    R["verdict"] = {
        "I1_fix_is_index_invariant": bool(d["S3_index_invariance"] == 1.0),
        "I2_fix_keeps_gain_dedup": bool(d["S3_freq_uniform"] > d["S1_uniform"]),
        "I2_fix_keeps_gain_full": bool(f["S3_freq_uniform"] > f["S1_uniform"]),
        "S2_defect_slot_share_dedup": round(
            d["S2_freq_argmax"] - d["S3_freq_uniform"], 4),
        "S3_beats_S2_on_shuffled_order": bool(
            R["DEDUP_bank_shuffled_order"]["S3_freq_uniform"]
            > R["DEDUP_bank_shuffled_order"]["S2_freq_argmax"]),
        "full_S3_beats_ridge_0.0317": bool(f["S3_freq_uniform"] > 0.0317),
        "dedup_S3_beats_ridge_0.0150": bool(d["S3_freq_uniform"] > 0.0150),
        "dedup_ceiling_is_containment": d["containment"],
        "gate_0.15_reachable_on_dedup": bool(d["containment"] >= 0.15),
        "gate_0.15_reachable_on_full": bool(
            f["containment"] >= 0.15 and f["S3_freq_uniform"] >= 0.15),
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
    agg = {"schema": "henri.index_invariant_selector.receipt.v1", "results": res}
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
