"""DEDUP SURVIVAL + INDEX INVARIANCE -- the decisive falsifier for the two-stage selector.

MEASURED ON ENTRY (exp_two_stage_selector.py, 3 pins, FULL 1092-row bank)
    A raw argmax                     0.0
    B uniform in max group           0.1247 / 0.1251 / 0.1254   (== within-group precision)
    C k1_trace_freq|max              0.1468 / 0.1464 / 0.1468   <- best
    C k4_novelty|max                 0.1402 / 0.1380 / 0.1402
    C k2_train_len|max               0.0534 / 0.0586 / 0.0586   (length does NOT help)
    D ridge ranker                   0.0317 / 0.0260 / 0.0260
    E row-uniform random             0.0549

TWO QUESTIONS THAT DECIDE EVERYTHING

Q1 DEDUP SURVIVAL. k1_trace_freq is a PREVALENCE PRIOR over a 5.3x-duplicated bank.
   On a DEDUPLICATED bank every trace has frequency 1, so k1 has ZERO spread and must
   be a no-op. If the gain is just duplication prevalence, it does not generalize.
   Test: run both stages on the dedup bank. Report the dedup ceiling.

Q2 INDEX INVARIANCE. A content selector must not depend on candidate array position.
   Test: apply cyclic permutations pi_k(i) = (i + k) mod N to the candidate axis and
   require the PICKED TRACE to be invariant. Also assert r(slot, correctness) ~ 0.

Q3 THE CONTROL I GOT WRONG. In the previous probe arm F was a duplicate of arm B.
   The correct size-matched control draws |G_q| candidates UNIFORMLY FROM THE BANK
   (not from the group) and asks for gold -- this is the right null for "the max
   group is special".

DEFECTS RECORDED IN MY PREVIOUS PROBE
   - arm F duplicated arm B (same estimator, different seed) -> F tested nothing
   - P1 compared raw-similarity argmax against the RIDGE number -> cross-arm, ill-posed

NO TRAINING. CPU, D=4096.
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


def run_pin(pin, n_draws=40, seed=20261010):
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

    tmap = {}
    for j, t in enumerate(tr_tgt):
        tmap.setdefault(tuple(t), set()).add(j)
    R = {"pin": str(pin), "n_train": n, "n_held": nq}

    freq = {}
    for j in range(n):
        freq[tuple(tr_tgt[j])] = freq.get(tuple(tr_tgt[j]), 0) + 1

    def evaluate(S, ix_map, tag):
        """S [nq, Kcol]; ix_map[c] -> global train index. Returns all arms."""
        K = S.shape[1]
        gold_here = []
        for q in range(nq):
            gold_here.append({c for c in range(K)
                              if ix_map[c] in tmap.get(tuple(ho_tgt[q]), set())})
        out = {}

        def hit(picks):
            return sum(1 for q in range(nq) if picks[q] in gold_here[q]) / nq

        Mx = S.max(1).values
        grp = (S == Mx.unsqueeze(1))
        sizes = grp.sum(1)
        members = [torch.nonzero(grp[q]).flatten().tolist() for q in range(nq)]

        out["mean_maxgroup_size"] = round(float(sizes.float().mean()), 4)
        cont = sum(1 for q in range(nq)
                   if gold_here[q] & set(members[q])) / nq
        out["P_gold_in_maxgroup"] = round(cont, 4)
        prec = sum(len(gold_here[q] & set(members[q])) / max(1, len(members[q]))
                   for q in range(nq)) / nq
        out["within_group_precision"] = round(prec, 5)
        out["argmax_top1"] = round(hit([members[q][0] for q in range(nq)]), 4)

        gp = torch.Generator().manual_seed(seed + 1)
        out["uniform_in_group_top1"] = round(hit(
            [members[q][int(torch.randint(0, len(members[q]), (1,),
                                          generator=gp))] for q in range(nq)]), 4)

        # CORRECT size-matched control: draw |G_q| from the WHOLE bank
        gc = torch.Generator().manual_seed(seed + 2)
        ctl_hits = []
        for _ in range(n_draws):
            h = 0
            for q in range(nq):
                k = len(members[q])
                pick = torch.randperm(K, generator=gc)[:k].tolist()
                if any(c in gold_here[q] for c in pick):
                    h += 1
            ctl_hits.append(h / nq)
        out["sizematched_bank_control"] = round(sum(ctl_hits) / len(ctl_hits), 4)

        # stage 2: frequency tie-break (prevalence prior)
        picks = []
        for q in range(nq):
            mm = members[q]
            vals = [freq[tuple(tr_tgt[ix_map[c]])] for c in mm]
            picks.append(mm[int(torch.tensor(vals).argmax())])
        out["freq_tiebreak_top1"] = round(hit(picks), 4)
        out["freq_key_spread"] = round(
            sum(max(freq[tuple(tr_tgt[ix_map[c]])] for c in members[q])
                - min(freq[tuple(tr_tgt[ix_map[c]])] for c in members[q])
                for q in range(nq)) / nq, 4)

        # index-permutation invariance on the freq tie-break
        gp4 = torch.Generator().manual_seed(seed + 7)
        inv = 0
        for q in range(nq):
            base_set = {tuple(tr_tgt[ix_map[c]]) for c in members[q]}
            sh = torch.randperm(K, generator=gp4).tolist()
            perm_members = [sh[c] for c in members[q]]
            vals = [freq[tuple(tr_tgt[ix_map[c]])] for c in perm_members]
            pick = perm_members[int(torch.tensor(vals).argmax())]
            if tuple(tr_tgt[ix_map[pick]]) in base_set:
                inv += 1
        out["index_permutation_invariant_frac"] = round(inv / nq, 4)
        return out, gold_here

    # ---- FULL bank -------------------------------------------------------
    full, _ = evaluate(Sm, list(range(n)), "full")
    R["FULL_bank"] = full

    # ---- DEDUP bank: one representative per distinct trace ---------------
    repmap = {}
    for j in range(n):
        repmap.setdefault(tuple(tr_tgt[j]), j)
    ded = sorted(repmap.values())
    R["dedup_candidates"] = len(ded)
    ded_res, _ = evaluate(Sm[:, ded], ded, "dedup")
    R["DEDUP_bank"] = ded_res

    # on the dedup bank every trace has frequency 1 -> k1 spread must be 0
    R["dedup_freq_key_spread_is_zero"] = bool(
        ded_res["freq_key_spread"] == 0.0)
    R["dedup_freq_tiebreak_equals_uniform"] = bool(
        abs(ded_res["freq_tiebreak_top1"]
            - ded_res["uniform_in_group_top1"]) < 1e-9)

    # ---- verdict ---------------------------------------------------------
    R["verdict"] = {
        "full_two_stage_top1": full["freq_tiebreak_top1"],
        "full_uniform_top1": full["uniform_in_group_top1"],
        "full_gain_over_uniform": round(
            full["freq_tiebreak_top1"] - full["uniform_in_group_top1"], 4),
        "full_uniform_beats_ridge": bool(
            full["uniform_in_group_top1"] > 0.0317),
        "full_maxgroup_beats_sizematched_control": bool(
            full["P_gold_in_maxgroup"] > full["sizematched_bank_control"]),
        "dedup_two_stage_top1": ded_res["freq_tiebreak_top1"],
        "dedup_uniform_top1": ded_res["uniform_in_group_top1"],
        "dedup_containment": ded_res["P_gold_in_maxgroup"],
        "dedup_ceiling_note": "any within-group rule caps at containment",
        "gain_survives_dedup": bool(
            ded_res["freq_tiebreak_top1"] > ded_res["uniform_in_group_top1"] + 1e-9),
        "index_invariant_full": full["index_permutation_invariant_frac"],
        "index_invariant_dedup": ded_res["index_permutation_invariant_frac"],
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
    agg = {"schema": "henri.dedup_survival.receipt.v1", "results": res}
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
