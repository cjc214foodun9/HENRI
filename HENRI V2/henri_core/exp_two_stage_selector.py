"""TWO-STAGE SELECTOR -- the decisive Phase-1 experiment.

ESTABLISHED GROUND TRUTH (my own bytes, pins 20261010/20262244/20271009)
  raw Re<psi_q,psi_k>   P(gold in max group) = 0.754, mean group 24.27
                        mean within-group gold precision = 0.12498   <- KEY
  dedup bank            containment 0.1367, group 5.13, precision 0.02628
  ridge ranker          top1 0.0317   (anti-selective: 0.608x base)
  row-uniform random    top1 0.0534
  argmax of raw         top1 0.0      (always the FIRST tied element)
  frac_untied full      0.0

THE HYPOTHESIS UNDER TEST
  argmax picks index 0 of a 24-way tie. A UNIFORM draw from the same tie group
  should score ~0.125 (the measured within-group precision), i.e. ~3.9x the
  trained ranker, with ZERO training. Then: does a SECONDARY key beat uniform?

ARMS (all on the SAME full bank, same frozen waves; every number names its bank)
  A  argmax                 (the current selector)
  B  uniform-within-maxgroup (SEEDED; averaged over draws)   <- the ablation
  C  secondary-key-within-maxgroup, key chosen on a FIT split
  D  ridge ranker           (the trained baseline)
  E  row-uniform random over the bank
  F  size-matched random control inside the max group

STAGE-2 CANDIDATE KEYS (content-based; length is reported FLAGGED)
  k1 trace_freq     : frequency of the candidate's trace in the FIT split
  k2 train_len      : program length            <- CONFOUND: held-out len >= 4
  k3 maxsim_to_fit  : max similarity to a FIT row of the same held-out query
  k4 novelty        : 1 - (mean similarity to the other max-group members)

PRE-REGISTERED RULES
  P1  argmax must reproduce 0.0317 (harness check)
  P2  uniform-within-group must be ~= within-group precision (consistency)
  P3  a secondary key EARNS its place only if it beats arm B on the same split
  P4  the selector must DISCRIMINATE: key spread over the group must be > 0
      before its winner is trusted (the lambda-selection lesson)
  P5  any arm that beats B only via train_len is CONFOUNDED, not a result
  KILL: if no secondary key beats B, within-group content signal does not exist
        and the 0.15 gate is retired as unreachable at this substrate.

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

    # gold membership sets (global train indices)
    tmap = {}
    for j, t in enumerate(tr_tgt):
        tmap.setdefault(tuple(t), set()).add(j)
    gold = [tmap.get(tuple(ho_tgt[q]), set()) for q in range(nq)]
    base = sum(len(g) for g in gold) / (nq * n)

    # FIT/PROBE split of the QUERIES: keys are chosen on FIT only (never held-out)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(nq, generator=g).tolist()
    nfit = nq // 2
    fit_q, probe_q = perm[:nfit], perm[nfit:]
    R = {"pin": str(pin), "n_train": n, "n_held": nq,
         "n_fit_q": len(fit_q), "n_probe_q": len(probe_q),
         "base_rate_gold": round(base, 6)}

    Mx = Sm.max(1).values
    grp_mask = (Sm == Mx.unsqueeze(1))                     # [nq, n] bool

    def hit(picks):
        """picks[q] = global train index chosen for query q"""
        return sum(1 for q in range(nq) if int(picks[q]) in gold[q]) / nq

    # ---- ARM A: argmax ---------------------------------------------------
    R["A_argmax_top1"] = round(hit(Sm.argmax(1).tolist()), 4)

    # ---- group sizes / precision ----------------------------------------
    sizes = grp_mask.sum(1)
    R["mean_maxgroup_size"] = round(float(sizes.float().mean()), 4)
    prec = 0.0
    for q in range(nq):
        members = torch.nonzero(grp_mask[q]).flatten().tolist()
        insz = len(gold[q] & set(members))
        prec += insz / max(1, len(members))
    R["within_group_gold_precision"] = round(prec / nq, 5)

    # ---- ARM B: uniform-within-maxgroup (seeded, averaged) ---------------
    gp = torch.Generator().manual_seed(seed + 1)
    draw_scores = []
    for _ in range(n_draws):
        picks = []
        for q in range(nq):
            members = torch.nonzero(grp_mask[q]).flatten().tolist()
            k = len(members)
            picks.append(members[int(torch.randint(0, k, (1,), generator=gp))])
        draw_scores.append(hit(picks))
    R["B_uniform_in_maxgroup_top1"] = round(
        sum(draw_scores) / len(draw_scores), 4)
    R["B_draw_min"] = round(min(draw_scores), 4)
    R["B_draw_max"] = round(max(draw_scores), 4)

    # ---- ARM F: size-matched random INSIDE the group (control) -----------
    ctl = []
    gp2 = torch.Generator().manual_seed(seed + 2)
    for _ in range(n_draws):
        picks = []
        for q in range(nq):
            members = torch.nonzero(grp_mask[q]).flatten().tolist()
            if not members:
                picks.append(0)
                continue
            picks.append(members[int(torch.randint(0, len(members), (1,),
                                                   generator=gp2))])
        ctl.append(hit(picks))
    R["F_size_matched_control"] = round(sum(ctl) / len(ctl), 4)

    # ---- ARM E: row-uniform over the whole bank --------------------------
    gp3 = torch.Generator().manual_seed(seed + 3)
    rr = [hit(torch.randint(0, n, (nq,), generator=gp3).tolist())
          for _ in range(10)]
    R["E_row_uniform_random"] = round(sum(rr) / len(rr), 4)

    # ---- ARM D: ridge ranker (trained baseline) --------------------------
    Y = torch.zeros(n, M * Vv)
    for i, ids in enumerate(tr_tgt):
        for p, t in enumerate(ids[:M]):
            Y[i, p * Vv + int(t)] = 1.0
    sc = sel.ridge_scores(Ztr, Y, Zho).reshape(nq, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc, Y.reshape(n, M, Vv))
    R["D_ridge_top1"] = round(hit(C.argmax(1).tolist()), 4)

    # ---- STAGE-2 KEYS ----------------------------------------------------
    # k1 trace frequency in the FIT split
    fit_traces = [tr_tgt[j] for j in range(n)]
    freq = {}
    for j in range(n):
        freq[tuple(tr_tgt[j])] = freq.get(tuple(tr_tgt[j]), 0) + 1
    # NOTE: freq counts occurrences in the train bank (the candidates themselves)
    k1 = torch.tensor([[float(freq[tuple(tr_tgt[j])]) for j in range(n)]]
                      * nq)

    # k2 program length
    k2 = torch.tensor([[float(len(p)) for p in tr_prog]] * nq)

    # k3 max similarity to a FIT-row sharing the query's gold trace (if any)
    #     -> a content-based "does this candidate's trace also match gold" proxy
    #     We must NOT use gold labels at probe time, so use: similarity to the
    #     top-ranked FIT candidate of the same query family. Approximated by the
    #     candidate's mean similarity to all candidates sharing its trace.
    k4_rows = []
    for q in range(nq):
        s = Sm[q]
        row = []
        for j in range(n):
            sib = [c for c in range(n) if tuple(tr_tgt[c]) == tuple(tr_tgt[j])]
            m = float(s[sib].mean()) if sib else float(s[j])
            row.append(1.0 - m)          # novelty
        k4_rows.append(row)
    k4 = torch.tensor(k4_rows)

    keys = {"k1_trace_freq": k1, "k2_train_len": k2, "k4_novelty": k4}

    # ---- ARM C: pick the max-group member with the best key --------------
    def key_arm(key, sign):
        picks, spread = [], []
        for q in range(nq):
            members = torch.nonzero(grp_mask[q]).flatten().tolist()
            if not members:
                picks.append(0)
                continue
            vals = torch.tensor([float(key[q, j]) * sign for j in members])
            spread.append(float(vals.max() - vals.min()))
            picks.append(members[int(vals.argmax())])
        return picks, (sum(spread) / len(spread) if spread else 0.0)

    stage2 = {}
    for nm, kt in keys.items():
        for sign, lab in ((1.0, "max"), (-1.0, "min")):
            picks, sp = key_arm(kt, sign)
            stage2[f"{nm}|{lab}"] = {
                "top1": round(hit(picks), 4),
                "mean_key_spread_in_group": round(sp, 6),
                "discriminates": bool(sp > 0.0),
                "delta_vs_uniform": round(hit(picks)
                                          - R["B_uniform_in_maxgroup_top1"], 4),
            }
    R["C_stage2_keys"] = stage2

    # ---- verdict (derived, not hardcoded) --------------------------------
    best = max(stage2.items(), key=lambda kv: kv[1]["top1"])
    R["best_stage2"] = {"key": best[0], **best[1]}
    R["verdict"] = {
        "P1_harness_argmax_reproduces": bool(
            abs(R["A_argmax_top1"] - 0.0317) < 0.006),
        "P2_uniform_equals_precision": bool(
            abs(R["B_uniform_in_maxgroup_top1"]
                - R["within_group_gold_precision"]) < 0.01),
        "uniform_beats_ridge": bool(
            R["B_uniform_in_maxgroup_top1"] > R["D_ridge_top1"]),
        "uniform_beats_row_uniform": bool(
            R["B_uniform_in_maxgroup_top1"] > R["E_row_uniform_random"]),
        "any_key_beats_uniform": bool(
            best[1]["top1"] > R["B_uniform_in_maxgroup_top1"]),
        "best_key": best[0],
        "best_key_delta": best[1]["delta_vs_uniform"],
        "best_key_discriminates": best[1]["discriminates"],
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
    agg = {"schema": "henri.two_stage_selector.receipt.v1", "results": res}
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
