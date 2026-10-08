"""BANK STRUCTURE / DEDUP PROBE -- is the top-1 metric measured over DUPLICATES?

MEASURED ON ENTRY (my own bytes)
    exp_ranker_tie_forensics: frac_size_1 = 0.0 at BOTH pins. Every argmax is tied
      (min tie 3, mean ~10.7, max 75-150). argmax = seeded-random-tiebreak =
      oracle-within-tie = 0.0401. The correct candidate is essentially never inside
      the max-tie group (T3 0.0317).
    exp_ranker_structure_probe: mean candidate trace frequency 46.518 over n=1092 rows
      -> ~1092/46.5 ~= 23 distinct traces. D1_distinct_argmax_candidates = 29.
    exp_ranker_conditional_info: AUC(C, exact) = 0.7942, residual 0.755, bias 0.6805.

HYPOTHESIS (falsifiable)
    The training bank is dominated by DUPLICATE traces. Identical traces give identical
    C values, so the top of the list is a plateau of copies. Argmax then picks an
    arbitrary member of a plateau, which is why top-1 equals random.

PRE-REGISTERED MEASUREMENTS (frozen)
    B1 n rows, n distinct traces (train), n distinct traces (held-out)
    B2 explicit RANDOM baselines at the same k, computed against the DISTINCT-trace
       universe (not the row universe)
    B3 ranker measured on the FULL bank vs a DEDUPLICATED bank (one row per distinct
       trace). Same ridge protocol, same lambda, same rows cap otherwise.
    B4 distinct-argmax count and its stability

FROZEN VERDICT
    PLATEAU_IS_THE_BINDING_CONSTRAINT  if top-1 on the full bank is within 0.01 of the
        random-over-distinct-traces baseline AND the dedup bank changes top-1 or
        coverage@5 by more than 0.01
    RANKER_WEAK_BUT_RANKING_VALID      otherwise
CPU, D=4096. DIAGNOSTIC ONLY. No model-performance claim.
"""
import argparse
import io
import json
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F

REPO = r"C:/Users/chan/henri-worktrees/phase1-transduction"
V = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, V)

import henri_core.exp_attributable_veto as vt                              # noqa: E402

sel = vt.sel


def run_pin(pin):
    inputs = sel.make_inputs(sel.N_INPUTS)
    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN, inputs=inputs)
    system, tok = sel.build_system(corpus, pin_seed=pin)
    tr_tgt = [tok.encode(corpus.traces[i]) for i in corpus.train_idx]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in corpus.heldout_idx]
    tr_spec = [corpus.specs[i] for i in corpus.train_idx]
    ho_spec = [corpus.specs[i] for i in corpus.heldout_idx]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace

    gen = sel.WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ztr = F.normalize(torch.nan_to_num(sel.flat(gen.wave(tr_spec))), dim=-1)
        Zho = F.normalize(torch.nan_to_num(sel.flat(gen.wave(ho_spec))), dim=-1)

    def Ymat(tg):
        Y = torch.zeros(len(tg), M * Vv)
        for i, ids in enumerate(tg):
            for p, t in enumerate(ids[:M]):
                Y[i, p * Vv + int(t)] = 1.0
        return Y

    R = {"n_train_rows": len(tr_tgt), "n_held_rows": len(ho_tgt)}
    tr_distinct = list({tuple(t) for t in tr_tgt})
    ho_distinct = list({tuple(t) for t in ho_tgt})
    R["B1_n_distinct_train"] = len(tr_distinct)
    R["B1_n_distinct_held"] = len(ho_distinct)
    R["B1_held_distinct_present_in_train"] = sum(
        1 for t in ho_distinct if t in set(tr_distinct))
    R["B1_row_redundancy_train"] = round(len(tr_tgt) / max(1, len(tr_distinct)), 2)

    def acc(picks, labels):
        h = t = 0
        for j, tg in zip(picks, labels):
            for x, y in zip(tr_tgt[j], tg):
                t += 1
                h += int(x == y)
        return round(h / max(1, t), 4)

    def cov5(Cm, labels):
        tk = torch.topk(Cm, min(5, Cm.shape[1]), dim=1).indices
        return round(sum(1 for q in range(len(labels))
                         if any(list(tr_tgt[int(j)]) == list(labels[q])
                                for j in tk[q])) / len(labels), 4)

    def eval_bank(rows_ix, tag):
        n = min(sel.ROWS, len(rows_ix))
        ix = rows_ix[:n]
        Yb = Ymat([tr_tgt[i] for i in ix])
        sc = sel.ridge_scores(Ztr[ix], Yb, Zho).reshape(len(Zho), M, Vv)
        Yr = Yb.reshape(n, M, Vv)
        C = torch.einsum("qmv,jmv->qj", sc, Yr)
        pmap = [ix[int(C[q].argmax())] for q in range(len(Zho))]
        R[f"B3_{tag}"] = {
            "n_rows": n,
            "top1_tok": acc(pmap, ho_tgt),
            "coverage5": cov5(C, ho_tgt),
            "top1_exact": round(sum(1 for q, j in enumerate(pmap)
                                    if list(tr_tgt[j]) == list(ho_tgt[q]))
                                / len(Zho), 4),
            "frac_untied_argmax": round(float(
                (C.max(1).values.unsqueeze(1) == C).sum(1).eq(1).float().mean()), 4),
        }
        return C

    rows_full = list(range(len(tr_tgt)))
    C_full = eval_bank(rows_full, "full_bank")

    # one representative row per distinct trace
    first = {}
    for i, t in enumerate(tr_tgt):
        first.setdefault(tuple(t), i)
    rows_dedup = sorted(first.values())
    R["B3_dedup_n_rows"] = len(rows_dedup)
    eval_bank(rows_dedup, "dedup_bank")

    # ---- B2 explicit random baselines over the DISTINCT-trace universe -------
    g = torch.Generator().manual_seed(pin)
    for tag, pool in (("full_bank", rows_full), ("dedup_bank", rows_dedup)):
        K = len(pool)
        hit1 = c5 = 0
        for _ in range(300):
            q = int(torch.randint(len(Zho), (1,), generator=g))
            picks = [pool[int(torch.randint(K, (1,), generator=g))] for _ in range(5)]
            for x, y in zip(tr_tgt[picks[0]], ho_tgt[q]):
                hit1 += 0
            hit1 += int(list(tr_tgt[picks[0]]) == list(ho_tgt[q]))
            c5 += int(any(list(tr_tgt[j]) == list(ho_tgt[q]) for j in picks))
        R[f"B2_random_{tag}"] = {"top1_exact": round(hit1 / 300, 4),
                                 "coverage5": round(c5 / 300, 4),
                                 "pool_size": K}

    full = R["B3_full_bank"]
    R["delta_dedup_top1_exact"] = round(R["B3_dedup_bank"]["top1_exact"]
                                        - full["top1_exact"], 4)
    R["delta_dedup_coverage5"] = round(R["B3_dedup_bank"]["coverage5"]
                                       - full["coverage5"], 4)
    near_random = (abs(full["top1_exact"]
                       - R["B2_random_full_bank"]["top1_exact"]) < 0.01)
    plate = (near_random
             and (abs(R["delta_dedup_top1_exact"]) > 0.01
                  or abs(R["delta_dedup_coverage5"]) > 0.01))
    R["verdict"] = ("PLATEAU_IS_THE_BINDING_CONSTRAINT" if plate
                    else "RANKER_WEAK_BUT_RANKING_VALID")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = {"schema": "henri.bank.structure.dedup.v1", "pins": [vt.PIN, vt.PIN + 1234]}
    for p in R["pins"]:
        print(f"[bank] pin {p}", file=sys.stderr)
        R[f"pin_{p}"] = run_pin(p)
    vs = [R[f"pin_{p}"]["verdict"] for p in R["pins"]]
    R["overall_verdict"] = vs[0] if len(set(vs)) == 1 else "MIXED"
    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/bank.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
