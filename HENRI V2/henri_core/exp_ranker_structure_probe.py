"""RANKER STRUCTURE PROBE -- why is slot 4 of the top-5 three times better than slot 0?

MEASURED ON ENTRY (my own bytes, exp_ranker_inversion_probe.py, 3 pins)
    corr(C, exact-match)          = +0.2824  (POSITIVE, identical at 3 pins)
    desc argmax exact             = 0.0317 / 0.026 / 0.026
    asc slot-4 exact              = 0.1142 / 0.1089 / 0.1111
    asc global argmin exact       = 0.022  / 0.0132 / 0.0185   (WORSE than random)
    random exact                  = 0.0498 / 0.0503 / 0.0525
    coverage@5 descending         = 0.1142 = the slot-4 rate exactly
    rows_with_tie_at_max          = 2268 / 2268  (EVERY row)
    slot 0,1,2 positive rates identical within each pin
    => the ranker is NOT inverted. It is WEAK, and the top-5 is DEGENERATE.

FOUR CANDIDATE CAUSES, each testable in this one probe
    D1 QUERY-INDEPENDENCE. If var over candidates >> var over queries, the ranking is
       nearly the same for every query and the readout is not conditioning on the query.
    D2 TIE DEGENERACY. Mean tie-group size at the maximum score.
    D3 TRACE-FREQUENCY DOMINANCE. Does C track how COMMON a candidate's trace is in
       train? A prototypicality term would reward generic traces over the exact answer.
    D4 SLOT STRUCTURE. Per-slot candidate trace frequency and bank index.

FROZEN READING
    QUERY_INDEPENDENT_RANKING        if candidate variance share > 0.90
    TIE_DEGENERATE_TOP5              if mean tie-group at max >= 3
    FREQUENCY_DOMINATED              if |corr(C, cand_freq)| > 0.30
    RANKER_STRUCTURALLY_WEAK         otherwise
    Several may hold at once; all are reported.

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


def pearson(a, b):
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    aa, bb = a - a.mean(), b - b.mean()
    den = aa.pow(2).sum().sqrt() * bb.pow(2).sum().sqrt()
    return float((aa * bb).sum() / den) if float(den) > 0 else 0.0


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

    n = min(sel.ROWS, len(Ztr))
    sc_ho = sel.ridge_scores(Ztr[:n], Ymat(tr_tgt[:n]), Zho).reshape(len(Zho), M, Vv)
    Ytr = Ymat(tr_tgt[:n]).reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc_ho, Ytr)
    KK = min(vt.K, n)
    R = {"n_train": n, "n_held": len(Zho), "k": KK}

    # ---- D1 query-independence --------------------------------------------
    mq = C.mean(0)                       # per-candidate mean over queries
    mj = C.mean(1)                       # per-query mean over candidates
    vq, vj = float(mq.var()), float(mj.var())
    R["D1_var_over_candidates"] = round(vq, 6)
    R["D1_var_over_queries"] = round(vj, 6)
    R["D1_candidate_variance_share"] = round(vq / (vq + vj + 1e-30), 4)
    am = C.argmax(1).tolist()
    modal, modal_ct = Counter(am).most_common(1)[0]
    R["D1_frac_queries_with_modal_argmax"] = round(modal_ct / len(am), 4)
    R["D1_distinct_argmax_candidates"] = len(set(am))

    # ---- D2 tie degeneracy ------------------------------------------------
    ties = [int((C[q] == C[q].max()).sum()) for q in range(len(Zho))]
    R["D2_mean_tie_group_at_max"] = round(sum(ties) / len(ties), 3)
    R["D2_frac_rows_with_tie_at_max"] = round(sum(1 for t in ties if t > 1) / len(ties), 4)
    # how many distinct values in the top 20
    R["D2_mean_distinct_in_top20"] = round(
        sum(len(torch.unique(C[q].topk(min(20, n)).values)) for q in range(len(Zho)))
        / len(Zho), 3)

    # ---- D3 trace-frequency dominance -------------------------------------
    freq = Counter(tuple(t) for t in tr_tgt[:n])
    fj = [float(freq[tuple(t)]) for t in tr_tgt[:n]]
    cs = C.reshape(-1).tolist()
    fj_pair = fj * len(Zho)
    R["D3_corr_C_vs_candidate_trace_freq"] = round(pearson(cs, fj_pair), 4)
    R["D3_mean_candidate_trace_freq"] = round(sum(fj) / len(fj), 3)

    # ---- D4 slot structure -------------------------------------------------
    tk = torch.topk(C, KK, dim=1).indices
    slots = {}
    for s in range(KK):
        js = [int(tk[q, s]) for q in range(len(Zho))]
        slots[f"slot{s}"] = {
            "mean_cand_trace_freq": round(sum(fj[j] for j in js) / len(js), 3),
            "mean_cand_index": round(sum(js) / len(js), 1),
            "mean_cand_trace_len": round(
                sum(len(tr_tgt[j]) for j in js) / len(js), 2),
        }
    R["D4_slot_stats"] = slots

    # ---- D5 query trace duplicate count -----------------------------------
    dup = [freq.get(tuple(t), 0) for t in ho_tgt]
    R["D5_mean_query_trace_dup_in_train"] = round(sum(dup) / len(dup), 3)
    R["D5_frac_queries_with_dup"] = round(sum(1 for d in dup if d > 0) / len(dup), 4)

    # ---- D6 candidate-constant component ----------------------------------
    # If C[q,j] ~ b_j (candidate bias only), the top-5 is nearly the same each query.
    top5_sets = [tuple(sorted(int(x) for x in tk[q].tolist())) for q in range(len(Zho))]
    R["D6_n_distinct_top5_sets"] = len(set(top5_sets))
    R["D6_frac_queries_in_modal_top5"] = round(
        Counter(top5_sets).most_common(1)[0][1] / len(top5_sets), 4)

    R["flags"] = {
        "QUERY_INDEPENDENT_RANKING": R["D1_candidate_variance_share"] > 0.90,
        "TIE_DEGENERATE_TOP5": R["D2_mean_tie_group_at_max"] >= 3,
        "FREQUENCY_DOMINATED": abs(R["D3_corr_C_vs_candidate_trace_freq"]) > 0.30,
    }
    R["verdict"] = ([k for k, v in R["flags"].items() if v]
                    or ["RANKER_STRUCTURALLY_WEAK"])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = {"schema": "henri.ranker.structure.probe.v1", "pins": [vt.PIN, vt.PIN + 1234]}
    for p in R["pins"]:
        print(f"[structure] pin {p}", file=sys.stderr)
        R[f"pin_{p}"] = run_pin(p)
    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/structure.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
