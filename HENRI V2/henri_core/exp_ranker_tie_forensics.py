"""TIE FORENSICS -- is top-1 a RANKING metric or a TIE-ARBITRATION metric?

MEASURED ON ENTRY (my own bytes)
    exp_ranker_structure_probe:  rows_with_tie_at_max = 2268/2268 (EVERY row)
                                 mean tie group at max 10.71; ~2.96 distinct per top-20
    exp_ranker_conditional_info: AUC(C,exact) = 0.7942, residual 0.755, bias-only 0.6805
    exp_veto_slot_prior_control: always-pick-slot4 = 0.1128; fitted arm 0.1097
    => C RANKS WELL (AUC 0.79) but the TOP is a plateau. If every argmax is a tie,
       then the reported top-1 (0.0401) measures the TIE-BREAK RULE, not the ranker.

HYPOTHESIS (falsifiable)
    H: the correct candidate is either strictly top-1, or it sits inside the max-tie
       group and the tie-break discards it. If so, no reranker working above this
       candidate set can exceed the max-tie-group hit rate -- which would place the
       veto's blocker UPSTREAM, in the candidate set.

PRE-REGISTERED MEASUREMENTS (frozen)
    T1 frac queries with strict (untied) argmax
    T2 accuracy on the strict subset only           (the only honest top-1 number)
    T3 frac queries whose correct candidate lies IN the max-tie group   (tie-break ceiling)
    T4 accuracy of argmax (current) vs a SEEDED RANDOM tie-break within the tie group,
       averaged over 5 seeds, on the SAME rows
    T5 max-tie-group size distribution (deciles)

FROZEN VERDICT
    TOP1_IS_TIE_ARBITRATION   if T1 < 0.50 AND T4_random > T4_argmax + 0.005
    TOP1_IS_A_RANKING_METRIC  otherwise
    (T3 reported regardless: it bounds every downstream reranker)

CPU, D=4096. DIAGNOSTIC ONLY. No model-performance claim.
"""
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
    R = {"n_train": n, "n_held": len(Zho)}

    def hit(q, j):
        h = t = 0
        for x, y in zip(tr_tgt[j], ho_tgt[q]):
            t += 1
            h += int(x == y)
        return h, t

    def acc(picks):
        h = t = 0
        for q, j in enumerate(picks):
            hh, tt = hit(q, j)
            h += hh
            t += tt
        return round(h / max(1, t), 4)

    mx = C.max(dim=1, keepdim=True).values
    tie_groups = (C == mx).nonzero()
    sizes = (C == mx).sum(1)
    R["T5_tie_group_size"] = {
        "mean": round(float(sizes.float().mean()), 3),
        "min": int(sizes.min()), "max": int(sizes.max()),
        "frac_size_1": round(float((sizes == 1).float().mean()), 4),
        "deciles": [int(torch.quantile(sizes.float(), q)) for q in
                    (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)],
    }

    strict = [q for q in range(len(Zho)) if int(sizes[q]) == 1]
    R["T1_n_strict_argmax"] = len(strict)
    R["T1_frac_strict"] = round(len(strict) / len(Zho), 4)
    argmax_picks = [int(C[q].argmax()) for q in range(len(Zho))]
    R["T2_accuracy_strict_subset"] = (
        acc([argmax_picks[q] for q in strict]) if strict else None)
    R["T2_n_strict"] = len(strict)

    # T3: is the correct candidate inside the max-tie group?
    in_tie = 0
    for q in range(len(Zho)):
        tied = (C[q] == mx[q, 0]).nonzero().flatten().tolist()
        if any(list(tr_tgt[j]) == list(ho_tgt[q]) for j in tied):
            in_tie += 1
    R["T3_frac_correct_in_max_tie_group"] = round(in_tie / len(Zho), 4)

    # T4: argmax vs seeded random tie-break within the tie group
    R["T4_argmax_acc"] = acc(argmax_picks)
    rnd_accs = []
    for seed in range(5):
        g = torch.Generator().manual_seed(pin + seed)
        picks = []
        for q in range(len(Zho)):
            tied = (C[q] == mx[q, 0]).nonzero().flatten()
            picks.append(int(tied[torch.randint(len(tied), (1,), generator=g)]))
        rnd_accs.append(acc(picks))
    R["T4_random_tiebreak_acc_by_seed"] = rnd_accs
    R["T4_random_tiebreak_acc_mean"] = round(sum(rnd_accs) / len(rnd_accs), 4)

    # T4b: deterministic worst and best tie-break (bounds)
    wb, bb = [], []
    for q in range(len(Zho)):
        tied = (C[q] == mx[q, 0]).nonzero().flatten().tolist()
        best = max(tied, key=lambda j: (list(tr_tgt[j]) == list(ho_tgt[q])))
        wb.append(tied[0])
        bb.append(best)
    R["T4_tiebreak_bounds"] = {"low_index_pick": acc(wb),
                               "oracle_within_tie": acc(bb)}

    ok = (R["T1_frac_strict"] < 0.50
          and R["T4_random_tiebreak_acc_mean"] > R["T4_argmax_acc"] + 0.005)
    R["verdict"] = "TOP1_IS_TIE_ARBITRATION" if ok else "TOP1_IS_A_RANKING_METRIC"
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = {"schema": "henri.ranker.tie.forensics.v1", "pins": [vt.PIN, vt.PIN + 1234]}
    for p in R["pins"]:
        print(f"[tie] pin {p}", file=sys.stderr)
        R[f"pin_{p}"] = run_pin(p)
    vs = [R[f"pin_{p}"]["verdict"] for p in R["pins"]]
    R["overall_verdict"] = vs[0] if len(set(vs)) == 1 else "MIXED"
    out = a.out or (os.environ.get("LOCALAPPDATA", ".") + "/Temp/tie.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
