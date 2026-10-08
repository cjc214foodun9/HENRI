"""LITERAL BLUEPRINT TEST -- the exact formula on Re<psi_q, psi_k>.

MY PRIOR PROBE TESTED A VARIANT. exp_mdl_length_probe.py used the LEARNED ridge
candidate score C as the data term. This probe tests the blueprint AS WRITTEN:

    L(z_k | psi_q) = |z_k| + (1/tau) * (1 - Re<psi_q, psi_k>)

and separately asks whether the RAW inner-product ranker -- the object the
blueprint calls "uncalibrated dot products" -- beats the learned ridge ranker.

MEASURED ALREADY (exp_mdl_length_probe.py, pin 20261010; reproduce = verified)
    ridge ranker      top1 0.0317  cov5 0.1142      <- matches prior forensics
    row-uniform rnd   top1 0.0514
    trace-len MDL     argmin==argmax_frac 1.0 for tau in [0.038,64]  -> NO-OP
    trace bits        CONSTANT (32)  -> that is WHY it is a no-op
    prog-len MDL      frac 0.679, top1 0.0428, cov5 0.3355  (confounded)
    Kraft sum         2.33e-10  (vacuously true)
    dedup bank        top1 0.015, rank4-excluded 0.015, random 0.0038

PRE-REGISTERED READS
  R1  raw inner product vs ridge ranker (is the "uncalibrated" score worse?)
  R2  literal MDL no-op check: argmin == argmax(Re<.>) at frac ~ 1.0
  R3  tau identifiability: count distinct top-1 maps over the tau grid
  R4  controls: row-uniform random top1 AND cov5 on the same bank
  R5  confound: candidate program length vs the train/held-out program split
      (train len < 4, held-out len >= 4) -- a length prior can exploit the split

NO TRAINING. Analysis only. CPU, D=4096, pin 20261010.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default="20261010")
    ap.add_argument("--taus",
                    default="0.038316,0.05,0.1,0.2,0.5,1,2,4,8,16,32,64,128,256")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    pin = int(a.pin) if a.pin.isdigit() else a.pin
    taus = [float(x) for x in a.taus.split(",") if x]

    inputs = sel.make_inputs(sel.N_INPUTS)
    corpus = sel.build_corpus(max_len=sel.MAX_LEN, holdout_len=sel.MAX_LEN,
                              inputs=inputs)
    system, tok = sel.build_system(corpus, pin_seed=pin)
    tr, ho = corpus.train_idx, corpus.heldout_idx
    tr_tgt = [tok.encode(corpus.traces[i]) for i in tr]
    ho_tgt = [tok.encode(corpus.traces[i]) for i in ho]
    tr_prog = [corpus.programs[i] for i in tr]
    ho_prog = [corpus.programs[i] for i in ho]
    Vv, M = tok.vocab_size, sel.M4Config().max_trace

    gen = sel.WaveTextGenerator(system, tok, train_body=True)
    with torch.no_grad():
        Ztr = F.normalize(torch.nan_to_num(sel.flat(gen.wave(
            [corpus.specs[i] for i in tr]))), dim=-1)
        Zho = F.normalize(torch.nan_to_num(sel.flat(gen.wave(
            [corpus.specs[i] for i in ho]))), dim=-1)

    nq, n = len(ho_tgt), len(tr_tgt)
    R: dict = {"pin": str(pin), "n_train_rows": n, "n_held_rows": nq}

    # ---- the RAW wave inner product: this IS Re<psi_q, psi_k> for real waves --
    Sm = (Zho @ Ztr.T)
    if torch.is_complex(Sm):
        Sm = Sm.real
    R["R1_sim_is_complex"] = bool(torch.is_complex(Zho @ Ztr.T))
    R["R1_sim_min"] = round(float(Sm.min()), 6)
    R["R1_sim_max"] = round(float(Sm.max()), 6)
    R["R1_sim_mean"] = round(float(Sm.mean()), 6)
    R["R1_sim_std"] = round(float(Sm.std()), 6)
    R["R1_sim_row_std_mean"] = round(float(Sm.std(1).mean()), 6)

    def top1_of(picks):
        return round(sum(1 for q in range(nq)
                         if tuple(tr_tgt[int(picks[q])]) == tuple(ho_tgt[q]))
                     / nq, 4)

    def covk_of(Mm, k):
        tk = torch.topk(Mm, min(k, Mm.shape[1]), dim=1).indices
        return round(sum(1 for q in range(nq)
                         if any(tuple(tr_tgt[int(j)]) == tuple(ho_tgt[q])
                                for j in tk[q])) / nq, 4)

    # ---- R1: raw inner-product ranker ------------------------------------
    R["R1_raw_ip_top1"] = top1_of(Sm.argmax(1).tolist())
    R["R1_raw_ip_cov5"] = covk_of(Sm, 5)

    # ridge ranker on the same rows (baseline for comparison)
    def Ymat(tg):
        Y = torch.zeros(len(tg), M * Vv)
        for i, ids in enumerate(tg):
            for p, t in enumerate(ids[:M]):
                Y[i, p * Vv + int(t)] = 1.0
        return Y
    Yall = Ymat(tr_tgt)
    sc = sel.ridge_scores(Ztr, Yall, Zho).reshape(nq, M, Vv)
    Yr = Yall.reshape(n, M, Vv)
    C = torch.einsum("qmv,jmv->qj", sc, Yr)
    R["R1_ridge_top1"] = top1_of(C.argmax(1).tolist())
    R["R1_ridge_cov5"] = covk_of(C, 5)

    # ---- R4: controls ----------------------------------------------------
    g = torch.Generator().manual_seed(12345)
    r1, r5 = [], []
    for _ in range(10):
        jj = torch.randint(0, n, (nq,), generator=g)
        r1.append(sum(1 for q in range(nq)
                      if tuple(tr_tgt[int(jj[q])]) == tuple(ho_tgt[q])) / nq)
        jk = torch.randint(0, n, (nq, 5), generator=g)
        r5.append(sum(1 for q in range(nq)
                      if any(tuple(tr_tgt[int(jk[q, c])]) == tuple(ho_tgt[q])
                             for c in range(5))) / nq)
    R["R4_row_uniform_random_top1"] = round(sum(r1) / len(r1), 4)
    R["R4_row_uniform_random_cov5"] = round(sum(r5) / len(r5), 4)

    # ---- R2 / R3: the literal blueprint formula --------------------------
    Lt = torch.tensor([float(len(corpus.traces[i].encode("utf-8")) * 8)
                       for i in tr])
    Lp = torch.tensor([float(len(p.encode("utf-8")) * 8) for p in tr_prog])
    R["R2_trace_bits_constant"] = bool(Lt.std().item() == 0.0)
    R["R2_trace_bits_values"] = sorted({int(x) for x in Lt.tolist()})
    R["R2_prog_bits_values"] = sorted({int(x) for x in Lp.tolist()})

    Delta = 1.0 - Sm                                            # [nq, n]
    R["R2_delta_min"] = round(float(Delta.min()), 6)
    R["R2_delta_max"] = round(float(Delta.max()), 6)
    R["R2_delta_mean"] = round(float(Delta.mean()), 6)

    ip_argmax = Sm.argmax(1)
    sweep, seen = {}, set()
    for tau in taus:
        data_bits = (Delta / tau) * (1.0 / math.log(2.0))
        for nm, Lv in (("trace", Lt), ("prog", Lp)):
            L = Lv.unsqueeze(0) + data_bits
            ag = L.argmin(1)
            key = f"tau={tau:g}|len={nm}"
            sweep[key] = {
                "top1": top1_of(ag.tolist()),
                "cov5": covk_of(-L, 5),
                "argmin_eq_argmax_frac": round(
                    float((ag == ip_argmax).float().mean()), 6),
                "mean_data_bits": round(float(data_bits.mean()), 4),
                "mean_len_bits": round(float(Lv.mean()), 4),
                "data_over_len": round(float(data_bits.mean())
                                       / max(1e-9, float(Lv.mean())), 4),
            }
            seen.add(tuple(ag.tolist()))
    R["R2_R3_mdl_sweep"] = sweep
    R["R3_distinct_top1_maps_over_tau"] = len(seen)

    # ---- R5: the length/split confound ----------------------------------
    R["R5_train_prog_len_max"] = max(len(p) for p in tr_prog)
    R["R5_held_prog_len_min"] = min(len(p) for p in ho_prog)
    ho_set = {tuple(t) for t in ho_tgt}
    # per candidate: does its trace match ANY held-out gold trace?
    match = torch.zeros(n, dtype=torch.float32)
    for j in range(n):
        if tuple(tr_tgt[j]) in ho_set:
            match[j] = 1.0
    R["R5_frac_candidates_matching_gold"] = round(float(match.mean()), 6)
    # correlation of candidate PROGRAM length with gold-match, per program len
    by_len = {}
    for ln in sorted({len(p) for p in tr_prog}):
        idx = [j for j, p in enumerate(tr_prog) if len(p) == ln]
        by_len[str(ln)] = {
            "n": len(idx),
            "frac_matching_gold": round(float(match[idx].mean()), 4),
        }
    R["R5_gold_match_by_prog_len"] = by_len
    if float(Lp.std()) > 0 and float(match.std()) > 0:
        R["R5_corr_proglen_goldmatch"] = round(
            float(torch.corrcoef(torch.stack([Lp, match]))[0, 1]), 4)
    else:
        R["R5_corr_proglen_goldmatch"] = None

    # ---- verdicts as DATA -----------------------------------------------
    noop_all = all(v["argmin_eq_argmax_frac"] > 0.999
                   for k, v in sweep.items() if k.endswith("len=trace"))
    R["verdict"] = {
        "literal_trace_mdl_is_strict_noop": noop_all,
        "tau_identifiable_for_trace_arm": not noop_all,
        "raw_ip_top1_beats_ridge": bool(R["R1_raw_ip_top1"] > R["R1_ridge_top1"]),
        "raw_ip_vs_row_uniform_random": round(
            R["R1_raw_ip_top1"] - R["R4_row_uniform_random_top1"], 4),
        "prog_arm_confounded_by_split": bool(
            R["R5_frac_candidates_matching_gold"] > 0
            and not noop_all),
    }

    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as f:
            json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
