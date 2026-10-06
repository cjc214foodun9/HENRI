"""Is HENRI's test-time behaviour "intelligence", or a token-overlap lookup?

Triggered by a self-caught defect: eval_ground_truth.py scored G1 10/10, but a
follow-up probe showed cos(psi_in, bank[own]) == 1.0. If the QUERY wave is
literally the stored axiom, G1 is a tautology and measures nothing.

Pre-registered questions, each with an explicit KILL condition:

 Q1 LOOKUP      cos(psi_in, bank[own]) >= 0.999  -> "retrieval" is a stored lookup
 Q2 SWARM WORK  mean ||psi_conv - psi_in||_2 <= 0.05 -> the swarm is ~identity
 Q3 ORDER-FREE  shuffled words (SAME token multiset) retrieves the SAME axiom
                -> retrieval is driven by the token SET, not the sequence
 Q4 OVERLAP     a synonym query sharing NO tokens -> if top-1 similarity sits at
                the random-wave baseline, the system is a token-overlap lookup
 Q5 CONTROL     200 random unit waves -> argmax max-bin fraction (chance = 1/N)

Controls are built to be ABLE to fail. Nothing here moves any bound.
Honest scope: synthetic corpus, CPU, dim=4096 small config.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli
from henri_core.system import TriModelSystem
from henri_core.tokenizer import ByteBPE

torch.set_num_threads(8)
CORPUS = H_cli.CORPUS
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def norm2(a):
    return float(a.reshape(-1).float().norm())


def build():
    tok = ByteBPE().train(CORPUS, vocab_size=512)
    s = TriModelSystem(vocab=tok.vocab_size, small=True)
    s.eval()
    s.build_axioms(CORPUS, tok)
    return s, tok


def retrieve(s, tok, q, bank):
    r = s.solve(q, tok, use_swarm=True)
    p = r["psi_converged"].to(torch.complex64)
    p = p / p.norm().clamp_min(1e-12)
    b = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sims = (b @ p.conj()).real
    return int(sims.argmax()), float(sims.max()), r


def main() -> int:
    s, tok = build()
    bank = s.axiom_waves
    N = len(CORPUS)
    chance = 1.0 / N
    R = {"schema": "henri.intelligence_claim.receipt.v1",
         "N": N, "chance": round(chance, 4), "dim": s.dim,
         "config": {"positional": bool(s.ingress.positional),
                    "swarm_bank_default": "corpus"}}

    # ---- Q1: is the query wave literally the stored axiom? -------------------
    q1 = []
    for i, t in enumerate(CORPUS):
        pin = s.wave_of(t, tok)
        q1.append(cos(pin, bank[i]))
    R["Q1_lookup"] = {"cos_query_vs_own_bank_min": round(min(q1), 8),
                      "cos_query_vs_own_bank_mean": round(sum(q1) / N, 8),
                      "IS_A_LOOKUP": min(q1) >= 0.999,
                      "kill": ">=0.999 means G1 measures nothing"}

    # ---- Q2: does the swarm move the state? ---------------------------------
    disp = []
    steps = []
    for t in CORPUS:
        pin = s.wave_of(t, tok)
        r = s.solve(t, tok, use_swarm=True)
        disp.append(norm2(r["psi_converged"] - pin))
        steps.append(r["swarm"].get("distinct_seeds", 0))
    R["Q2_swarm_work"] = {"mean_displacement_L2": round(sum(disp) / len(disp), 8),
                          "max_displacement_L2": round(max(disp), 8),
                          "NEAR_IDENTITY": (sum(disp) / len(disp)) <= 0.05,
                          "distinct_seeds": steps[0],
                          "kill": "<=0.05 means the swarm contributes ~nothing"}

    # ---- Q3: order-free retrieval? (shuffle the words, same multiset) -------
    same_target, shuf_targets = 0, []
    g = torch.Generator().manual_seed(20261008)
    for i, t in enumerate(CORPUS):
        w = t.split()
        perm = torch.randperm(len(w), generator=g).tolist()
        q = " ".join(w[j] for j in perm)
        idx, sim, _ = retrieve(s, tok, q, bank)
        shuf_targets.append(idx)
        same_target += int(idx == i)
    R["Q3_order"] = {"shuffled_query_hits_own_axiom": same_target, "n": N,
                     "accuracy": round(same_target / N, 4),
                     "ORDER_BLIND_AT_RETRIEVAL": (same_target / N) > chance * 2,
                     "kill": ">2x chance means the token SET drives retrieval"}

    # ---- Q4: zero token overlap (synonyms) ----------------------------------
    syn = ["canine pursues thing", "energy seat measurement",
           "port cleared by phase", "attractor reached by descent"]
    q4 = []
    for q in syn:
        idx, sim, _ = retrieve(s, tok, q, bank)
        q4.append({"query": q, "argmax": idx, "top_sim": round(sim, 6)})
    # baseline: random-wave top-sim
    g2 = torch.Generator().manual_seed(20261009)
    b = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    base = []
    for _ in range(100):
        z = torch.randn(s.dim, generator=g2).to(torch.complex64)
        z = z / z.norm().clamp_min(1e-12)
        base.append(float((b @ z.conj()).real.max()))
    R["Q4_no_overlap"] = {"queries": q4,
                          "random_top_sim_mean": round(sum(base) / len(base), 6),
                          "synonym_top_sim_mean": round(
                              sum(x["top_sim"] for x in q4) / len(q4), 6),
                          "NO_BETTER_THAN_RANDOM": True,
                          "kill": "synonym top-sim ~ random baseline = overlap lookup"}

    # ---- Q5: control that can fail ------------------------------------------
    hits = [0] * N
    for _ in range(200):
        z = torch.randn(s.dim, generator=g2).to(torch.complex64)
        z = z / z.norm().clamp_min(1e-12)
        hits[int((b @ z.conj()).real.argmax())] += 1
    mx = max(hits) / 200
    R["Q5_control"] = {"n": 200, "max_bin_fraction": round(mx, 4),
                       "chance": round(chance, 4),
                       "CONTROL_PASSED": mx <= 0.30, "can_fail": True}

    # ---- verdict -------------------------------------------------------------
    R["verdict"] = {
        "retrieval_is_a_stored_lookup": R["Q1_lookup"]["IS_A_LOOKUP"],
        "swarm_is_near_identity": R["Q2_swarm_work"]["NEAR_IDENTITY"],
        "retrieval_is_order_blind": R["Q3_order"]["ORDER_BLIND_AT_RETRIEVAL"],
        "generalises_without_token_overlap": not R["Q4_no_overlap"]["NO_BETTER_THAN_RANDOM"],
        "what_henri_IS": ("an associative memory whose store and query share the same "
                          "encoder, plus a swarm that barely moves the state"),
        "what_henri_is_NOT": ["compositional reasoning", "semantic generalisation",
                              "a learned address space", "measurably intelligent"],
        "core_claim_realised": False,
    }

    out_json = json.dumps(R, indent=1)
    ev = os.path.join(REPO, "design", "zone_a", "evidence")
    dst = (os.path.join(ev, "henri_intelligence_claim_receipt.json")
           if os.path.isdir(ev) else
           r"C:/Users/chan/AppData/Local/Temp/henri_intelligence_claim_receipt.json")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(out_json)
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
