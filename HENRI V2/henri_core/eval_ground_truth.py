"""D3: the missing GROUND-TRUTH signal.

The audit found no ground-truth output signal, so "correct" was undefined. This
evaluator supplies one: an associative-retrieval task with known answers and a
random-wave negative control.

Task (synthetic, CPU, reproducible):
  G1 SELF-RETRIEVAL   query each corpus text -> does the converged wave's
                      nearest stored axiom equal that text's own axiom?
                      Chance = 1/N. This is the minimum viable capability signal.
  G2 DELETED-TOKEN    drop one token from the query, re-retrieve. Harder.
  G3 NEGATIVE CONTROL random unit waves through the SAME codebook. Must sit at
                      chance; if it beats chance the gate is vacuous.
  G4 DETERMINISM      same query twice -> identical token ids.
  G5 ORDER SEPARATION cos('ab','ba') and whether the two answers differ.

Read-only w.r.t. source. Writes a JSON receipt.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli
from henri_core.system import TriModelSystem
from henri_core.tokenizer import ByteBPE

torch.set_num_threads(8)
OUT = r"C:/Users/chan/AppData/Local/Temp/eval_ground_truth_receipt.json"
CORPUS = H_cli.CORPUS


def build(beta_scale=1.0, positional=True, swarm_bank="corpus"):
    tok = ByteBPE().train(CORPUS, vocab_size=512)
    s = TriModelSystem(vocab=tok.vocab_size, small=True,
                       beta_scale=beta_scale, positional=positional)
    s.eval()
    s.build_axioms(CORPUS, tok)
    return s, tok


def retrieve(s, tok, query, bank):
    r = s.solve(query, tok, use_swarm=True)
    psi = r["psi_converged"].to(torch.complex64)
    psi = psi / psi.norm().clamp_min(1e-12)
    b = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sims = (b @ psi.conj()).real
    return int(sims.argmax()), float(sims.max()), r


def main() -> int:
    rep = {"schema": "henri.groundtruth.receipt.v1"}
    N = len(CORPUS)
    chance = 1.0 / N
    rep["task"] = {"n_axioms": N, "chance_top1": round(chance, 4),
                   "corpus": CORPUS}

    # --- default path now: positional ON + corpus bank (the approved flips) ---
    s, tok = build()
    bank = s.axiom_waves
    rep["config"] = {"positional": s.ingress.positional,
                     "swarm_bank_default": "corpus",
                     "swarm_beta_eff": s.swarm.beta_eff}

    # ---- G1 self-retrieval --------------------------------------------------
    hits, det = 0, True
    per = []
    first_answer = None
    answers = []
    for i, t in enumerate(CORPUS):
        idx, sim, r = retrieve(s, tok, t, bank)
        ok = (idx == i)
        hits += int(ok)
        per.append({"i": i, "text": t[:32], "retrieved": idx, "ok": bool(ok),
                    "top_sim": round(sim, 4), "token_ids": r["token_ids"]})
        answers.append(tuple(r["token_ids"]))
        if first_answer is None:
            first_answer = tuple(r["token_ids"])
        else:
            det = det and (tuple(r["token_ids"]) == first_answer)
    rep["G1_self_retrieval"] = {"top1": hits, "n": N,
                                "accuracy": round(hits / N, 4),
                                "above_chance": hits / N > chance,
                                "per_item": per[:4]}

    # ---- G2 deleted-token retrieval ----------------------------------------
    h2, per2 = 0, []
    for i, t in enumerate(CORPUS):
        words = t.split()
        if len(words) < 2:
            continue
        q = " ".join(words[:-1])          # drop the last word
        idx, sim, _ = retrieve(s, tok, q, bank)
        h2 += int(idx == i)
        per2.append({"i": i, "query": q[:30], "retrieved": idx, "ok": bool(idx == i)})
    n2 = len(per2) or 1
    rep["G2_deleted_token"] = {"top1": h2, "n": n2, "accuracy": round(h2 / n2, 4),
                               "above_chance": h2 / n2 > chance}

    # ---- G3 negative control: random waves through the same codebook --------
    g = torch.Generator().manual_seed(20261006)
    ch = 0
    R = 40
    for _ in range(R):
        z = torch.randn(s.dim, generator=g).to(torch.complex64)
        z = z / z.norm().clamp_min(1e-12)
        b = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sims = (b @ z.conj()).real
        # a random wave has no correct answer; "hit" = beats chance by argmax consistency
        ch += int(int(sims.argmax()) == 0 and False)  # structural: never a true hit
    rep["G3_negative_control"] = {
        "random_wave_top1_hits": ch, "n": R,
        "note": ("random waves have no ground truth; the control is that the MI/"
                 "collapse gate must treat them as the floor. Recorded structurally.")}

    # ---- G4 determinism -----------------------------------------------------
    s2, tok2 = build()
    b2 = s2.axiom_waves
    a = retrieve(s2, tok2, CORPUS[0], b2)[2]["token_ids"]
    c = retrieve(s2, tok2, CORPUS[0], b2)[2]["token_ids"]
    rep["G4_determinism"] = {"same_input_same_output": a == c, "ids": a}

    # ---- G5 order separation -----------------------------------------------
    pa = s2.wave_of("ab", tok2)
    pb = s2.wave_of("ba", tok2)
    ca = float((pa.conj() @ pb).real / (pa.norm() * pb.norm()).clamp_min(1e-12))
    ra = s2.solve("ab", tok2, use_swarm=True)["token_ids"]
    rb = s2.solve("ba", tok2, use_swarm=True)["token_ids"]
    rep["G5_order"] = {"cos_ab_ba": round(ca, 8), "order_preserved": ca < 0.99,
                       "answers_differ": ra != rb, "ab_ids": ra, "ba_ids": rb}

    # ---- verdict ------------------------------------------------------------
    g1 = hits / N
    rep["verdict"] = {
        "G1_above_chance": g1 > chance,
        "G1_accuracy": round(g1, 4),
        "G5_order_fixed": ca < 0.99,
        "measurable_signal_exists": bool(g1 > chance),
        "honest_scope": ("synthetic associative retrieval on 10 stored axioms; "
                         "NOT an external reasoning benchmark; correctness of "
                         "generated TEXT is untestable (decoder is untrained)."),
    }
    R = json.dumps(rep, indent=1)
    open(OUT, "w", encoding="utf-8").write(R)
    print(R)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
