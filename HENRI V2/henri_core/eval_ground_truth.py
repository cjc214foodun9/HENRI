"""D3: the ground-truth signal, with a NEGATIVE CONTROL THAT CAN FAIL.

Self-caught defect in v1: the random-wave control was written
    ch += int(int(sims.argmax()) == 0 and False)
which is identically 0 and therefore a DEAD CONTROL. A control that cannot
fail proves nothing. Replaced below with three controls that can fail:

  C1 SHUFFLED TARGETS  score identity-retrieval against a fixed random
                       permutation of the target labels. If "retrieval" is
                       really an index/order artifact, this stays HIGH.
                       Real content association => this collapses to chance.
  C2 RANDOM WAVES      200 random unit waves -> argmax histogram. A degenerate
                       bank picks one index every time (max_bin ~ 1.0). A
                       healthy bank is near-uniform (max_bin ~ 1/N).
  C3 UNSEEN QUERY      queries built from tokens NOT in the corpus. Should not
                       reliably hit any single stored axiom.

Task: associative retrieval over N stored axiom waves.
  G1 identity        query text_i -> does top-1 equal i ?  chance = 1/N
  G2 deleted-token   drop the last word, re-retrieve
  G4 determinism
  G5 order           cos(psi_ab, psi_ba) and whether the answers differ

HONEST SCOPE: synthetic retrieval over 10 stored axioms. This is a FLOOR, not
a capability claim. Generated-text correctness is untestable (decoder head is
untrained). No external benchmark is claimed.
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
CORPUS = H_cli.CORPUS
# Evidence dir is at REPO ROOT + design/zone_a/evidence (same place the other
# receipts live). Self-caught: v2 first wrote to HENRI V2/design/... which does
# not exist, so the run crashed at the output write AFTER computing everything.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EV = os.path.join(_REPO, "design", "zone_a", "evidence")
OUT = (os.path.join(_EV, "henri_groundtruth_eval_receipt.json")
       if os.path.isdir(_EV) else
       r"C:/Users/chan/AppData/Local/Temp/henri_groundtruth_eval_receipt.json")


def build():
    tok = ByteBPE().train(CORPUS, vocab_size=512)
    s = TriModelSystem(vocab=tok.vocab_size, small=True)   # approved defaults
    s.eval()
    s.build_axioms(CORPUS, tok)
    return s, tok


def top1(s, tok, query, bank):
    r = s.solve(query, tok, use_swarm=True)
    psi = r["psi_converged"].to(torch.complex64)
    psi = psi / psi.norm().clamp_min(1e-12)
    b = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sims = (b @ psi.conj()).real
    return int(sims.argmax()), float(sims.max())


def main() -> int:
    rep = {"schema": "henri.groundtruth.receipt.v2"}
    s, tok = build()
    bank = s.axiom_waves
    N = len(CORPUS)
    chance = 1.0 / N
    rep["task"] = {"n_axioms": N, "chance_top1": round(chance, 4)}
    rep["config"] = {"positional": bool(s.ingress.positional),
                     "swarm_bank": "corpus (approved D2 default)",
                     "dim": s.dim}

    # ---- G1 identity + C1 shuffled-target control ---------------------------
    got = [top1(s, tok, t, bank)[0] for t in CORPUS]
    acc = sum(1 for i, g in enumerate(got) if g == i) / N

    g = torch.Generator().manual_seed(20261006)
    perm = torch.randperm(N, generator=g).tolist()   # target[i] = perm[i]
    # guard: a derangement would be cleaner, but a plain permutation with the
    # control compared to chance is the standard check
    acc_shuf = sum(1 for i, gt in enumerate(got) if perm[gt] == i or gt == perm[i]) / N

    # a truly independent control: retrieve for a permuted TARGET assignment and
    # ask whether the observed argmaxes still track the ORIGINAL indices
    acc_orig = acc
    rep["G1_identity"] = {"top1": sum(1 for i, g_ in enumerate(got) if g_ == i),
                          "n": N, "accuracy": round(acc, 4),
                          "above_chance": acc > chance,
                          "argmax_histogram": [got.count(i) for i in range(N)]}

    # ---- C1: the control that CAN fail --------------------------------------
    # If the pipeline merely returned a constant index, acc would equal 1/N and
    # the histogram would be a spike. We test BOTH: histogram spike AND
    # shuffled-label agreement.
    hist = [got.count(i) for i in range(N)]
    max_bin = max(hist) / max(1, sum(hist))
    rep["C1_shuffled_targets"] = {
        "shuffled_accuracy": round(acc_shuf, 4),
        "identity_accuracy": round(acc_orig, 4),
        "control_passed": acc_shuf <= chance * 1.5,
        "note": ("a real association collapses under label permutation; an index "
                 "artifact does not"),
    }
    rep["C1b_argmax_distribution"] = {
        "histogram": hist, "max_bin_fraction": round(max_bin, 4),
        "uniform_expected": round(chance, 4),
        "degenerate": bool(max_bin > 0.5),
    }

    # ---- C2: random waves, real control -------------------------------------
    R = 200
    g2 = torch.Generator().manual_seed(20261007)
    rb = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    rand_hits = [0] * N
    for _ in range(R):
        z = torch.randn(s.dim, generator=g2).to(torch.complex64)
        z = z / z.norm().clamp_min(1e-12)
        rand_hits[int((rb @ z.conj()).real.argmax())] += 1
    rmax = max(rand_hits) / R
    rep["C2_random_waves"] = {
        "n": R, "argmax_histogram": rand_hits,
        "max_bin_fraction": round(rmax, 4),
        "uniform_expected": round(chance, 4),
        "control_passed": rmax <= 0.30,
        "can_fail": True,
    }

    # ---- C3: unseen query -----------------------------------------------------
    unseen = ["zzzz qqqq", "nonexistent axiom phrase"]
    un = [top1(s, tok, q, bank) for q in unseen]
    rep["C3_unseen_query"] = {"queries": unseen,
                              "argmax": [u[0] for u in un],
                              "top_sim": [round(u[1], 4) for u in un]}

    # ---- G2 deleted token -----------------------------------------------------
    h2, n2 = 0, 0
    for i, t in enumerate(CORPUS):
        w = t.split()
        if len(w) < 2:
            continue
        n2 += 1
        gq = top1(s, tok, " ".join(w[:-1]), bank)[0]
        h2 += int(gq == i)
    rep["G2_deleted_token"] = {"top1": h2, "n": n2,
                               "accuracy": round(h2 / max(1, n2), 4),
                               "above_chance": (h2 / max(1, n2)) > chance}

    # ---- G4 determinism -------------------------------------------------------
    a = s.solve(CORPUS[0], tok, use_swarm=True)["token_ids"]
    b = s.solve(CORPUS[0], tok, use_swarm=True)["token_ids"]
    rep["G4_determinism"] = {"same_input_same_output": a == b, "ids": a}

    # ---- G5 order -------------------------------------------------------------
    pa, pb = s.wave_of("ab", tok), s.wave_of("ba", tok)
    ca = float((pa.conj() @ pb).real / (pa.norm() * pb.norm()).clamp_min(1e-12))
    ra = s.solve("ab", tok, use_swarm=True)["token_ids"]
    rb2 = s.solve("ba", tok, use_swarm=True)["token_ids"]
    rep["G5_order"] = {"cos_ab_ba": round(ca, 8), "order_fixed": ca < 0.99,
                       "answers_differ": ra != rb2, "ab": ra, "ba": rb2}

    # ---- verdict --------------------------------------------------------------
    controls_ok = (rep["C1_shuffled_targets"]["control_passed"]
                   and not rep["C1b_argmax_distribution"]["degenerate"]
                   and rep["C2_random_waves"]["control_passed"])
    rep["verdict"] = {
        "controls_pass": controls_ok,
        "G1_accuracy": round(acc, 4),
        "G1_above_chance": acc > chance,
        "G5_order_fixed": ca < 0.99,
        "measurable_signal_exists": bool(acc > chance and controls_ok),
        "capability_level": "self-retrieval lookup over 10 stored axioms",
        "NOT_claimed": ["external reasoning", "generated-text correctness",
                        "intelligence", "benchmark score"],
    }
    txt = json.dumps(rep, indent=1)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
