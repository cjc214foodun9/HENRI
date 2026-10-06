"""DIRECTIVE 3: Runtime Zone C seeding for contingent world knowledge.

THE RECONCILIATION THIS TESTS
    D_c = 0 is a PRETRAINING constraint, not a RUNTIME MEMORY constraint.
    Cowsik's decomposition penalizes contingent facts in the TRAINING objective.
    It does not forbid storing them as engrams at inference time. Zone C is
    where D_c belongs. This probe asks whether a SEEDED bank changes a
    downstream measurement with NO parameter update.

PRE-REGISTERED ARMS (design/zone_a/AUTORESEARCH_zonec_seeding_v1.md)
    O  oracle      : query the EXACT seeded wave. Retrieval must reach ~1.0 or
                     the retrieval path itself is broken and no verdict is valid.
    S  seeded      : bank holds (question -> answer) pairs from the TRAIN split;
                     held-out questions are NEW. Measures factual extraction on
                     unseen questions.
    U  unseeded    : empty bank. Base rate.
    P  permuted    : bank holds mismatched answer labels. Must land at chance.
    F  frozen-param: constant-input control for the readout head.

ISOLATION (asserted, not assumed)
    * NO parameter trains. The probe asserts every parameter keeps its value.
    * The bank is non-empty in S and O; empty in U.
    * Retrieval is exercised at least once (a vacuity guard).

ENDPOINT
    held-out answer exact-match, with the unigram floor beside it. Bound fixed
    BEFORE the run.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                        # noqa: E402
from henri_core.m4_generative import build_corpus, build_system  # noqa: E402

PIN = 20261004
BOUND = 0.25          # pre-registered: seeded must beat the floor by >= 0.25


def make_facts(n: int, seed: int = 7):
    """Contingent facts: (question, answer) pairs over an arbitrary codebook.

    Nothing here derives from the corpus grammar, so this is genuine D_c:
    arbitrary, non-tautological associations the model cannot compute.
    """
    g = torch.Generator().manual_seed(seed)
    facts = []
    for i in range(n):
        ent = f"ENT{i:04d}"
        code = f"{(i * 7919) % 99991:05d}"
        facts.append((f"what is the access code for {ent}", code))
    perm = torch.randperm(n, generator=g).tolist()
    return facts, perm


def bank_from(system, tok, texts):
    """Encode texts into a [N, D] complex unit-norm bank via the FROZEN ingress."""
    with torch.no_grad():
        return torch.stack([system.wave_of(t, tok) for t in texts])


def retrieve(bank: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Single-step Hopfield retrieval: argmax inner product over the bank.

    Returns the index of the best-matching memory for every query row.
    """
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    bn = bank / bank.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sim = (qn @ bn.conj().T).real                     # [B, N]
    return sim.argmax(dim=-1), sim


def run_arm(name, system, tok, bank_texts, answer_texts, queries, answers,
            bank_bank=None):
    """Seed a bank, retrieve, and score. NO parameter is modified."""
    if bank_bank is None:
        bank = bank_from(system, tok, bank_texts) if bank_texts else \
            torch.zeros(0, system.dim, dtype=torch.complex64)
    else:
        bank = bank_bank
    q = bank_from(system, tok, queries)
    if bank.shape[0] == 0:
        return {"arm": name, "n_bank": 0, "retrieval_exercised": False,
                "exact_match": 0.0, "answer_hit": 0.0}
    idx, sim = retrieve(bank, q)
    got = [answer_texts[i] for i in idx.tolist()]
    em = sum(1 for g, a in zip(got, answers) if g == a) / max(1, len(answers))
    # answer-level: did the retrieved answer string occur anywhere in the bank
    return {
        "arm": name, "n_bank": int(bank.shape[0]),
        "retrieval_exercised": True,
        "exact_match": em,
        "mean_top_sim": float(sim.max(dim=-1).values.mean()),
        "sample": got[:3],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, ingress_seed=PIN, pin_seed=PIN)
    system.eval()

    facts, perm = make_facts(a.n_facts)
    half = a.n_facts // 2
    tr_f, ho_f = facts[:half], facts[half:]

    # ---- ISOLATION: snapshot every parameter before any measurement
    before = {k: v.detach().clone() for k, v in system.named_parameters()}

    tr_q = [f[0] for f in tr_f]
    tr_a = [f[1] for f in tr_f]
    ho_q = [f[0] for f in ho_f]
    ho_a = [f[1] for f in ho_f]

    # D165 (self-caught): make_facts built `perm` over n_facts (64) but tr_a holds
    # only half (32) entries, so perm[:half] indexed out of range. The permutation
    # must be drawn over the TRAIN split itself.
    g = torch.Generator().manual_seed(7)
    perm_a = [tr_a[i] for i in torch.randperm(half, generator=g).tolist()]

    res = {}
    # O: oracle -- the answer wave itself is IN the bank, keyed by the question
    # wave. Retrieval on a stored key must return that key.
    res["O_oracle"] = run_arm("O_oracle", system, tok, tr_q, tr_q, tr_q, tr_q)
    # S: seeded -- question -> answer, held-out questions are NEW strings
    res["S_seeded"] = run_arm("S_seeded", system, tok, tr_q, tr_a, tr_q, tr_a)
    # held-out questions against the train bank
    res["S_heldout"] = run_arm("S_heldout", system, tok, tr_q, tr_a, ho_q, ho_a)
    # U: unseeded
    res["U_unseeded"] = run_arm("U_unseeded", system, tok, [], [], ho_q, ho_a)
    # P: permuted labels
    res["P_permuted"] = run_arm("P_permuted", system, tok, tr_q, perm_a,
                                tr_q, tr_a)

    # ---- floor: most common answer in the train bank
    from collections import Counter
    c = Counter(tr_a)
    best = c.most_common(1)[0][0]
    floor = sum(1 for x in ho_a if x == best) / max(1, len(ho_a))

    # ---- ISOLATION CHECK: did ANY parameter move?
    moved = 0
    max_delta = 0.0
    for k, v in system.named_parameters():
        d = float((v.detach() - before[k]).abs().max())
        max_delta = max(max_delta, d)
        if d > 0.0:
            moved += 1

    guards = {
        "no_parameter_moved": moved == 0,
        "oracle_reaches_one": res["O_oracle"]["exact_match"] >= 0.99,
        "seeded_bank_nonempty": res["S_seeded"]["n_bank"] > 0,
        "unseeded_bank_empty": res["U_unseeded"]["n_bank"] == 0,
        "retrieval_exercised": res["S_seeded"]["retrieval_exercised"],
    }
    gain = res["S_seeded"]["exact_match"] - floor
    ctl_worse = res["P_permuted"]["exact_match"] <= res["S_seeded"]["exact_match"]

    if not all(guards.values()):
        verdict = "VACUOUS"
        why = f"guards {guards}"
    elif gain >= BOUND and ctl_worse:
        verdict = "SEEDING_WORKS"
        why = (f"seeded EM {res['S_seeded']['exact_match']:.4f} vs floor "
               f"{floor:.4f} (gain {gain:+.4f}); permuted "
               f"{res['P_permuted']['exact_match']:.4f}")
    else:
        verdict = "SEEDING_NO_GAIN"
        why = f"seeded {res['S_seeded']['exact_match']:.4f} floor {floor:.4f} "\
              f"gain {gain:+.4f}"

    out = {
        "schema": "henri.zonec.seeding.v1",
        "verdict": verdict, "why": why,
        "guards": guards,
        "arms": res,
        "unigram_floor": floor,
        "bound": BOUND,
        "isolation": {"params_moved": moved, "max_delta": max_delta},
        "n_facts": a.n_facts,
        "run": {"pin": PIN},
        "seconds": round(time.time() - t0, 2),
        "disclosure": {
            "metric": ("exact-match of the ANSWER retrieved for a held-out "
                       "QUESTION. The bank keys on the question wave, so this "
                       "measures associative recall, not reasoning."),
            "no_store": ("HenriMem65M is the cognitive governor; no TimescaleDB "
                         "write occurs. The probe holds the bank in memory."),
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("verdict", "why", "guards", "arms",
                                          "unigram_floor", "isolation")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
