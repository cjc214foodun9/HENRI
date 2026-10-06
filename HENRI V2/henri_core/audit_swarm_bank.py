"""WHY does the repair only move 1 -> 2? The bank the swarm matches against.

Hypothesis (from reading system.__init__):
  self.axiom_bank = unit_norm(randn(8, dim))    <-- RANDOM, seed 20261004
  build_axioms(texts) -> self.veto.load_axioms(waves)   <-- sets the VETO only
  solve() uses `bank = patterns if patterns is not None else self.axiom_bank`

So the SWARM matches against 8 RANDOM vectors that are INDEPENDENT of the
corpus and of the input. The veto sees the corpus waves; the swarm does not.
If true, the swarm's attractor is semantically arbitrary: it picks whichever
random pattern each psi happens to align with, and different inputs can land on
the same arbitrary pattern.

Predictions:
  P1  axiom_bank is UNCHANGED by build_axioms (same object values before/after)
  P2  cos(axiom_bank_i, corpus_wave_j) ~ 1/sqrt(D) for all i,j (unrelated)
  P3  passing the CORPUS waves as the swarm bank changes the distinct count

Read-only.
"""
from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli
from henri_core.system import TriModelSystem
from henri_core.tokenizer import ByteBPE

torch.set_num_threads(8)
QS = ["ab", "ba", "dog", "god", "cat", "act"]


def cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def sep(t):
    print("\n" + "-" * 76)
    print(t)
    print("-" * 76)


def main() -> int:
    tok = ByteBPE().train(H_cli.CORPUS, vocab_size=512)
    sysm = TriModelSystem(vocab=tok.vocab_size, small=True)
    sysm.eval()
    D = sysm.dim

    print("=" * 76)
    print(f"WHICH BANK DOES THE SWARM MATCH AGAINST?  (dim={D})")
    print("=" * 76)

    bank_before = sysm.axiom_bank.clone()
    sep("P1  does build_axioms change the swarm's pattern bank?")
    waves = sysm.build_axioms(H_cli.CORPUS, tok)
    bank_after = sysm.axiom_bank
    same = bool(torch.equal(bank_before, bank_after))
    print(f"  axiom_bank identical before/after build_axioms : {same}")
    print(f"  axiom_bank shape {tuple(bank_after.shape)}   corpus waves shape {tuple(waves.shape)}")
    print(f"  -> the swarm bank is {'UNCHANGED (random, corpus-independent)' if same else 'updated'}")
    print(f"  veto.n_axioms = {sysm.veto.n_axioms}  (this DOES get the corpus)")

    sep("P2  are the random bank vectors related to the corpus waves at all?")
    cross = (bank_after @ waves.conj().T).abs()
    print(f"  bank vs corpus |cos| : mean={float(cross.mean()):.6f} "
          f"max={float(cross.max()):.6f}   1/sqrt(D)={1/math.sqrt(D):.6f}")
    print(f"  -> {'UNRELATED (as random vectors)' if float(cross.max()) < 4/math.sqrt(D) else 'related'}")

    sep("P3  does the swarm land on the SAME random pattern for different inputs?")
    with torch.no_grad():
        psi = {q: sysm.wave_of(q, tok) for q in QS}
        print(f"  {'query':7s} {'best bank idx':>14s} {'align':>9s}   (bank = random)")
        picks = {}
        for q in QS:
            logits = (sysm.swarm.beta_eff * (psi[q] @ bank_after.conj().T).real)
            # the CCCP fixed point normalizes to the softmax-weighted mean; use
            # the dominant pattern as the semantic attractor
            idx = int(logits.argmax())
            picks[q] = idx
            print(f"  {q!r:7s} {idx:14d} {float(logits[idx]):9.4f}")
        print(f"  distinct attractors chosen : {len(set(picks.values()))} of {len(QS)}")
        print(f"  -> {'SAME arbitrary attractor for different inputs' if len(set(picks.values()))<2 else 'inputs pick different attractors'}")

    sep("P4  FIX CHECK: give the swarm the CORPUS waves as its bank")
    with torch.no_grad():
        for label, bank in (("random axiom_bank (current)", bank_after),
                            ("corpus waves (proposed)", waves)):
            ids = {}
            for q in QS:
                out = sysm.swarm(psi[q].unsqueeze(0)[0], patterns=bank)
                win = sysm.consensus(out)
                i, _ = sysm.decoder.snap_text(win["psi"].unsqueeze(0))
                ids[q] = tuple(i.tolist())
            pks = {}
            for q in QS:
                lg = sysm.swarm.beta_eff * (psi[q] @ bank.conj().T).real
                pks[q] = int(lg.argmax())
            print(f"  {label:30s} distinct answers {len(set(ids.values()))}/{len(QS)}   "
                  f"distinct attractors {len(set(pks.values()))}/{len(QS)}")

    print("\nDONE (read-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
