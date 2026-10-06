"""A/B: the two measured test-time repairs, combined, with regressions.

  REPAIR A  swarm.beta_scale  (logit domain: <z,p> ~ 1/sqrt(D), not ~1)
  REPAIR B  solve(swarm_bank='corpus')  (swarm matched a RANDOM bank)

Both are DEFAULT-PRESERVING: beta_scale defaults to 1.0 and swarm_bank
defaults to "random", so the shipped default path is byte-identical.

Pre-registered:
  H1  default: few distinct answers (baseline collapse)
  H2  corpus bank: MORE distinct answers than default
  H3  corpus bank + beta_scale sqrt(D): most distinct
  H4  order sensitivity improves (swapped pairs differ)
Kill: if distinctness does not increase, attribution is WRONG -> report FALSIFIED.
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
PAIRS = [("ab", "ba"), ("dog", "god"), ("cat", "act")]


def build(beta_scale=1.0):
    tok = ByteBPE().train(H_cli.CORPUS, vocab_size=512)
    s = TriModelSystem(vocab=tok.vocab_size, small=True, beta_scale=beta_scale)
    s.eval()
    s.build_axioms(H_cli.CORPUS, tok)
    return s, tok


def run(s, tok, bank_kind, label, beta_scale):
    ids, dec, allow, dlt = {}, {}, {}, {}
    for q in QS:
        r = s.solve(q, tok, use_swarm=True, swarm_bank=bank_kind)
        ids[q] = tuple(r["token_ids"])
        dec[q] = r["decoded"]
        allow[q] = r["sagnac"]["allow"]
        dlt[q] = r["sagnac"]["delta"]
    n_ids = len(set(ids.values()))
    n_dec = len(set(dec.values()))
    order = sum(1 for a, b in PAIRS if ids[a] != ids[b])
    dark = sum(1 for q in QS if not allow[q])
    print(f"\n  {label:44s} distinct ids {n_ids}/6  decoded {n_dec}/6  "
          f"order {order}/3  veto-dark {dark}/6")
    return {"n_ids": n_ids, "n_dec": n_dec, "order": order, "dark": dark, "ids": ids}


def main() -> int:
    print("=" * 78)
    print("A/B: D-ZO-COLLAPSE (beta_scale) + D-BANK (corpus swarm bank)")
    print("=" * 78)

    s1, tok = build(1.0)
    D = s1.dim
    base = run(s1, tok, "random", "1. DEFAULT (beta_scale=1, random bank)", 1.0)

    s2, _ = build(1.0)
    bfix = run(s2, tok, "corpus", "2. corpus bank only (beta_scale=1)", 1.0)

    s3, _ = build(math.sqrt(D))
    both = run(s3, tok, "corpus", f"3. corpus bank + beta_scale={math.sqrt(D):.0f}", math.sqrt(D))

    s4, _ = build(math.sqrt(D))
    bsonly = run(s4, tok, "random", f"4. beta_scale={math.sqrt(D):.0f} only (random bank)", math.sqrt(D))

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    h1 = base["n_ids"] <= 2
    h2 = bfix["n_ids"] > base["n_ids"]
    h3 = both["n_ids"] >= bfix["n_ids"]
    h4 = both["order"] > base["order"]
    print(f"  H1 baseline collapses            : {h1}  ({base['n_ids']}/6)")
    print(f"  H2 corpus bank increases         : {h2}  ({base['n_ids']} -> {bfix['n_ids']})")
    print(f"  H3 corpus+scale >= corpus        : {h3}  ({bfix['n_ids']} -> {both['n_ids']})")
    print(f"  H4 order sensitivity improves    : {h4}  ({base['order']}/3 -> {both['order']}/3)")
    print()
    print(f"  RANKED: default={base['n_ids']}  beta_only={bsonly['n_ids']}  "
          f"bank_only={bfix['n_ids']}  bank+beta={both['n_ids']}")
    print(f"  veto stays dark for all queries  : {both['dark']}/6  "
          f"(expected: no dispatcher exists; disclosure, not success)")

    if h2:
        print("\n  MEASURED FIX EFFECTIVE (input dependence restored, not intelligence).")
        print("  No ground-truth output signal exists, so correctness is untested.")
    else:
        print("\n  ATTRIBUTION WRONG: report FALSIFIED, do not ship.")
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
