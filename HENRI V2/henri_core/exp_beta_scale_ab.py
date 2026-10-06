"""A/B test of the D-ZO-COLLAPSE repair (beta_scale) at test time.

Pre-registered:
  H1  at beta_scale=1 (current), solve() returns few distinct answers
      (measured baseline: 1 of 6 queries).
  H2  at beta_scale=sqrt(D), solve() returns MORE distinct answers.
  H3  the repair must NOT simply randomize: the Sagnac veto and the
      gate regressions must still behave. Report both.

Kill condition: if the distinct-answer count does not increase, the
root-cause attribution is WRONG and I report that, not a fix.

Read-only w.r.t. source; builds the system twice with different beta_scale.
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


def build(beta_scale):
    tok = ByteBPE().train(H_cli.CORPUS, vocab_size=512)
    sysm = TriModelSystem(vocab=tok.vocab_size, small=True, beta_scale=beta_scale)
    sysm.eval()
    sysm.build_axioms(H_cli.CORPUS, tok)
    return sysm, tok


def run(beta_scale, label):
    sysm, tok = build(beta_scale)
    D = sysm.dim
    ids, dec, ener, allow, delta = {}, {}, {}, {}, {}
    for q in QS:
        r = sysm.solve(q, tok, use_swarm=True)
        ids[q] = tuple(r["token_ids"])
        dec[q] = r["decoded"]
        ener[q] = r["swarm"]["energy"]
        allow[q] = r["sagnac"]["allow"]
        delta[q] = r["sagnac"]["delta"]
    n_ids = len(set(ids.values()))
    n_dec = len(set(dec.values()))
    espr = (max(ener.values()) - min(ener.values())) if all(
        e is not None for e in ener.values()) else float("nan")
    print(f"\n  --- {label}  (beta_scale={beta_scale:.4g}, dim={D}) ---")
    print(f"  {'query':8s} {'ids':>10s} {'decoded':>10s} {'energy':>12s} {'delta':>9s} {'allow':>6s}")
    for q in QS:
        print(f"  {q!r:8s} {str(ids[q]):>10s} {str(dec[q])[:10]:>10s} "
              f"{ener[q]:12.6f} {delta[q]:9.5f} {str(allow[q]):>6s}")
    print(f"  distinct ids     : {n_ids} of {len(QS)}")
    print(f"  distinct decoded : {n_dec} of {len(QS)}")
    print(f"  energy spread    : {espr:.3e}")
    order_ok = sum(1 for a, b in [("ab", "ba"), ("dog", "god"), ("cat", "act")]
                   if ids[a] != ids[b])
    print(f"  order-sensitivity: {order_ok}/3 swapped pairs now differ")
    return {"scale": beta_scale, "n_ids": n_ids, "n_dec": n_dec,
            "energy_spread": espr, "order_ok": order_ok, "ids": ids}


def main() -> int:
    print("=" * 78)
    print("A/B: D-ZO-COLLAPSE repair (swarm CCCP beta_scale)")
    print("=" * 78)
    base = run(1.0, "BASELINE (unchanged default)")
    D = build(1.0)[0].dim
    fixed = run(math.sqrt(D), f"REPAIR (beta_scale=sqrt(D)={math.sqrt(D):.1f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    h1 = base["n_ids"] <= 2
    h2 = fixed["n_ids"] > base["n_ids"]
    print(f"  H1 baseline collapses      : {h1}  (distinct ids {base['n_ids']} of 6)")
    print(f"  H2 repair increases        : {h2}  ({base['n_ids']} -> {fixed['n_ids']})")
    print(f"  order sensitivity          : {base['order_ok']}/3 -> {fixed['order_ok']}/3")
    print(f"  energy spread (was ~5e-6)  : {base['energy_spread']:.2e} -> {fixed['energy_spread']:.2e}")
    if h2:
        print("\n  ROOT CAUSE CONFIRMED + REPAIR EFFECTIVE")
        print("  (this is NOT yet 'intelligence': the answers must also be CORRECT,")
        print("   and no output-vs-ground-truth signal exists here. Distinctness is")
        print("   necessary, not sufficient.)")
    else:
        print("\n  ATTRIBUTION WRONG: distinctness did not increase. Report as")
        print("  FALSIFIED; do not ship the repair as a fix.")
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
