"""ROOT CAUSE of the input-independent answer.

Hypothesis: the CCCP step
    logits = beta * <z, patterns>;  att = softmax(logits);  return att @ patterns
collapses to the BANK MEAN because the axiom bank is UNIT-NORM RANDOM vectors.
In D=4096 two random unit vectors have <z,p> ~ N(0, 1/D), i.e. |inner| ~ 0.016.
With beta=26.10 the logit spread is only ~0.4, so softmax over N axioms stays
near-uniform and `att @ patterns` converges to (1/N) sum(patterns) -- the SAME
vector for every query. Input identity is destroyed at the swarm, exactly where
the audit located it (distinct answers: 1 with swarm, 4 without).

The doc's beta=26.10 assumes <z,p> is a COSINE near 1 for the matching pattern.
Random unit vectors in 4096-d do not satisfy that. So beta acts on the wrong
scale. This is the D82/D85 defect class: a Hopfield block whose logits do not
match the formula's stated domain.

Predictions to test:
  P1  axiom bank pairwise |cos| should be small (~1/sqrt(D))
  P2  max |logit| per query should be small (< ~1), so softmax is near-uniform
  P3  softmax entropy should be close to ln(N) (uniform)
  P4  att @ patterns should be nearly identical across queries
  P5  scaling beta by sqrt(D) (or normalizing patterns to a dense bank) should
      restore input dependence

Read-only.
"""
from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli
from henri_core import substrate as sub

torch.set_num_threads(8)
QS = ["ab", "dog", "cat", "entity acts on object in context"]


def cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def sep(t):
    print("\n" + "-" * 76)
    print(t)
    print("-" * 76)


def main() -> int:
    system, tok = H_cli.build(small=True, vocab=512)
    system.build_axioms(H_cli.CORPUS, tok)
    bank = system.axiom_bank                       # [N, D] complex, unit norm
    N, D = bank.shape
    beta = system.swarm.beta
    print("=" * 76)
    print(f"ROOT CAUSE  (N={N} axioms, D={D}, beta={beta})")
    print("=" * 76)

    sep("P1  axiom bank cross-similarity")
    g = (bank @ bank.conj().T).real
    off = g - torch.diag(torch.diagonal(g))
    mask = ~torch.eye(N, dtype=torch.bool)
    offv = off[mask]
    print(f"  ||axiom_i||_2        : {float(bank.norm(dim=-1).mean()):.6f} (unit)")
    print(f"  mean |cos(i,j)|      : {float(offv.abs().mean()):.6f}")
    print(f"  max  |cos(i,j)|      : {float(offv.abs().max()):.6f}")
    print(f"  1/sqrt(D) reference  : {1.0 / math.sqrt(D):.6f}   "
          f"--> {'random orthogonal as predicted' if float(offv.abs().max()) < 4/math.sqrt(D) else 'NOT random'}")

    sep("P2/P3  per-query logits and softmax entropy")
    with torch.no_grad():
        psi = {q: system.wave_of(q, tok) for q in QS}
        print(f"  {'query':38s} {'max|logit|':>10s} {'H/Hmax':>8s} {'H':>8s} {'ln N':>8s}")
        ents = {}
        for q in QS:
            z = psi[q]
            logits = beta * (z @ bank.conj().T).real          # [N]
            att = torch.softmax(logits, dim=-1)
            H = float(-(att * att.clamp_min(1e-12).log()).sum())
            Hmax = math.log(N)
            ents[q] = (float(logits.abs().max()), H, Hmax)
            print(f"  {q[:38]:38s} {float(logits.abs().max()):10.4f} "
                  f"{H/Hmax:8.4f} {H:8.4f} {Hmax:8.4f}")
    print(f"  -> softmax is {'NEAR-UNIFORM (no selection)' if all(v[1]/v[2] > 0.95 for v in ents.values()) else 'peaked'}")

    sep("P4  the CCCP output  (att @ patterns) across queries")
    with torch.no_grad():
        outs = {}
        for q in QS:
            z = psi[q]
            logits = beta * (z @ bank.conj().T).real
            att = torch.softmax(logits, dim=-1)
            outs[q] = att.to(torch.complex64) @ bank
        print(f"  {'pair':38s} {'cos(cccp_out)':>14s}")
        for i, a in enumerate(QS):
            for b in QS[i + 1:]:
                print(f"  {a[:18]:18s} vs {b[:18]:18s} {cos(outs[a], outs[b]):14.8f}")
        mean_bank = bank.mean(0)
        print(f"\n  cos(cccp_out('ab'), mean(bank)) = {cos(outs['ab'], mean_bank):.8f}")
        print(f"  -> CCCP converges to the BANK MEAN, not to a per-input attractor")

    sep("P5  FIX CHECK: rescale beta to the real logit domain")
    with torch.no_grad():
        for scale_name, b in (("beta=26.10 (current)", beta),
                              (f"beta=26.10*sqrt(D)={beta*math.sqrt(D):.1f}", beta * math.sqrt(D)),
                              ("beta=26.10*D", beta * D)):
            ans = {}
            for q in QS:
                z = psi[q]
                logits = b * (z @ bank.conj().T).real
                att = torch.softmax(logits, dim=-1)
                out = sub.unit_norm((att.to(torch.complex64) @ bank).unsqueeze(0))[0]
                ans[q] = out
            uniq = len({round(cos(ans[QS[0]], ans[q]), 4) for q in QS[1:]})
            # how input-dependent is the result? spread of pairwise cos
            pc = [cos(ans[a], ans[b]) for i, a in enumerate(QS) for b in QS[i+1:]]
            print(f"  {scale_name:34s} pairwise cos spread: "
                  f"min={min(pc): .6f} max={max(pc): .6f}  "
                  f"{'INPUT-DEPENDENT' if min(pc) < 0.99 else 'COLLAPSED'}")

    print("\nDONE (read-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
