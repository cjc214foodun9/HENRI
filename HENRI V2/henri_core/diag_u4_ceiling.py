"""Why does G-U4 stall at ~0.91? Locate the binding constraint.

G-U4 asks: can the 256 macro-tokens predict the 4-slot energy profile of the
wave? After the D85 routing fix the mechanism is verifiably sound (G-U4a: the
v1 rank-1 pool fails at |cos| 0.99 / rank 1.02, the fix passes at 0.025 / 187.65),
yet retention moved 0.9128 -> 0.9060.

That FALSIFIES the routing-saturation hypothesis. This tool finds what binds
instead, by measuring three separate ceilings:

  CEILING 1  estimator. A feature set that GENERATES the target (256 bins of
             |Psi|^2, whose 4 block sums ARE the target) must reach R^2 ~ 1.0.
             If it does not, the estimator is broken, not the architecture.
  CEILING 2  features. The pooling path projector -> wave_proj -> pool. Sweep
             the PCA dimension. If R^2 saturates well below ceiling 1, the
             tokens do not carry the slot-profile information.
  CEILING 3  target. The per-slot variance across texts. A near-constant target
             makes R^2 noise-dominated regardless of feature quality.

DIAGNOSTIC ONLY. This does not change the pre-registered gate or its 0.95 bound.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                 # noqa: E402
from henri_core.tokenizer import ByteBPE                # noqa: E402
from henri_core.system import TriModelSystem            # noqa: E402

N = 512
PREFIX_CORPUS = ["the rotor advances the entity across the plane",
                 "a stack pushes and pops the nested frame",
                 "recursion unfolds the copied subtree",
                 "the gripper closes on the object edge"]
DIVERSE = [
    "the rotor advances the entity across the plane",
    "a stack pushes and pops the nested frame",
    "recursion unfolds the copied subtree",
    "the gripper closes on the object edge",
    "the swarm descends the energy saddle to rest",
    "axioms ground the candidate wave in fact",
    "the veto annihilates a destructive phase",
    "arithmetic composes the integer successor",
    "copy map permute recurse nest and return",
    "spectral decay bounds the subspace drift",
]


def r2(F, Y, n_comp, ridge=1.0):
    half = len(Y) // 2
    tr, te = slice(0, half), slice(half, len(Y))
    mu = F[tr].mean(0, keepdim=True)
    sd = F[tr].std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (F[tr] - mu) / sd, (F[te] - mu) / sd
    ybar = Y[tr].mean(0, keepdim=True)
    Ytrc, Yte = Y[tr] - ybar, Y[te]
    k = max(1, min(int(n_comp), Xtr.shape[0] - 1, Xtr.shape[1]))
    _, _, vh = torch.linalg.svd(Xtr, full_matrices=False)
    P = vh[:k]
    Ztr, Zte = Xtr @ P.T, Xte @ P.T
    gram = Ztr.T @ Ztr + ridge * torch.eye(k)
    w = torch.linalg.solve(gram, Ztr.T @ Ytrc)
    pred = Zte @ w + ybar
    ss_res = ((Yte - pred) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)


def build(texts, tag):
    system = TriModelSystem()
    tok = ByteBPE().train(texts, vocab_size=512)
    with torch.no_grad():
        psi = torch.stack([system.wave_of(t, tok) for t in texts])
        prof = (psi.abs() ** 2).reshape(len(texts), sub.N_SLOTS, -1).sum(-1)
        prof = prof / prof.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        filt = system.decoder.projector(psi)
        _, tokens = system.decoder.encode_wave(psi)
        content = tokens.mean(dim=1).float()
        routing = tokens.mean(dim=-1).float()
        base = torch.cat([content, routing], dim=-1)
        feats = torch.cat([base, base ** 2], dim=-1)
        # CEILING 1: 256 bins of |Psi|^2. The 4 slot sums are exact aggregates
        # of these bins, so this feature set GENERATES the target.
        oracle = (psi.abs() ** 2).reshape(len(texts), 256, -1).sum(-1)
    print(f"\n=== [{tag}] target statistics (CEILING 3) ===")
    print(f"  per-slot mean      {[round(v, 5) for v in prof.mean(0).tolist()]}")
    print(f"  per-slot std       {[round(v, 6) for v in prof.std(0).tolist()]}")
    cv = (prof.std(0) / prof.mean(0).clamp_min(1e-9)).mean()
    print(f"  mean coeff of var  {float(cv):.5f}")
    print("  CEILING 1 oracle |Psi|^2 bins (must be ~1.0):")
    print(f"     R2 = {r2(oracle, prof, 32):.6f}")
    print("  CEILING 2 pooling-path features, PCA sweep (the gate uses 32):")
    for k in (4, 8, 16, 32, 64, 128, 256):
        print(f"     n_comp={k:<4} R2 = {r2(feats, prof, k):.6f}")
    print("  content-only features, n_comp=32 (does routing help at all?):")
    print(f"     R2 = {r2(content, prof, 32):.6f}")
    return prof, feats


def main() -> int:
    prefix_texts = [f"retrieval transfer result {i} on the ladder" for i in range(N)]
    diverse_texts = [f"{DIVERSE[i % len(DIVERSE)]}" for i in range(N)]
    p_prof, p_feat = build(prefix_texts, "pre-registered corpus")
    d_prof, d_feat = build(diverse_texts, "diverse corpus (diagnostic)")

    print("\n=== VERDICT ===")
    print("  If the oracle reaches ~1.0 and the pooling path saturates low, the")
    print("  tokens do not carry the slot profile. Routing is NOT the constraint.")
    print("  Compare the pre-registered corpus against the diverse corpus: a big")
    print("  gap means a near-constant target, not a weak mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
