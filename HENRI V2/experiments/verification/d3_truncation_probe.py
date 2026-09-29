"""Discriminating probe: is the low-rank rollout error TRUNCATION or a BUG?

Hypothesis: `make_synthetic_triples` builds block-diagonal 2x2 ROTATIONS, which
are orthogonal => ALL singular values equal 1 (FLAT spectrum). Truncating a
flat-spectrum operator to rank r < d destroys it. If so:
   * r = d  reproduces the dense solve EXACTLY        (algebra correct)
   * r < d  error grows as sqrt(discarded energy)     (truncation, expected)
   * a DECAYING-spectrum truth is recovered well at the same rank
That triple separates "my factorisation is wrong" from "the test premise was
wrong". Prints numbers only.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import henri_action_koopman as K  # noqa: E402

DIM, N_A, N_PER = 32, 4, 64


def spectrum(M):
    s = torch.linalg.svdvals(M.to(torch.float64))
    return s


def fit_rank(rank):
    triples, truth = K.make_synthetic_triples(N_A, DIM, N_PER, 11, rng_scale=0.02)
    m = K.ActionConditionedKoopman(dim=DIM, n_actions=N_A, rank=rank).fit(triples)
    return m, truth, triples


print("=== 1. SPECTRUM OF THE TRUTH OPERATOR (flat or decaying?) ===")
_, truth, _ = fit_rank(None)
s0 = spectrum(truth[0])
print("  truth[0] singular values: min=%.6f max=%.6f  ratio=%.6f" %
      (float(s0.min()), float(s0.max()), float(s0.min() / s0.max())))
print("  -> FLAT spectrum (orthogonal operator): truncation is worst-case."

      if float(s0.min() / s0.max()) > 0.99 else
      "  -> decaying spectrum: truncation is benign.")

print()
print("=== 2. FITTED OPERATOR ERROR vs RANK (rebuilt K_r vs dense K) ===")
m_dense, _, _ = fit_rank(None)
for r in (8, 16, 24, 32):
    m, _, _ = fit_rank(r)
    rel = 0.0
    for a in sorted(m.U):
        rebuilt = (m.U[a] @ m.Vs[a].t()).to(torch.float64)
        ref = m_dense.K[a].to(torch.float64)
        rel = max(rel, float((rebuilt - ref).norm() / (ref.norm() + 1e-12)))
    err = K.rollout_error(m, truth, DIM, horizon=1)
    print("  rank=%2d  rel||K_r-K||=%.6f   1-step rollout err=%.4f" % (r, rel, err))

print()
print("=== 3. DECAYING-SPECTRUM TRUTH: does the SAME rank fit recover it? ===")
g = torch.Generator().manual_seed(21)
Q, _ = torch.linalg.qr(torch.randn(DIM, DIM, generator=g, dtype=torch.float64))
truth2 = {}
for a in range(N_A):
    k = 8                                    # true rank only 8
    sv = torch.tensor([0.9 ** i for i in range(k)], dtype=torch.float64)
    block = (Q[:, :k] * sv) @ Q[:, :k].t()   # symmetric, decaying, rank 8
    th = (a + 1) * 0.13
    truth2[a] = (math.cos(th) * block
                 + math.sin(th) * (torch.eye(DIM, dtype=torch.float64) * 0.05)).to(torch.float32)
print("  truth2[0] singular values (top 10):",
      [round(float(x), 4) for x in spectrum(truth2[0])[:10]])
triples2 = []
for a in range(N_A):
    for _ in range(N_PER):
        s = torch.randn(DIM, generator=g).to(torch.float32)
        s = s / s.norm()
        s1 = truth2[a] @ s + 0.02 * torch.randn(DIM, generator=g).to(torch.float32) / math.sqrt(DIM)
        triples2.append((s, a, s1))
for r in (8, 16, 32):
    m2 = K.ActionConditionedKoopman(dim=DIM, n_actions=N_A, rank=r).fit(triples2)
    print("  rank=%2d  1-step rollout err=%.6f   3-step=%.6f   eff_rank=%s" %
          (r, K.rollout_error(m2, truth2, DIM, horizon=1),
           K.rollout_error(m2, truth2, DIM, horizon=3),
           m2.effective_rank[0]))
print()
print("=== 4. VERDICT INPUTS ===")
print("  r=d exact reproduction  -> algebra correct (see ==2== rank=32 row)")
print("  flat-spectrum r<d large -> TRUNCATION, not a bug (==1== + ==2==)")
print("  decaying r>=8 small     -> LOW-RANK IS USABLE when the spectrum decays (==3==)")
