"""Calibration for zone_c_transfer_r1. NOT the test. Chooses LR, OBS_NOISE, budget.

Fixes two vacuity defects found by smoke run:
  D1. For a DIAGONAL operator, the least-squares estimate sum(y*conj(x))/sum|x|^2
      recovers m EXACTLY when observations carry no noise. Then m_hat == m_true,
      every arm starts at tau, and all arms return step 1. The test would be
      vacuous. Observation noise makes m_hat a genuine proxy.
  D2. Fit-from-random-init did not converge to tau=0.90 within 300 steps at
      lr=0.05, so no fit task could be ingested; ingested=0.

Chooses the smallest observation noise whose cold start needs a real number of
steps (room for A1 to help, baseline can still fail), then the smallest LR at
which fit-from-random-init converges (ingestion is possible).
"""
import os
import sys

HERE = os.path.abspath(__file__)
HENRI_V2 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HENRI_V2)

import torch

D = 1024
TWO_PI = 6.283185307179586
SEEDS = (20261002, 20261003, 20261004)
FIT_BUDGET = 400


def unit(m):
    return m / (m.abs() + 1e-12)


def make_task(base, g, noise=0.30):
    return torch.exp(1j * (base + noise * torch.randn(D, generator=g)))


def estimate(m_true, g, n_demo=4, obs_noise=0.0):
    x = torch.randn(n_demo, D, generator=g, dtype=torch.cfloat)
    y = x * m_true + obs_noise * torch.randn(n_demo, D, generator=g,
                                             dtype=torch.cfloat)
    return unit((y * x.conj()).sum(0) / ((x.abs() ** 2).sum(0) + 1e-9))


def corr(a, b):
    return float((a * b.conj()).sum().abs()
                 / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12))


def steps(m_true, m_init, g, lr, tau=0.90, budget=FIT_BUDGET):
    m = m_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([m], lr=lr)
    x = torch.randn(64, D, generator=g, dtype=torch.cfloat)
    y = x * m_true
    yn = torch.linalg.vector_norm(y)
    for s in range(1, budget + 1):
        opt.zero_grad()
        ((x * m - y).abs() ** 2).mean().backward()
        opt.step()
        with torch.no_grad():
            m.data = unit(m.data)
            xm = x * m
            c = float((xm * y.conj()).sum().abs()
                      / (torch.linalg.vector_norm(xm) * yn + 1e-9))
            if c >= tau:
                return s
    return budget


def task(seed, off=0):
    g = torch.Generator().manual_seed(seed + off)
    return g, torch.rand(D, generator=g) * TWO_PI


print("== D2: fit-from-random-init vs LR (sigma=0.30, budget=%d) ==" % FIT_BUDGET)
best_lr = None
for lr in (0.05, 0.15, 0.30, 0.60):
    got = []
    for s in SEEDS:
        g, base = task(s)
        mt = make_task(base, g)
        r = unit(torch.randn(D, generator=torch.Generator().manual_seed(s + 1),
                             dtype=torch.cfloat))
        got.append(steps(mt, r, torch.Generator().manual_seed(s + 2), lr))
    conv = sum(1 for v in got if v < FIT_BUDGET)
    print("  lr=%.2f  steps=%s  converged=%d/3" % (lr, got, conv))
    if best_lr is None and conv == 3:
        best_lr = lr
print("  -> best_lr =", best_lr)

print("\n== D1: cold-start steps vs observation noise (lr=%.2f) ==" % best_lr)
for noise in (0.0, 1.0, 2.0, 3.0):
    cs, st = [], []
    for s in SEEDS:
        g, base = task(s)
        mt = make_task(base, g)
        gh = torch.Generator().manual_seed(s + 3)
        mh = estimate(mt, gh, 4, noise)
        cs.append(corr(mh, mt))
        st.append(steps(mt, mh, torch.Generator().manual_seed(s + 4), best_lr))
    print("  obs_noise=%.1f  corr=%.4f  A0_steps=%s"
          % (noise, sum(cs) / len(cs), st))
