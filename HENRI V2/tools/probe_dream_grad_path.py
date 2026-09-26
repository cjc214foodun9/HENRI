"""Probe the live Zone A surfaces the facade and dreamer depend on.

Answers, by EXECUTION (not assumption):
  Q1  Does `HenriSwarmOrchestrator.sagnac_coherence` differentiate w.r.t. its
      TARGET argument? The dreamer's SGLD creep needs a live gradient path
      (GATE-C). If the Clifford product is not differentiable in arg 2, the
      creep must use a plain-torch objective and say so.
  Q2  What is the observed distribution of the live delta family, and what are
      candidate_action_waves shapes? Decides whether the .md thresholds
      (0.35 / 0.0431) are plausible on the live [0, 2] family.
  Q3  Is the facade's encode -> candidate -> coherence chain executable?

Read-only. No writes to the repo.
"""

from __future__ import annotations

import os
import sys
import traceback

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import torch  # noqa: E402


def hdr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ---------------------------------------------------------------- Q1 grad path
hdr("Q1  does sagnac_coherence differentiate w.r.t. the TARGET argument?")
try:
    from darwinian_phase_swarm import HenriSwarmOrchestrator

    torch.manual_seed(0)
    K, W = 8, 8
    orch = HenriSwarmOrchestrator(num_experts=4, d_model=K * W, r_rank=4, num_blocks=K)
    active = torch.randn(K, W)
    target = torch.randn(K, W, requires_grad=True)

    coh = orch.sagnac_coherence(active, target)
    print(f"  coherence type={type(coh).__name__} requires_grad={getattr(coh, 'requires_grad', None)}")
    g = torch.autograd.grad(coh, target, allow_unused=True, retain_graph=False)[0]
    if g is None:
        print("  Q1 RESULT: grad is NONE -> target not in the graph")
    else:
        print(f"  Q1 RESULT: grad norm = {float(g.norm()):.6e}  LIVE GRADIENT PATH")
except Exception as exc:
    print(f"  Q1 FAIL {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=3)


# ---------------------------------------------------------------- Q2 distribution
hdr("Q2  live delta family distribution + candidate shapes")
try:
    from darwinian_phase_swarm import HenriSwarmOrchestrator

    torch.manual_seed(1)
    K, W = 16, 8
    orch = HenriSwarmOrchestrator(num_experts=8, d_model=K * W, r_rank=4, num_blocks=K)
    ref = torch.randn(K, W)
    cands = orch.candidate_action_waves(top_k=6)
    print(f"  n_candidates={len(cands)}")
    if cands:
        a0, w0 = cands[0]
        print(f"  cand[0] action={a0!r} wave={tuple(w0.shape)} dtype={w0.dtype}")
    deltas = []
    for a, w in cands:
        d = 1.0 - float(orch.sagnac_coherence(w, ref))
        deltas.append(d)
    if deltas:
        t = torch.tensor(deltas)
        print(f"  deltas={[round(x,4) for x in deltas]}")
        print(f"  min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}")
        print(f"  fraction > entry(0.35): {float((t > 0.35).float().mean()):.3f}")
        print(f"  fraction <= wake(0.0431): {float((t <= 0.0431).float().mean()):.3f}")
except Exception as exc:
    print(f"  Q2 FAIL {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=3)


# ---------------------------------------------------------------- Q3 facade chain
hdr("Q3  facade chain: encode -> candidate_waves")
try:
    from henri_zone_a_core import ZoneACore

    torch.manual_seed(2)
    core = ZoneACore(d_model=128, num_blocks=16, num_experts=8, r_rank=4)
    print(f"  orchestrator params = {core.parameter_count()}")
    grid = torch.randint(0, 10, (5, 5)).tolist()
    wave = core.encode(grid)
    print(f"  encode -> {tuple(wave.shape)} norm={float(wave.norm()):.4f}")
    pairs = core.candidate_waves(wave, wave, top_k=4)
    print(f"  pairs = {len(pairs)}")
    for p in pairs[:3]:
        print(f"    action={p.action} delta={p.sagnac_delta:.4f} coh={p.coherence:.4f}")
except Exception as exc:
    print(f"  Q3 FAIL {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=3)

print("\nPROBE COMPLETE")
