"""Calibrate the dream entry/wake gates on the LIVE Sagnac family.

WHY THIS EXISTS
---------------
My probe (2026-09-26) measured the live `HenriSwarmOrchestrator.sagnac_coherence`
deltas for a random context at 0.55-1.19 (mean 0.97). Against the .md thresholds
this is a DEAD SYSTEM: entry (0.35) admits 100% of candidates and wake (0.0431)
is met 0% of the time, so the dream enters always and NEVER wakes.

That is the same scale error the .md's own snippet has (D-SAGNAC class): a
threshold carried over from a different metric family. This probe measures the
ACTUAL dynamic range of the live family so the gates can be set from data.

Questions answered by execution:
  Q1  Is `sagnac_coherence` differentiable w.r.t. its TARGET? (GATE-C validity)
  Q2  What is the null distribution of deltas across contexts?
  Q3  What is the ACHIEVABLE FLOOR? (reference := the candidate itself, which is
      perfect resonance -> delta should be ~0). This sets the dynamic range.

Read-only. Writes nothing to the repo.
"""

from __future__ import annotations

import os
import sys
import traceback

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import torch  # noqa: E402

K, W = 16, 8
EXPERTS = 8  # NOTE: m=4 in construct_ba_adjacency requires num_nodes >= 5


def hdr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def build(seed=0):
    from darwinian_phase_swarm import HenriSwarmOrchestrator

    torch.manual_seed(seed)
    return HenriSwarmOrchestrator(num_experts=EXPERTS, d_model=K * W, r_rank=4, num_blocks=K)


# ------------------------------------------------------------------ Q1
hdr("Q1  gradient path: does sagnac_coherence differentiate w.r.t. the TARGET?")
try:
    orch = build(0)
    active = torch.randn(K, W)
    target = torch.randn(K, W, requires_grad=True)
    coh = orch.sagnac_coherence(active, target)
    print(f"  coherence={float(coh):.6f} requires_grad={coh.requires_grad}")
    g = torch.autograd.grad(coh, target, allow_unused=True)[0]
    if g is None:
        print("  RESULT: grad is NONE -> target NOT in the graph (GATE-C must use own objective)")
    else:
        print(f"  RESULT: grad norm={float(g.norm()):.6e} -> LIVE GRADIENT PATH present")
except Exception as exc:
    print(f"  FAIL {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=3)

# ------------------------------------------------------------------ Q2
hdr("Q2  null distribution of deltas across contexts")
try:
    all_d, per_ctx = [], []
    for seed in range(6):
        orch = build(seed)
        ref = torch.randn(K, W)
        ds = [1.0 - float(orch.sagnac_coherence(w, ref))
              for _, w in orch.candidate_action_waves(top_k=6)]
        all_d.extend(ds)
        per_ctx.append((min(ds), max(ds), sum(ds) / len(ds)))
    t = torch.tensor(all_d)
    print("  per-context (min, max, mean):")
    for i, (lo, hi, mu) in enumerate(per_ctx):
        print(f"    ctx{i}: min={lo:.4f} max={hi:.4f} mean={mu:.4f}")
    q = torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    vals = torch.quantile(t, q)
    print("  quantiles:", {f"p{int(a*100)}": round(float(b), 4) for a, b in zip(q, vals)})
    print(f"  n={len(all_d)} min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}")
    print(f"  fraction > .md entry 0.35 : {float((t > 0.35).float().mean()):.3f}")
    print(f"  fraction <= .md wake 0.0431: {float((t <= 0.0431).float().mean()):.3f}")
except Exception as exc:
    print(f"  FAIL {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=3)

# ------------------------------------------------------------------ Q3
hdr("Q3  achievable floor: reference := candidate itself (perfect resonance)")
try:
    orch = build(0)
    floors, selfs = [], []
    for _, w in orch.candidate_action_waves(top_k=6):
        d = 1.0 - float(orch.sagnac_coherence(w, w))
        floors.append(d)
        # a small perturbation off-resonance
        pert = w + 0.05 * torch.randn_like(w)
        selfs.append(1.0 - float(orch.sagnac_coherence(pert, w)))
    print(f"  self-resonance deltas (expect ~0): {[round(x, 6) for x in floors]}")
    print(f"  perturbed deltas                 : {[round(x, 6) for x in selfs]}")
    print(f"  FLOOR (max self-resonance) = {max(floors):.6f}")
    print(f"  -> dynamic range is [~{max(floors):.4f}, ~1.2]; .md wake 0.0431 sits "
          f"{'INSIDE' if max(floors) > 0.0431 else 'OUTSIDE'} the achievable floor band")
except Exception as exc:
    print(f"  FAIL {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=3)

print("\nCALIBRATION PROBE COMPLETE")
