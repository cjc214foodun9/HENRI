"""Local API audit: every symbol and call signature the GPU harness uses.

Runs on CPU at tiny D.  A wrong kwarg here costs GPU dollars later, so this
exercises the exact construction and call sites of phase1_gpu_latency_harness.py
without touching CUDA.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HENRI_ZONE_A_BACKBONE", "1")

_V2 = Path(r"C:/Users/chan/henri-worktrees/phase1-transduction/HENRI V2")
sys.path.insert(0, str(_V2))

import torch  # noqa: E402


def check(label, fn):
    try:
        out = fn()
        print(f"  OK    {label}  -> {out}")
        return True
    except Exception as exc:
        print(f"  FAIL  {label}  -> {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        return False


results = []
D = 256
MIX = 4

from henri_zone_a_backbone import (          # noqa: E402
    PCALMInferenceState,
    PreSnapCovarianceProbe,
    ZoneATransitionOperator,
)
from arc_sagnac_veto import evaluate_veto      # noqa: E402
from hopfield_cleanup import ContinuousHopfieldCleanup  # noqa: E402

g = torch.Generator().manual_seed(7)

# --- ZoneATransitionOperator(dim, mixing_rank, seed).to(dev); .apply(x) ------
op = ZoneATransitionOperator(dim=D, mixing_rank=MIX, seed=20261002)
results.append(check(
    "ZoneATransitionOperator(dim=, mixing_rank=, seed=).apply([1,D])",
    lambda: tuple(op.apply(torch.randn(1, D, generator=g).to(torch.complex64)).shape),
))
results.append(check(
    "operator.apply([B,D]) batched",
    lambda: tuple(op.apply(torch.randn(8, D, generator=g).to(torch.complex64)).shape),
))

# --- evaluate_veto(candidate, axiom, world) -> 4-tuple ----------------------
c = torch.randn(D, generator=g).to(torch.complex64)
a = torch.randn(D, generator=g).to(torch.complex64)
w = torch.randn(D, generator=g).to(torch.complex64)
results.append(check(
    "evaluate_veto(cand, axiom, world) -> (delta_ax, delta_ep, veto, status)",
    lambda: evaluate_veto(c, a, w),
))

# --- ContinuousHopfieldCleanup(dim) .store_engrams([M,dim]) .retrieve ------
hp = ContinuousHopfieldCleanup(dim=2 * D)
results.append(check(
    "ContinuousHopfieldCleanup(dim=2D).store_engrams([64,2D])",
    lambda: hp.store_engrams(torch.randn(64, 2 * D, generator=g)),
))
results.append(check(
    "hopfield.retrieve([1,D]) complex wave (cleanup dim=2D)",
    lambda: tuple(hp.retrieve(torch.randn(1, D, generator=g).to(torch.complex64))[0].shape),
))

# --- PCALMInferenceState(W, rho=, eta_h=, steps=).run(x, y) ----------------
# NOTE: the module exposes .run(x, y) -> dict; there is no .infer.
W = [torch.randn(64, 64, generator=g) for _ in range(4)]
st = PCALMInferenceState(W, rho=1.0, eta_h=0.05, steps=8)
results.append(check(
    "PCALMInferenceState(...).run(x,y)[h][-1]",
    lambda: tuple(st.run(torch.randn(8, 64, generator=g),
                         torch.randn(8, 64, generator=g))["h"][-1].shape),
))
results.append(check("PCALM exposes .run, not .infer",
                     lambda: hasattr(st, "run") and not hasattr(st, "infer")))

# --- PreSnapCovarianceProbe(dim, k, ema) .observe([N,D]) .state_shapes -----
pr = PreSnapCovarianceProbe(dim=D, k=8, ema=0.3)
results.append(check(
    "PreSnapCovarianceProbe(dim,k,ema).observe([16,D]) top-k spectrum",
    lambda: tuple(pr.observe(torch.randn(16, D, generator=g).to(torch.complex64)).shape),
))
results.append(check("probe.state_shapes()", lambda: pr.state_shapes()))

print()
print(f"API AUDIT: {sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
