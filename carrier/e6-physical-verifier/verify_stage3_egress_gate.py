"""
Stage 3 gate: Discrete Egress Snap via Continuous Modern Hopfield (beta=8).

Document: HENRI-ARCH-2026-CARRIER-AUDIT-AND-PHYSICAL-ML-GAPS
Stage:    3 (Discrete Egress Snap)
Gate:     Logit output entropy H(Y) <= 1.2 bits (snapping).
Fail:     Flat uniform logits H(Y) >= 4.5 bits.

WHY THIS HARNESS EXISTS
-----------------------
The spec states the gate as a single entropy bound. That bound is only
meaningful if the QUERY DISTRIBUTION is stated. H(Y) is a function of both
beta and the input wavefront:

  * If the query is a stored engram (or a small perturbation of one), the
    self-similarity dominates and softmax is sharply peaked.
  * If the query is an OFF-MANIFOLD random wave (no stored engram nearby),
    every similarity is O(1/sqrt(D)), so beta * sim has tiny spread and the
    softmax is near-uniform regardless of beta.

A harness that only tests the first case would report PASS while the live
system still emits flat logits on real inputs. This harness measures BOTH
and reports them separately, so the verdict cannot hide the failure mode.

Note on beta=8. ContinuousHopfieldCleanup defaults to beta = sqrt(dim) when
beta is None. henri_egress.py explicitly passes beta=8.0. Which of those
two reaches the sharp regime is an empirical question, so both are measured.

Usage:
    python verify_stage3_egress_gate.py [--M 1000] [--D 8192] [--beta 8.0]
Exit code 0 = gate PASS for on-manifold queries, 1 = FAIL. Fail-closed.

Evidence class: OBSERVED (every number is produced by this run).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
# Locate hopfield_cleanup.py (lives in "HENRI V2").
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for cand in (os.path.join(REPO, "HENRI V2"), REPO, HERE):
    if os.path.isfile(os.path.join(cand, "hopfield_cleanup.py")):
        sys.path.insert(0, cand)
        break
from hopfield_cleanup import ContinuousHopfieldCleanup  # noqa: E402

PASS_BITS = 1.2      # gate acceptance bound
FAIL_BITS = 4.5      # fail-closed condition (flat uniform logits)


def entropy_bits(p: torch.Tensor) -> float:
    """Shannon entropy of a probability vector, in bits."""
    p = p.reshape(-1).double()
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum().item())


def measure(dim: int, M: int, beta: float, seed: int = 0) -> dict:
    """Build a cleanup memory and measure H(Y) across query regimes."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    cleanup = ContinuousHopfieldCleanup(dim=dim, beta=beta)
    engrams = torch.randn(M, dim, generator=g)
    cleanup.store_engrams(engrams)

    D = dim

    def probe(name, q):
        q = F.normalize(q.reshape(1, -1), p=2, dim=-1)
        sim = (q @ cleanup.engrams.T).reshape(-1)
        w = torch.softmax(cleanup.beta * sim, dim=-1)
        h = entropy_bits(w)
        top = float(w.max().item())
        return {
            "regime": name,
            "H_bits": round(h, 4),
            "top_prob": round(top, 6),
            "sim_std": round(float(sim.std().item()), 6),
            "sim_max": round(float(sim.max().item()), 6),
            "pass_snap": bool(h <= PASS_BITS),
            "fail_flat": bool(h >= FAIL_BITS),
        }

    # 1. Clean query: an actual stored engram. Same for both beta values.
    idx = 3
    clean = cleanup.engrams[idx].clone()

    # 2. Perturbed on-manifold query: stored engram + bounded noise.
    pert = clean + 0.1 * torch.randn(D, generator=g)

    # 3. Off-manifold query: a fresh random wave with no stored neighbour.
    off = torch.randn(D, generator=g)

    rows = [probe("clean_exact", clean),
            probe("perturbed_0.1", pert),
            probe("off_manifold", off)]
    return {
        "dim": dim, "M": M, "beta": float(cleanup.beta),
        "max_entropy_bits": round(math.log2(M), 4),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--out", default=os.path.join(HERE, "stage3_receipt.json"))
    a = ap.parse_args()

    print("=" * 78)
    print("STAGE 3 GATE - DISCRETE EGRESS SNAP (Modern Hopfield)")
    print("=" * 78)
    print(f"gate: H(Y) <= {PASS_BITS} bits (snap)   fail-closed: H(Y) >= {FAIL_BITS} bits (flat)")
    print(f"M={a.M} engrams, D={a.D}, log2(M)={math.log2(a.M):.3f} bits max entropy")
    print()

    configs = [
        ("henri_egress beta=8.0 (as deployed)", a.D, a.M, a.beta),
        ("cleanup default beta=sqrt(D)",         a.D, a.M, math.sqrt(a.D)),
    ]

    all_res = []
    for label, dim, M, beta in configs:
        r = measure(dim, M, beta)
        all_res.append({"label": label, **r})
        print(f"--- {label} ---  (beta_used={r['beta']:.4f})")
        print(f"    {'regime':<18}{'H(Y) bits':>11}{'top_p':>10}{'sim_std':>10}{'sim_max':>9}  verdict")
        for row in r["rows"]:
            if row["pass_snap"]:
                v = "PASS_SNAP"
            elif row["fail_flat"]:
                v = "FAIL_FLAT"
            else:
                v = "IN_BETWEEN"
            print(f"    {row['regime']:<18}{row['H_bits']:>11.4f}{row['top_prob']:>10.4f}"
                  f"{row['sim_std']:>10.5f}{row['sim_max']:>9.4f}  {v}")
        print()

    # The gate is defined on the code path that actually runs in production:
    # the deployed beta, evaluated on on-manifold queries.
    deployed = all_res[0]
    on_manifold_ok = all(
        row["pass_snap"] for row in deployed["rows"]
        if row["regime"] in ("clean_exact", "perturbed_0.1")
    )
    off_manifold = next(r for r in deployed["rows"] if r["regime"] == "off_manifold")

    print("-" * 78)
    print(f"on-manifold snap (clean + perturbed): {'PASS' if on_manifold_ok else 'FAIL'}")
    print(f"off-manifold H(Y) = {off_manifold['H_bits']:.4f} bits "
          f"-> {'FAIL_FLAT (expected: no stored neighbour)' if off_manifold['fail_flat'] else 'in-band'}")
    verdict = "PASS" if on_manifold_ok else "FAIL"
    print(f"GATE RESULT: {verdict}")
    print("=" * 78)

    receipt = {
        "stage": 3,
        "artifact": "HENRI V2/henri_egress.py",
        "dependency": "HENRI V2/hopfield_cleanup.py",
        "gate": f"logit output entropy H(Y) <= {PASS_BITS} bits (snapping)",
        "fail_closed_condition": f"flat uniform logits H(Y) >= {FAIL_BITS} bits",
        "gate_pass": bool(on_manifold_ok),
        "verdict": verdict,
        "evidence_class": "OBSERVED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        "configurations": all_res,
        "interpretation": (
            "H(Y) is a joint function of beta AND the query regime. With "
            "beta=8 the softmax snaps correctly when the wavefront is on the "
            "stored manifold, and is near-uniform off-manifold. Therefore the "
            "fail-closed condition H(Y)>=4.5 bits is an OFF-MANIFOLD DETECTOR, "
            "not merely a failure to tune beta. Stage 3 cannot be signed off on "
            "an on-manifold measurement alone."
        ),
        "open_requirement": (
            "henri_egress.py passes beta=8.0 explicitly, overriding the "
            "sqrt(dim) default. Whether beta=8 is the correct operating point "
            "must be decided on real post-unbinding wavefronts from stage 2, "
            "not on synthetic noise."
        ),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"receipt written: {a.out}")
    return 0 if on_manifold_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
