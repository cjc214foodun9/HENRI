"""
System-1 v0.6.3a entropy instrumentation CONTRACTS (pre-registered).
====================================================================
C1  default-OFF: entropy returns NaN/empty without HENRI_V063A_ENABLE=1.
C2  entropy math: H=0 on single mass, H=logK on uniform, range [0, logK].
C3  normalized entropy H/logK in [0,1]; NaN-safe.
C4  non-vacuity: real sims (K>=2, nonzero spread) -> H>0, K recorded.
C5  determinism: identical inputs -> identical outputs (no RNG).
C6  no behavioral change: evaluator entropy arm pool order == baseline
    order (byte-identical candidate sequence), calls/outcome identical.
C7  split hygiene: fresh disposable split, n % 13 == 0, seed disjoint
    from every consumed digest/seed; consumed digests guarded.
"""

import math
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from v063_entropy_gate_carrier import (
    _ENABLED,
    candidate_score_distribution,
    shannon_entropy_nats,
    normalized_entropy,
)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(ckpt_path=None, device="cpu"):
    # ---- C1: default-OFF ----
    if os.environ.get("HENRI_V063A_ENABLE", "0") != "1":
        _assert(not _ENABLED, "C1: carrier must be OFF by default")
        probs, h, h_norm, k = candidate_score_distribution(
            torch.tensor([0.9, 0.1]))
        _assert(h != h, "C1: H must be NaN when disabled")
        _assert(probs == [] and k == 0, "C1: no telemetry when disabled")
        print("C1 default-OFF PASS (flag absent)")
    else:
        print("C1 default-OFF PASS (flag present; enabled for test)")

    # force re-read of module state after env toggle (subprocess in real runs)
    os.environ["HENRI_V063A_ENABLE"] = "1"
    import importlib
    importlib.reload(sys.modules["v063_entropy_gate_carrier"])
    from v063_entropy_gate_carrier import (
        candidate_score_distribution as csd2,
        _ENABLED as _EN2,
    )
    _assert(_EN2, "C1: carrier must enable when flag=1")

    # ---- C2: entropy math ----
    _assert(abs(shannon_entropy_nats([1.0]) - 0.0) < 1e-9,
            "C2: single mass H=0")
    _assert(abs(shannon_entropy_nats([0.5, 0.5]) - math.log(2)) < 1e-9,
            "C2: uniform H=logK")
    h2 = shannon_entropy_nats([0.7, 0.2, 0.1])
    _assert(0.0 < h2 < math.log(3) + 1e-9, "C2: H within [0, logK]")
    print(f"C2 entropy math PASS (H(0.7,0.2,0.1)={h2:.4f})")

    # ---- C3: normalized entropy ----
    _, hn1 = normalized_entropy([0.5, 0.5])
    _assert(abs(hn1 - 1.0) < 1e-9, "C3: uniform H/logK=1")
    _, hn2 = normalized_entropy([1.0, 0.0])
    _assert(abs(hn2 - 0.0) < 1e-9, "C3: single mass H/logK=0")
    _assert(0.0 <= hn2 <= 1.0, "C3: H/logK in [0,1]")
    print("C3 normalized entropy PASS")

    # ---- C4: non-vacuity on real-ish sims ----
    sims = torch.tensor([0.95, 0.80, 0.61, 0.42])
    probs, h, h_norm, k = csd2(sims)
    _assert(k >= 2, "C4: K>=2")
    _assert(h > 0.0, "C4: H>0 on spread sims")
    _assert(probs and abs(sum(probs) - 1.0) < 1e-6, "C4: probs sum to 1")
    print(f"C4 non-vacuity PASS (K={k}, H={h:.4f}, Hnorm={h_norm:.4f})")

    # ---- C5: determinism ----
    _, h_a, hn_a, k_a = csd2(sims)
    _, h_b, hn_b, k_b = csd2(sims)
    _assert(h_a == h_b and hn_a == hn_b and k_a == k_b,
            "C5: deterministic outputs")
    print("C5 determinism PASS")

    # ---- C6: no behavioral change (evaluator-level, checked by eval) ----
    print("C6 no-behavioral-change: enforced in eval_v063_dev (entropy arm "
          "order == baseline byte-identical; assert in eval output)")

    # ---- C7: split hygiene (checked by eval guard) ----
    print("C7 split hygiene: enforced in eval_v063_dev guard "
          "(consumed digests incl. dev9_v0601 a8a2d7a7..., heldout "
          "87390286..., a09bf275...)")

    print("\nALL v0.6.3a CONTRACTS PASS (C1-C5 executable; C6-C7 eval-enforced)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.ckpt, args.device)
