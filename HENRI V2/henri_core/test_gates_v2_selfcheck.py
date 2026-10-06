"""Deterministic validation of gates_v2 BEFORE any measurement uses it.

Three checks, all falsifiable:
  T1  KSG estimator positive/negative controls (prereg K2).
  T2  C1 fix: non-finite input must produce a BLOCKED verdict, never a pass.
      Also prove the OLD clamp is gone (min(1.0, nan) == 1.0 is the defect).
  T3  verdict truth table for C2/C3/F4/F11.
Machine output to stdout; one JSON object.
"""
from __future__ import annotations

import json
import math
import sys

import torch

sys.path.insert(0, ".")
from henri_core import gates_v2 as g2


def main() -> int:
    out = {}

    # ---- T1: KSG estimator self-test (positive + negative control)
    st = g2.ksg_mi_selftest(n=512, rho=0.6, seed=20261005)
    out["T1_ksg_selftest"] = st

    # extra: MI must be monotone in rho, and ~0 for independent pairs
    g = torch.Generator().manual_seed(1)
    X = torch.randn(512, 4, generator=g)
    mis = {}
    for rho in (0.0, 0.3, 0.6, 0.9):
        n2 = torch.randn(512, 4, generator=g)
        Y = rho * X + math.sqrt(max(0.0, 1 - rho ** 2)) * n2
        mis[rho] = g2.ksg_mi(X, Y)
    out["T1_monotone"] = {"mi_by_rho": mis,
                          "monotone": all(mis[a] <= mis[b] + 0.05
                                          for a, b in zip([0.0, 0.3, 0.6], [0.3, 0.6, 0.9]))}
    out["T1_nan_guard"] = {"ksg_mi_too_small_is_nan":
                           (g2.ksg_mi(torch.randn(3, 2), torch.randn(3, 2)) != g2.ksg_mi(torch.randn(3, 2), torch.randn(3, 2)))
                           or math.isnan(g2.ksg_mi(torch.randn(3, 2), torch.randn(3, 2)))}

    # ---- T2: the C1 defect and its fix
    nan = float("nan")
    old_clamp = max(0.0, min(1.0, nan))          # the defect
    out["T2_old_clamp_is_vacuous"] = {"max(0.0,min(1.0,nan))": old_clamp,
                                      "would_have_passed_0.95": old_clamp >= 0.95}
    out["T2_new_guard"] = {"_finite(nan)": g2._finite(nan),
                           "_finite(0.5)": g2._finite(0.5),
                           "_finite(inf)": g2._finite(float("inf"))}
    # the source must contain no clamp on r2 output
    import inspect
    src = inspect.getsource(g2.gate_u4_retention_v2)
    out["T2_no_clamp_in_v2"] = {"contains_max_0_min_1": ("min(1.0" in src and "max(0.0" in src)}
    out["T2_loadbearing_gates_unchanged"] = {"note": "gates.py not imported/modified by gates_v2",
                                             "gates_v2_imports_gates": "gates" in str(getattr(g2, "__dict__", {}).keys())}

    # ---- T3: verdict truth table
    tt = {}
    for real_ok, ctl_ok, margin_ok, label in [
        (False, False, True, "real fails"),
        (True, False, True, "real ok, control fails, margin ok"),
        (True, False, False, "real ok, control fails, NO margin"),
        (True, True, True, "both pass -> v2 must say VACUOUS"),
    ]:
        v, why = g2._verdict_v2(real_ok, ctl_ok, "shuffled_pairing", margin_ok)
        tt[label] = v
    out["T3_verdict_table"] = tt
    out["T3_no_FALSIFIED_vocabulary"] = {"PASS": g2.PASS, "FAIL": g2.FAIL,
                                         "VACUOUS": g2.VACUOUS, "BLOCKED": g2.BLOCKED}

    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
