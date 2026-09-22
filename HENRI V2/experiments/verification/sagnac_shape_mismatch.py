#!/usr/bin/env python3
"""Is the production Sagnac shape mismatch PRE-EXISTING, and what does it cost?

MEASURED ROOT CAUSE (experiments/verification/_veto_rootcause.sh, run from the
WORKTREE so the fixes ARE present):

    FIRST CALL SHAPES: ((65536,), (512,), (512,), 'torch.complex64', 'torch.float32', 0.35)
    RuntimeError: The size of tensor a (65536) must match the size of tensor b (512)
      at sagnac_mcts_planner.py:273 in _norm_consistent_similarity
        val = torch.abs((a.conj() * b).sum()).item() / (na * nb)

`_psi_macro` is a 65536-element complex wave; `_axiom_ref` (boundary_batch[0]) and
`_world_ref` (state_wave) are 512-element real waves. The veto cannot multiply them.

THE QUESTION THIS SETTLES
    The LEGACY expression was `torch.abs(torch.mean(w_cand.conj() * w_ax))` -- the
    SAME mismatched elementwise multiply. If it also raises, then the Sagnac veto has
    NEVER executed in production, for ANY scale, and:
      * the 0.0% ARC score cannot be attributed to the scale defect (nor to the
        adaptive-epsilon quirk I fixed): the veto never ran to have an effect;
      * `engaged` was always decided by `g_macro >= g_single` alone;
      * my epsilon_hard fix is NECESSARY but NOT SUFFICIENT -- it cannot change
        behaviour while the call still raises.
    Both are tested here so the claim is measured, not argued.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "sagnac_shape_mismatch_observed.json"

BIG = 65536
SMALL = 512


def legacy_similarity(w_cand: torch.Tensor, w_ax: torch.Tensor) -> float:
    """The EXACT legacy expression, from the pre-fix dual_channel_sagnac_veto."""
    if w_cand.is_complex():
        inner = torch.abs(torch.mean(w_cand.conj() * w_ax))
    else:
        inner = torch.abs(torch.mean(w_cand * w_ax))
    return float(1.0 - inner.item())


def main() -> int:
    res: dict = {}
    fails: list = []

    g = torch.Generator().manual_seed(0)
    psi_macro = torch.randn(BIG, generator=g).to(torch.complex64)     # production shape
    axiom_ref = torch.randn(SMALL, generator=g)                        # production shape
    world_ref = torch.randn(SMALL, generator=g)

    res["production_shapes"] = {
        "psi_macro": list(psi_macro.shape), "axiom_ref": list(axiom_ref.shape),
        "world_ref": list(world_ref.shape), "psi_dtype": str(psi_macro.dtype),
        "ref_dtype": str(axiom_ref.dtype),
        "dimension_ratio": BIG // SMALL,
    }

    # ---------------------------------------------------------------- L: legacy
    legacy_exc = None
    try:
        legacy_similarity(psi_macro, axiom_ref)
    except Exception as e:  # noqa: BLE001
        legacy_exc = f"{type(e).__name__}: {e}"
    res["legacy_raises_on_production_shapes"] = legacy_exc is not None
    res["legacy_exception"] = legacy_exc

    # ---------------------------------------------------------------- N: new form
    from sagnac_mcts_planner import SagnacMCTSPlanner
    planner = SagnacMCTSPlanner(d_model=SMALL, k_blocks=64, tau_veto=0.35,
                                device="cpu")
    new_exc = None
    try:
        planner._norm_consistent_similarity(psi_macro.flatten(), axiom_ref.flatten())
    except Exception as e:  # noqa: BLE001
        new_exc = f"{type(e).__name__}: {e}"
    res["new_form_raises_on_production_shapes"] = new_exc is not None
    res["new_exception"] = new_exc

    # ---------------------------------------------------------------- both raise?
    if not res["legacy_raises_on_production_shapes"]:
        fails.append("the LEGACY form did NOT raise; the mismatch is not pre-existing "
                     "and the never-executed claim must be withdrawn")
    if not res["new_form_raises_on_production_shapes"]:
        fails.append("the NEW form did NOT raise; my fix removed the failure and the "
                     "root-cause conclusion is wrong")

    # ------------------------------------------------- matched shapes still work?
    m = torch.randn(SMALL, generator=g)
    matched_exc = None
    try:
        legacy_similarity(m, m)
    except Exception as e:  # noqa: BLE001
        matched_exc = f"{type(e).__name__}: {e}"
    res["matched_shapes_work"] = matched_exc is None
    if matched_exc is not None:
        fails.append(f"matched shapes also raised ({matched_exc}); the function is "
                     f"broken beyond the shape issue")

    # ---------------------------------------------------------------- impact
    # In the gauntlet, `except Exception` sets `_veto = {"error": ...}` and leaves
    # `_hard_vetoed` at its initialised False, so:
    #     engaged = (g_macro >= g_single) and not False = (g_macro >= g_single)
    # From the observed run: g_single=0.0000, g_macro=0.1139 -> engaged=True.
    res["impact"] = {
        "hard_vetoed_stays": False,
        "engaged_reduces_to": "g_macro >= g_single",
        "observed_g_single": 0.0,
        "observed_g_macro": 0.1139,
        "observed_engaged": True,
        "conclusion": ("the Sagnac veto never ran, so no veto outcome influenced the "
                       "run; `engaged` was decided by the gain comparison alone"),
    }

    verdict = "PASS" if not fails else "FAIL"
    out = {
        "module": "sagnac_shape_mismatch", "evidence_class": "OBSERVED",
        "results": res, "gate_failures": fails, "verdict": verdict,
        "claim": (
            "The production Sagnac veto call passes waves of 65536 and 512 elements. "
            "Both the LEGACY and the current similarity expressions raise RuntimeError "
            "on that mismatch, so the veto has NEVER executed in production; the "
            "handler swallowed it and the engagement gate silently failed open. This "
            "is PRE-EXISTING, not introduced by the scale or epsilon fixes."),
        "consequence": (
            "The observed 0.0% ARC score cannot be attributed to the scale defect or "
            "to the adaptive-epsilon quirk, because the veto never ran. The epsilon "
            "fix is necessary but not sufficient: it cannot change behaviour while the "
            "call still raises."),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 84)
    print("PRODUCTION SAGNAC SHAPE MISMATCH")
    print("=" * 84)
    print(f"  psi_macro {res['production_shapes']['psi_macro']} "
          f"{res['production_shapes']['psi_dtype']}  vs  "
          f"axiom_ref {res['production_shapes']['axiom_ref']} "
          f"{res['production_shapes']['ref_dtype']}")
    print(f"  dimension ratio: {res['production_shapes']['dimension_ratio']}x")
    print()
    print(f"  LEGACY expression raises : {res['legacy_raises_on_production_shapes']}")
    print(f"    -> {legacy_exc}")
    print(f"  CURRENT form raises      : {res['new_form_raises_on_production_shapes']}")
    print(f"    -> {new_exc}")
    print(f"  matched shapes work      : {res['matched_shapes_work']}")
    print()
    print("  IMPACT:")
    for k, v in res["impact"].items():
        print(f"    {k}: {v}")
    print()
    if fails:
        print("GATE FAILURES:")
        for f in fails:
            print(f"  - {f}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
