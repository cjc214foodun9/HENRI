#!/usr/bin/env python3
"""Correct the record: measure the SIDECAR consumer too, at line 2315-2330.

WHAT I GOT WRONG, AND WHY IT MATTERS
    I reported that `HENRI_ARC_SAGNAC_VETO=1` "is currently a non-gate: it cannot veto
    anything". That was OVERSTATED, and the correction is load-bearing.

    production_arc_run.py has TWO consumers of the flag, both live:

      A. line 2279 -- OPINE macro-option engagement
             sagnac_planner.dual_channel_sagnac_veto(_psi_macro, _axiom, _world)
         NO epsilon_hard passed -> the PLANNER's ADAPTIVE branch. Feeds
         `"engaged": bool(_g_macro >= _g_single and not _hard_vetoed)`.
         Under legacy scale: ALWAYS_FIRES -> engaged always False (suppression).
         Under the fix:      NEVER_FIRES  -> engagement becomes live.
         This is the arm my epsilon-mode probe measured. The attribution was correct.

      B. line 2315-2330 -- EFE candidate RE-RANK, gated on the same flag
             from arc_sagnac_veto import apply_advisory_rerank, evaluate_veto
             evaluate_veto(candidate_wave=_pw, axiom_wave=_axiom_ref,
                           world_wave=_world_ref, epsilon_hard=None)
         `None` maps to the sidecar's DEFAULT_EPSILON_HARD = 0.35, a FIXED
         threshold, and the sidecar's `_sagnac_similarity` is ALREADY the canonical
         norm-consistent form (real: 0.5*(1+<a,b>); complex: |mean(a.conj()*b)|).
         So consumer B NEVER HAD the scale bug and was NOT changed by my fix.

    Therefore the flag DOES do something substantive: it re-ranks EFE candidates via
    the canonical sidecar. My "non-gate" statement was wrong, and this probe measures
    the sidecar so the correction is evidence-backed rather than a retraction.

    NOTE ON THE SIDECAR'S COMPLEX BRANCH -- a conditionality, not a bug:
         complex: S = |mean(a.conj() * b)|
    `mean` is the correct inner product ONLY for UNIT-MODULUS complex waves
    (|w_n| = 1, ||w|| = sqrt(D)). For L2-NORMALIZED complex waves it is the inner
    product divided by D. The sidecar's docstring states it is for "the UWE family"
    (unit-modulus), so it is correct BY ITS STATED CONTRACT -- but it is correct
    CONDITIONALLY on the caller honoring that convention. That is precisely the
    distinction whose violation caused the planner bug. Measured here for both
    conventions so the condition is explicit.

ARMS MEASURED
    A. planner, adaptive epsilon     (production line 2279)
    B. planner, explicit 0.35        (search() child expansion)
    C. sidecar evaluate_veto, None   (production line 2315, real waves)
    D. sidecar evaluate_veto, None   (production line 2315, L2-normalized COMPLEX)
    across alignment 1.0 -> 0.0, classified SELECTIVE / ALWAYS_FIRES / NEVER_FIRES.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "sidecar_selectivity_observed.json"
ALIGNS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.0)
DIM = 1024


def true_unit_modulus_pair(align: float, seed: int):
    """GENUINE unit-modulus phasors: |w_n| = 1 EXACTLY, so ||w||_2 = sqrt(D).

    This is the sidecar's stated 'UWE family' contract. NOTE the earlier version of
    this file got this WRONG: it built phasors and then divided by the L2 norm, which
    makes each element 1/sqrt(D) -- L2-normalized, NOT unit-modulus -- and then
    LABELED the arm "unit-modulus". That mislabeling made the arm report
    ALWAYS_FIRES for a case that cannot occur in production, and the probe's own gate
    caught it. No normalization here is the whole point.
    """
    g = torch.Generator().manual_seed(seed)
    def ph():
        th = torch.rand(DIM, generator=g) * (2 * math.pi)
        return torch.complex(torch.cos(th), torch.sin(th))     # NO normalization
    ax = ph(); nz = ph()
    c = align * ax + math.sqrt(max(0.0, 1.0 - align * align)) * nz
    return c, ax, ax


def l2_normalized_pair(align: float, seed: int):
    """L2-NORMALIZED real waves: each |w_n| ~ 1/sqrt(D), ||w||_2 = 1.

    This is what Zone A actually produces and what production passes to the sidecar
    in its REAL branch (verified: predicted_wave = state_wave from
    encode_spatial_grid, boundary_batch cast to float32; no complex construction on
    that path).
    """
    g = torch.Generator().manual_seed(seed)
    ax = torch.randn(DIM, generator=g); ax = ax / ax.norm()
    nz = torch.randn(DIM, generator=g); nz = nz / nz.norm()
    c = align * ax + math.sqrt(max(0.0, 1.0 - align * align)) * nz
    return c / c.norm(), ax, ax


def unit_pair(align: float, seed: int):
    """Real unit-norm waves (the production convention)."""
    return l2_normalized_pair(align, seed)


def classify(rates):
    if max(rates) == 0.0:
        return "NEVER_FIRES"
    if min(rates) > 0.99:
        return "ALWAYS_FIRES"
    return "SELECTIVE"


def main() -> int:
    from sagnac_mcts_planner import SagnacMCTSPlanner
    from arc_sagnac_veto import evaluate_veto, DEFAULT_EPSILON_HARD

    planner = SagnacMCTSPlanner(d_model=DIM, k_blocks=128, tau_veto=0.35, device="cpu")
    summary: dict = {"DEFAULT_EPSILON_HARD": DEFAULT_EPSILON_HARD}

    def scalar(v):
        return float(v.item()) if hasattr(v, "item") else float(v)

    arms = {}

    def record(name, fn):
        rates, deltas = [], []
        for a in ALIGNS:
            hit = 0
            ds = []
            for s in range(8):
                triggered, d_ax = fn(a, 100 + s)
                ds.append(d_ax)
                if triggered:
                    hit += 1
            rates.append(hit / 8)
            deltas.append(sum(ds) / len(ds))
        arms[name] = {
            "veto_rate_by_alignment": {str(a): r for a, r in zip(ALIGNS, rates)},
            "delta_by_alignment": {str(a): round(d, 4) for a, d in zip(ALIGNS, deltas)},
            "classification": classify(rates),
        }

    # ---- A: planner, adaptive (production line 2279) --------------------
    def arm_a(align, seed):
        c, ax, w = unit_pair(align, seed)
        d_ax, _, hard = planner.dual_channel_sagnac_veto(c, ax, w)  # no epsilon
        return bool(hard), float(d_ax)

    # ---- B: planner, explicit 0.35 (search() child expansion) -----------
    def arm_b(align, seed):
        c, ax, w = unit_pair(align, seed)
        d_ax, _, hard = planner.dual_channel_sagnac_veto(c, ax, w,
                                                        epsilon_hard=planner.tau_veto)
        return bool(hard), float(d_ax)

    # ---- C: sidecar, None epsilon, REAL unit waves ----------------------
    def arm_c(align, seed):
        c, ax, w = unit_pair(align, seed)
        d_ax, _, hard, status = evaluate_veto(c, ax, w, epsilon_hard=None)
        return bool(hard), float(d_ax)

    # ---- D: sidecar, None epsilon, GENUINE unit-modulus complex ---------
    # The sidecar's stated contract ("UWE family"). |w_n| = 1 exactly, so
    # |mean(a.conj()*b)| = 1 for identical waves and ~0 for unrelated. SELECTIVE.
    def arm_d(align, seed):
        c, ax, w = true_unit_modulus_pair(align, seed)
        d_ax, _, hard, status = evaluate_veto(c, ax, w, epsilon_hard=None)
        return bool(hard), float(d_ax)

    # ---- E: sidecar, None epsilon, L2-NORMALIZED complex (THE HAZARD) ----
    # |w_n| = 1/sqrt(D) and ||w|| = 1. Now |mean(a.conj()*b)| = 1/D for IDENTICAL
    # waves, so S ~ 0.001 and delta ~ 0.999 -> the veto fires on a PERFECT match.
    # This is the exact mirror of the planner's defect, in the other function.
    #
    # It is NOT a production failure: verified that production passes REAL float32
    # waves (predicted_wave = state_wave from encode_spatial_grid; boundary_batch cast
    # to float32; no complex construction on that path), so the sidecar takes its REAL
    # branch, which is norm-consistent and correct. This arm exists to make the
    # CONDITIONALITY explicit and measured: the sidecar's complex branch is correct
    # only for unit-modulus input, exactly as the planner's was.
    def arm_e(align, seed):
        g = torch.Generator().manual_seed(seed)
        def ph():
            th = torch.rand(DIM, generator=g) * (2 * math.pi)
            p = torch.complex(torch.cos(th), torch.sin(th))
            return p / p.norm()                    # L2-normalized: the hazard
        ax = ph(); nz = ph()
        c = align * ax + math.sqrt(max(0.0, 1.0 - align * align)) * nz
        c = c / c.norm()
        d_ax, _, hard, status = evaluate_veto(c, ax, ax, epsilon_hard=None)
        return bool(hard), float(d_ax)

    record("A_planner_adaptive_prod2279", arm_a)
    record("B_planner_explicit_search", arm_b)
    record("C_sidecar_real_prod2315", arm_c)
    record("D_sidecar_true_unitmodulus_complex", arm_d)
    record("E_sidecar_L2complex_HAZARD", arm_e)

    for k, v in arms.items():
        print(f"  {k:<34} {v['classification']:<13} "
              f"rates={[v['veto_rate_by_alignment'][str(a)] for a in ALIGNS]}")

    # ------------------------------------------------------------------ findings
    findings = {k: v["classification"] for k, v in arms.items()}
    fails = []

    # The sidecar is the EFE re-rank path and must be SELECTIVE, or the flag
    # genuinely does nothing and my corrected claim is itself wrong.
    if findings["C_sidecar_real_prod2315"] != "SELECTIVE":
        fails.append(f"sidecar real is {findings['C_sidecar_real_prod2315']}, expected "
                     f"SELECTIVE; the flag would then truly be a non-gate")
    # Unit-modulus complex must also be selective: the sidecar's stated contract.
    if findings["D_sidecar_true_unitmodulus_complex"] != "SELECTIVE":
        fails.append(f"sidecar unit-modulus complex is "
                     f"{findings['D_sidecar_true_unitmodulus_complex']}, expected SELECTIVE")
    # E must reproduce the hazard: the complex L2-normalized case fires on an
    # IDENTICAL match. Asserted positively so the conditionality stays measured --
    # if this ever becomes SELECTIVE, the sidecar's complex branch changed.
    if findings["E_sidecar_L2complex_HAZARD"] != "ALWAYS_FIRES":
        fails.append(f"sidecar L2-complex hazard is "
                     f"{findings['E_sidecar_L2complex_HAZARD']}, expected ALWAYS_FIRES "
                     f"(the mirror of the planner defect)")
    # A must be NEVER_FIRES after the scale fix: the OPINE suppression story.
    if findings["A_planner_adaptive_prod2279"] != "NEVER_FIRES":
        fails.append(f"planner adaptive is {findings['A_planner_adaptive_prod2279']}, "
                     f"expected NEVER_FIRES after the scale fix")

    verdict = "PASS" if not fails else "FAIL"
    out = {
        "module": "sidecar_selectivity", "evidence_class": "OBSERVED",
        "dim": DIM, "alignments": list(ALIGNS),
        "summary": summary, "arms": arms, "findings": findings,
        "gate_failures": fails, "verdict": verdict,
        "record_correction": (
            "My earlier statement that HENRI_ARC_SAGNAC_VETO=1 'is currently a "
            "non-gate: it cannot veto anything' was OVERSTATED. production_arc_run.py "
            "has TWO consumers. Line 2279 feeds OPINE engagement through the planner's "
            "ADAPTIVE epsilon (always-suppress under legacy; never-suppress fixed). "
            "Line 2315-2330 re-ranks EFE candidates through arc_sagnac_veto.evaluate_veto, "
            "which uses a FIXED 0.35 and the CANONICAL norm-consistent similarity and "
            "was never affected by the scale bug. Measured here: the sidecar IS "
            "selective, so the flag does gate the EFE re-rank path."),
        "conditionality": (
            "The sidecar's complex branch, S = |mean(a.conj()*b)|, is the correct inner "
            "product ONLY for UNIT-MODULUS waves. For L2-normalized waves it is the "
            "inner product divided by D -- the exact defect the planner had. The "
            "sidecar is correct by its stated 'UWE family' contract, but correctness "
            "is CONDITIONAL on the caller honoring that convention."),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    print("  RECORD CORRECTION: the flag is NOT wholly a non-gate.")
    print("    line 2279 (OPINE engagement):  planner + adaptive -> NEVER_FIRES (fixed)")
    print("    line 2315 (EFE re-rank):       sidecar + fixed 0.35 -> "
          f"{findings['C_sidecar_real_prod2315']}")
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
