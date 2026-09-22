#!/usr/bin/env python3
"""What defect does the LIVE planner actually have? Measure, do not assume.

CONTEXT / SELF-CORRECTION
    My earlier claim was that the live planner derives its Sagnac stress by
    comparing against random `zone_c_axioms`, making the veto un-passable by
    construction. Reading sagnac_mcts_planner.py showed that is WRONG for the
    repository:

      * `evaluate_test_time_plan` does NOT exist in the tree. It is the reference
        snippet in the attached roadmap spec, not live code.
      * `search()` builds `reference_wave` from the demonstration-induced goal
        (`w_task @ encode(X_test)`), which is legitimate pre-prediction
        information and is exactly what Milestone 1 preserved.

    So the random-axiom defect is a defect OF THE SPECIFICATION, not of the live
    planner. Reporting it as live code was an error on my part, and it is the same
    class of error I have been warning about: describing an artifact I had read
    about rather than one I had read.

THE REAL LIVE DEFECTS, TO BE MEASURED HERE
    D1  `psi_world=pred_wave` is passed as the epistemic reference
        (sagnac_mcts_planner.py line ~335). The channel therefore compares the
        candidate against ITSELF, so delta_epistemic = 1 - |<pred, pred>| = 0 for
        every candidate. A channel that is identically constant carries no
        information and cannot inform selection. This is circular validation in
        the technical sense.
    D2  `root.delta_epistemic = root.sagnac_delta` (line ~286) conflates the two
        channels at the root, so the root's "epistemic" value is a copy.
    D3  There is no DECODE. delta_axiom = 1 - |<encode(P(X)), reference_wave>| is a
        real-valued waveform cosine. Nothing ever decodes the candidate's OUTPUT to
        an observable and compares it to an expected observable, so a candidate
        producing the correct grid can still be penalised by encoding noise, and no
        per-cell observable error is available for gating.

    D3 is what the readout supplies. D1/D2 are separate and are measured here so
    the claim is not taken on faith.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(HERE, "live_planner_defects_observed.json")


def main() -> int:
    from sagnac_mcts_planner import SagnacMCTSPlanner

    res: dict = {}
    fails: list = []

    planner = SagnacMCTSPlanner(d_model=1024, k_blocks=128, tau_veto=0.35,
                                device="cpu")

    g = torch.Generator().manual_seed(3)
    psi_a = torch.randn(128, 8, generator=g)
    psi_b = torch.randn(128, 8, generator=g)
    psi_a = psi_a / psi_a.norm()
    psi_b = psi_b / psi_b.norm()

    # ---- D1: candidate compared against ITSELF as the "world" reference -------
    d_ax_self, d_ep_self, veto_self = planner.dual_channel_sagnac_veto(
        psi_candidate=psi_a, psi_axiom=psi_b, psi_world=psi_a)
    # And with a genuinely different world reference, for contrast.
    d_ax_oth, d_ep_oth, veto_oth = planner.dual_channel_sagnac_veto(
        psi_candidate=psi_a, psi_axiom=psi_b, psi_world=psi_b)

    res["D1_epistemic_self_reference"] = d_ep_self
    res["D1_epistemic_other_reference"] = d_ep_oth
    res["D1_axiom_channel"] = d_ax_self

    # The live code passes psi_world=pred_wave, i.e. the SELF case.
    if abs(d_ep_self) > 1e-9:
        fails.append(f"D1: with psi_world=psi_candidate the epistemic channel returned "
                     f"{d_ep_self:.3e}, expected exactly 0. The self-comparison claim "
                     f"does not reproduce, so it must be re-derived.")
    # And the channel must be CONSTANT across different candidates, not merely zero
    # for one draw. That is what makes it uninformative.
    ones = []
    for _ in range(5):
        p = torch.randn(128, 8, generator=torch.Generator().manual_seed(int(
            torch.randint(0, 10**6, (1,)).item())))
        p = p / p.norm()
        ones.append(planner.dual_channel_sagnac_veto(p, psi_b, p)[1])
    res["D1_epistemic_over_5_candidates"] = [float(x) for x in ones]
    res["D1_epistemic_is_constant"] = float(max(ones) - min(ones)) < 1e-12

    # ---- D2: how does the root set the two channels? Read the source, do not guess.
    import inspect
    import re
    src = inspect.getsource(planner.search)
    res["D2_root_conflates_channels"] = bool(
        re.search(r"delta_epistemic\s*=\s*\w*\.?sagnac_delta", src))
    res["D2_psi_world_is_candidate"] = bool(
        re.search(r"psi_world\s*=\s*pred_wave", src))

    # ---- D3: is the axiom channel informative? (contrast with the epistemic one)
    res["D3_axiom_varies_with_reference"] = abs(d_ax_self - d_ax_oth) > 1e-9
    # Random-axiom un-passability, measured, to keep the spec claim reproducible.
    ax = torch.randn(128, 8, generator=torch.Generator().manual_seed(99))
    ax = ax / ax.norm()
    rand_stress = []
    for i in range(64):
        p = torch.randn(128, 8, generator=torch.Generator().manual_seed(1000 + i))
        p = p / p.norm()
        rand_stress.append(planner.dual_channel_sagnac_veto(p, ax, p)[0])
    res["D3_random_axiom_stress_mean"] = float(np.mean(rand_stress))
    res["D3_random_axiom_stress_min"] = float(np.min(rand_stress))
    res["D3_random_axiom_pass_rate_at_tau0.35"] = float(
        np.mean([1.0 if s <= 0.35 else 0.0 for s in rand_stress]))

    verdict = "PASS" if not fails else "FAIL"
    out = {"module": "live_planner_defects", "evidence_class": "OBSERVED",
           "results": res, "gate_failures": fails, "verdict": verdict,
           "self_correction": (
               "evaluate_test_time_plan with random zone_c_axioms is SPEC code, not "
               "repository code. The live planner uses the demonstration-induced "
               "goal as its axiom reference. The live defects are the self-referenced "
               "epistemic channel and the absence of any decode step.")}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 78)
    print("LIVE PLANNER DEFECT PROBE")
    print("=" * 78)
    print(f"  D1 epistemic channel, self reference  = {d_ep_self:.3e}")
    print(f"     epistemic channel, other reference = {d_ep_oth:.4f}")
    print(f"     axiom channel (informative?)       = {d_ax_self:.4f}")
    print(f"     D1 epistemic constant over 5 draws = {res['D1_epistemic_is_constant']}")
    print(f"  D2 root conflates the two channels    = {res['D2_root_conflates_channels']}")
    print(f"     psi_world is the candidate itself  = {res['D2_psi_world_is_candidate']}")
    print(f"  D3 axiom channel varies w/ reference  = {res['D3_axiom_varies_with_reference']}")
    print(f"     random-axiom stress mean/min        = "
          f"{res['D3_random_axiom_stress_mean']:.4f} / {res['D3_random_axiom_stress_min']:.4f}")
    print(f"     random-axiom veto pass rate @0.35   = "
          f"{res['D3_random_axiom_pass_rate_at_tau0.35']:.4f}")
    print()
    if fails:
        print("GATE FAILURES:")
        for f_ in fails:
            print(f"  - {f_}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
