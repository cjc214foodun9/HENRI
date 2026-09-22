#!/usr/bin/env python3
"""MEASURE the live Sagnac scale defect before changing production code.

THE FINDING (from live_planner_defect_probe.py, which FALSIFIED my first hypothesis)
    My hypothesis was: `psi_world=pred_wave` makes the epistemic channel compare a
    candidate against ITSELF, so delta_epistemic = 1 - |<pred,pred>| = 0 for every
    candidate. MEASURED: 9.990e-01, not 0. Hypothesis FALSIFIED.

    The measurement says something worse. `dual_channel_sagnac_veto` computes

        inner = torch.abs(torch.mean(w_cand.conj() * w_ref))
        delta = 1.0 - inner

    For L2-normalized vectors of length D, sum(cand.conj()*ref) IS the proper inner
    product (bounded by 1), so MEAN is the inner product divided by D. Therefore

        delta = 1 - |<a,b>| / D

    which is 1 - O(1/D) for ALL inputs. At D = 1024 that is 1 - 0.000977 = 0.999023,
    and the probe measured 9.990e-01. The stress is pinned near 1.0 no matter how
    well the candidate matches.

    CONSEQUENCE: `hard_veto_triggered = delta_axiom > epsilon_hard` with
    epsilon_hard = tau_veto = 0.35 fires on 100% of candidates, so every child node
    is pruned and the MCTS cannot explore at all. Measured random-axiom veto pass
    rate: 0.0000.

WHAT THIS SCRIPT ESTABLISHES
    S1  reproduce the scale identity numerically: delta == 1 - |<a,b>|/D
    S2  show delta is ~constant across identical / orthogonal / random pairs
    S3  show the CORRECT normalized correlation separates those same pairs widely
    S4  run the LIVE search and count how many children get pruned
    S5  list every caller of dual_channel_sagnac_veto, so a fix is scoped

    Nothing is patched here. This is the measurement that justifies a patch.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(HERE, "sagnac_scale_defect_observed.json")


def unit(n: int, seed: int, D: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, D, generator=g)
    return v / v.norm(dim=-1, keepdim=True)


def main() -> int:
    from sagnac_mcts_planner import SagnacMCTSPlanner

    D = 1024
    planner = SagnacMCTSPlanner(d_model=D, k_blocks=128, tau_veto=0.35, device="cpu")
    res: dict = {}

    a = unit(1, 1, D)[0]
    b = unit(1, 2, D)[0]

    # ------------------------------------------------------------------ S1
    inner_proper = torch.abs((a.conj() * b).sum()).item()
    inner_mean = torch.abs((a.conj() * b).mean()).item()
    d_meas, _, _ = planner.dual_channel_sagnac_veto(a, b, b)
    predicted = 1.0 - inner_proper / D
    res["S1_inner_proper"] = inner_proper
    res["S1_inner_mean"] = inner_mean
    res["S1_ratio_mean_over_proper"] = (inner_mean / inner_proper) if inner_proper else None
    res["S1_delta_measured"] = d_meas
    res["S1_delta_predicted_as_1_minus_inner_over_D"] = predicted
    res["S1_identity_holds"] = abs(d_meas - predicted) < 1e-9
    res["S1_one_over_D"] = 1.0 / D

    # ------------------------------------------------------------------ S2
    cases = {
        "identical": (a, a),
        "orthogonal_pair": (a, b),
        "random_1": (unit(1, 11, D)[0], unit(1, 12, D)[0]),
        "random_2": (unit(1, 21, D)[0], unit(1, 22, D)[0]),
    }
    live_deltas = {}
    for name, (x, y) in cases.items():
        live_deltas[name] = planner.dual_channel_sagnac_veto(x, y, y)[0]
    res["S2_live_delta_by_case"] = live_deltas
    res["S2_live_delta_range"] = max(live_deltas.values()) - min(live_deltas.values())
    # The decisive statement: even an EXACT match is not distinguished.
    res["S2_identical_match_stress"] = live_deltas["identical"]
    res["S2_identical_match_should_be"] = 0.0

    # ------------------------------------------------------------------ S3
    fixed = {}
    for name, (x, y) in cases.items():
        ip = torch.abs((x.conj() * y).sum()).item() / (
            x.norm().item() * y.norm().item())
        fixed[name] = 1.0 - ip
    res["S3_normalized_delta_by_case"] = fixed
    res["S3_normalized_range"] = max(fixed.values()) - min(fixed.values())

    # ------------------------------------------------------------------ S4
    grid = np.array([[1, 2], [3, 4]])
    demos = [(np.array([[1, 2], [3, 4]]), np.array([[3, 1], [4, 2]]))]
    try:
        prog, delta = planner.search(grid, num_simulations=3, demo_pairs=demos)
        res["S4_search_returned_op"] = getattr(prog, "op_name", str(prog))
        res["S4_search_delta"] = float(delta)
    except Exception as e:  # noqa: BLE001
        res["S4_search_error"] = f"{type(e).__name__}: {e}"

    # Count pruned children directly by reproducing one root expansion.
    try:
        ref = None
        from sagnac_mcts_planner import SpelkeDSLNode
        # Rebuild the induced reference the way search() does, via the public scorer
        root_ast = SpelkeDSLNode(op_name="Identity")
        ref_wave = planner.vision_encoder.encode_grid(root_ast.execute(grid))
        n_pruned = 0
        n_total = 0
        for op in planner.primitive_ops:
            c = SpelkeDSLNode(op_name=op)
            pw = planner.vision_encoder.encode_grid(c.execute(grid))
            d_ax, d_ep, veto = planner.dual_channel_sagnac_veto(
                psi_candidate=pw, psi_axiom=ref_wave, psi_world=pw,
                epsilon_hard=planner.tau_veto)
            n_total += 1
            if veto:
                n_pruned += 1
        res["S4_children_total"] = n_total
        res["S4_children_pruned"] = n_pruned
        res["S4_prune_rate"] = (n_pruned / n_total) if n_total else None
    except Exception as e:  # noqa: BLE001
        res["S4_prune_probe_error"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ S5
    callers = []
    for p in Path(ROOT).rglob("*.py"):
        if "_archive" in p.parts or ".git" in p.parts:
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        if "dual_channel_sagnac_veto" in src:
            for i, line in enumerate(src.splitlines(), 1):
                if "dual_channel_sagnac_veto" in line:
                    callers.append({"file": str(p.relative_to(ROOT)), "line": i,
                                    "text": line.strip()[:110]})
    res["S5_callers"] = callers
    res["S5_caller_count"] = len(callers)

    # Verdict logic: the defect is confirmed if the identity holds AND the live
    # delta is near 1.0 even for an exact match AND normalized correlation
    # separates the cases well.
    defect_confirmed = (
        res["S1_identity_holds"]
        and res["S2_live_delta_range"] < 0.01
        and res["S2_identical_match_stress"] > 0.9
        and res["S3_normalized_range"] > 0.5
    )
    res["DEFECT_CONFIRMED"] = defect_confirmed

    out = {"module": "sagnac_scale_defect", "evidence_class": "OBSERVED",
           "dim": D, "results": res,
           "claim": (
               "dual_channel_sagnac_veto computes 1 - |<a,b>|/D instead of "
               "1 - |<a,b>|/(||a|| ||b||), so every stress is pinned near 1.0 and "
               "the hard veto fires on 100% of candidates. This is the live "
               "'un-passable by construction' defect, and it is a DIVISION BY D, "
               "not the random-axiom construction the spec blamed."),
           "hypothesis_falsified": (
               "My earlier D1 hypothesis -- that self-referencing psi_world pins the "
               "EPISTEMIC channel to 0 -- is FALSIFIED by measurement (0.999, not 0). "
               "The real cause is the mean-vs-inner-product scaling, and it pins BOTH "
               "channels near 1.0.")}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 80)
    print(f"SAGNAC SCALE DEFECT  (D={D}, tau_veto=0.35)")
    print("=" * 80)
    print(f"  S1 proper inner product <a,b>          = {inner_proper:.6f}")
    print(f"     mean(a.conj()*b)                    = {inner_mean:.8f}")
    print(f"     1/D                                 = {1.0/D:.8f}")
    print(f"     measured delta                      = {d_meas:.6f}")
    print(f"     1 - <a,b>/D  (predicted)            = {predicted:.6f}")
    print(f"     identity holds                      = {res['S1_identity_holds']}")
    print()
    print("  S2 LIVE delta by case (should separate; does not):")
    for k_, v_ in live_deltas.items():
        print(f"       {k_:<16} {v_:.6f}")
    print(f"     live delta RANGE                    = {res['S2_live_delta_range']:.2e}")
    print(f"     EXACT-MATCH stress                  = {res['S2_identical_match_stress']:.6f} "
          f"(should be 0.0)")
    print()
    print("  S3 normalized delta by case (separates):")
    for k_, v_ in fixed.items():
        print(f"       {k_:<16} {v_:.6f}")
    print(f"     normalized RANGE                    = {res['S3_normalized_range']:.4f}")
    print()
    if "S4_prune_rate" in res:
        print(f"  S4 live root expansion: {res['S4_children_pruned']}/"
              f"{res['S4_children_total']} children PRUNED "
              f"(rate {res['S4_prune_rate']:.3f})")
        print(f"     search() returned                   = {res.get('S4_search_returned_op')} "
              f"delta {res.get('S4_search_delta')}")
    else:
        print(f"  S4 prune probe error: {res.get('S4_prune_probe_error', res.get('S4_search_error'))}")
    print()
    print(f"  S5 callers of dual_channel_sagnac_veto: {res['S5_caller_count']}")
    for c in callers:
        print(f"       {c['file']}:{c['line']}")
    print()
    print(f"DEFECT_CONFIRMED = {defect_confirmed}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
