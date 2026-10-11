"""Measure directive 3's persistence-invariance claim on real NMR ensembles.

CAN FAIL. The directive asserts beta_0/beta_1 barcodes and Ramachandran basins are
STRICTLY invariant across conformational ensembles (d_B == 0.000, gate G-D1). Persistent
homology is STABLE (Lipschitz in the bottleneck metric), not invariant, so equality is
expected to FAIL on a disperse ensemble. This probe measures whether it does, and
includes a CROSS-PROTEIN separation control: an invariant that does not separate two
different proteins would be useless even if it were constant.

Runs on the SAME two ensembles that falsified the 8 A contact map (ledger 1991).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# HERE is .../HENRI V2/experiments/verification, so parents[1] is "HENRI V2".
# FIRST WRITE USED parents[2], which is the REPO ROOT -- the same off-by-one that put
# stage_arxiv_corpus.py output in an untracked location earlier. Caught by the
# ModuleNotFoundError on the first run, not by reading it.
HENRI_V2 = HERE.parents[1]
for sub in ("henri_core", os.path.join("scripts", "staging")):
    p = str(HENRI_V2 / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from henri_persistence_invariants import (compute_invariants, ensemble_report,  # noqa: E402
                                          rank_curve_distance)
from stage_protein_dinv import fetch_pdb, parse_models                          # noqa: E402

# Coarse grid: 8 epsilons over the inter-CA scale. A coarse grid is DELIBERATE and is
# itself a tautology risk (Reference-2 trap): quantising everything onto identical values
# would make d_B == 0 trivially. The cross-protein control is what keeps this honest.
EPS = np.arange(4.0, 12.1, 1.0)
MAX_MODELS = 8


def main() -> int:
    t0 = time.time()
    out = {
        "schema": "henri.persistence.invariance.observed.v1",
        "evidence_class": "OBSERVED",
        "what": ("Directive-3 persistence-invariance test on real NMR ensembles. Replaces "
                 "the FALSIFIED 8 A contact map (ledger 1991) with topological invariants."),
        "implementation_scope": {
            "beta_0": "EXACT single-linkage connected components (union-find).",
            "beta_1": ("EXACT for the complex: |E| - |V| + beta_0 - rank(boundary_2), the "
                       "rank taken over GF(2) by Gaussian elimination on the 2-simplex "
                       "boundary matrix. This is homology, NOT the 1-skeleton cycle-rank "
                       "surrogate."),
            "curves": "Betti curves over a shared epsilon grid; curve L1 is a conservative "
                      "stand-in for the bottleneck distance d_B (d_B=0 implies this is 0, "
                      "not conversely).",
            "dihedrals": ("CA-ONLY pseudo-dihedrals from consecutive C-alpha quadruples. "
                          "These are NOT true (phi, psi) Ramachandran angles, which need "
                          "N, CA, C. Labelled as pseudo so no reader mistakes them."),
            "grid_note": ("epsilon grid is coarse (8 points). A coarse grid can force "
                          "spurious d_B == 0 by quantisation, which is why the cross-protein "
                          "control is required before any invariance claim."),
        },
        "config": {"epsilons": EPS.tolist(), "max_models": MAX_MODELS},
        "entries": [],
        "failures": [],
    }

    coords_by_pid = {}
    for pid in ("1D3Z", "2K39"):
        print(f"--- {pid}")
        try:
            models = parse_models(fetch_pdb(pid))
            Xs = [m[1] for m in models]
            coords_by_pid[pid] = Xs
            rep = ensemble_report(Xs, EPS, max_models=MAX_MODELS)
            rep["pdb_id"] = pid
            rep["n_models_total"] = len(Xs)
            out["entries"].append(rep)
            print(f"    n_models={len(Xs)} used={rep['n_models_used']} "
                  f"strict={rep['strict_invariance']} "
                  f"d_b0={rep['max_b0_curve_distance']} d_b1={rep['max_b1_curve_distance']} "
                  f"d_basin={rep['max_basin_histogram_L1']}")
        except Exception as exc:                                    # noqa: BLE001
            out["failures"].append({"pdb_id": pid, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"    FAIL {type(exc).__name__}: {exc}")

    # ---- CROSS-PROTEIN separation control (the falsifiable part) --------------------
    if len(coords_by_pid) == 2:
        a = compute_invariants(coords_by_pid["1D3Z"][0], EPS)
        b = compute_invariants(coords_by_pid["2K39"][0], EPS)
        cross_b0 = rank_curve_distance(a["b0_curve"], b["b0_curve"])
        cross_b1 = rank_curve_distance(a["b1_curve"], b["b1_curve"])
        within = out["entries"][0]["max_b0_curve_distance"] + \
            out["entries"][0]["max_b1_curve_distance"]
        out["cross_protein_control"] = {
            "cross_b0": int(cross_b0), "cross_b1": int(cross_b1),
            "cross_total": int(cross_b0 + cross_b1),
            "within_1D3Z_total": int(within),
            "separates": bool((cross_b0 + cross_b1) > within),
            "note": ("If cross-protein distance is not clearly ABOVE the within-protein "
                     "distance, the invariant does not discriminate folds and invariance "
                     "would be worthless."),
        }
        print(f"  cross-protein d_b0={cross_b0} d_b1={cross_b1} vs within-1D3Z {within}")

        # ---- G-D1 AS RESTATED BY RULING 3 (Lipschitz stability separation) --------
        #   ratio = max same-fold d_B / min different-fold d_B  <=  0.8000
        # The RULING'S OWN FORMULA uses max-over-same-fold, not the first entry's
        # within-distance. With two entries that is max(7, 13) = 13 against a single
        # cross pair of 9. Encoded exactly as written so the number is honest.
        within_max = max(e["max_b0_curve_distance"] + e["max_b1_curve_distance"]
                         for e in out["entries"])
        out["g_d1_separation"] = {
            "within_max_same_fold": int(within_max),
            "cross_min_different_fold": int(cross_b0 + cross_b1),
            "ratio": round(within_max / max(1, cross_b0 + cross_b1), 6),
            "bar": 0.8000,
            "passes": bool(within_max / max(1, cross_b0 + cross_b1) <= 0.8000),
            "statistic": ("Betti-curve L1 is a conservative stand-in for the bottleneck "
                          "distance d_B: d_B = 0 implies L1 = 0, not conversely."),
            "mechanism_if_failed": ("The DISPERSE ensemble (2K39, rmsd 3.29 A) spreads its "
                                    "filtration further than the two folds differ, so "
                                    "conformational variation outruns fold identity on "
                                    "this 8-point epsilon grid."),
        }
        out.setdefault("checks", {})["g_d1_separation_passes"] = bool(
            out["g_d1_separation"]["passes"])
        print(f"  G-D1 ratio = {out['g_d1_separation']['ratio']:.4f} "
              f"(within_max {within_max} / cross_min {cross_b0 + cross_b1}) "
              f"bar 0.8000 -> {out['g_d1_separation']['passes']}")

    strict_any = any(e.get("strict_invariance") for e in out["entries"])
    not_strict = [e["pdb_id"] for e in out["entries"] if not e.get("strict_invariance")]
    out["verdict"] = (
        "STRICT_INVARIANCE_HOLDS_ON_ALL_ENTRIES" if out["entries"] and not not_strict
        else ("STRICT_INVARIANCE_FAILS_ON_%s" % "_".join(not_strict) if not_strict
              else "NO_DATA"))
    out["finding"] = (
        "Persistent homology is STABLE, not invariant. On a tight ensemble the barcodes can "
        "coincide exactly; on a disperse ensemble thermal breathing moves the filtration "
        "itself and d_B > 0. The directive's d_B == 0.000 guarantee therefore cannot be "
        "asserted as written and must be restated as a separation test with a stated "
        "tolerance." if not_strict else
        "All entries measured strictly invariant on this grid; the cross-protein control "
        "must still separate or the result is vacuous.")
    out["not_claims"] = [
        "no model-performance claim",
        "CA-only pseudo-dihedrals are not true Ramachandran angles",
        "curve L1 is not bottleneck distance",
        "coarse epsilon grid can force spurious zero distances",
        "two ensembles do not characterise the tight/disperse boundary",
        "subsampled to at most %d models per ensemble" % MAX_MODELS,
    ]
    out["elapsed_s"] = round(time.time() - t0, 1)

    dest = pathlib.Path(__file__).resolve().parents[3] / "design" / "zone_a" / "evidence" \
        / "henri_persistence_invariance_observed.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("PYTEST_CURRENT_TEST") and "--out" not in sys.argv:
        print(f"[receipt] NOT written under pytest: {dest}")
    else:
        dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
        print(f"[receipt] {dest}")
    print(f"  VERDICT: {out['verdict']}")
    print(json.dumps({k: out[k] for k in ("verdict", "cross_protein_control") if k in out},
                     indent=1)[:600])
    return 0 if not out["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
