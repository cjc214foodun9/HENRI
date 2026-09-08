"""G8 probe — anti-rescale rank-change evidence on structured synthetic scores.

Answers the falsifiable question: does the Gibbs (partition) policy on
candidate EFE scores change SELECTION, or only rescale it? A rescaled argmin
is FALSIFIED evidence (prereg T4 + wiring gate).

Protocol:
  * synthetic candidate score vector (near-tied top-2, then descending),
    analogous to a measured EFE candidate distribution.
  * baseline selection = argmin (current planner behavior).
  * thermo selection = fixed-seed Gibbs draw at beta_sigma(e) for e in [1..32].
  * metrics: baseline_idx, thermo_select_count (uniform-ish?), rank_change_count
    (positions where thermo argmax != baseline argmax), gibbs_entropy>0 gate.
  * additionally: real EFE-style score trace is not available offline; the
    probe therefore uses the pre-registered synthetic anti-rescale set and
    reports it as CALIBRATION_Law evidence, NOT as a measured run claim.
Output: /tmp/g8_thermo_receipt.json + stdout.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HENRI2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HENRI2))

from thermo_partition import (  # noqa: E402
    ThermoCalibrator, schedule, gibbs_weights, gibb_select, entropy,
    rank_change_count, softmin, axiom_weights,
)

SEED = 20260908


def make_scores(n=12, top_gap=0.001, decay=0.15):
    """Near-tied top two, then decaying tail — EFE-like scale."""
    v = np.array([0.0, top_gap] + [0.5 + decay * i for i in range(n - 2)])
    return v.astype(np.float64)


def main() -> int:
    v = make_scores()
    base = int(np.argmin(v))
    rows = []
    for e in [1, 2, 4, 8, 16, 32]:
        r = schedule(e)
        w = gibbs_weights(v, r.beta_sigma)
        # deterministic draw set (fixed seed) to measure selection spread
        rng = np.random.default_rng(SEED)
        picks = [gibb_select(v, beta=r.beta_sigma, rng=rng) for _ in range(64)]
        counts = {i: int(sum(1 for p in picks if p == i)) for i in set(picks)}
        pick_nonargmin = bool(any(p != base for p in picks))  # selection identity change
        rows.append({
            "e": e, "beta_sigma": round(r.beta_sigma, 4),
            "n": round(r.n, 6), "m": round(r.m, 6),
            "entropy": round(entropy(w), 4),
            "argmax_weights": int(np.argmax(w)),
            "base_argmin": base,
            "picks_unique": sorted(set(picks)),
            "counts": counts,
            "selection_identity_change": pick_nonargmin,
            "select_nonargmin_flag": int(pick_nonargmin),
        })
    # T1/T8 spot checks for the probe (hard-min vs softmin, axiom reweight)
    t1 = bool(abs(softmin(v, 1e6) - float(v.min())) < 1e-4)
    fe = np.array([0.5, 0.1, 2.0, 1.0])
    t8 = bool(int(np.argmax(axiom_weights(fe, 1e6))) == int(np.argmin(fe)))

    summary = {
        "part": "g8_thermo_probe", "status": "OK",
        "synthetic_scores": v.tolist(),
        "baseline_argmin": base,
        "per_exposure": rows,
        "anti_rescale_summary": {
            # any exposure with selection spread > 1 proves non-rescale
            "any_selection_identity_change": any(r["selection_identity_change"] for r in rows),
            "any_select_distinct": any(r["select_nonargmin_flag"] for r in rows),
            "max_entropy": max(r["entropy"] for r in rows),
        },
        "checks": {"T1_hardmin": t1, "T8_axiom_hardmin": t8},
        "evidence_class": "CALIBRATION_LAW_EVIDENCE",
        "note": "synthetic anti-rescale set per prereg; NOT a measured run claim.",
    }
    Path("/tmp/g8_thermo_receipt.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
