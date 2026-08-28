"""M2-successor standing-order reducer (HENRI-ORD-2026-08-M2-COHERENCE-REDUCTION,
doc 3242f6ee; adopted 2026-08-28, ledger #76a92d6f).

Reduction contract (standing order §3, supersedes prereg #c109524b acceptance):
- STEP 1 LIVENESS: all 18 cells RC=0 (verified from aggregate.log by caller).
- STEP 2 ENGAGEMENT: m2_engaged==true in >= 95% of valid step rows across all
  18 cells AND non-zero resolved-delta count for every k in 1..8.
- STEP 3 COHORT-WIDE per-horizon mean: mean over ALL cells of delta values at
  horizon k <= 0.15 for EVERY k in 1..8. Per-env means retained as DIAGNOSTIC.
- STEP 4 VERDICT: M2_HORIZON_COHERENCE_VERIFIED | M2_HORIZON_COHERENCE_FALSIFIED
  (degraded horizon = k with max cohort mean when FAIL).
Telemetry read from LIVE keys: m2_sagnac_by_horizon, m2_engaged.
"""
import glob
import json
import os
import sys

from henri_m2_coherence import M2_HORIZON

ENVS = ["ka59-38d34dbb", "sk48-d8078629", "sc25-635fd71a",
        "g50t-5849a774", "sb26-7fbdac44", "vc33-5430563c"]

ENGAGEMENT_GATE = 0.95
COHERENCE_GATE = 0.15


def load_cells(root):
    """Return {env: {seed: {"rows": n, "engaged": n, "per_k": {k: [deltas]}}}}."""
    cells = {}
    for env in ENVS:
        cells[env] = {}
        for path in sorted(glob.glob(os.path.join(root, env, "*", "*.jsonl"))):
            seed = os.path.basename(os.path.dirname(path))
            per_k = {k: [] for k in range(1, M2_HORIZON + 1)}
            rows = 0
            engaged = 0
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("step") is None:
                        continue
                    rows += 1
                    if rec.get("m2_engaged") is True:
                        engaged += 1
                    m2 = rec.get("m2_sagnac_by_horizon")
                    if m2:
                        for k in range(1, M2_HORIZON + 1):
                            v = m2[k - 1]
                            if v is not None:
                                per_k[k].append(float(v))
            cells[env][seed] = {"rows": rows, "engaged": engaged, "per_k": per_k}
    return cells


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/chan/AppData/Local/henri_r2_next/m2_gauntlet_v3"
    cells = load_cells(root)

    # STEP 2: engagement
    total_rows = sum(c["rows"] for e in cells.values() for c in e.values())
    total_engaged = sum(c["engaged"] for e in cells.values() for c in e.values())
    engagement_rate = (total_engaged / total_rows) if total_rows else 0.0

    # Cohort-wide per-horizon means
    cohort = {k: [] for k in range(1, M2_HORIZON + 1)}
    per_env_means = {}
    for env, seeds in cells.items():
        env_means = {}
        for k in range(1, M2_HORIZON + 1):
            vals = [v for c in seeds.values() for v in c["per_k"][k]]
            cohort[k].extend(vals)
            env_means[k] = (sum(vals) / len(vals)) if vals else None
        per_env_means[env] = env_means

    cohort_means = {k: (sum(v) / len(v)) if v else None for k, v in cohort.items()}
    nonzero_all_k = all(len(cohort[k]) > 0 for k in range(1, M2_HORIZON + 1))
    engagement_ok = engagement_rate >= ENGAGEMENT_GATE and nonzero_all_k
    coherence_ok = all(m is not None and m <= COHERENCE_GATE
                       for m in cohort_means.values())

    report = {
        "standing_order": "HENRI-ORD-2026-08-M2-COHERENCE-REDUCTION (3242f6ee)",
        "cells": {env: {s: {"rows": c["rows"], "engaged_rows": c["engaged"],
                            "n_deltas": {k: len(c["per_k"][k]) for k in c["per_k"]}}
                        for s, c in seeds.items()} for env, seeds in cells.items()},
        "step2_engagement": {
            "total_rows": total_rows,
            "engaged_rows": total_engaged,
            "rate": round(engagement_rate, 4),
            "gate": ENGAGEMENT_GATE,
            "nonzero_all_k": nonzero_all_k,
            "ok": engagement_ok,
        },
        "step3_cohort_means": {k: (round(v, 6) if v is not None else None)
                               for k, v in cohort_means.items()},
        "step3_per_env_diagnostic": {
            env: {k: (round(v, 6) if v is not None else None)
                  for k, v in m.items()} for env, m in per_env_means.items()},
        "step3_coherence_ok": coherence_ok,
        "step4_verdict": ("M2_HORIZON_COHERENCE_VERIFIED"
                          if (engagement_ok and coherence_ok)
                          else "M2_HORIZON_COHERENCE_FALSIFIED"),
        "degraded_horizon": (max((k for k, m in cohort_means.items()
                                  if m is not None), key=lambda k: cohort_means[k])
                             if not coherence_ok else None),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
