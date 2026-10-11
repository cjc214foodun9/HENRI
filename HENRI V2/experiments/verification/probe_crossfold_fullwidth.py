"""probe_crossfold_fullwidth.py -- re-measure the cross-fold statistic at FULL ensemble width.

THESIS ITEM 3. The prior probe (probe_crossfold_statistic.py) truncated both ensembles to
MAX_MODELS = 10 and recorded the limit plainly: "2K39 truncated to 10 of 116 models; the
cited rmsd 3.29 A is for the FULL 2K39 ensemble." That makes the recorded extremum ratio
1.571 and mean ratio 0.266 PROVISIONAL. This probe re-measures at full width.

PRE-REGISTERED BEFORE THE RUN
    H_representative: the full-width 2K39 median CA distance lies within 20% of the
        truncated-10 value. If so, truncation preserved the filtration scale and the earlier
        ratios stand as scale-representative.
    H_artifact: if the full-width median differs by more than 20%, the truncated ratios are
        a sampling artifact and the FULL numbers supersede them.
    ARM A raw Betti, ARM C landscape (diagram shape). ARM B is NOT repeated: it is an exact
        no-op, already recorded -- a Betti curve is ALREADY scale-parameterised by the
        epsilon grid, so dividing by the ensemble median changes nothing.
    Bars unchanged: extremum ratio < 0.8000 (the inherited bar). The mean ratio is reported
        separately and is NOT a substitute for the bar.
    Ensemble spread: report mean pairwise superposed RMSD, to test the cited 3.29 A for the
        full 2K39 ensemble. Label it as a SUPERPOSED-pairwise proxy, not the cited statistic.

SCOPE
    1D3Z and 2K39 are BOTH ubiquitin NMR ensembles, so this is a same-protein
    cross-ensemble test: the stringent case, not a cross-fold-of-different-proteins test.
    Single seed. CPU. No training. No capability claim.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
V = REPO / "HENRI V2"
for sub in ("henri_core", os.path.join("scripts", "staging"),
            os.path.join("experiments", "verification")):
    p = str(V / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from henri_persistence_invariants import (          # noqa: E402
    persistence_pairs, pairwise_distances, betti_curves, rank_curve_distance,
)
from stage_protein_dinv import fetch_pdb, parse_models      # noqa: E402

EV = REPO / "design" / "zone_a" / "evidence"
EV.mkdir(parents=True, exist_ok=True)

EPS = np.linspace(0.0, 1.0, 24)
TRUNC = 10
ENTRIES = ["1D3Z", "2K39"]
BAR_EXTREMUM = 0.8000
PREREG_TOL = 0.20

R = {
    "schema": "henri.crossfold.fullwidth.observed.v1",
    "evidence_class": "OBSERVED",
    "what": "cross-fold statistic re-measured at FULL ensemble width",
    "repairs": "henri_crossfold_statistic_observed.json (MAX_MODELS = 10 truncation)",
    "preregistered": {
        "truncation_models": TRUNC,
        "bar_extremum_ratio": BAR_EXTREMUM,
        "representative_tolerance": PREREG_TOL,
        "arm_B_omitted_because": "exact no-op: a Betti curve is already scale-parameterised",
        "same_protein": "1D3Z and 2K39 are both ubiquitin; this is a same-protein test",
    },
    "fetched": {},
}


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Superposed RMSD with the optimal rotation. Element i of P corresponds to i of Q."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    Rm = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    Pr = (Rm @ Pc.T).T
    return float(np.sqrt(((Pr - Qc) ** 2).sum() / len(P)))


def landscape(diagram, tgrid, k_max=3) -> np.ndarray:
    """Persistence landscape lambda_k(t), k = 1..k_max. A set-level invariant."""
    pts = [(float(b), float(d)) for (b, d) in diagram if np.isfinite(d)]
    out = np.zeros((k_max, tgrid.size), dtype=float)
    if not pts:
        return out
    vals = np.zeros((len(pts), tgrid.size), dtype=float)
    for i, (b, d) in enumerate(pts):
        vals[i] = np.maximum(0.0, np.minimum(tgrid - b, d - tgrid))
    vals.sort(axis=0)
    for k in range(k_max):
        idx = len(pts) - 1 - k
        if idx >= 0:
            out[k] = vals[idx]
    return out


print("=" * 78)
print("STEP 1 -- fetch real NMR ensembles at FULL width (fail-closed)")
print("=" * 78)

raw_models: dict[str, list] = {}
blocked: list[str] = []
for pid in ENTRIES:
    try:
        text = fetch_pdb(pid, timeout=180)
        models = parse_models(text)
        models = [m for m in models if len(m[1]) >= 8]
        raw_models[pid] = models
        R["fetched"][pid] = {"models_parsed": len(models)}
        print(f"  {pid}: {len(models)} models parsed")
    except Exception as exc:                                        # noqa: BLE001
        blocked.append(f"{pid}:{type(exc).__name__}:{exc}")
        print(f"  {pid}: FAILED {type(exc).__name__}: {str(exc)[:80]}")

if len(raw_models) < 2:
    R["verdict"] = "BLOCKED_NO_USABLE_ENSEMBLES"
    R["blocked"] = blocked
    (EV / "henri_crossfold_fullwidth_observed.json").write_text(
        json.dumps(R, indent=2) + "\n", encoding="utf-8")
    print(f"\n  VERDICT {R['verdict']}")
    sys.exit(0)

# ---- align every model onto the residue labels COMMON to all models of its entry ----
print()
print("STEP 2 -- align residues onto the common set, then build per-model invariants")
aligned: dict[str, list] = {}
for pid, models in raw_models.items():
    labs = [tuple(str(x) for x in m[0]) for m in models]
    common = set(labs[0])
    for L in labs[1:]:
        common &= set(L)
    common = sorted(common)
    index = {lab: k for k, lab in enumerate(common)}
    keep = []
    for m, L in zip(models, labs):
        pos = [index[lab] for lab in common]
        X = np.asarray(m[1], dtype=float)[pos]
        keep.append(X)
    aligned[pid] = keep
    R["fetched"][pid]["common_residues"] = len(common)
    R["fetched"][pid]["models_used_full"] = len(keep)
    print(f"  {pid}: {len(keep)} models aligned on {len(common)} common residues")

per: dict[str, list] = {}
for pid, models in aligned.items():
    rows = []
    for i, X in enumerate(models):
        D = pairwise_distances(X)
        iu = np.triu_indices(D.shape[0], k=1)
        med = float(np.median(D[iu]))
        Dn = D / (med if med > 1e-9 else 1.0)
        h0, h1 = persistence_pairs(Dn)
        b0n = np.asarray([sum(1 for pr in h0 if pr[0] <= e < pr[1]) for e in EPS], dtype=int)
        b1n = np.asarray([sum(1 for pr in h1 if pr[0] <= e < pr[1]) for e in EPS], dtype=int)
        b0r, b1r = betti_curves(D, EPS * med)
        rows.append({
            "model": i,
            "b0_raw": np.asarray(b0r, dtype=int), "b1_raw": np.asarray(b1r, dtype=int),
            "b0_norm": b0n, "b1_norm": b1n,
            "land_h0": landscape(h0, EPS), "land_h1": landscape(h1, EPS),
            "scale_median": med, "X": X,
        })
    per[pid] = rows
    print(f"  {pid}: invariants for {len(rows)} models | mean median distance "
          f"{np.mean([r['scale_median'] for r in rows]):.3f} A")

# ---- ensemble spread: mean pairwise superposed RMSD (tests the cited 3.29 A) ----
print()
print("STEP 3 -- ensemble spread (superposed pairwise RMSD proxy)")
spread = {}
for pid, rows in per.items():
    Xs = [r["X"] for r in rows]
    n = len(Xs)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(kabsch_rmsd(Xs[i], Xs[j]))
    vals = np.asarray(vals, float)
    spread[pid] = {
        "n_models": n, "n_pairs": int(vals.size),
        "mean_rmsd_A": float(vals.mean()), "max_rmsd_A": float(vals.max()),
        "p95_rmsd_A": float(np.percentile(vals, 95)),
        "note": "SUPERPOSED pairwise RMSD proxy; not the cited ensemble statistic",
    }
    print(f"  {pid}: n={n} pairs={vals.size} mean RMSD {vals.mean():.3f} A  "
          f"max {vals.max():.3f} A  p95 {np.percentile(vals, 95):.3f} A")
R["ensemble_spread"] = spread

# ---- the arms, at truncated-10 and at full width ----
print()
print("STEP 4 -- arms: TRUNCATED-10 vs FULL WIDTH, identical code path")
print("=" * 78)


def pairs_within(rows):
    return [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))]


def pairs_cross(rows_a, rows_b):
    return [(i, j) for i in range(len(rows_a)) for j in range(len(rows_b))]


def dist(kind, ra, rb):
    if kind == "raw_betti":
        return (rank_curve_distance(ra["b0_raw"], rb["b0_raw"])
                + rank_curve_distance(ra["b1_raw"], rb["b1_raw"]))
    if kind == "landscape":
        return float(np.abs(ra["land_h0"] - rb["land_h0"]).sum()
                     + np.abs(ra["land_h1"] - rb["land_h1"]).sum())
    raise ValueError(kind)


def summarize(kind, ra, rb, label):
    w = np.asarray([dist(kind, ra[i], ra[j]) for (i, j) in pairs_within(ra)], float)
    c = np.asarray([dist(kind, ra[i], rb[j]) for (i, j) in pairs_cross(ra, rb)], float)
    ratio = float(w.max() / c.min()) if c.min() > 0 else float("inf")
    mr = float(w.mean() / c.mean()) if c.mean() > 1e-12 else float("inf")
    pooled = float(np.sqrt((w.var(ddof=1) + c.var(ddof=1)) / 2.0)) if len(w) > 1 and len(c) > 1 else 0.0
    z = float((c.mean() - w.mean()) / pooled) if pooled > 1e-12 else 0.0
    rec = {"label": label, "n_within": len(w), "n_cross": len(c),
           "mean_within": float(w.mean()), "mean_cross": float(c.mean()),
           "max_within": float(w.max()), "min_cross": float(c.min()),
           "ratio_within_over_cross": ratio, "passes_bar": bool(ratio < BAR_EXTREMUM),
           "mean_ratio": mr, "passes_bar_mean": bool(mr < BAR_EXTREMUM),
           "effect_z": z}
    print(f"  {label:16} within {w.mean():9.2f}  cross {c.mean():9.2f}  "
          f"ext-ratio {ratio:7.3f} {'PASS' if ratio < BAR_EXTREMUM else 'fail'}  "
          f"mean-ratio {mr:6.3f} {'PASS' if mr < BAR_EXTREMUM else 'fail'}  z {z:+.2f}")
    return rec


A, B = "1D3Z", "2K39"
res = {"truncated": {}, "full": {}}
for tag, amap, bmap in (("truncated", per[A][:TRUNC], per[B][:TRUNC]),
                        ("full", per[A], per[B])):
    print(f"  -- {tag.upper()}  ({len(amap)} x {len(bmap)} models) --")
    for kind, label in (("raw_betti", "A_raw_betti"), ("landscape", "C_landscape")):
        res[tag][label] = summarize(kind, amap, bmap, label)
R["arms"] = res

# ---- truncation-representativeness test (pre-registered) ----
print()
print("STEP 5 -- truncation representativeness (pre-registered tolerance %.2f)" % PREREG_TOL)
med_trunc = {p: float(np.mean([r["scale_median"] for r in per[p][:TRUNC]])) for p in per}
med_full = {p: float(np.mean([r["scale_median"] for r in per[p]])) for p in per}
reldiff = {p: (abs(med_full[p] - med_trunc[p]) / med_trunc[p]) for p in per}
representative = all(v <= PREREG_TOL for v in reldiff.values())
for p in per:
    print(f"  {p}: truncated-10 median {med_trunc[p]:.3f} A | full {med_full[p]:.3f} A | "
          f"rel diff {reldiff[p]*100:5.1f}%")
R["truncation_test"] = {
    "median_truncated10_A": med_trunc, "median_full_A": med_full,
    "relative_difference": reldiff, "tolerance": PREREG_TOL,
    "representative": bool(representative),
    "outcome": ("H_representative" if representative else "H_artifact"),
}

# ---- verdict ----
print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
full_best = min(res["full"].values(), key=lambda a: a["ratio_within_over_cross"])
trunc_best = min(res["truncated"].values(), key=lambda a: a["ratio_within_over_cross"])
R["best_arm_full"] = full_best["label"]
R["bar_cleared_full"] = bool(full_best["passes_bar"])
R["comparison"] = {
    "truncated_best_ratio": trunc_best["ratio_within_over_cross"],
    "full_best_ratio": full_best["ratio_within_over_cross"],
    "truncated_best_arm": trunc_best["label"],
    "full_best_arm": full_best["label"],
}
print(f"  truncated-10 best : {trunc_best['label']:16} ext-ratio {trunc_best['ratio_within_over_cross']:.3f}")
print(f"  FULL width best   : {full_best['label']:16} ext-ratio {full_best['ratio_within_over_cross']:.3f}")
print(f"  bar {BAR_EXTREMUM}           : {'CLEARED' if full_best['passes_bar'] else 'NOT CLEARED'}")
print(f"  truncation test   : {'REPRESENTATIVE' if representative else 'ARTIFACT'}")

if R["truncation_test"]["outcome"] == "H_artifact":
    R["verdict"] = "FULL_WIDTH_SUPERSEDES_TRUNCATED"
elif full_best["passes_bar"]:
    R["verdict"] = "CROSSFOLD_SEPARATES_AT_FULL_WIDTH"
else:
    R["verdict"] = "CROSSFOLD_STILL_NOT_SEPARATED_AT_FULL_WIDTH"
R["limits"] = [
    "same-protein test: 1D3Z and 2K39 are both ubiquitin; this is the stringent case",
    "superposed pairwise RMSD is a PROXY for the cited ensemble statistic, not the citation",
    "landscape uses k_max = 3 on a 24-point grid; refinement could move ARM C",
    "single seed; no training; CPU only",
]
print(f"  VERDICT: {R['verdict']}")

out = EV / "henri_crossfold_fullwidth_observed.json"
out.write_text(json.dumps(R, indent=2, default=str) + "\n", encoding="utf-8")
print(f"\n  receipt -> {out.relative_to(REPO)}")
print("  OBSERVED by execution. No bar moved. No CUDA. No AAII claim.")
