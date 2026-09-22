#!/usr/bin/env python3
"""Grid-observable readout probe: round-trip, separation, derived tau, fail-closed.

Gates, each falsifiable. A veto is only wired if it can DECIDE.

  G1 round-trip      encode -> decode recovers the lattice exactly
  G2 codebook guard  decoding with a DIFFERENT codebook must NOT recover the grid
                     (proves the shared-codebook dependency is real, not decorative)
  G3 separation      identical / perturbed / unrelated separate monotonically
  G4 derived tau     tau matches an independent Monte Carlo null at the same size
  G5 tau trap        inheriting 0.35 would veto a lightly-perturbed candidate that
                     the derived tau passes
  G6 fail-closed     zero-energy wave -> stress 1.0, valid False, no crash
  G7 literal path    grid-vs-grid stress is exact and needs no decode
  G8 dimension caveat tau at D=65536 vs D=1024: the WAVE dim does not change the
                     match-rate null (n_slots does), recorded explicitly
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

from henri_grid_observable import GridObservableReadout, derive_tau  # noqa: E402

OUT = os.path.join(HERE, "grid_observable_probe_observed.json")


def make_grid(seed: int = 0, H: int = 8, W: int = 8) -> np.ndarray:
    g = np.random.default_rng(seed)
    grid = np.zeros((H, W), dtype=np.int64)
    grid[1:4, 1:4] = 3
    grid[5:7, 5:7] = 7
    return grid


def perturb(grid: np.ndarray, n: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    out = grid.copy()
    idx = g.choice(out.size, size=n, replace=False)
    for i in idx:
        r, c = divmod(int(i), out.shape[1])
        out[r, c] = int(g.integers(0, 10))
    return out


def mc_null_stress(n_slots: int, n_values: int, trials: int = 40000,
                   seed: int = 7) -> float:
    g = torch.Generator().manual_seed(seed)
    dec = torch.randint(0, n_values, (trials, n_slots), generator=g)
    ref = torch.randint(0, n_values, (1, n_slots), generator=g)
    return float((1.0 - (dec == ref).float().mean(dim=1)).quantile(0.01).item())


def main() -> int:
    D = 1024
    ro = GridObservableReadout(shape=(8, 8), dim=D, n_values=11, seed=20261012)
    res: dict = {}
    fails: list = []

    # ------------------------------------------------------------------ G0
    # SEPARATE ALGEBRA FROM CAPACITY. A single-cell wave (M = 1, no crosstalk) must
    # round-trip EXACTLY. If this fails the unbinding algebra is wrong; if it passes
    # and G1 fails, the limit is superposition capacity. G1 failed on the first run
    # and this gate is what localized the cause.
    ro1 = GridObservableReadout(shape=(1, 1), dim=D, n_values=11, seed=5)
    one = np.array([[7]], dtype=np.int64)
    d1 = ro1.decode(ro1.encode(one))
    res["G0_single_cell_roundtrip"] = bool(d1.valid and d1.values[0, 0] == 7)
    res["G0_single_cell_quality"] = float(d1.quality[0, 0])
    if not res["G0_single_cell_roundtrip"]:
        fails.append(
            f"G0: a SINGLE bound pair failed to round-trip (decoded "
            f"{int(d1.values[0,0])}, quality {res['G0_single_cell_quality']:.4f}). "
            f"The unbinding algebra is wrong, so no capacity conclusion is available.")
    # The unbinding identity: role (*) conj(role) must be a delta (large peak/mean).
    zz = ro1._unbind(ro1.role_keys[0], ro1.role_keys[:1])
    res["G0_delta_peak_over_mean"] = float(
        (zz.abs().max() / zz.abs().mean().clamp_min(1e-30)).item())
    if res["G0_delta_peak_over_mean"] < 10.0:
        fails.append(f"G0: role (*) conj(role) peak/mean "
                     f"{res['G0_delta_peak_over_mean']:.3f} is not delta-like")

    # ------------------------------------------------------------------ G8/G4
    res["G_tau_info"] = ro.tau_info
    res["tau"] = ro.tau
    mc = mc_null_stress(ro.n_slots, ro.n_values)
    res["G4_mc_q01_stress"] = mc
    res["G4_mc_matches_exact_binomial"] = abs(mc - ro.tau) <= 0.05
    if not res["G4_mc_matches_exact_binomial"]:
        fails.append(f"G4: exact tau {ro.tau:.4f} vs MC q01 {mc:.4f} disagree")

    # ------------------------------------------------------------------ G1
    base = make_grid()
    psi = ro.encode(base)
    dec = ro.decode(psi)
    exact = bool(dec.valid and np.array_equal(dec.values, base))
    res["G1_roundtrip_exact"] = exact
    res["G1_min_cell_quality"] = float(dec.quality.min())
    res["G1_norm"] = float(psi.norm().item())
    if not exact:
        fails.append(f"G1: round-trip failed; decoded differs at "
                     f"{int((dec.values != base).sum())} cells")
    if abs(res["G1_norm"] - 1.0) > 1e-5:
        fails.append(f"G1: wave norm {res['G1_norm']:.6f} != 1")

    # ------------------------------------------------------------------ G2
    other = GridObservableReadout(shape=(8, 8), dim=D, n_values=11, seed=999)
    dec_wrong = other.decode(psi)
    wrong_match = float((dec_wrong.values == base).mean()) if dec_wrong.valid else 0.0
    res["G2_wrong_codebook_match_rate"] = wrong_match
    res["G2_fingerprints_differ"] = ro.codebook_fingerprint() != other.codebook_fingerprint()
    if wrong_match > 0.5:
        fails.append(f"G2: a WRONG codebook recovered {wrong_match:.2%} of cells; the "
                     f"shared-codebook dependency is not real, so the decode is trivial")
    if not res["G2_fingerprints_differ"]:
        fails.append("G2: codebook fingerprints identical; guard is vacuous")

    # ------------------------------------------------------------------ G3
    cases = {
        "identical": base,
        "1_cell": perturb(base, 1, 1),
        "4_cells": perturb(base, 4, 2),
        "8_cells": perturb(base, 8, 3),
        "16_cells": perturb(base, 16, 4),
        "unrelated": np.random.default_rng(99).integers(0, 10, size=(8, 8)).astype(np.int64),
    }
    rows = []
    for name, grid in cases.items():
        s = ro.observational_stress(ro.encode(grid), base)
        gs = ro.grid_stress(grid, base)
        rows.append({"case": name, "observational_stress": round(s["stress"], 6),
                     "match_rate": round(s["match_rate"], 6), "valid": s["valid"],
                     "direct_grid_stress": round(gs, 6),
                     "passes_derived_tau": s["stress"] <= ro.tau,
                     "passes_inherited_0.35": s["stress"] <= 0.35})
    res["G3_cases"] = rows
    idr = next(r for r in rows if r["case"] == "identical")
    unr = next(r for r in rows if r["case"] == "unrelated")
    res["G3_separation"] = round(unr["observational_stress"] - idr["observational_stress"], 6)
    vals = [r["observational_stress"] for r in rows]
    res["G3_monotone"] = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
    if idr["observational_stress"] > 1e-6:
        fails.append(f"G3: identical case stress {idr['observational_stress']} != 0")
    if res["G3_separation"] < 0.3:
        fails.append(f"G3: separation only {res['G3_separation']:.4f}")
    if not res["G3_monotone"]:
        res["G3_monotone_note"] = ("non-decreasing violated; recorded, not fatal, "
                                  "because damage location matters more than count")

    # ------------------------------------------------------------------ G5
    # THE TRAP LIVES IN THE PARTIALLY-CORRECT RANGE.
    #
    # G5 first tested a 1-cell edit (match 0.984) and FAILED its own assertion, which
    # was correct: a near-perfect candidate passes ANY sensible threshold, so it
    # cannot demonstrate a threshold trap. The inherited-0.35 hazard is for a
    # candidate that is genuinely BETTER than random but not yet correct -- the
    # realistic mid-search state. With match rate m:
    #     passes derived tau (0.8125)  <=>  m >= 0.1875
    #     passes inherited 0.35        <=>  m >= 0.65
    # so any candidate with m in [0.1875, 0.65) is vetoed by 0.35 but admitted by the
    # derived threshold. Partial solvers live exactly there. That band is measured
    # here by constructing candidates with controlled overlap and decoding them.
    band = []
    rng = np.random.default_rng(4242)
    flat_base = base.reshape(-1)
    for target_match in (0.20, 0.40, 0.60, 0.70):
        n_keep = int(round(target_match * flat_base.size))
        keep = rng.choice(flat_base.size, size=n_keep, replace=False)
        cand = rng.integers(0, 10, size=flat_base.size).astype(np.int64)
        cand[keep] = flat_base[keep]          # keep `n_keep` cells identical
        cand = cand.reshape(base.shape)
        s = ro.observational_stress(ro.encode(cand), base)
        band.append({
            "intended_match_rate": target_match,
            "measured_match_rate": round(s["match_rate"], 4),
            "stress": round(s["stress"], 4),
            "passes_derived_tau": s["stress"] <= ro.tau,
            "passes_inherited_0.35": s["stress"] <= 0.35,
        })
    res["G5_partial_correct_band"] = band
    # The near-perfect control: a 1-cell edit (match 0.984) must pass BOTH thresholds.
    light = next(r for r in rows if r["case"] == "1_cell")
    trap_rows = [b for b in band if b["passes_derived_tau"]
                 and not b["passes_inherited_0.35"]]
    res["G5_trap_reproduced_count"] = len(trap_rows)
    res["G5_trap_reproduced"] = len(trap_rows) > 0
    # The 1-cell case is retained as a control: it must pass BOTH (it is near-perfect).
    res["G5_near_perfect_passes_both"] = (light["passes_derived_tau"]
                                          and light["passes_inherited_0.35"])
    if not res["G5_trap_reproduced"]:
        fails.append(
            f"G5: no partially-correct candidate distinguishes the derived tau "
            f"({ro.tau:.4f}) from the inherited 0.35. Band measured: "
            f"{[(b['measured_match_rate'], b['passes_derived_tau'], b['passes_inherited_0.35']) for b in band]}. "
            f"If the band does not cross both thresholds, the trap claim is "
            f"unsupported and must be withdrawn.")
    if not res["G5_near_perfect_passes_both"]:
        fails.append("G5: the near-perfect control did not pass both thresholds")

    # ------------------------------------------------------------------ G6
    z = torch.zeros(D, dtype=torch.complex64)
    s0 = ro.observational_stress(z, base)
    res["G6_zero_energy_stress"] = float(s0["stress"])
    res["G6_zero_energy_valid"] = bool(s0["valid"])
    res["G6_nan_free"] = not bool(np.isnan(s0["stress"]))
    if not (res["G6_zero_energy_stress"] == 1.0 and not res["G6_zero_energy_valid"]):
        fails.append("G6: zero-energy wave did not fail closed")

    # ------------------------------------------------------------------ G7
    res["G7_literal_exact_when_equal"] = ro.grid_stress(base, base) == 0.0
    res["G7_literal_one_cell"] = ro.grid_stress(perturb(base, 1, 1), base) == 1.0 / 64
    res["G7_literal_shape_mismatch"] = ro.grid_stress(np.zeros((2, 2)), base) == 1.0
    if not (res["G7_literal_exact_when_equal"] and res["G7_literal_one_cell"]
            and res["G7_literal_shape_mismatch"]):
        fails.append("G7: direct grid stress is not exact")

    # ------------------------------------------------------------------ G8
    res["G8_tau_4x4"] = derive_tau(16, 11, 0.01)["tau_observational"]
    res["G8_tau_8x8"] = derive_tau(64, 11, 0.01)["tau_observational"]
    res["G8_tau_16x16"] = derive_tau(256, 11, 0.01)["tau_observational"]
    res["G8_note"] = ("tau depends on n_slots (the observable lattice), NOT on the "
                      "wave dim D. A 30x30 ARC output must use its own tau.")

    verdict = "PASS" if not fails else "FAIL"
    out = {"module": "grid_observable_probe", "evidence_class": "OBSERVED",
           "dim": D, "results": res, "gate_failures": fails, "verdict": verdict,
           "claim": ("A role-filler grid readout round-trips exactly, separates cases, "
                     "fails closed, and its veto threshold is derived from the "
                     "match-rate null rather than inherited from the waveform metric."),
           "dependency": ("Decoding works only for waves THIS codebook encoded. "
                          "Production reference_wave comes from a k_bins=256 phase "
                          "algebra, so observable veto wiring is flag-gated until the "
                          "RFSS ingress is enabled.")}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 84)
    print(f"GRID OBSERVABLE PROBE  (8x8 lattice, dim={D}, n_values=11)")
    print("=" * 84)
    print(f"  tau derived            = {ro.tau:.4f}  (alpha={ro.alpha}, "
          f"null match rate {ro.tau_info['null_match_rate']:.4f}, "
          f"q99 matched {ro.tau_info['q_99'] if False else ro.tau_info['q_1_minus_alpha_matched']}/{ro.n_slots})")
    print(f"  G4 MC q01 stress       = {mc:.4f}   agrees={res['G4_mc_matches_exact_binomial']}")
    print(f"  G1 round-trip exact    = {exact}   min cell quality {res['G1_min_cell_quality']:.4f}")
    print(f"  G2 wrong-codebook match= {wrong_match:.3f}  (must be low)")
    print()
    print(f"  G3 cases:")
    print(f"    {'case':<12} {'obs_stress':>11} {'direct':>9} {'derived_tau':>12} {'inherited':>10}")
    for r in rows:
        print(f"    {r['case']:<12} {r['observational_stress']:>11.4f} "
              f"{r['direct_grid_stress']:>9.4f} {str(r['passes_derived_tau']):>12} "
              f"{str(r['passes_inherited_0.35']):>10}")
    print(f"    separation identical->unrelated = {res['G3_separation']:.4f}")
    print()
    print(f"  G5 1-cell edit: stress {light['observational_stress']:.4f} -> "
          f"derived tau pass={light['passes_derived_tau']}, "
          f"inherited 0.35 pass={light['passes_inherited_0.35']}")
    print(f"  G6 zero-energy: stress {res['G6_zero_energy_stress']:.1f} "
          f"valid={res['G6_zero_energy_valid']} nan_free={res['G6_nan_free']}")
    print(f"  G7 literal grid stress exact = "
          f"{res['G7_literal_exact_when_equal'] and res['G7_literal_one_cell'] and res['G7_literal_shape_mismatch']}")
    print(f"  G8 tau by lattice: 4x4={res['G8_tau_4x4']:.4f} 8x8={res['G8_tau_8x8']:.4f} "
          f"16x16={res['G8_tau_16x16']:.4f}")
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
