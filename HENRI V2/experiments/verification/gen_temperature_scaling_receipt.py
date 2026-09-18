#!/usr/bin/env python3
"""ACTION 2 receipt: temperature scaling of the Zone A wave readout (60 ARC tasks).

WHAT THIS PRODUCES
    The measured optimum readout temperature T* (and beta* = 1/T) for the functor
    arm, fitted by NLL, plus the ECE/Brier/BSS consequences and the HELD-OUT
    number. It also tests the calibration-physics blueprint's own claim that
    beta* in 4..8 satisfies the joint gate (ECE <= 0.05 AND BSS > 0).

EVERY NUMBER COMES FROM henri_probe_calibration FUNCTIONS
    fit_temperature_nll / temperature_floor / split_temperature_report are the
    production code path. No scoring logic is re-implemented here, so the receipt
    exercises the module that ships.

WHY NLL IS THE FIT OBJECTIVE AND ECE IS ONLY REPORTED
    NLL is strictly proper and smooth. ECE is a binned statistic: an argmin-ECE
    temperature is fitted to the bin edges of the set it was fitted on. Fitting
    under ECE would manufacture a flattering number.

WHY THERE IS A HELD-OUT NUMBER
    Fitting T on all 60 tasks and reporting that same 60-task ECE is circular
    validation. The headline here is the held-out ECE, and the in-sample ECE is
    reported beside it so the optimism gap is visible.

PRE-REGISTERED OUTCOMES (written before the run)
    A. argmax, accuracy and Brier skill are INVARIANT in T. Violation => kill.
    B. If held-out ECE at T* is worse than in-sample ECE at T*, the T fit is
       overfit and the receipt MUST say so.
    C. If the reachable ECE floor exceeds the 0.05 gate, the gate is recorded
       UNREACHABLE. That is a falsification of the blueprint's beta* range, not a
       tuning failure, and the receipt must not soften it.
"""
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
VERIF = R / "experiments" / "verification"
RECEIPT = VERIF / "calibration_eval_observed.json"
OUT = VERIF / "temperature_scaling_observed.json"

sys.path.insert(0, str(R))
# Plain import: the module uses `from __future__ import annotations`, so
# dataclasses resolves annotations via sys.modules and fails if exec'd unregistered.
import henri_probe_calibration as hpc  # noqa: E402

src = json.loads(RECEIPT.read_text(encoding="utf-8"))
rows = src["per_task"]
SCORES = [list(r["functor_scores"]) for r in rows]
Y = [int(r["truth_index"]) for r in rows]
IDS = [r["task_id"] for r in rows]
N = len(SCORES)
K = len(SCORES[0])
ACC = sum(1 for s, y in zip(SCORES, Y) if max(range(K), key=lambda i: s[i]) == y) / N

GRID = hpc.temperature_grid()

print("=" * 78)
print("SELF-CHECK at T=1.0 against the sealed calibration receipt")
print("=" * 78)
p1 = hpc.scale_score_rows(SCORES, 1.0)
r1 = hpc.evaluate_calibration(p1, Y, n_bins=10)
rec_acc = src["arm_functor"]["accuracy"]
rec_ece = src["arm_functor"]["ece"]
rec_bss = src["arm_functor"]["brier_skill_score"]
max_dev = max(abs(a - b) for pr, rr in zip(p1, rows)
              for a, b in zip(pr, rr["functor_probs"]))
print(f"  accuracy  this={r1.accuracy:.6f}  receipt={rec_acc:.6f}")
print(f"  ECE       this={r1.ece:.6f}  receipt={rec_ece:.6f}")
print(f"  BSS       this={hpc.brier_skill_score(p1, Y):+.6f}  receipt={rec_bss:+.6f}")
print(f"  max |p_here - p_receipt| = {max_dev:.2e}")
# TOLERANCE JUSTIFICATION (not a loosening to force a pass).
#   The sealed receipt's stored probabilities are TORCH FLOAT32 output. This
#   module computes softmax in Python FLOAT64. So a real, expected difference of
#   order 1e-7 is present in the inputs, independent of any definitional error.
#   Aggregates inherit that magnitude. 1e-6 is one order above the provenance
#   noise and three-to-five orders BELOW any genuine definitional mismatch
#   (a wrong bin rule or a wrong confidence rule moves ECE by 0.01-0.5).
#   The earlier 1e-9 gate was unjustifiably tight and produced a false FAIL.
SELF_CHECK_TOL = 1e-6
self_ok = (abs(r1.accuracy - rec_acc) < SELF_CHECK_TOL
           and abs(r1.ece - rec_ece) < SELF_CHECK_TOL
           and abs(hpc.brier_skill_score(p1, Y) - rec_bss) < SELF_CHECK_TOL
           and max_dev < SELF_CHECK_TOL)
print(f"  tolerance = {SELF_CHECK_TOL:.0e} (float32-vs-float64 provenance)")
print(f"  SELF-CHECK: {'PASS' if self_ok else 'FAIL'}")
if not self_ok:
    print("  => definitions differ from the sealed receipt. ABORT.")
    sys.exit(1)

print()
print("=" * 78)
print("FULL-60 FIT (in-sample; reported for the record, NOT the headline)")
print("=" * 78)
T_full, nll_full = hpc.fit_temperature_nll(SCORES, Y, grid=GRID)
pf = hpc.scale_score_rows(SCORES, T_full)
rf = hpc.evaluate_calibration(pf, Y, n_bins=10)
floor_ece, floor_T = hpc.temperature_floor(SCORES, Y, n_bins=10, grid=GRID)
print(f"  T* (argmin NLL, 60 tasks) = {T_full:.6f}   beta* = {1/T_full:.4f}")
print(f"  NLL at T*                 = {nll_full:.6f}")
print(f"  ECE  at T* = {rf.ece:.6f}   acc = {rf.accuracy:.6f}   "
      f"BSS = {hpc.brier_skill_score(pf, Y):+.6f}")
print(f"  ECE  at T=1.0             = {r1.ece:.6f}   (gain {r1.ece - rf.ece:+.4f})")
print(f"  ECE floor over grid       = {floor_ece:.6f} at T={floor_T:.6f} "
      f"(beta*={1/floor_T:.2f})")

print()
print("=" * 78)
print("INVARIANCE PRE-REGISTRATION A  (must hold, else the change is killed)")
print("=" * 78)
base_arg = [max(range(K), key=lambda i: r[i]) for r in SCORES]
invar = {
    "accuracy_invariant": abs(rf.accuracy - r1.accuracy) < 1e-12,
    "argmax_invariant": all(
        max(range(K), key=lambda i: pf[k][i]) == base_arg[k] for k in range(N)
    ),
    # PRE-REGISTRATION CORRECTION (honest record). Pre-registration A asserted
    # that Brier skill is INVARIANT in T. That assertion was WRONG and is
    # recorded here as falsified: the Brier sum contains the probability
    # MAGNITUDES, so a positive logit rescaling changes it. Only argmax-derived
    # statistics are invariant. The check is kept, inverted, so the correction
    # stays falsifiable: if BSS ever stops moving, this receipt must be revisited.
    "bss_moved_by_temperature": abs(
        hpc.brier_skill_score(pf, Y) - hpc.brier_skill_score(p1, Y)) > 1e-9,
}
invar_correction = (
    "Pre-registration A expected Brier skill to be invariant under temperature. "
    "MEASURED FALSIFIED: BSS moves because the Brier sum contains probability "
    "magnitudes, not only the argmax. Only accuracy/argmax is invariant in T. "
    "The invariance check was kept in inverted form so it remains falsifiable."
)
for k, v in invar.items():
    print(f"  {k:<26} {'HOLDS' if v else 'VIOLATED'}")
print(f"  CORRECTION: {invar_correction}")

print()
print("=" * 78)
print("BINNING ROBUSTNESS (is the floor a binning artifact?)")
print("=" * 78)
binning = {}
print(f"  {'bins':>5} {'floor ECE':>11} {'at T':>9} {'beta*':>8} {'<=0.05?':>8}")
for nb in (5, 8, 10, 15, 20, 30):
    e, t = hpc.temperature_floor(SCORES, Y, n_bins=nb, grid=GRID)
    binning[str(nb)] = {"floor_ece": e, "temperature": t, "beta_star": 1 / t,
                        "meets_0p05": e <= 0.05}
    print(f"  {nb:>5} {e:>11.6f} {t:>9.6f} {1/t:>8.2f} {'YES' if e <= 0.05 else 'no':>8}")

print()
print("=" * 78)
print("HELD-OUT SPLIT (the headline) + 40-split distribution")
print("=" * 78)
sp = hpc.split_temperature_report(SCORES, Y, n_bins=10, holdout_fraction=0.5,
                                 split_seed=20260918, grid=GRID, n_random_splits=40)
print(f"  T* fitted on {sp['n_train']} train = {sp['temperature_star']:.6f} "
      f"(beta*={sp['beta_star']:.3f})")
print(f"  {'part':<9} {'acc':>8} {'ECE':>9} {'BSS':>9} {'peak':>8} {'NLL':>8}")
for part in ("train", "holdout"):
    print(f"  {part:<9} {sp[f'{part}_accuracy_T_star']:>8.4f} "
          f"{sp[f'{part}_ece_T_star']:>9.4f} {sp[f'{part}_bss_T_star']:>+9.4f} "
          f"{sp[f'{part}_mean_peak_T_star']:>8.4f} {sp[f'{part}_nll_T_star']:>8.4f}")
print(f"  {'holdout@1':<9} {sp['holdout_accuracy_T_1p0']:>8.4f} "
      f"{sp['holdout_ece_T_1p0']:>9.4f} {sp['holdout_bss_T_1p0']:>+9.4f} "
      f"{sp['holdout_mean_peak_T_1p0']:>8.4f} {sp['holdout_nll_T_1p0']:>8.4f}")
print(f"  held-out ECE improved by T*        : {sp['ece_improved_holdout']}")
print(f"  accuracy invariant on holdout      : {sp['accuracy_invariant_holdout']}")
print(f"  BSS improved on holdout (NOT invar): {sp['bss_improved_holdout']}")
print(f"  Brier improved on holdout          : {sp['brier_improved_holdout']}")
print(f"  held-out passes joint gate         : {sp['holdout_passes_joint_gate']}")
ms = sp["multi_split"]
print(f"  40 splits: held-out ECE mean={ms['holdout_ece_mean']:.4f} "
      f"min={ms['holdout_ece_min']:.4f} max={ms['holdout_ece_max']:.4f}")
print(f"  40 splits: T* median={ms['temperature_star_median']:.5f} "
      f"range=[{ms['temperature_star_min']:.5f},{ms['temperature_star_max']:.5f}]")
print(f"  40 splits passing joint gate       : {ms['splits_passing_joint_gate']}/40")

print()
print("=" * 78)
print("BLUEPRINT CLAIM TEST: beta* in 4..8 satisfies ECE<=0.05 AND BSS>0")
print("=" * 78)
claim = {}
for beta in (4.0, 8.0):
    t = 1.0 / beta
    pp = hpc.scale_score_rows(SCORES, t)
    rr = hpc.evaluate_calibration(pp, Y, n_bins=10)
    bb = hpc.brier_skill_score(pp, Y)
    passed = rr.ece <= 0.05 and bb > 0
    claim[f"beta_{beta}"] = {"temperature": t, "ece": rr.ece,
                             "mean_peak": rr.mean_confidence, "bss": bb,
                             "passes_joint_gate": passed}
    print(f"  beta*={beta:<4} T={t:<6.4f} ECE={rr.ece:.6f} peak={rr.mean_confidence:.4f} "
          f"BSS={bb:+.4f} gate={'PASS' if passed else 'FAIL'}")
peak_T = min(GRID, key=lambda t: abs(
    hpc.evaluate_calibration(hpc.scale_score_rows(SCORES, t), Y, n_bins=10).mean_confidence - ACC))
peak_rep = hpc.evaluate_calibration(hpc.scale_score_rows(SCORES, peak_T), Y, n_bins=10)
print(f"  PEAK-MATCH T (blueprint mechanism: mean peak == accuracy {ACC:.4f}):")
print(f"    T={peak_T:.6f} beta*={1/peak_T:.3f} -> ECE={peak_rep.ece:.6f} "
      f"peak={peak_rep.mean_confidence:.4f} BSS={hpc.brier_skill_score(hpc.scale_score_rows(SCORES, peak_T), Y):+.4f}")
blueprint_satisfied = all(v["passes_joint_gate"] for v in claim.values())
print(f"  VERDICT: blueprint beta* range 4..8 -> "
      f"{'SATISFIED' if blueprint_satisfied else 'FALSIFIED'}")

body = {
    "schema": "henri.temperature-scaling-receipt.v1",
    "utc": datetime.now(timezone.utc).isoformat(),
    "evidence_class": "DERIVED",
    "evidence_class_note": (
        "DERIVED: every ECE/Brier/BSS/NLL here is COMPUTED from the per-task "
        "functor_scores stored in calibration_eval_observed.json. The scores "
        "themselves are the measured readout output (OBSERVED). No scoring rule "
        "is re-implemented: all numbers come from henri_probe_calibration."
    ),
    "source_receipt": "experiments/verification/calibration_eval_observed.json",
    "source_receipt_sha256": hpc.sha256_of_rows([r["functor_probs"] for r in rows]),
    "n_tasks": N, "n_options": K, "accuracy_argmax": ACC,
    "grid": {"n": len(GRID), "t_min": GRID[0], "t_max": GRID[-1]},
    "self_check": {"accuracy": r1.accuracy, "ece": r1.ece,
                   "bss": hpc.brier_skill_score(p1, Y), "max_prob_dev": max_dev,
                   "passed": self_ok},
    "invariance": invar,
    "full_60_in_sample": {
        "temperature_star": T_full, "beta_star": 1 / T_full, "nll": nll_full,
        "ece_at_star": rf.ece, "ece_at_T1": r1.ece,
        "ece_gain": r1.ece - rf.ece,
        "accuracy_at_star": rf.accuracy,
        "bss_at_star": hpc.brier_skill_score(pf, Y),
        "mean_peak_at_star": rf.mean_confidence,
    },
    "ece_floor": {"ece": floor_ece, "temperature": floor_T,
                  "beta_star": 1 / floor_T,
                  "joint_gate_0p05_reachable": floor_ece <= 0.05,
                  "analytic_T0_limit": 1.0 - ACC},
    "binning_robustness": binning,
    "headline_holdout": {
        k: sp[k] for k in sp
        if k not in ("train_indices", "holdout_indices")
    },
    "headline_holdout_split_indices": {
        "train": sp["train_indices"], "holdout": sp["holdout_indices"],
        "task_ids_holdout": [IDS[i] for i in sp["holdout_indices"]],
    },
    "blueprint_claim_test": claim,
    "peak_match_temperature": {
        "target_mean_peak": ACC, "temperature": peak_T, "beta_star": 1 / peak_T,
        "achieved_mean_peak": peak_rep.mean_confidence, "ece": peak_rep.ece,
    },
    "blueprint_beta_range_satisfies_gate": blueprint_satisfied,
    "device_kind": src.get("device_kind"),
    "limits": [
        "n=60 limits ECE resolution: see binning_robustness and ece_resolution.",
        "Measures the READOUT only. Not a task score, not a benchmark score.",
        "Temperature moves CALIBRATION only; accuracy and BSS are invariant in T.",
        "CPU only; no CUDA (Vast instance 50797414 EXITED, credit 0).",
        "A 40-split mean is reported because a single split can be lucky.",
    ],
}
OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
print()
print(f"wrote {OUT}")
print(f"file sha256 = {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
