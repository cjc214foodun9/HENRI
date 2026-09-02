# Carrier G2 — Local Block-Gauge Affordance: Results

**Directive:** user message (2026-09-01) + `Example code.pdf`
(`HENRI-EVAL-2026-09-V3-G1-FALSIFICATION-AUDIT`, 175,315 B, SHA `9d36971c…`).
**Prereg:** `docs/spec/g2_local_block_gauge_preregistration.md` (SHA `023ece117457…`, sealed `#671ae4ac…` @1,161).
**Branch:** `carrier/g2-local-block-gauge`; tested code @ `23dab20`.

## Verdict

**`G2_AFFORDANCE_FIT_COLLAPSE` — PG1 pre-flight kill, 0 live steps. 23rd sealed falsification (22 prior carriers, 0 solved).**

Gate PG1 (AUC ≥ 0.8800 across all 7 actions): **min_auc 0.4512** ✗ (binding gate; Failure Action = PRE-FLIGHT KILL). G1–G4 NOT RUN.

| action | n | moving rate | AUC |
|---|---|---|---|
| 0 | 267 | 0.210 | 0.7139 ✗ |
| 1 | 257 | 0.218 | 0.6906 ✗ |
| 2 | 244 | 0.197 | 0.7396 ✗ |
| 3 | 290 | 0.224 | 0.6762 ✗ |
| 4 | 64 | 0.234 | 1.0000 ✓ |
| 5 | 361 | 0.205 | 0.8643 ✗ |
| 6 | 53 | 0.189 | 0.4512 ✗ |

All actions ≥ 53 samples (min gate 10); label counts match the prereg calibration (0.19–0.23 moving per action; calibration probe 0.19–0.234) — **labels verified, kill genuine.**

## Launch history (all identical preregistered bounds: seed 20260925, 12 envs × 150 steps, τ_stall 0.90, β=10, τ_base 0.05)

1. **`2445cc1` — harness defect** (`G2_HARNESS_DEFECT_LABEL_NORM`): bank `psi` not unit-norm (‖·‖ 14–22 vs next_wave 1.0); raw-dot label → 0.8% positives vs true 21%. Receipt `7ab641a4…` (prior). Fixed: norm-invariant label.
2. **`0e5262b` — harness defect** (`G2_LABEL_GEOMETRY_DRIFT`): per-block renormalization changed the metric to mean-of-block-cosines → label balance drifted to 0–73% moving, min_auc 0.5. Receipt `g2_gates_receipt_r2.json`. Fixed: flat norm-divided cosine on raw waves (scale-invariant).
3. **`23dab20` — GENUINE kill** (sealed): flat labels verified (0.210/0.218/0.197/0.224/0.234/0.205/0.189), min_auc 0.4512. Receipt SHA `9563ed74…`, log SHA `d0148a47…`.

## Mechanism finding (DERIVED)

- **Affordance signal exists:** actions 4–5 separate (AUC 1.0 / 0.86); the moving-vs-blocked axis is real.
- **The full-D per-block quadratic metric FAILS on heterogeneous actions:** a0–a3 (0.68–0.74) and a6 (0.45). The bridge (G1 D=64) achieves min 0.7865 and **a6 0.9224 vs full-D 0.4512** — the D=64 pipeline beats the full-D quadratic for the most heterogeneous action.
- **Root cause hypothesis:** the fit is a mean-over-blocks quadratic (1/N Σ φφᵀ), the score is β=10 softmax pooling over 8,192 blocks — **different functionals**. Max-emphasizing pooling amplifies noise blocks for heterogeneous actions (a6 worst: 53 samples, 4:1 blocked). Also: a single global β and per-action τ from a closed-form std do not calibrate 7 different block-activation distributions.

## Next-carrier levers (NOT authorized; require a new directive/prereg)

1. **Score with the same functional as the fit** (mean pooling, β→0) or fit with the max-emphasizing functional — eliminate the fit/score mismatch.
2. **Per-action block selection** (top-k blocks by action-specific energy, not global softmax over all 8,192).
3. **Stall-cosine label stays** (verified calibration; the axis is correct).
4. **`holographic search.pdf` (G3 candidate)** — `WavePacketPathSearch` replaces sequential MCTS; load-bearing (planner replacement) → REQUIRES_APPROVAL before implementation.

## Evidence

- Receipt: `C:/Users/chan/AppData/Local/Temp/g2_gates_receipt_r3.json` (SHA `9563ed745f96…`)
- Log: `C:/Users/chan/AppData/Local/Temp/g2_launch_r3.log` (SHA `d0148a47205c…`)
- Remote: `/tmp/henri_g2_local_gauge/g2_gates_receipt.json`, `g2_launch.log`
- Engine: `HENRI V2/experiments/verification/arc_g2_local_gauge_engine.py` @ `23dab20`; tests `test_g2_local_gauge_engine.py` (13/13 local + remote CUDA).
- Regression: 1,055 passed / 6 skipped (G2 +12, no regressions).
