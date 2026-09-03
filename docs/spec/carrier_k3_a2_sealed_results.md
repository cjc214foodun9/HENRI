# Carrier K3 — A2 Measurement Instrument: Sealed Results

Document Identifier: `HENRI-SPEC-2026-09-V3-CARRIER-K3-A2-SEALED-RESULTS`
Date: 2026-09-03
Branch: `feat/carrier-k3-empirical-koopman`
Instrument commit: `f7ff6e5` (probe SHA-256 `d5daabc7…`, 5,802 B; prereg `docs/spec/carrier_k3_a2_measurement_instrument.md`)
Remote: vast-5090, dispatch tree `/workspace/henri-k3-dispatch` @ `ddc950f`; digest gate PASS; GPU idle at launch (2 MiB / 32,607 MiB, 0 %, no CUDA apps).
Run: `/venv/main/bin/python /tmp/a2/k3_a2_decomp.py` → RC=0, verdict line `A2_DECOMP_DONE`.
Runfile: local `a2run1.txt`, SHA-256 `7febf341f74ae911a85626868860bbd9c1955f61d9a67923ce42c6551ba5a08c` (849 B).

## Verdict

**A2 PARTIAL.** Five of six acceptance anchors PASS. The composite-straddle
anchor FAILED → the composite timed-region site is dispositioned
`BLOCKED_INSTRUMENT` (support-coverage mismatch, below). The measured-dominant
site of the sealed KG5 2.4738 ms mean is identified: the spectral screen.

## Measurements (CUDA-event median of 15 after 3 warmups, seed 20260904)

| Site | n=128 | n=248 |
|---|---|---|
| accum_ms | 0.3506 | 0.6296 |
| solve_ms (accum + cholesky ridge) | 0.4635 | 0.7568 |
| screen_ms (`_screen_sigma_max`, 2-start power + top-64) | 1.392 | 1.400 |
| full_fit_ms (`BlockRidgeKoopmanFit.fit` = accum+solve+screen+scale+post) | 1.9838 | 2.2898 |

| Site | ms |
|---|---|
| apply_ms | 0.0204 |
| affordance_g7_predict_ms (full support: 7 actions × 64 blocks) | 10.9629 |
| composite_region_1fit_ms | 12.9553 |
| composite_region_2fit_ms | 15.6915 |

## Anchor evaluation

- screen ≥ 0.650 (old single-start lower bound): PASS — 1.392 / 1.400, i.e.
  2.14–2.15× the old lower bound. The production two-start screen is
  n-independent (per-K).
- apply ∈ [0.010, 0.040]: PASS — 0.0204.
- full_fit ≥ solve + screen − 0.3 at same n: PASS — 1.9838 ≥ 1.5555;
  2.2898 ≥ 1.8568.
- composite straddle of sealed mean 2.4738: **FAIL** — 1-fit composite 12.9553
  > 2.4738 + 0.7. 2-fit composite 15.6915 ≥ 1.7738 PASS (one-sided only).

## Interpretation

- **Screen is the measured dominant site.** 1.392–1.400 ms per fit = ~57 % of
  the sealed 2.4738 ms timed-step mean at one fit per timed step. Engine refit
  cadence (137 fit calls / 980 K3 score calls, sealed results doc) concentrates
  screen cost on refit steps.
- **Solve is sub-dominant** (0.4635–0.7568 ms at full fill incl. accum; smaller
  at engine fill). The A1-class solve swap is confirmed NOT the lever (A1
  FALSIFIED at 18.85 ms; torch solve ceiling here 0.76 ms).
- **Apply is negligible** (0.0204 ms).
- **Affordance at full support is 10.9629 ms = 24.5 µs per populated block**
  (448 = 7 × 64). This is an UPPER BOUND: the sealed 2.4738 ms mean implies the
  engine's timed steps ran sparse per-action support (~2–4 blocks/action;
  INFERRED from the aggregate mean minus measured components — engine store
  coverage is not in sealed telemetry).
- **Composite overshoot is OBSERVED, not engine cost.** 1-fit composite 12.96 ms
  and 2-fit 15.69 ms both far exceed the sealed mean → the probe's
  `transitions_g4` over-population (64 blocks × 7 actions) does not mirror the
  live store at the measured steps → `BLOCKED_INSTRUMENT` on the composite site
  only. Per the sealed kill criteria this is a probe-fixture mismatch, not a
  mechanism verdict; all single-site measurements stand (their anchors passed).

## Remedy scope (constrains any follow-on carrier)

- The spectral screen `_screen_sigma_max` is the measured dominant site
  (≥ 0.5 ms, ≥ ~40 % of the sealed band). Any screen-targeted remedy is a NEW
  carrier requiring its own sealed prereg + approval before remote execution.
- The solve swap is NOT re-opened (measured sub-dominant).
- KG5' (solve-only component accuracy vs torch ≤ 1e-5, sealed `51273a0`)
  remains separate and uninstrumented; this run measured latency, not accuracy.

## Disclosed limitations

1. Composite-site rerun would require a coverage-mirroring fixture (sparse
   per-action support matching the live store at measured steps + the engine's
   fit_n distribution) under a fresh instrument amendment — NOT executed.
2. Engine per-action support at timed steps is INFERRED (not directly
   observed; sealed telemetry carries only the aggregate mean).
3. Local-only commit; no push; zero code change to the dispatch tree or
   worktree. The co-scientist claim-clip fired: the 12.96 ms composite was
   classified BLOCKED_INSTRUMENT, not reported as engine timed-region cost.
