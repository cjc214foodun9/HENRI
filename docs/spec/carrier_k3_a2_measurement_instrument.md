# Carrier K3 — A2 Measurement Instrument (Sealed 2026-09-03)

Status: SEALED (pre-registration)
ID: HENRI-SPEC-2026-09-V3-CARRIER-K3-A2-INSTRUMENT
Approval basis: user "Approved proceed with a2" (2026-09-03) and bundle instruction "Proceed with option (a) proceed to A2 Stage-1 ... scp + remote probe" (2026-09-03).
Repo HEAD (local, no push): `51273a0` (KG5' gate amendment).
Remote reference tree: `/workspace/henri-k3-dispatch` @ `ddc950f`.
Digest gate: PASS — remote `git hash-object` == local `git rev-parse HEAD:<path>` for the three probed paths (`27020b67bb771901452740e08d57ce4191b6ffab`, `c1bc7e1d813f5c41df709b386338da69db365e31`, `8dda77ff0dc072310007d36e40a5f9ff1a5c4798`).
GPU exclusivity: PASS at dispatch time — 2 MiB / 32,607 MiB, 0 %, no CUDA compute apps.

## Purpose

Decompose the sealed KG5 engine latency mean (2.4738 ms, timed in
`arc_k3_steering_engine.py score_all_actions`, region ~295-344) at its
measured-dominant sites using PRODUCTION code imported from the dispatch tree
— never reimplementations:

- `arc_k3_koopman_generator._screen_sigma_max` (two power-iteration starts, top-64 svdvals, floor/headroom logic)
- `arc_k3_koopman_generator.BlockRidgeKoopmanFit.fit/apply` (cholesky ridge path, K3_ALPHA=1e-4)
- `arc_g7_calibrated_engine.G7CalibratedAffordanceEngine.predict_affordance`
  (the exact method the K3 engine times via MRO K3 -> P1 -> G7; residuals via
  G4 `aligned_mean_quadratic` over each action's top-k support)
- `arc_g4_aligned_engine.aligned_mean_quadratic` (shared C1 functional)

Instrument: `k3_a2_decomp.py` (SHA-256 `d5daabc72e96df5c199b22ea6dc24778cdf5169d513d1fdfa1624a643c8b0484`, 5,802 B), imported as above. Pre-seal patches applied after the Stage-1 signature audit: (1) added `bridge_transitions={}` and `bridge_route_flags={}` to the G7 constructor call (live signature requires them; probe omitted them); (2) transition matrices moved `.to(DEV)` (CPU-generator tensors would otherwise mismatch the CUDA consumer); (3) `accum()` now slices both operands to `X[:nfit]` (previously einsummed `X` vs `X[:nfit]` — n-dim mismatch).
as above. Measurement method: CUDA-event median of 15 after 3 warmups
(mirrors the sealed KG5 timing convention), seed 20260904, fp16-stored ->
fp32-ordered reads (K3RingAccumulator parity), per-block unit rows (live wave
boundary), device cuda on vast-5090, interpreter `/venv/main/bin/python`.

## Measured sites (JSON keys)

accum_ms_n{128,248}; solve_ms_n{128,248}; screen_ms_n{128,248};
full_fit_ms_n{128,248}; apply_ms; affordance_g7_predict_ms;
composite_region_{1,2}fit_ms; gpu. Verdict line: `A2_DECOMP_DONE`.

## Instrument acceptance anchors (fail-closed)

- screen_ms >= 0.650 ms (production two-start screen must exceed the prior
  single-start lower bound measured on the old probe).
- apply_ms within [0.010, 0.040] ms (prior anchor ~0.019 ms).
- full_fit_ms >= solve_ms + screen_ms - 0.3 ms at the same nfit (fit contains
  accum + solve + screen + scale + post-svdvals).
- composite band straddles the sealed mean: composite_region_1fit_ms <= 2.4738
  + 0.7 and composite_region_2fit_ms >= 2.4738 - 0.7.

## Kill criteria (verdict classes)

- BLOCKED_PROBE_DEFECT: import error, constructor TypeError, device mismatch,
  shape error in the probe. Fix the probe and rerun (disposable verification);
  never a verdict on the mechanism, never a seal reopen.
- BLOCKED_INSTRUMENT: anchor violations above after a defect-free run.
  Re-audit the probe against live signatures; rerun; not a mechanism verdict.
- BLOCKED_CONTENTION: any CUDA compute app present on vast-5090 during the run.
- No capability verdict is claimed. This is a component-latency decomposition
  only; task outcomes and benchmark scores are out of scope.

## Remedy scope (constrains any follow-on carrier)

A remedy may target ONLY the measured dominant site (largest measured positive
share of the sealed 2.4738 ms mean; a site is dominant when its measured cost
>= ~0.5 ms and >= ~40 % of the sealed band). All other sites stay
byte-identical. The A1-class solve swap is NOT re-opened unless the solve is
measured dominant. Any remedy is a NEW carrier requiring its own seal and
approval before remote execution.

## Out of scope

- KG5' (solve-only component accuracy vs torch <= 1e-5, sealed at `51273a0`)
  remains a separate gate with no instrument yet. This run does not measure it.
- No code change to the dispatch tree or the worktree is made by this run.
