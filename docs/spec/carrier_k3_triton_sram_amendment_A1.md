# Carrier K3 — Amendment A1: In-SRAM Fused Triton Solve (deficit #1 remedy)

Document Identifier: `HENRI-SPEC-2026-09-V3-CARRIER-K3-AMENDMENT-A1`
Amends: `HENRI-SPEC-2026-09-V3-CARRIER-K3-SEALED-PREREG` (sealed 2026-09-02, 8 visible amendments)
Parent results record: `HENRI-SPEC-2026-09-V3-CARRIER-K3-SEALED-RESULTS` (commit `591b526`, verdict `K3_GATE_KG5_LATENCY_FAILED`, 31st sealed falsification)
Date: 2026-09-02. Branch: `feat/carrier-k3-empirical-koopman`. Base: `591b526`.

## 1. Authorization

User approval (verbatim, 2026-09-02 session): "Approved to execute — In-SRAM Triton Cholesky carrier — audit deficit #1 is now measured (2.47 ms batched-torch solve, not 45 us prose). A Triton rewrite is a bounded fix against a real bound; it amends the sealed prereg → needs your sealed instrument."
This document IS the sealed instrument for the A1 implementation + remote CUDA measurement run: `HENRI-AUTH-2026-09-V3-CARRIER-K3-A1-DISPATCH` (approval level APPROVE_REMOTE_RUN, granted by the quoted user message). No push is authorized; the branch stays local (user's standing "no push" posture).

## 2. Stage-0 measurement (OBSERVED, vast-5090, RTX 5090, torch 2.12.0+cu130)

Probe: `/tmp/k3_latency_decomp.py` (scp'd, LF). CUDA-event median, 15 reps, M=8192, d=8, n=256, alpha=1e-4.

| Stage | ms |
|---|---|
| einsum accum (A=XᵀX, B=YᵀX) | 0.655 |
| accum + batched cholesky + cholesky_solve | 0.768 (solve component ≈ 0.11) |
| spectral screen (power-16 estimate + exact top-64 SVD) | 0.650 |
| full fit (accum + solve + screen + scale) | 1.410 |
| per-action apply (K·ψ per block) | 0.019 |
| score-path proxy (1 fit + 7 applies) | 1.491 |
| sealed KG5 engine mean (receipt `ecb01252…`) | 2.474 |

Reading: the torch solve is ~4.5% of the sealed KG5 mean. A solve-only kernel substitution cannot meet the 2.0 ms bound (floor ≈ 2.47 − 0.11). The supplied kernel's own design fuses the covariance accumulation with the solve; A1 therefore implements the FUSED in-SRAM kernel (deficit #1 remedy as scoped), keeping the sealed KG5 bound and verdict-by-receipt unchanged.

## 3. Scope (bounded, default-OFF)

1. New file `HENRI V2/experiments/verification/arc_k3_triton_sram_solve.py`: one Triton kernel per block m ∈ [0, 8192): load X rows (n × 8 fp32) and Y rows (n × 8) once each, accumulate A[m] = Σ x xᵀ + αI₈ and B[m] = Σ y xᵀ in registers/SRAM, in-kernel 8×8 Cholesky solve K = B A⁻¹ (equivalently solve K A = B via the in-register Cholesky factors), store K [M,8,8] fp32. One launch, no [M,8,8] intermediate round-trips for A/B.
2. Wire behind `HENRI_K3_TRITON_SOLVE=1` inside `BlockRidgeKoopmanFit.fit` (generator module). Default path (einsum + torch cholesky_solve) byte-identical when the flag is absent. Triton imported lazily inside the flag branch only.
3. The supplied staged kernel `carrier_k3_supplied_kernel.py` remains byte-identical (SHA `bff0174955e5eea7d22be222c4a8056f2e04d02ad077de272776a7bf8ce66e4e`), untouched — the audit-doc premise that the supplied file already contains the solve is FALSIFIED by read (it accumulates A/B only; solve is torch in its class).
4. Spectral screening + contraction scaling (KG4 machinery) stays torch — out of A1 scope.
5. No gate thresholds change. KG5 bound stays ≤ 2.00 ms on the sealed score-path CUDA-event mean (`k3_kernel_latency_ms` receipt field, unchanged measurement site in `arc_k3_steering_engine.score_all_actions`).

## 4. Verification gates (A1)

| Gate | Metric | Bound |
|---|---|---|
| A1-EQ | max per-block relative error ‖K_tri − K_torch‖_F / ‖K_torch‖_F over the fired + unfired batch (deterministic seed, n=256) | ≤ 1e-4 (fp32 reference = the sealed torch path) |
| A1-ENG | kernel launched and K consumed by the KG4 screen (receipt fields) | fired-batch non-empty |
| A1-KG5 | sealed score-path CUDA-event mean over the remote run | ≤ 2.00 ms (unchanged) |
| A1-DFLT | flag-absent differential: fit output byte-identical to the pre-A1 torch path | max diff 0.0 |

Verdict classes: `A1_EQUIVALENCE_FAILED` (fail-closed, no run) | `K3_GATE_KG5_LATENCY_FAILED` (measured mean > 2.0; decomposition receipts preserved; NO bound relaxation) | `A1_SRAM_SOLVE_VERIFIED` (equivalence + engagement + KG5 pass). A1 passing does NOT reopen KG2/KG6 — those stay FALSIFIED on the 31st record; a full K3 re-dispatch is a separate instrument.

## 5. Disclosures

- The engine's KG5 mean (2.474) includes the G7 base `predict_affordance` term (~1.0 ms inferred), which A1 does not touch; A1's expected ceiling is the honest outcome of the remote receipt.
- A1-EQ runs on the remote CUDA target only (Triton). Local CPU suite exercises the flag-absent differential and module import only.
- Kill criteria pre-registered: any non-finite K from the kernel → `K3NumericalAbort` fail-closed (unchanged); equivalence violation → abort before any run.

## 6. A1 measurement-run contract (dispatch topology)

Run: the sealed K3 launch topology, unchanged (launcher `arc_g7_calibrated_engine.py`, 12 envs, 150 steps/env, seed 20260930, `HENRI_K3_KOOPMAN=1`) PLUS `HENRI_K3_TRITON_SOLVE=1`, on vast-5090 `/workspace/henri-k3-dispatch` after the CAP12 capture completes (GPU-exclusive sequencing; never concurrent).

- Trajectory bank for goal binding: `f3_bank_capture_v3/trajectories_production_run_k3cohort.npz` when CAP12 = ENTROPY_GATE_PASS, else the sealed f3v2 bank (apples-to-apples KG5 baseline = receipt `ecb01252…` mean 2.474).
- Receipt fields read: `k3_kernel_latency_ms` (A1-KG5 gate, ≤ 2.00), plus the engine's full verdict symbol and KG2/KG6/KG1/KG3/KG4 values.
- Verdict classes (engine precedence, unchanged): latency ≤ 2.0 with equivalence+engagement → `A1_SRAM_SOLVE_VERIFIED` recorded alongside the engine's full symbol. A remaining KG2/KG6 failure on the 12-env basis is a NEW sealed falsification (32nd chain member), not an A1 failure — the A1 gate is the latency defect only.
- No push. Results committed to this branch as `docs/spec/carrier_k3_triton_sram_amendment_A1_results.md` with artifact SHAs.
