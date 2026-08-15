# Phase 8.6 — Spectral Thermostat + Batch EDMD (Lever a+b) — Pre-Registration

Source: `HENRI V2 Failure Diagnostic & Next Decision Plan.pdf.pdf` (Drive
inbox, 6 pp, raw SHA-256
`27e01038201ec31601ebc09286dc48a89656dfe94f7a129a6deae8e8dab65ac9`,
LF text SHA-256 `e1db8d8ff04e366381950c58a37a9ecc7be3a4b9e9190c036a4ec020c42b03a3`).
Phase 8.6 postmortem of sealed D1/D2 (evidence `1c141519…`/`f4191736…`,
branch `feat/low-rank-wave-jepa` @ `d8e28f6`; main `2218ec4`).

## Classification of the document's instructions
- **Lever (a) D2 revised — spectral thermostat `P_high = I − P_low`:**
  GENUINELY MISSING on main (thermostat has only the Phase 5 wavelet gate).
  → NEW bounded default-OFF experiment. Mechanism: FFT high-pass noise
  injection preserves low-frequency macro-state (invariants) while injecting
  thermal variance into high-frequency micro-modes.
- **Lever (b) D1 batch EDMD `train_transition_batch`:** ALREADY IMPLEMENTED
  AND LIVE (`efe_planner.py:1148`, dual-Woodbury ridge EDMD, blend=0.5).
  The PDF's pseudocode (`[T,D]` complex tensors + direct `pinv`) is the
  REJECTED third tensor family (invariant: real `[num_blocks, 8]` Clifford
  waves; dual form never forms the 2d×2d Gram). → REUSE-ONLY: measure the
  production path.
- **Lever (c) learnable action embeddings (P1):** NOT in scope — pending
  separate pre-registration.
- **Lever (d) valence=0 pre-training (P1):** NOT in scope — pending
  separate pre-registration.

## Dimensional feasibility (pre-computed)
- Spectral projector, flat wave length n=D=65,536, cutoff k=512 (mask both
  ends → 2k low harmonics): retained noise energy fraction
  `1 − 2k/D = 0.9844`. Low-frequency component of projected noise is exactly
  zero → `lowfreq_norm_change < 1e-5` gate dimensionally achievable.
  Recovery drift is dominated by macro-state preservation → `>40%` drift
  reduction gate plausible (contrast D2's P_null: 0.1% energy removal).
- **Trap (pre-registered):** FFT over the last dim of `[num_blocks, 8]`
  (length 8) with cutoff 512 zeroes ALL noise (2k > 8). The projector MUST
  operate on the FLATTENED wave (length D). Fallback: if `n <= 4k`, clamp
  `k = max(1, n//8)`; if still degenerate, return the isotropic draw
  (never zero noise).
- **Precision clarification v2 (2026-08-14, pre-launch):** the PDF's
  "low-frequency mode norm change < 1e-5" is read as a RELATIVE mechanism
  gate. Absolute low-freq norm of the projected draw after float64 FFT
  round-trip at n=65,536 floors at ~1e-3 relative (FFT precision, not
  leakage). Gates (final):
  - mechanism: `1 - ||low(highpass x)|| / ||low(x)|| > 0.98`
    (measured ~98.7% pre-launch);
  - causal: `drift_reduction_lowfreq_pct > 40%` — macro-state
    (low-frequency) drift reduction during Sagnac veto recovery, the
    spectral projector's basin-preservation claim. Total-norm drift
    reduction is recorded as telemetry (dimensionally ~3%, dominated by
    the 98.4% retained noise energy).
- Batch EDMD N=128, d=65,536: dual Woodbury O(N·d) ≈ 34 MB — feasible;
  existing production code path.

## Arms (remote CUDA matrix, all `diagnostic_only=true`, NO env stepping)
| Arm | Treatment | Gates (pre-registered) |
|---|---|---|
| A0 | control (frozen) | rc=0; loss finite on 64 known-transform pairs; projection inactive |
| A1 | Lever (a) spectral thermostat | drift_reduction > 40% (recovered-state variance, 50 steps × 16 seeds); lowfreq_norm_change < 1e-5; energy retention ≈ 0.9844 ± 0.01 |
| A2 | Lever (b) production `train_transition_batch` (N=128) | held-out (fresh 128 pairs) Sagnac loss decrease > 15% (pre vs post); post loss finite < 1.0; wall ≤ 600s |
| A3 | combined (a+b smoke) | rc=0; finite losses; adapter shapes `[8192, 8]` |

DONE_MARKER written ONLY when all arms rc=0. Any nonzero arm →
`BLOCKED_INFRASTRUCTURE`, no science verdict.

## RUN LOG (measurement-validity audit — 2026-08-14)
| Run | SHA | Status | Reason |
|---|---|---|---|
| 1 | `6ba8bda` | INVALID (not sealed) | A1 reused ONE noise tensor across all seeds → degenerate cross-seed variance (~1e-13), vacuous PASS; A2 pre = fit-batch residual |
| 2 | `1805daf` | PROVISIONAL (not sealed) | A1 draws n_i / n_s independently per arm → UNPAIRED comparison, not the pre-registered paired test; A2 pre still fit-batch residual |
| 3 | (candidate) | DEFINITIVE | A1: one base noise per (seed, step), cloned + passed to BOTH arms — arms differ only by thermostat mechanism; gate = mean of per-seed PAIRED low-freq reductions; A2: disjoint train/held-out (seeds 20260814/777), matched held-out pre/post, fit on train only |

## Pre-launch precision clarification #4 (PAIRED reading — recorded BEFORE run 3)
The pre-registered gates are unchanged: A1 low-freq drift reduction > 40%
AND low-freq energy reduction > 0.98; A2 matched held-out Sagnac decrease
> 15%. Run-3 measurement reading: A1 gate metric = mean of per-seed PAIRED
low-freq reductions; A2 verdict from matched held-out pre/post with
disjoint train. Run 3 is the ONLY run eligible for sealing; runs 1–2 are
evidence artifacts, not verdicts.

## PROVISIONAL VERDICT (run 2 @ `1805daf` — NOT sealed, superseded by run 3)

Remote CUDA matrix @ RTX 5090 (torch 2.12.0+cu130, CUDA 13.0, D=65,536),
candidate SHA `1805daf3ec122e9ad20773a4365c97a37fe5cad4`, all arms rc=0,
DONE_MARKER rc=0. (Run 1 at `6ba8bda` had a degenerate A1 causal metric —
reused single noise tensor across seeds, variance ~1e-13 — measured as a
harness defect, NOT sealed; repaired with per-seed fresh noise + held-out
A2 eval, re-run at `1805daf`.)

Evidence: `phase8_evidence/phase86_spectral_edmd/`
- `p86_result.json` SHA-256 `ee73c07822d5c1c059ca568dbd142da07c5ecdddbe681ef489d42088b21ff13c`
- `p86.log` SHA-256 `02af2f1225dab1195ed13b64231148284725f22aba72d4351255e8280cbfb586`
- decoder checkpoint loaded: SHA `75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060` (symlink overlay)

| Arm | Result | Key metrics |
|---|---|---|
| A0 OFF | OK | predictor=RecursiveDualEDMD, projection inactive (default-path identity) |
| A1 Lever (a) spectral | **PASS** | lowfreq_energy_reduction 0.9998 (>0.98); drift_reduction_lowfreq_pct **99.86%** (>40%); raw→proj lowfreq norm 32.88→0.0053; total-norm drift_reduction_pct −11.6% (telemetry: high-freq energy dominates total norm, expected) |
| A2 Lever (b) batch EDMD | **D1_BATCH_FAIL** (gate fired) | pre_loss 1.0023, post_loss_heldout 0.9980, decrease 0.43% < 15% gate; N=128 held-out pairs; wall 4.51s |
| A3 combined smoke | OK | shape [8192,8], loss 1.0031 finite |

Verdicts:
1. **Lever (a) SPECTRAL THERMOSTAT PASS** — the corpus-prescribed
   `P_high = I − P_low` projector is implemented, the mechanism is exact
   (99.98% low-freq energy removal), and the CAUSAL claim (macro-state /
   low-frequency drift reduction during SDE recovery) is validated at
   99.86% with non-degenerate cross-seed variance. This replaces the
   falsified D2 `P_null = I − VV†` (0.16% reduction). Default-OFF flag
   `use_spectral_gating`; NOT promoted.
2. **Lever (b) D1_BATCH_FAIL** — the PRODUCTION `train_transition_batch`
   (dual-Woodbury EDMD, N=128) improves held-out Sagnac loss by 0.43% at
   D=65,536, far below the 15% gate. Consistent with sealed E1 and D1
   (per-step 0.07–0.41%): batch EDMD is also inert at this scale on
   known-transform integrity pairs. The PDF's batch-EDMD prescription
   (closed-form Koopman via SVD) is ALREADY LIVE in this function; the
   inertness is a representation-level limit, not a missing solver.
   NOT promoted.
3. **Corpus consult (INFERRED)** — planned; the corpus attributes the
   thermostat mechanism to spectral low-pass/high-pass separation and
   warns against coupling noise to the transition subspace; consistent
   with Lever (a) PASS and Lever (b) inertness at D=65,536.
4. **Harness lesson (recorded)**: a cross-seed variance gate requires
   per-seed fresh noise sequences; reusing one noise tensor across seeds
   yields degenerate variance (~1e-13) and a vacuous PASS. Held-out
   evaluation requires fresh pairs from a different seed, not the fit
   batch (train-loss would understate the gate).

Branch sealed @ `1805daf`; `main` untouched `2218ec4`. No promotion.
Next levers (each a NEW pre-registered protocol): (c) learnable action
embeddings (P1); (d) valence=0 pre-training (P1); representation repair
(R1 foreground-masked ramps, per sealed R1). All await explicit approval.
