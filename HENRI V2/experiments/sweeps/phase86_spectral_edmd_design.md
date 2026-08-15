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
| 3 | `667a74f` | DEFINITIVE — **SEALED** | A1: one base noise per (seed, step), cloned + passed to BOTH arms — arms differ only by thermostat mechanism; gate = mean of per-seed PAIRED low-freq reductions; A2: disjoint train/held-out (seeds 20260814/777), matched held-out pre/post, fit on train only. Evidence `5a5f0eb8…` / `965c2f3a…` |

## Pre-launch precision clarification #4 (PAIRED reading — recorded BEFORE run 3)
The pre-registered gates are unchanged: A1 low-freq drift reduction > 40%
AND low-freq energy reduction > 0.98; A2 matched held-out Sagnac decrease
> 15%. Run-3 measurement reading: A1 gate metric = mean of per-seed PAIRED
low-freq reductions; A2 verdict from matched held-out pre/post with
disjoint train. Run 3 is the ONLY run eligible for sealing; runs 1–2 are
evidence artifacts, not verdicts.

## SEALED VERDICT (2026-08-14) — Phase 8.6 (feat/batch-edmd-spectral-thermostat)

Definitive RUN 3 @ `667a74fa50b8aa22203bd6f48afd9a0293c9e4f8`; remote CUDA
matrix @ RTX 5090 (torch 2.12.0+cu130, CUDA 13.0, D=65,536); all arms
rc=0; DONE_MARKER rc=0. Runs 1–2 were measurement-invalid (RUN LOG above)
and are NOT sealed; run 3 is the ONLY sealed verdict.

Evidence: `phase8_evidence/phase86_spectral_edmd/run3/`
- `p86_result.json` SHA-256 `5a5f0eb82be07e15eec6d1d5249650a3b9571321e9fe3bcca8c330ce2ea4ca8b`
- `p86.log` SHA-256 `965c2f3acb4348018ebd627dd4f54093580f963dd28146f352dde1c2aa93ad47`
- decoder checkpoint loaded: SHA `75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060` (symlink overlay)

| Arm | Result | Key metrics (run 3, PAIRED) |
|---|---|---|
| A0 OFF | OK | predictor=RecursiveDualEDMD, projection inactive |
| A1 Lever (a) spectral | **PASS** | PAIRED per-seed low-freq drift reduction mean **93.27%** (>40% gate; per-seed 92.93–93.66%, 16/16 > gate); lowfreq_energy_reduction 0.9998 (>0.98); raw→proj lowfreq norm 32.88→0.0053; total-norm paired reduction mean +0.78% (high-freq energy dominates total norm) |
| A2 Lever (b) batch EDMD | **D1_BATCH_FAIL** (gate fired) | matched held-out pre 1.00208 → post 1.00207, decrease **0.0014%** << 15% gate; train_loss 0.99823 (0.18% fit reduction, does NOT generalize); N=128 disjoint train/held-out |
| A3 combined | OK | shape [8192,8], loss 1.0031 finite |

Verdicts:
1. **Lever (a) SPECTRAL THERMOSTAT PASS** — PAIRED test: identical noise
   draws per (seed, step), arms differ ONLY by thermostat mechanism. Mean
   low-frequency drift reduction 93.27% (16/16 seeds), mechanism energy
   removal 99.98%. Replaces the falsified D2 `P_null = I − VV†` (0.16%
   reduction). Default-OFF `use_spectral_gating`; NOT promoted. NOTE: the
   run-2 UNPAIRED estimate (99.86%) was inflated; the honest PAIRED value
   is 93.3% — still far above the 40% gate. Corpus (INFERRED) attributes
   this to spectral scale separation preserving Kuramoto basins; predicts
   degradation at higher cutoff (k > 512) and logic-lock risk at T → 0.
2. **Lever (b) D1_BATCH_FAIL** — matched held-out decrease 0.0014%
   (run-2's 0.43% was partly fit-batch optimism). Production
   `train_transition_batch` (dual-Woodbury EDMD, N=128) is inert at
   D=65,536 on known-transform integrity pairs. Corpus consult (INFERRED)
   attributes the floor to a REPRESENTATION bottleneck, not the solver:
   (i) actions as random vectors (Fallacy #3) — no action→morphism
   coupling; (ii) cross-block leakage without a coordinate-equivariant
   carrier basis; (iii) episode reset starvation (planner + loss_ema
   reset every env). NOT promoted.
3. **Corpus prescription (INFERRED — next-lever candidates; each requires
   a NEW pre-registered protocol and explicit approval)**: (c) learnable /
   typed action embeddings bound to spatial carriers (P1); (d) valence=0
   pre-training on task-agnostic offline transitions (P1); transition
   persistence across episodes; representation repair (R1 foreground-
   masked ramps, per sealed R1).
4. **Harness lessons (recorded)**: (i) cross-seed variance gates require
   per-seed fresh noise — reuse ⇒ degenerate variance ~1e-13, vacuous
   PASS (run 1); (ii) paired comparisons require IDENTICAL draws per
   (seed, step) — independent draws inflate the effect size (run 2:
   99.86% vs paired 93.27%); (iii) held-out pre/post must use the SAME
   disjoint set — fit-batch residual understates the gate (A2: 0.43% →
   0.0014%).

Branch sealed @ `667a74f`; `main` untouched `2218ec4`. No promotion.
Next levers (c)/(d)/(persistence)/(representation repair) each require a
new pre-registered protocol and explicit approval.
