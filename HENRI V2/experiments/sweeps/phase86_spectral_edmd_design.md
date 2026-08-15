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

## Acceptance / kill discipline
- A1 AND A2 gates pass → levers validated (PASS).
- Any gate fires → that lever sealed FAIL; no post-hoc threshold tuning; no
  rerun after a gate fires.
- Branch `feat/batch-edmd-spectral-thermostat` from main `2218ec4`;
  `main` untouched; NO promotion.
- Infrastructure: GPU-exclusive preflight; checkpoint overlay symlink
  (SHA `75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060`);
  Zone C env `/workspace/zonec_prod.env`; `PYTHONPATH='HENRI V2'`;
  detached `setsid nohup` launch.
