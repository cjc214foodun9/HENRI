# System-1 Stage-0b — Frozen Deterministic Encoder Contract (Pre-registration)

**Date:** 2026-08-24 · **Reference 3 (gpt-5.6-sol) binding**
**Authorization:** User Option 1 — CPU verification only; default OFF; zero learning.
**Carrier:** `vla_stage0b_encoder.py` (written AFTER this contract is frozen).
**Boundary:** live `[1,16,384]`, d_slot=384, float32 REAL. No new tensor family
(corpus #22 65,536-D claim recorded as conflict and overridden by live code +
Reference 3).

## Geometry (explicit — sphere vs torus)

- Canonical consumer output: real slot tensor `z ∈ R^{[1,16,384]}`, unit-norm
  sphere per slot: `||z_s||_2 = 1`, error ≤ 1e-6.
- Diagnostic torus: phasor `E = exp(iθ)` with `|E_sd| = 1` elementwise, error
  ≤ 1e-6. `E` is DERIVED ONLY and is never passed to any consumer.
- The "unit-modulus error ≤ 1e-6" contract is valid ONLY because the torus is
  explicitly implemented as the diagnostic; the canonical tensor is
  sphere-normalized real. Both labels are reported in telemetry.

## Contracts (C1–C10)

- **C1 Default OFF:** without `HENRI_STAGE0B_ENABLE=1`, `encode()` returns
  byte-identical input passthrough; verify asserts output == input.
- **C2 Frozen:** no `nn.Parameter`; zero optimizer membership; constants are
  module-level frozen tensors; static scan for Parameter/optimizer.
- **C3 Deterministic init hash:** SHA-256 of the frozen constant tensors
  recorded; identical across two separate python processes.
- **C4 Replay determinism:** same obs → byte-identical `[1,16,384]` output
  across (a) two instances in one process, (b) two separate processes.
- **C5 Sensitivity:** obs from different seeds → encodings with L2 distance
  > 1e-3 on ≥ 90% of distinct pairs; min pairwise distance > 0 over episode.
- **C6 Non-collapse:** SVD of stacked per-slot encodings: numerical rank ≥ 2;
  per-slot std > 1e-6.
- **C7 Finite/shape/dtype:** all finite; shape `(1,16,384)`; dtype float32.
- **C8 Geometry:** max|‖z_s‖₂ − 1| ≤ 1e-6 per slot; max||E_sd| − 1| ≤ 1e-6
  elementwise; explicit sphere/torus labels in telemetry.
- **C9 No mutation / no learner:** encoder source contains no env access, no
  `backward()`, no optimizer, no policy/action output (static scan).
- **C10 Diagnostics:** report min/mean pairwise L2 over episode, SVD spectrum
  (top 5), collision count (pairwise L2 < 1e-6).

## Gates

- ALL C1–C10 PASS on CPU over ≥ 2 fresh deterministic episodes (seeds 4242 and
  90909) AND a cross-process byte-identical check → verdict
  `FROZEN_REPRESENTATION_VERIFIED`.
- Any fail → `REPRESENTATION_FAILED` + fix loop on CPU only.
- No CUDA launch. No Stage-0c learner. Global VLA gate stays **0/12**.

## Stage-0c precondition (recorded, NOT executed)

- R-EDMD requires N ≥ r independent action-conditioned lifted tuples
  `(ψ(s_t,a_t), ψ(s_{t+1}))`. Stage-0b must demonstrate distinct lifted states
  across a real episode (C5/C6/C10) — this is evidence for 0c design, not
  authorization. Stage-0c requires its own pre-registered contract and user
  authorization.
