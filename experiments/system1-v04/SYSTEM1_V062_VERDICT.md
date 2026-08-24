# System-1 v0.6.2-dev — Heterogeneous Sub-Swarm Partition: PARTITION_DIVERSITY_ONLY

**Date:** 2026-08-24 (CUDA 5090, vast-5090, PID 405642)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121...` (frozen)
**Split:** dev8_v062 (DISPOSABLE, seed 81828, n=60, sha `44657c7f`)
**Arms (matched budget 64 / beam 64):** A token beam · B13 uniform CEGIS-first · E partition rotation (3 sweeps × in-vocab arg rotation)

## OBSERVED (CUDA)

| Metric | A | B13 | E (sweeps=3) |
|---|---|---|---|
| outcome pass | 0.283 | 0.967 | 0.967 |
| distinct programs/task | — | 7.08 | **21.25 (3.0×)** |
| verifier calls (mean) | — | 3.867 | 5.217 (+34.9%) |
| family 10 | — | 0.0 | 0.0 (structural, frozen grammar) |

Paired E vs B13: both 58, E_only 0, B13_only 0, neither 2 → McNemar p=1.0. Gates: G0 identity TRUE · G1 TRUE · G2 TRUE · G3 TRUE (5.217 ≤ 5.80) · **G4 diversity-2× TRUE**. Verdict: **`PARTITION_DIVERSITY_ONLY`**.

## Verdict semantics

- **Diversity claim VERIFIED:** 3.0× distinct programs/task (7.08 → 21.25) at matched total budget 64 — the upload's "3.63 → 15.0+" direction is confirmed on the live substrate (different absolute baseline, ours measured).
- **Efficacy UNCHANGED (p=1.0):** rotation changes code strings with identical semantics (positional call convention); the semantic pool is unchanged → outcome pass identical.
- **Cost claim FALSIFIED:** "identical total compute overhead" is wrong — verifier calls +34.9% (5.217 vs 3.867). Rotation interleaves arg-name variants, so the correct rule is scanned later. Measured, not assumed.

## Honest bound

Distinct programs per task is bounded by (#rules × #arg-sets) per arity (here 13 rules × up to 7 rotations → 21.25 mean). Not unbounded; `15+` from the proposal was a HYPOTHESIS on a different substrate.

## Integrity

- Fresh disposable split; consumed-digest guard held (incl. quarantined `392ce03e3b35`, `d6b79d51`, `a09bf275`).
- No heldout created or consumed. No replay.

## Next

Partition's value is diversity (support breadth), not outcome pass at this baseline. Its real test is whether 3× distinct programs raises ANY_PASS@K under tighter verifier budgets (different gate set — pre-registered separately). v0.5.5 rule-10 fix remains separate with semantic per-family closure + NEW seal.
