# Path B2 Gate A Verdict — FALSIFIED (2026-08-21)

Doc ID: HENRI-CLASS44-PATHB2-GATEA-FALSIFIED-2026-08-21
Sealed: PATH_B2_GATE_A_FALSIFIED
Spec: `Project_HENRI__Isometry_vs._Semantic_Margin_Synthesis___Literature_Integration.md` (SHA `fa145c46…`)

## Gate A result (OBSERVED, RTX 5090, D=65,536, exact-SHA worktree 3df1ed4)

| Target | oracle_rank (≤5 req) | true_cos | best_other_cos | margin (≥0.25 req) | C1 | C2 |
|---|---|---|---|---|---|---|
| HumanEval/23 | **5**/71 | 0.8484 | 0.8703 | **−0.0218** | PASS | FAIL |
| HumanEval/35 | **13**/71 | 0.7869 | 0.8211 | **−0.0342** | FAIL | FAIL |

Verdict: **FALSIFIED** (both conditions must pass). Gate B skipped per protocol.

Probe validity: checkpoint `a6609347daacf153` (path_b2_codec.pt, 4,825,883 bytes,
val_contrastive_acc 0.9583, gram_max 1.37e-06); dataset HumanEval
`b796127e635a67f9`; 71-candidate grammar pool; CUDA 5090; production dims.

## What the run proved

- Isometry HELD: gram_max 1.37e-06 < 1e-5 (Cholesky retraction, dual-sided).
- Training learned: val_contrastive_acc 0.9583 (chance 0.5) — non-collapse.
- Hard-negative InfoNCE + qFHRR-IDF moved the oracle from Path B1 ranks
  31/32 → **5/13** — real metric deformation toward the goal, but
  lookalikes STILL outscore the oracle (best_other 0.8703/0.8211 vs true
  0.8484/0.7869). Margin negative on both targets.

## Root cause (DERIVED + HYPOTHESIS)

The learned metric now sees the oracle near the top but cannot push
lookalikes below a +0.25 margin. Skeleton-carrying candidates still occupy
the goal neighborhood after IDF weighting — the shared AST frame dominates
the residual. The margin gate is the binding constraint; rank alone improved.

## Kill executed (pre-registered, binding)

- Gate A fail ⇒ revert + seal + halt. Gate B NOT run.
- Path B2 files reverted (module, trainer, probe, contract tests, runner
  wiring). HOPS-VSA module (Class 4.5, separate user-authorized phase)
  PRESERVED.
- Evidence preserved: `experiments/verification/path_b2_evidence/path_b2_codec_gate_a_ckpt.pt`
  (SHA `a6609347daacf153`, gitignored overlay).

## Future reopen condition

No further Path B2-class work without a NEW pre-registered packet (with a
different margin mechanism, e.g. hard-negative margin loss directly on the
71-pool + skeleton-subtracted scoring) + explicit user approval.
