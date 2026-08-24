# System-1 v0.5.2 Heldout Verification — CEGIS_VERIFIER_ASSISTED_CAPABILITY_PROMOTED

**Date:** 2026-08-24 (CUDA 5090, vast-5090, 05:57 UTC, PID 389387)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121...` (frozen)
**Heldout split:** `heldout51_v052` — FRESH single-use, seed 271828, sha `5e5f4a00...`,
sealed 05:53:38Z BEFORE any evaluation (receipt `seal_heldout51_v052.json`), evaluated ONCE.
**Evaluator:** `eval_v052_heldout.py` (committed `f81ca20`), guard extended to 16 consumed digests.

## Directive disposition

- `887d0d6c...` = consumed/quarantined `smoke40_v04`. NOT a fresh holdout. Executing it
  would be replay → classified `INVALID_VERIFIER_REPLAY` by the extended guard (refusal
  tested locally before sealing). Disposition: `CONFLICTS_WITH_LIVE_INTEGRITY_POLICY`.
- Upload `HENRI_V0.5.1_CEGIS_Evaluation___Architectural_Synthesis.md` = 3-line ASCII flow
  diagram (297 bytes, sha `6a651801...`); matches the implemented v0.5.1 pipeline; nothing
  to reconcile beyond the target-hash conflict above.

## OBSERVED (CUDA, 40 tasks, budget 64, beam-width 64, disjoint verifier/outcome 4+4)

| Metric | A: token beam | B: uniform CEGIS-first |
|---|---|---|
| outcome pass rate (disjoint tests) | 0.375 | **1.000** |
| delta B−A | — | **+0.625** (task-blocked CI90 [0.50, 0.75]) |
| paired B_vs_A | both 15, B_only 25, A_only 0, neither 0 | **McNemar p = 5.96e-08** |
| admitted programs | 15 | 40 |
| AST validity (admitted only) | **1.0** (15/15) | **1.0** (40/40) |
| oracle outcome support | — | 1.0 (measurement only) |

## Correction disclosure (governance win, not a silent fix)

The first receipt computed `min_ast_valid=0.375` → `CONDITIONAL_VERIFIER_ASSISTED_IMPROVEMENT`.
Diagnosis: `ast_rate` counted NON-ADMITTED tasks as invalid (0), conflating "no admission"
with "invalid program". Per-task telemetry shows 0 admitted-but-invalid programs in either
arm. Fixed `ast_rate` (admitted-only) in the evaluator; verdict recomputed deterministically
from the SAVED per-task telemetry — the heldout split was NOT re-evaluated (single-use).
Corrected verdict: **`CEGIS_VERIFIER_ASSISTED_CAPABILITY_PROMOTED`**
(McNemar p<0.05, delta ≥ +0.10, CI lb > 0, admitted-validity ≥ 0.95 — all pre-registered gates).

## Interpretation (all OBSERVED)

1. The uniform CEGIS-first egress chain generalizes to unseen heldout tasks: 1.0 vs 0.375,
   significant, validity preserved. The v0.5.1 dev2 result was NOT an artifact of the
   dev split or of test replay (disjoint verifier/outcome enforced on both).
2. This is VERIFIER-ASSISTED capability (sandbox admission), not unguided top-1 decoding —
   classification kept explicit per reference `system1-structural-egress-cegis.md`.
3. The learned discriminator remains out of the promotion comparison (v0.5.1: zero outcome
   change); call-cost improvement was 10.8% < 20% gate → still marginal.
4. Heldout split now CONSUMED (single-use). Any future heldout requires a new seal.

## Next (separate cycle, NOT combined with heldout)

Grammar manifold expansion as a NEW development cycle on fresh disposable splits, with
run14 rank-dilution warning: more grammar shapes displaced simple correct programs.
Audit: task-family coverage, semantic support, effective diversity, pass@K,
calls-to-first-pass, ranking dilution.

## Artifacts

- Seal receipt: `seal_heldout51_v052.json` (`5e5f4a00...` full sha in receipt)
- Receipt: `eval_v052_heldout.json` `99cb04d1...` (contains the ORIGINAL buggy verdict — kept
  as historical record; authoritative = `verdict_corrected.json`)
- Corrected: `verdict_corrected.json`; results `5558f11f...`; log `fbc33c08...`
- Lineage: `f81ca20` (evaluator+seal) + telemetry/verdict commit; bundle r10
