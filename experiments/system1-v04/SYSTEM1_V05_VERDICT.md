# System-1 v0.5 Structural-Egress Verdict — SUPPORT_RESTORED_NOT_PROMOTED

**Date:** 2026-08-24 (CUDA 5090, vast-5090, 03:14 UTC, 150.8s)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121...` (frozen, calibrated)
**Split:** dev50_v05 (fresh disposable, seed 55123, sha `9d4c29ad...`)
**Budget:** 64 candidates/arm; sandbox executions 334; arms A/B/C

## OBSERVED (CUDA)

| Metric | A token beam | B skeleton uniform | C skeleton+energy |
|---|---|---|---|
| pass@1 | **0.55** | 0.325 | 0.325 |
| any_pass@2 | 0.55 | 0.60 | 0.675 |
| any_pass@4 | 0.55 | **1.0** | **1.0** |
| any_pass@64 | 0.55 | **1.0** | **1.0** |
| S = any_pass@64 − pass@1 | 0.0 | **0.675** (CI lb 0.55) | 0.675 |
| CEGIS admit (first passer in pool) | 0.55 | **1.0** | **1.0** |
| mean distinct finals/task | 1.0 | 3.67 | 3.67 |
| energy Spearman (exact v0.5 states) | — | — | **0.0128** (147 pairs) |

Transitions: A_vs_B — both 7, A_only 15, B_only 6, neither 12, McNemar p=0.0784.
B_vs_C — 0 discordant pairs (identical outcomes on all 40 tasks), p=1.0.

## Pre-registered gate results

- SUPPORT (S_B ≥ 0.15, CI lb > 0): **PASSED** — 0.675 / 0.55. Correct programs
  now ENTER the candidate pool for every task (was: never, v0.4.3).
- CAPABILITY (paired pass@1 improvement, McNemar < 0.05, delta ≥ 0.10):
  **NOT MET** — pass@1 dropped 0.55 → 0.325; McNemar 0.0784 (n.s.), wrong direction.
- ENERGY_FILTER (C ≥ B): nominally TRUE (equal), but B ≡ C on ALL 40 tasks —
  the filter is **inert**; energy carries zero signal on skeleton states.
- PROMOTION: **NOT MET** → `SUPPORT_RESTORED_NOT_PROMOTED`.

## Claim dispositions (v0.5 upload `3dfd53f0...`)

- "Guarantees S > 0.15" — outcome TRUE (0.675) but via the FAITHFUL build,
  not the upload's engine (input-independent random-weight mock; FALSIFIED).
- "Calibrated E_φ repurposed (ρ=0.4383)" — **FALSIFIED by re-measurement**:
  ρ=0.0128 on exact v0.5 candidate states. The v0.4.1 head was calibrated on
  token-decoder core-unrolled states; skeleton-code states are a different
  distribution (the provenance trap, now measured).
- "Run System-1_Kernel_v0.5_Engine.py" — NOT run (mock `torch.randint`
  `__main__`, no tasks/sandbox/checkpoint). The faithful kernel
  `system1_kernel_v05_ast_skeleton.py` was built and run instead.

## Architectural meaning (all OBSERVED)

1. Beam collapse is broken: 1.0 → 3.67 distinct programs/task.
2. Support failure is FIXED at the pool level: correct program enters the
   pool for 40/40 tasks (CEGIS admit 1.0).
3. The bottleneck MOVED to ranking: top-1 correct only 32.5% (rule-prob
   ordering); energy adds nothing.
4. A trivial CEGIS policy ("run candidates in pool order until one passes")
   achieves 1.0 on this dev split — the exact-replay verifier is the
   operational mechanism, exactly as the corpus prescribed.
5. Caveat: the grammar is DSL-specific (7 task families); this is structural
   egress for the System-1 task DSL, not general program synthesis.

## Next grounded steps

(a) Train a NEW discriminator on the egress-state family (skeleton codes)
    with binary sandbox labels — the v0.4.1 attribution lesson applied to
    the new distribution (supervision must match the candidate family).
(b) CEGIS-first: return first sandbox-passer (measured 1.0 admit); make
    ordering optimize expected sandbox calls (rule-prob + learned prior).
(c) Then re-seal a fresh heldout ONLY after gates pass on a second dev split.

## Artifacts

- eval_v05_support.json `aefc625d...`; results `fc9d21a6...`;
  dev50_v05.json `9d4c29ad...`; log `c55650a3...`
- lineage: `98fb9ec` (impl) + telemetry commit; bundle r8
