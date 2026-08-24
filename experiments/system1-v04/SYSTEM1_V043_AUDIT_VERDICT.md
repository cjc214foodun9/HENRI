# System-1 v0.4.3 Oracle Candidate-Support Audit — Verdict

**Date:** 2026-08-24 (CUDA 5090, vast-5090, completed 01:10 UTC)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121...` (frozen, calibrated)
**Split:** dev43_v04 (fresh disposable, seed 77031, sha `db027f9c...`)
**Run:** width 64, arms A(β=0.0) / B(β=0.40), ALL finals sandboxed, 292s

## Results (OBSERVED)

| Metric | Arm A (β=0.0) | Arm B (β=0.40) |
|---|---|---|
| pass@1 | 0.425 | 0.425 |
| any_pass@K (K=2..64) | 0.425 | 0.425 |
| mean distinct finals / task | **1.0** | **1.0** |
| mean unique valid / task | 1.0 | 1.0 |
| passer min energy rank | 1.0 (17/40 tasks with passer) | 1.0 |
| energy Spearman (pooled, 80 pairs) | 0.6092 | 0.6092 |
| S = any_pass@64 − pass1 | 0.0 | 0.0 |
| CI90 delta LB | −0.25 | −0.25 |

## Decision (pre-registered rule)
S = 0.0 ≤ 0.05 → **SUPPORT_FAILURE**

## Key finding: beam diversity collapse
mean_distinct_finals = 1.0 at width 64 → the beam collapses to ONE unique
program per task. Width is decorative. The frozen decoder's generative
support is effectively a single program per signature; correct programs for
failing tasks are outside the support.

## Evidence chain (all OBSERVED)
1. v0.4 stochastic swarm: 128 particles → 3.65 unique programs/task, pass
   rate unchanged → sampling does not reach correct programs.
2. v0.4.2 CEGIS beam β=0.40: 100% trajectory engagement, 0 paired efficacy
   (21/40 both arms) → reordering a 1-element set changes nothing.
3. v0.4.3 oracle audit (this run): any_pass@64 = pass@1 = 0.425, distinct
   finals = 1.0 → widened search cannot surface correct programs.

## Implication
Structural egress is the grounded next mechanism (reference
system1-calibrated-probe-search-integration.md: SUPPORT_FAILURE →
"unfreeze alone poorly grounded; implement structural AST egress").
Token-level unfreezing adds a sparse signal to a collapsed distribution; the
principled fix is signature → AST-skeleton → FSA-instantiation with CEGIS
verification. The energy head stays a secondary ranking signal (passer
energy rank 1.0 confirms calibration at the top).

## Artifacts
- audit_support.json sha `e5c83188...`
- audit_support_results.json sha `50fbbd16...`
- dev43_v04.json sha `db027f9c...`
- audit_v043.log sha `202462b8...`
- lineage commit: (recorded below)
