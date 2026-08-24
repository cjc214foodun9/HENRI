# System-1 v0.6.0.1 — CANDIDATE_RETRIEVAL_NO_EFFECT (CUDA dev9_v0601)

**Date:** 2026-08-24 · **CUDA 5090, vast-5090, PID 425556**
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121…` (frozen, unchanged)
**Carrier:** system1_kernel_v055_ast_skeleton.py sha `d9a976ad…` (frozen v0.5.5)
**Split:** dev9_v0601 — DISPOSABLE dev split, sha `a8a2d7a7…`, seed 90909,
n=65 (13 families × 5, divisible stratification). No heldout created or
consumed; sealed `87390286…` and `a09bf275…` in the consumed-digest guard.

## OBSERVED (CUDA, single evaluation, β=0.15)

| Metric | B13 (baseline) | R0 (β=0) | R1 (β=0.15) |
|---|---|---|---|
| outcome pass (disjoint tests) | 1.0 (65/65) | 1.0 (65/65) | 1.0 (65/65) |
| family support | 13/13 at 5/5 | 13/13 at 5/5 | 13/13 at 5/5 |
| verifier calls mean / median / p90 / max | 4.23 / 4 / 8 / 9 | 4.23 / 4 / 8 / 9 | 4.23 / 4 / 8 / 9 |
| R0 byte-identical to B13 | — | 65/65 tasks | — |
| R1 reordered admits | — | — | 60/65 tasks |
| within-task sim variance > 0 | — | — | 65/65 tasks (min 1.5e-4) |

Paired R1 vs B13: both 65, R1_only 0, B13_only 0 → **p=1.0, zero discordance**.
Call delta B13−R1: mean 0.0, 95% CI [-1.08, 1.11] → includes 0.
Gates: G1 TRUE (integrity + R0 identity) · G2 TRUE (capability preserved) ·
G3 FALSE (no cost reduction) · G5 TRUE (variance positive).

## Verdict

**`CANDIDATE_RETRIEVAL_NO_EFFECT`** (pre-registered chain; zero paired
discordance AND zero call-delta significance).

## Mechanism explanation

The retrieval mechanism is ACTIVE (R1 reorders the candidate pool on 60/65
tasks; within-task sim variance positive everywhere) but COST-INERT at this
operating point: the correct candidate is already admitted within ~4 verifier
calls under uniform order, and the z-scored β=0.15 similarity reorder does
not move the first-passing candidate earlier relative to verifier admission.
Reordering permutes WHICH candidates are tried first without changing the
position of the correct one — net zero calls. This is the third consecutive
order-preserving null (v0.6.0 uniform scalar, v0.6.1 failure-only fast
weights, v0.6.2 syntactic rotation): ranking interventions cannot extract
cost where the verifier already finds the answer at the same rank.

## Integrity record

- Pre-registration + frozen manifest: commit `7937c51`; correction commit
  `e128a55` (n=60→65; n=60 crashed the stratifier at startup before any
  task ran, out dir empty → zero exposure, no quarantine).
- Contracts C1–C7: ALL PASS (non-vacuous; variance > 0 across 13 families;
  β=0 byte-identical; C3 static leak scan clean after AST-dedent fix).
- Remote dependency preflight: all 6 dependency shas byte-identical; GPU
  idle before launch; evaluator/bridge/contract remote shas matched manifest.
- Smoke (plumbing only): smoke601_disposable n=13 → NO_IMPROVEMENT (R1
  reorders 12/13, calls unchanged).
- No consumed-split replay; no quarantine event this cycle.

## Attribution boundary

- Endpoint per Reference 3 ceiling correction: verifier-call reduction under
  EXACT capability preservation (B13 already 52/52 on heldout53_v055).
- No heldout was created or consumed. No pass-rate claim is made.
- EFFICACY_PROMOTED remains reserved for a harder disposable condition
  (future work).
- The 5-stage VLA roadmap is a planning artifact, not capability evidence.

## Post-measurement corpus interpretation (INFERRED, consult #19)

Bank `ca4bb787…`, 2026-08-24, after telemetry existed; 16 sources cited.

- **The null is predicted, not anomalous:** under perfect candidate support
  (S=1.0) and low pre-reasoning entropy (H→0), the correct candidate is
  already at Rank 1; the verifier terminates on the first passing candidate,
  so reordering the remaining K−1 incorrect tail candidates is bypassed by
  the execution thread → zero physical/operational effect on verifier calls
  [5,9,18]. "Engagement without efficacy is the natural signature of a
  saturated search space."
- **Where retrieval CAN pay:** Scenario A (high-entropy rank dilution,
  H ≥ 1.3) — correct candidate buried at rank ~5; a calibrated z-scored
  similarity term pulls it to Rank 1, reducing expected verifier calls
  O(K)→O(1) [13,17]. This is the falsifiable condition for a future test,
  NOT present in the current 13-family baseline.
- **Remediation direction (corpus):** a Pre-Reasoning Entropy Gate
  (H < 0.4 → commit immediately; reserve verifier calls for high-entropy
  boundaries) — commit-early triaging saves 30–47% of evaluation cost at
  zero accuracy loss in the cited forecasters study [9,18].
- **Offered JAX/Triton schema work DECLINED** (mock-loop pattern; recorded,
  not accepted).
