# System-1 v0.5.1 CEGIS-First + Egress Discriminator — Verdict

**Date:** 2026-08-24 (CUDA 5090, vast-5090, 03:49 UTC, 166.8s eval + ~10s train)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121...` (frozen, calibrated)
**Discriminator:** disc_v051.pt (Brier, verifier-test labels only, 49,409 trainable)
**Split:** dev2_v051 (fresh disposable, seed 90837, sha `1f81e4d0...`) — sealed BEFORE training, evaluated once
**Protocol:** reference `system1-structural-egress-cegis.md` v0.5.1: DISJOINT verifier/outcome tests
(4 verifier + 4 outcome per task, input-uniqueness-guaranteed partition)

## OBSERVED (CUDA, 40 tasks, budget 64/arm)

| Metric | A token beam | B skeleton uniform | C disc-order | D random |
|---|---|---|---|---|
| **outcome pass (disjoint tests)** | 0.50 | **1.00** | **1.00** | 1.00 |
| verifier calls mean | 1.0 | 2.325 | **2.075** | 2.575 |
| calls median / p90 / max | 1/1/1 | 2/4/4 | 2/3/4 | 2/4/4 |
| success@1 | 0.50 | 0.375 | 0.35 | 0.15 |
| success@2 | 0.50 | 0.575 | **0.65** | 0.575 |
| success@4 | 0.50 | 1.0 | 1.0 | 1.0 |

Paired: C_vs_A — both 20, C_only 20, A_only 0, **McNemar p=1.9e-6, delta +0.50**.
B_vs_C — 0 discordant pairs (identical outcomes), p=1.0. Call delta C−B = −0.25/task,
task-blocked CI90 [−0.475, −0.025] (excludes zero). Oracle outcome support = 1.0 (measurement only).

**Calibration (exact egress states, C-arm, n=149, pos 40 / neg 109):**
AUROC **0.5806**, Brier **0.1945 < baseline 0.1964**, Spearman **+0.124** (correct sign),
prob_var 0.00103, frozen audit 0 backbone / 49,409 disc. → calibration gate PASS.

## Pre-registered gates

- CALIBRATION: **PASSED** (AUROC>0.5, Brier<baseline, +sign Spearman, both classes, variance>0, frozen).
- CEGIS_OPERATIONAL: **NOT MET** — cost reduction −10.8% (2.075 vs 2.325) < pre-registered −20%;
  CI excludes zero (real but small). Outcome not worse (p=1.0).
- PROMOTION metric: **TRUE** (C vs A delta +0.50, McNemar 1.9e-6, CI lb > 0 on disjoint outcome tests).
- **Verdict: `CEGIS_VERIFIER_ASSISTED_NOT_CAPABILITY_PROMOTED`** (hierarchy: calib ∧ outcome-not-worse
  but ¬cost-gate).

## Interpretation (all OBSERVED)

1. **The 1.0 CEGIS admission is NOT tautological replay.** With disjoint verifier/outcome
   tests, the skeleton pool contains a program that passes verifier tests AND outcome tests
   for 40/40 tasks. Prior v0.5 concern (same-test replay) is retired by measurement.
2. **CEGIS-first on the skeleton pool = verifier-assisted capability** (1.0 outcome pass vs
   token beam 0.50, McNemar significant). The egress mechanism is the capability carrier.
3. **The discriminator is a genuinely calibrated but WEAK search-shaping signal**: ordering
   cost is monotone (disc 2.075 < uniform 2.325 < random 2.575 — the signal is real), CI
   excludes zero, but the improvement is 10.8%, below the pre-registered 20% gate, and it
   changes NO outcomes (B≡C).
4. **Ranking remains secondary**: uniform order already finds a passer in 2.3 calls; the
   discriminator's marginal value is small. The v0.4.1 token-state head (AUROC 0.75) is
   irrelevant here — this head is trained on egress states (AUROC 0.58, honest re-measure).

## Claim dispositions

- "Calibrated E_phi repurposed (ρ=0.4383)" — superseded: the v0.5.1 discriminator is a NEW
  head trained on egress states with verifier labels. Measured AUROC 0.581 (weak-positive).
- v0.5 CEGIS admit 1.0 — now PROVEN non-tautological via disjoint tests (was labeled
  CONDITIONAL pending this audit).
- "Unfreeze decoder" — NOT needed for this DSL: egress+CEGIS reaches 1.0 outcome pass.

## DSL bound (stated limitation)

7-rule grammar is tailored to the System-1 task DSL. This establishes structural egress +
verifier-efficient selection WITHIN the DSL, not general program synthesis.

## Next grounded steps

(a) Heldout re-seal is now DEFENSIBLE for the egress+CEGIS chain (gates passed on dev2_v051):
    seal a NEW single-use heldout split, run B (uniform CEGIS-first) vs A once.
(b) If call-cost matters (silicon budget): stronger discriminator (bigger head / more
    training data / better pooling) to clear the 20% gate; currently marginal.
(c) General-program capability requires grammar expansion beyond the 7-family DSL.

## Artifacts

- eval_v051_cegis.json `86428946...`; results `6879e5e6...`; dev2_v051.json `1f81e4d0...`;
  train_v051.json (split) `181cc59b...`; logs `e738b796...` / `c085dcb3...`
- lineage: `0be7794` (impl) + telemetry commit; bundle r9
