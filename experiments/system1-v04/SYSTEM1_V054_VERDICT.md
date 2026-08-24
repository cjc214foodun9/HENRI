# System-1 v0.5.4 Heldout Verification — HELDOUT_13_RULE_CARRIER_PROMOTED (as frozen, rule-10 defect disclosed)

**Date:** 2026-08-24 (CUDA 5090, vast-5090, ~06:59 UTC, PID 394381)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121...` (frozen)
**Split:** heldout52_v054 (FRESH single-use, seed 99991, sha `a09bf275...`, n=52 stratified 13 families × 4, disjoint verifier/outcome 4+4)
**Option B authorized by user.** Pre-registered before seal (commit `2a18ec2`).

## OBSERVED (CUDA, budget 64, beam 64)

| Metric | A token beam | B13 (13 rules) | B7 (diagnostic) |
|---|---|---|---|
| outcome pass (disjoint) | 0.231 | **0.923** | 0.538 |
| ci90 | [0.135, 0.327] | [0.865, 0.981] | [0.423, 0.654] |
| verifier calls (mean) | 1.0 | 4.385 | 3.0 |
| AST validity (admitted) | 1.0 | 1.0 | 1.0 |
| family coverage | 3/13 families | 12/13 families | — |

Paired B13 vs A: both 12, B13_only 36, A_only 0, neither 4 — **McNemar p=2.9e-11**. Delta +0.692 (CI90 [0.577, 0.788]). Oracle pool support 0.923 (48/52).

**Gates: promotion TRUE, validity_preserved TRUE (min_ast_valid 1.0), integrity TRUE (fresh seal, pinned sha, one run).**

## Verdict: `HELDOUT_13_RULE_CARRIER_PROMOTED` — AS FROZEN, WITH DEFECT

The 13-rule carrier (frozen set, hashes in eval docstring) beats the token beam on a fresh single-use heldout. This is the heldout promotion claim for the carrier as implemented.

## DEFECT DISCLOSURE (post-measurement, OBSERVED)

- Rule 10 (`range_sum`, family 10) template body: `return sum(range({a}))` — generates `sum(range(xs))`, which raises TypeError (list passed to `range`). Canonical reference: `sum(range(len(xs)))`.
- Consequence: family 10 has 0/4 heldout support and 0 dev support — structural, NOT selection. All 4 fid-10 tasks are non-admissions (correctly excluded from the validity denominator).
- Confirmed locally: rule-10 candidate sandbox verifier=0 / outcome=0; canonical verifier=1 / outcome=1.
- v0.5.3 record CORRECTION: "new-family 0.95" was 5/6 families at 1.0 + family 10 at 0.0. The 0.95 aggregate hid a family-level 0.0. The defect was latent in dev3_v053; I did not catch it during that closure.
- The 0.923 heldout pass is a FLOOR: 48/52 = all 12 healthy families (1.0) + family 10 (0.0). A fixed rule 10 is expected to reach 52/52 but is NOT covered by this seal.

## Attribution boundary (pre-registered, honored)

- A vs B13 promotes the 13-rule carrier as frozen.
- B7 diagnostic only; B13-vs-B7 incremental attribution NOT claimed from this run (monotone again: B13_only 20, B7_only 0 — consistent with dev).
- Heldout-isolated grammar-expansion attribution beyond the carrier is NOT claimed.

## Next (pre-registered)

- **v0.5.5 (proposed):** one-line fix rule 10 body → `return sum(range(len({a})))`; re-freeze + hash; NEW single-use heldout seal (a09bf275 is consumed); single A vs B13 run; expected 52/52 if no other defect. No re-evaluation of a09bf275.

## Artifacts

- eval_v054_heldout.json `ad037335...`; results `b2704cf3...`; log `a3aa7055...`; split `a09bf275...`; seal receipt (UTC 06:54:26Z, single_use)
- lineage: `2a18ec2` (pre-registration + seal), verdict commit, bundle r12
