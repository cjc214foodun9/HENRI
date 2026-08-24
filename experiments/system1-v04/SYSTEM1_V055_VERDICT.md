# System-1 v0.5.5 — HELDOUT_13_RULE_CARRIER_CLEAN_PROMOTED (52/52)

**Date:** 2026-08-24 (CUDA 5090, vast-5090, PID 421309)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121…` (frozen, unchanged)
**Carrier:** `system1_kernel_v055_ast_skeleton.py` sha
`d9a976adff4146a11950c51218ca32af1cde4b3db59431ce370a45913ff8d870`
(rule-10 one-line fix: `sum(range({a}))` → `sum(range(len({a})))`; diff proof 101c101)
**Split:** `heldout53_v055` — FRESH single-use, sha
`873902867001b7f19abf3e2641b10913f2c57c4fb2e23d609078a471edc9c9ed`,
seed 60013, n=52 stratified 13 families × 4, `single_use=true`,
sealed in generation-only mode before any evaluation; pinned via
`--expect-sha` on the CUDA run. Guard (full SHA-256 where available)
passed at seal time and at load time; never loaded in smoke/contracts.

## OBSERVED (CUDA, single evaluation)

| Metric | A: token beam | B13: corrected 13-rule CEGIS-first |
|---|---|---|
| outcome pass (disjoint tests) | 0.2308 (12/52) | **1.0 (52/52)** |
| CI90 | [0.135, 0.327] | [1.0, 1.0] |
| AST validity (admitted) | 1.0 | 1.0 |
| verifier calls mean / median / p90 / max | 1.0 / 1 / 1 / 1 | 4.23 / 4 / 8 / 9 |
| oracle outcome support | — | 1.0 |
| family 10 (range_sum) | 0/4 | **4/4** |
| all other families | — | 4/4 each |

Paired B13 vs A: both 12, B13_only 40, A_only 0, neither 0 →
**McNemar p = 1.82e-12**. Delta B13−A = **+0.7692**, task-blocked CI90
[0.6731, 0.8654] (lb > 0).

## Gates (pre-registered, all TRUE)

- integrity: TRUE — fresh seal, pinned sha, single CUDA run, no prior load
- validity: TRUE — admitted-program AST validity = 1.0 in both arms
- family gate: TRUE — minimum per-family B13 outcome support = 1.0
  (13/13 families at 4/4 on DISJOINT outcome tests)
- efficacy: TRUE — McNemar p < 0.05 AND delta CI lb > 0

## Verdict

**`HELDOUT_13_RULE_CARRIER_CLEAN_PROMOTED`**

The v0.5.4 disclosed defect (rule-10 structural TypeError) is REPAIRED and
the corrected carrier passes every pre-registered gate on a fresh single-use
heldout. No replay of `a09bf275…` (heldout52_v054) or any consumed digest.

## Integrity record

- Pre-registration + frozen manifest: commit `1c8ab6e` (+ manifest
  correction `9e04645`: contract final sha `45f2d6ba…` after `_build_tests`
  patch — hashed post-patch, pre-seal).
- Semantic closure contract C1–C9: **ALL PASS 13/13** before seal.
- Seal receipt: commit `089b7b5`.
- Remote dependency-closure preflight: evaluator `10bcd8a8…`, carrier
  `d9a976ad…`, split `87390286…` verified byte-identical on remote; rule-10
  `sum(range(len(` probed remotely; GPU idle before launch.
- Verdict + telemetry: commit (next); bundle r15; Drive `v055_heldout/`.

## Attribution boundary

- Promotes the CORRECTED 13-rule carrier (A vs B13) on a bounded 13-family
  DSL. General synthesis beyond the DSL is NOT claimed.
- v0.6.0 (retrieval), v0.6.1 (fast weights), v0.6.2 (partition) remain
  separate carriers with their own NO_EFFECT / DIVERSITY_ONLY verdicts;
  nothing from them is attributed to this promotion.
