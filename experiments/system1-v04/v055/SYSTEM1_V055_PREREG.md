# System-1 v0.5.5 — Pre-registration, Frozen Manifest, and Verdict Gates

**Date:** 2026-08-24 (before any seal)
**Carrier:** corrected 13-rule structural egress (rule-10 defect repaired)

## 1. Carrier change (ONE line, rule 10 only)

```
frozen v0.5.4:  rule 10 body: "return sum(range({a}))"
v0.5.5:         rule 10 body: "return sum(range(len({a})))"
```

- Frozen v0.5.4 kernel `system1_kernel_v05_ast_skeleton.py` is NOT modified
  (sha `a237d4239f256c349676bbe2…` preserved).
- New carrier file: `system1_kernel_v055_ast_skeleton.py`
  (sha `d9a976adff4146a11950c51218ca32af1cde4b3db59431ce370a45913ff8d870`).
- `diff` proof: exactly `101c101`, one line changed, nothing else.

## 2. Frozen implementation closure (hashed BEFORE seal)

| File | SHA-256 (full) |
|---|---|
| system1_kernel_v055_ast_skeleton.py | d9a976adff4146a11950c51218ca32af1cde4b3db59431ce370a45913ff8d870 |
| system1_kernel_v041_energy_refactored.py | 754cbe4910e35c8f018f9b2fc411b3798d7b0ca83e7a42e2d37522358aa6fa1a |
| system1_kernel_v042_cegis_beam.py | 057e0ec1f39c58cfee5043c601c0e767cdc9e0dd43d6c2bf352a9e185f09f94a |
| train_system1_kernel_v04.py | 35025cfa5d40819d815ca4ec7f49b5bce16ab707fc3effab593907232e1cf67f |
| train_v051_discriminator.py | dd596e93327fb994b83407d0b7156cc491ccf65a233a643f1cec4925c0e6d651 |
| v041_energy_checkpoint.pt | 11d56121e4b091e2162078eb4cae71ce213dacc01397d8f8209bc9e2152a8f4d |
| eval_v055_heldout.py | 10bcd8a8bfe1ba73776393ca025e57b21cacfb41e47c6d72f1ad896a1da76f58 |
| contract_v055_closure.py | 45f2d6baf05c33680258c6e7 (patched before seal: added `_build_tests` building disjoint verifier/outcome fixtures exactly like `build_split` — `gen_task` output lacks `verifier_tests`; hashed post-patch) |

## 3. Semantic closure contract (must PASS before seal)

`contract_v055_closure.py`, gates C1–C9: pool non-empty; AST parses; FSA
closure; canonical candidate passes verifier fixtures; passes DISJOINT
outcome fixtures; min per-family support = 1.0; aggregate = 1.0; other 12
rules byte-identical to frozen; rule 10 emits `sum(range(len(…)))`.

## 4. Integrity guard (consumed digests, full SHA-256 where available)

Includes: `a09bf275…` (heldout52_v054, full), `5e5f4a00…`, `9a17af61…`,
`635c2aaa…`, `ce2a76fb…`, `888809df…`, `8ea34261…`, `d6b79d51…` (dev6_v060,
full), `392ce03e3b35…` (dev7_v061 exposed, full), `338cbda6…` (dev7b_v061,
full), `44657c7f…` (dev8_v062, full), `bed7368d…` (smoke60, full),
`928e40af…` (smoke61, full), `4b7d854d…` (smoke62, full), plus every earlier
smoke/contract/train/heldout prefix. Any match → `INVALID_VERIFIER_REPLAY`.

## 5. Heldout seal (generation-only, no checkpoint)

- Stratified: 13 families × 4 tasks = 52.
- Fresh seed and tag (recorded in seal receipt).
- Receipt fields: full SHA-256, seed, UTC, n, family counts,
  verifier/outcome partition sizes (4+4), generator identity,
  `single_use=true`, evaluator sha.
- CPU smoke uses a DIFFERENT disposable seed/tag; sealed bytes never loaded
  in smoke/contracts.

## 6. Evaluation arms (primary, matched budget 64 / beam 64)

- **A:** token beam + CEGIS-first (verifier tests).
- **B13:** corrected 13-rule uniform CEGIS-first carrier.
- NO other arms in the primary verdict (v0.6.0.1/v0.6.1/v0.6.2 are separate
  carriers; nothing attributed to them).

## 7. Pre-registered gates and verdict

`HELDOUT_13_RULE_CARRIER_CLEAN_PROMOTED` requires ALL of:

1. **Integrity:** fresh single-use seal, pinned `--expect-sha`, digest not in
   consumed list, single CUDA evaluation, no prior loading of the split.
2. **Validity:** admitted-program AST validity = 1.0 in BOTH arms (admitted
   denominator; non-admission is not invalidity).
3. **Family gate:** minimum per-family B13 outcome support = 1.0
   (all 13 families at 4/4 on DISJOINT outcome tests).
4. **Efficacy:** paired McNemar p < 0.05 AND task-blocked delta CI lower
   bound > 0.
5. **Cost accounting:** exact verifier-call mean/median/p90/max per arm.

Else: family gate failed → `HELDOUT_FAMILY_GATE_FAILED`; validity failed →
`HELDOUT_VALIDITY_FAILED`; efficacy partial → `CONDITIONAL_VERIFIER_…`;
delta ≤ 0 → `HELDOUT_PROMOTION_NOT_ESTABLISHED`; guard/leak →
`INVALID_VERIFIER_REPLAY`.

**No expected 52/52 is claimed before measurement.** This is a fresh
single-use heldout; `a09bf275…` is consumed and will NEVER be re-evaluated.
