# Tier_I / Tier_II gauntlet fixtures — UNVERIFIED INSTRUMENTATION

Status: `UNVERIFIED_INSTRUMENTATION`
Class: **not a result**. These telemetry records must not be cited as benchmark evidence.
Date identified: 2026-09-16. Identified by: direct re-read of the on-disk JSONL bytes.

## What was found (`OBSERVED`)

Seven `telemetry/gauntlet_*.jsonl` files, each 8 rows, each with `status: "PASSED"`.

Across all seven runs, **five** metrics carry byte-identical values — and **two of the
five are SCORE claims**:

| Metric | Tier | Constant value across all 7 runs | Status carried |
|---|---|---|---|
| `AA_Composite_Intelligence_Index` | Tier_IV | `0.7262000000000001` | **PASSED** |
| `ARC_AGI_3_RHAE_Score` | Tier_III | `0.7346938775510203` | **PASSED** |
| `Viscoelastic_Adapted_Loss` | Tier_I | `0.04` | PASSED |
| `Geodesic_Goal_Distance` | Tier_I | `0.0412` (`initial_distance = 0.854`) | PASSED |
| `Conservation_Error_Final` | Tier_I | `0.018` (`sample_steps = 35`) | PASSED |

Identified by `experiments/verification/telemetry_invariance_guard.py`:

```
[FLAG] AA_Composite_Intelligence_Index   n=7 distinct=1  status=['PASSED'] -> NONVARYING_FIXTURE
[FLAG] ARC_AGI_3_RHAE_Score              n=7 distinct=1  status=['PASSED'] -> NONVARYING_FIXTURE
[FLAG] Conservation_Error_Final          n=7 distinct=1  status=['PASSED'] -> NONVARYING_FIXTURE
[FLAG] Geodesic_Goal_Distance            n=7 distinct=1  status=['PASSED'] -> NONVARYING_FIXTURE
[FLAG] Viscoelastic_Adapted_Loss         n=7 distinct=1  status=['PASSED'] -> NONVARYING_FIXTURE
[OK  ] Holographic_Prefetch_Latency_ms   n=7 distinct=7  status=['PASSED'] -> PASSED
[OK  ] Online_Update_Latency_ms          n=7 distinct=7  status=['PASSED'] -> PASSED
[OK  ] Sagnac_Veto_Phase_Delta           n=7 distinct=2  status=['PASSED'] -> PASSED
```

Three of the eight metrics DO vary between runs. The file is not wholly fabricated; the
five flagged metrics are.

## Why this is the most serious finding in the set

`ARC_AGI_3_RHAE_Score = 0.7346938775510203` and
`AA_Composite_Intelligence_Index = 0.7262000000000001` are **benchmark score claims
carrying `status: "PASSED"`**, and they do not move across seven independent runs.

The project's own record contradicts them:
`experiments/verification/aaii_v42_testtime_sufficiency_audit.md` states HENRI is
`NOT_EVALUATED` on all 10 AAII members and that no hosted endpoint exists — and the
capability-audit gauntlet (`gauntlet_artificial-analysis_*_9345d6`) reports `0.0` with
`UNFIT` on every member. A constant `0.7262` "PASSED" composite cannot coexist with a
measured `0.0 / NOT_EVALUATED`.

This is the exact defect class the arbiter filter exists to reject: a plausible-looking
score that is a literal, not a measurement.

## Disposition

1. **Do not rewrite the JSONL.** Evidence is write-once (CLASS52). The bytes stay exactly
   as recorded and this notice is the correction. The original record remains auditable.
2. **Downgrade the class.** Every claim resting on these seven files is
   `UNVERIFIED_INSTRUMENTATION`, never `OBSERVED` capability.
3. **The score metrics are withdrawn.** `ARC_AGI_3_RHAE_Score = 0.7347…` and
   `AA_Composite_Intelligence_Index = 0.7262…` must not be cited anywhere. The live
   position remains: ARC 60-task `diag_ls` 0.4215 vs oracle ceiling 0.7536; AAII
   `NOT_EVALUATED`.
4. **The producer is already quarantined.** `grep` over the live tree finds
   `efficiency_gain_x` (a `details` field carrying only two distinct values,
   `286102.294921875` and `9155273.4375`) only in these JSONL files plus
   `_archive/henri_benchmark_gauntlet.py` and
   `_archive/invalid_evaluators/henri_benchmark_gauntlet.py`. **No live module imports
   either** (`0` live importers). The constant emitter is dead code and cannot emit a new
   fixture today.
5. **Guard the future.** `experiments/verification/telemetry_invariance_guard.py` is the
   reusable check: before emitting `status: PASSED`, a writer must compare its metric
   against prior runs and refuse the PASS when variance across runs is zero. Wire it into
   any revived gauntlet writer.

## Not claimed

This notice concerns only the seven `gauntlet_<epoch>_<hash>.jsonl` files. The
`gauntlet_artificial-analysis_*`, `gauntlet_full-production_*`, and
`gauntlet_robotics-deepmind_*` files use `evidence_class` labels and are not covered by
this finding.

