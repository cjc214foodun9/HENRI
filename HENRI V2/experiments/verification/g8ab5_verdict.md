# G8AB5 Verdict — thermo serialization + 3-arm re-run (measured)

**Carrier:** carrier/g8-thermo-timescale @ `4856f3fd` (base c59aa0e)
**Prereg (sealed):** g8_thermo_serialization_prereg.md sha256 `45f6f005…`; governance `#265bd9ed`
**Run:** G8AB5, env ka59-38d34dbb, 60 steps, seed 20260908, overlay `7557238908` (verified)

## Preconditions (OBSERVED)
- SHA_PREFIX_OK 4856f3fd; WT_DIRTY=0; HAS_DSN=True
- FULLSUITE_RC=0 (1346 passed / 12 skipped)
- CONTRACT_RC=0 (17 passed: thermo_partition + g8_wiring + runner serialization)

## Predicates (from pulled telemetry, /tmp/g8ab5_{arm}/*.jsonl -> local g8ab5/")
| Arm | rows | key_present | ratios_nonnull | gibbs_true | explored | admissible |
|---|---|---|---|---|---|---|
| off  | 68 | 60/60 | 0   (partition OFF -> null, correct) | 0 | 60/60 | 4 (60 steps), 8 rows non-step |
| prior| 68 | 60/60 | 60/60 | 0 | 60/60 | 4 |
| post | 68 | 60/60 | 60/60 | 0 | 60/60 | 4 |

ratio sample (prior, e=4): beta_sigma=8.0, beta_s=0.0625, beta_j=16.970563, n=0.007812, m=0.003683
ratio sample (post, e=4): beta_sigma=8.0, beta_s=0.176777, beta_j=16.970563, n=0.022097, m=0.010417
-> P1–P4 all satisfied; timescale-separation invariants (beta_sigma>beta_s, beta_j>beta_s) hold per row.

## Action divergence (64 common steps)
off vs prior: 38 diffs; off vs post: 36; prior vs post: 45. Action identity (payload-aware) identical counts.
levels_completed: 0 occurrences in all arms (no external outcome; diagnostic run).

## Verdict per prereg class
`G8_DIAGNOSTIC_HOLE_CLOSED_GIBBS_PREEMPTED`
- Hole closed: thermo_ratios + thermo_gibbs now serialize into every runner record.
- Gibbs firing = 0/60/arm: the pre-existing T4 epistemic arm preempts the Gibbs branch
  (documented precedence in test_g8_wiring.py test_on_high_loss_ema_prefers_explore_arm).
- Selection divergence measured (38–45 diffs) but NOT attributable to Gibbs (gibbs=0);
  attribution of the divergence source is a separate item (likely softmin/shaping side
  effects of the partition path; not claimed here).
- G8 promotion remains withheld per prereg (no Gibbs-engagement selection-identity
  evidence). No main change (10f5f23 untouched).

## Sealing
Governance: `G8_DIAGNOSTIC_HOLE_CLOSED_GIBBS_PREEMPTED` (#sealed via henri_audit.py record).
Telemetry: C:/Users/chan/henri-telemetry/g8ab5/{g8ab5_off,prior,post}.jsonl (119–127 KB each).
