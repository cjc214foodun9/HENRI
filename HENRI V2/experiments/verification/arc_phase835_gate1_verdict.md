# Phase 8.35 Sprint a — Gate 1 Sealed Verdict (2026-08-19)

Verdict: **PARTIAL** (transition fixed; egress FALSIFIED at representation level)

## Run identity
- Branch: feat/phase835-analog-traveling-wave-vla @ 5e348ac
- Bank: stratified harvest_1787182225 (66 records, 54/12 split, seed 20260819,
  all 6 actions >= 10, digest verified, npz_sha 6097a13b854c69803933a1062f0f556f5f45ccb93520e2d01b1d5d9b7d9f2b94)
- Device: CUDA (RTX 5090), D=65,536
- Gate (HENRI-SPEC-MI-TRAJECTORY-2026 addendum): ACCEPT iff D <= 0.15 AND
  I_norm >= 0.85 AND acc >= 0.80

## Metrics (OBSERVED)
| Arm | Holdout (1-cos) | acc | I_norm |
|---|---|---|---|
| A linear r=16 | 0.9861 | 0.1667 | 6.7e-12 |
| B coupled r=128 | **0.0119** | 0.1667 | 2.2e-08 |
| C control | 0.0534 | 0.1667 | 2.2e-08 |
| D AP r=128 | 0.0646 | 0.1667 | 4.2e-08 |
| E PA r=128 | 0.0573 | 0.1667 | 2.8e-08 |

- Transition gate MET: B 0.0119 (best in HENRI history; 8.34 = 0.3153),
  D 0.0646 <= 0.15.
- Egress gate UNMET: acc 0.1667 = constant-classifier score; I_norm ~1e-8.

## Kill experiment (OBSERVED, same bank, next-wave space)
- A env-mixed prototype acc: 0.500 (env-partition artifact)
- B env-conditioned match fraction: 1.000 (12/12)
- C within-env same-action cosine: 0.9961 (n=334)
- D within-env cross-action cosine: 0.9958 (n=1811)
- E action separation (C-D): 0.0003

## Inference
Within an environment, the six actions produce near-identical next-waves at
D=65,536 (cosine separation 0.0003). The UWE spatial-grid wave encodes
environment state, not action identity. Action-egress from passive
next-wave prototypes is FALSIFIED at the representation level on this data.
The random/quota-forced harvest policy is action-balanced but NOT
action-informative (grid deltas dominated by env dynamics). The
reference-bladed I_norm gate is working correctly (uniform confs -> ~0.0,
strictly bounded [0,1]).

## Disposition
- Transition machinery (coupled EDMD + directional traveling wave) is the
  strongest production result in HENRI history: ACCEPT for continuation.
- Egress via next-wave prototype snap: FALSIFIED. Do NOT promote.
- Next bounded step (8.35b per staged plan): Kuramoto early-stop (R >= 0.85),
  field r 128->256, P_null thermostat coupling — representation-side
  work; then egress direction requires action-informative trajectories
  (policy that produces action-distinctive grid deltas) or an
  action-conditioned codebook that includes env context.

## Evidence artifacts
- Bank: /root/henri-835-stratified-bank/trajectories_harvest_1787182225.{npz,jsonl,_manifest.json}
- Benchmark: HENRI V2/experiments/verification/arc_phase835_gate1_benchmark.py
- Harvester: HENRI V2/cegis_self_play_sandbox.py
- Prereg: HENRI V2/experiments/verification/arc_phase835_gate1_prereg.md (with addendum)
- Tests: HENRI V2/tests/unit/test_henri_phase835_reference_bladed_mi.py (8/8)
- Local suite: 585 passed / 3 skipped
