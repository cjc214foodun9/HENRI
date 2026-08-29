# F2-M3 K2' Amendment — Comparator Replacement Proposal (DRAFT, REQUIRES APPROVAL)

Spec: SPEC-2026-08-29-F2-EGRESS (sealed)
Amendment ID: AMEND-2026-08-29-F2-K2PRIME
Status: DRAFT — no gate runs until approved and both hashes recorded.

## Problem (OBSERVED)

K2 (prereg): "margin ≥ +0.05 over legacy linear head on identical frozen waves."
The legacy head (`down_proj 65536→2048 + lm_head 32000`, checkpoint `75572389…`)
decodes 32k BPE TEXT tokens. The only fresh calibration corpus is the authorized
ARC trajectory bank whose labels are `GameAction` enums (ACTION1..6). No
legitimate token↔action mapping exists (`ACTION_HEAD_NOT_CALIBRATED`, sealed
Phase 6/7 lineage). A fabricated comparator is prohibited. Therefore K2 as
preregistered is `BLOCKED` on this corpus — this is a data-source constraint,
not a code defect.

## Proposed amendment (K2')

Replace the K2 comparator with the Phase 8.32b **calibrated algebraic action
head** (`henri_calibrator_ingest.py`, artifact schema `henri.calibrated-action-head.v1`),
which is trained/calibrated on the SAME bank schema (ACTION1..6, same wave
boundary `[M, 65536]`), on the SAME calibration split, evaluated on the SAME
held-out envs.

- K2' gate: `heldout_P1(F2) − heldout_P1(calibrated_head) ≥ +0.05` on identical
  frozen waves, plus per-action coverage on both sides.
- Both heads remain zero-trainable at evaluation (no-BPTT; frozen calibration).
- K1, G1, G2, K3, K4 unchanged.
- The legacy-head comparator stays `BLOCKED` (documented in the gates receipt),
  never replaced silently.

## Kill criteria (unchanged where applicable)

- K1: G1 held-out P@1 < 0.99 → carrier fails.
- K2': margin < +0.05 → fails the amended gate.
- K3: no engagement telemetry → fails.

## Boundary

Amendment applies ONLY to the comparator of gate K2. All other gates and the
split seal stand byte-identical. If K2' is rejected, the carrier reports
`GATES_PARTIAL: G1/G2 pass, K2 BLOCKED_BY_DESIGN` — never a fabricated margin.
