# E1 Verdict — calibrated egress projection head (measured)

**Carrier:** carrier/e1-egress-calibration @ `d32409c7`
**Prereg (sealed):** e1_egress_calibration_prereg.md @ `297d25a1`, sha256 `02560a9a…`
**Date:** 2026-09-08

## Remote verify (Vast 5090, GPU-exclusive, exact SHA, overlay `7557238908`)

| Stage | Result |
|---|---|
| Full suite | RC=0 (1342+ passed, overlay staged; bucket-4 resolved) |
| E1 contract tests | 7/7 (incl. per-block rotation geometry) |
| Teacher harvest | RC=0, `[151936, 896]`, shard sha `88c14255…` == pin |
| Scaffold | RC=0, 16 pairs, loss 2.714 → 1.085 (descent), 3 steps |

## Full run (real waves via G7 HighOrderCodec + frozen Qwen2.5-0.5B targets)

Corpus: Corberi 2609.04732 text, 226 sentences → 180 calib / 46 eval pairs
(ordered 80/20 split, seed 20260908, 3 epochs).

| Gate | Criterion (prereg) | Measured | Verdict |
|---|---|---|---|
| G1 isometry | ‖WᵀW − I‖_F ≤ 1e-4 | 1.24e-11 | **PASS** |
| G2 loss descent | avg last < 0.9 × avg first | 1.580 < 0.9 × 3.473 | **PASS** |
| G3 retrieval | P@1 > untrained + 0.05 AND > random + 0.05 | trained −0.102, untrained −0.133, random −0.118 (margins +0.031 / +0.016) | **FAIL** |
| G4 causal consumption | ≥ 8/16 probes change argmax under O(8) rotation | 16/16 | **PASS** |

**Verdict: E1_GATES_FAIL. Checkpoint NOT exported. Main untouched. Default-OFF.**

## Diagnostics (DERIVED from receipt)

- All arms ordered correctly (trained > untrained > random), but the metric floor is
  negative cosine (~−0.12): the eval metric measures cosine of the NEAREST SINGLE
  teacher row against a multi-token CENTROID target. The training objective aligns
  features to centroids; the gate measures nearest-row-to-centroid. Metric/target
  mismatch, not necessarily a mechanism falsification. Second limiting factor:
  180-pair corpus is far below the prereg's stated ≥4096 calibration-pair design
  intent for a 151,936-row retrieval space; effective contrastive support is tiny.
- G1/G2/G4 pass → head geometry, retraction, descent, and causal consumption are
  all functional. The carrier machinery is verified; the semantic gate did not
  clear under the sealed criteria at this corpus scale.

## Next actions (pre-registered for a NEW carrier, not a rerun)

1. Diagnose retrieval metric: replace nearest-single-row cosine with (a) k-NN
   embedding-space retrieval against the same centroid target, or (b) cross-batch
   InfoNCE P@1 on the eval split — under a NEW sealed prereg.
2. Scale the real corpus (≥10k pairs; e.g. staged text shards) with provenance
   pins; keep the frozen teacher revision immutable.
3. Re-run gates only after amendment sealed; never replay this 46-pair eval as a
   fresh heldout (it is consumed).

## Evidence

- `e1_full_receipt.json`, `e1_scaffold_receipt.json` (this dir)
- Governance: `#4397e49f1886f589` E1_GATES_FAIL; `#a30e0dc6e80f58eb` E1_PREREG_SEALED
- Telemetry raw: `/root/e1-calib/out_full/full_receipt.json` (Vast), mirrored
  `C:/Users/chan/henri-telemetry/e1/`
