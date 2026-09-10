# E3 Addendum — Root-Cause Diagnostic (read-only, no training)

**Companion to:** `e3_verdict.md` (E3_FAIL both arms, receipt `f738ff7a…`)
**Diagnostic:** `experiments/verification/e3_diag_target_gap.py` (read-only; no weights modified)
**Artifact:** `/root/e3/diag_target_gap.json`, sha256 `bdf589e39bf2318c9f678292…` (pulled locally)

## Measured (OBSERVED, n=64 eval windows, sealed split)

| Quantity | Value |
|---|---|
| mean cos(z_hat, **window-centroid target**) — E2's own training objective | **0.399208** |
| mean cos(z_hat, **gold next-token embedding**) — what token P@1 needs | **0.020492** |
| median gold-token rank under z_hat | **19,364** |
| mean gold-token rank | 41,577.8 |
| gold top-1 rate | 0.0 |
| gold ∈ probe top-16 teacher rows | **0.0625** (4/64) |
| vocab | 151,936 |
| reading | `E2 objective satisfied / token objective not` |

## What this establishes (and what it falsifies)

1. **The frozen E2 head satisfies its own objective and does not satisfy the token objective.**
   cos to the window-centroid target is 0.399 (≈20× the token cosine, and far above the
   ~0.0039 random-ring null for this dimension). The head is not broken; it is doing
   exactly what the contrastive loss asked of it.
2. **The "readout construct mismatch" diagnosis is FALSIFIED.** Arm A read the *same*
   frozen checkpoint through the mechanism's own k-NN probe (k=16, τ=0.07) → cluster
   centroid → token scores, and scored **P@1 0.015** versus Gate 3.1's raw-point
   **0.016** — statistically indistinguishable. Changing the readout does not change
   the outcome.
3. **The real gap is the TARGET DEFINITION, and it is an information problem.**
   The probe's top-16 teacher rows contain the gold next token only **6.25%** of the
   time. The cluster centroid therefore *cannot* rank the gold token first: the
   correct answer is absent from the candidate set 93.75% of the time. Median gold
   rank 19,364 of 151,936 is consistent with a centroid that encodes a *window-level
   semantic blend*, not a next-token prediction.
4. **Arm B's 4.9× improvement (0.016 → 0.079) is real but bounded by supervision.**
   CE began at 11.9346 against `ln(151936) = 11.9312` (uniform) and ended at 11.6916:
   a 0.24-nat movement in 3 epochs over 10,000 windows against a 151,936-way target.
   Token-level CE is learning, just far too slowly for a 1-epoch-scale budget; G2
   (needs `last < 0.9 × first`) correctly failed.

## Honest conclusion

The E2 contrastive objective is **not** the defect, and neither is the readout
function. A window-centroid target cannot be decomposed into a next-token
distribution by re-reading it — the information is not present in the representation.
Any successor carrier must either (a) supply token-level supervision at a scale
commensurate with the vocabulary (larger corpus, more epochs, or a curriculum), or
(b) change the contract explicitly (frozen revision-pinned backbone) rather than
retuning the same probe.

Bounds 0.285 / 0.640 were **not** met, were **not** renegotiated, and no export was
written. Main remains `10f5f23`.
