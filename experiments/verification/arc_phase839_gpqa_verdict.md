# Phase 8.39 — GPQA Diamond wave-rank verdict

Status: **EVALUATED — FALSIFIED (pre-registered kill)**
Branch: `phase839/humaneval-wave-ast` @ `10c254f`
Date: 2026-08-20

## Result (OBSERVED, remote RTX 5090, D=65,536)

| metric | value |
|---|---|
| item_count | 198 (canonical simple-evals subset) |
| correct | 59 |
| accuracy | 0.2980 |
| chance | 0.2500 |
| margin | +0.0480 |
| accept_margin | +0.0500 |
| verdict | FALSIFIED |
| dataset_sha256 | `41d1213c…fcd305` |
| scorecard | `experiments/verification/gpqa_839_scorecard.json` |
| wall_clock | 22.6 s (encode + rank, CUDA) |

## Mechanism

Zero-demo structured qFHRR (character-position) codec wave ranking: question wave vs
each option wave, cosine on ring-to-real unit vectors, argmax = selection.
Canonical exact-match checker on the Correct Answer string.

## Geometry control (codec quality floor)

- mean distinct-option cosine: **0.4529** (random baseline 0.0039)
- mean correct-option cosine: 0.2185
- mean wrong-option cosine: 0.2017

The codec is far from orthogonal at D=65,536: shared characters dominate pair
similarity. The correct-vs-wrong cosine gap (+0.0168) is a weak but real lexical
signal — it produces the +4.8% over chance, which is BELOW the pre-registered
+5.0% acceptance gate.

## Inference

1. Character-position qFHRR similarity is not a task-competent MCQ ranking
   mechanism at zero demo. Kill stands.
2. The kill is informative: ranking signal exists (correct > wrong, n=198),
   it is just below gate. A calibrated semantic head or in-context demo pairs
   (WaveAST-style grammar + ranking) is required to cross the gate — consistent
   with the HumanEval 2/50 wave-AST result where structure came from grammar
   enumeration, not raw cosine.
3. Registry: GPQA Diamond → `EVALUATED` (verdict FALSIFIED). No score promotion.

## Governance

EVENT_ID/AUDIT_HASH appended via `agentic_event_store.py` (verification.jsonl).
