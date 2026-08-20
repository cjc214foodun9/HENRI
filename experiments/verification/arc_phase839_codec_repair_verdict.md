# Phase 8.39 — Codec repair lever verdict (Evolution I remedy)

Status: **FALSIFIED (pre-registered kill fired)**
Branch: `phase839/humaneval-wave-ast` @ `ae90acf` + probe commit
Date: 2026-08-20

## Kill experiment (200-item MMLU slice @ D=65,536, CPU)
Pre-registered launch gate: proxy accuracy >= 0.31 (full-run gate 0.30; slice SE ~0.03).

| position_mode | acc | correct>wrong gap |
|---|---|---|
| full (status quo) | **0.2950** | +0.0050 |
| none | 0.2800 | +0.0197 |
| independent (compositional) | 0.2700 | −0.0002 |
| shuffled | 0.2600 | +0.0087 |
| word_engram | 0.2150 | +0.0003 |

**No mode clears 0.31. Kill fired. No GPU run launched.**

## Geometry facts (OBSERVED)
- `full` distinct-option cosine 0.5513 @ D=65,536 vs random baseline 0.0039 (collinear position carrier; shared chars dominate)
- `independent` distinct-option cosine −0.0017 (orthogonal; at baseline) — compositional fix is geometrically real but destroys the only chance-beating signal (acc 0.295 → 0.270)
- `word_engram` acc 0.215 — word atoms lose the character-overlap bias with no lexical gain

## Inference
The structured character-position codec's chance-beating signal (+4.8%) IS the shared-character collinearity, not semantic composition. All repairs that orthogonalize or abstract away character overlap remove the signal. Static-vector cosine ranking of question→option is at its ceiling (~0.26–0.30) for this codec family. **The egress path for MCQ text ranking must change representation, not parameters** (per PDF: Hopfield lexical snap over Zone C engrams; run-20: learned projection head).

## State
- Default `position_mode="full"` unchanged (no default-path change).
- New modes (`independent`, `word_engram`) remain available but are NOT promoted.
- Unit suite: 170 passed / 1 skipped.
- Campaign registry: GPQA + MMLU EVALUATED (FALSIFIED); codec-repair lever CLOSED.
