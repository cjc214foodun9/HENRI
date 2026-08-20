# Phase 8.39 — HumanEval grammar-extension rerun verdict

Status: **EVALUATED — NO GAIN (observed negative)**
Branch: `phase839/humaneval-wave-ast` @ `4e3f84b`
Date: 2026-08-20

## Result (OBSERVED, remote RTX 5090 CUDA)
- solved = **2 / 50** — identical to sealed baseline (02d6d45)
- expressible = 50/50; not_expressible = 0; infra_errors = 0
- total candidates generated = 4,661 (up from ~2,500 baseline pool via string-op + list-indexing templates)
- wall_clock 21.1 s; avg 416 ms/item; dataset SHA `b796127e…`; egress WAVE_AST_GRAMMAR_SANDBOX
- Scorecard: `experiments/verification/humaneval_839b_scorecard.json`

## Inference
Adding grammar bodies expands the candidate pool but does not change wave-ranking winners: the same 2 tasks solved. Coverage is not the binding constraint; **candidate ranking by the transformation-relative wave similarity is**. Grammar-extension as a lever = FALSIFIED for score improvement at this ranking quality.

## Next lever (from bottlenecks PDF + run-20 verdict)
Improve the ranking signal itself (compositional token-level codec) rather than the grammar enumeration.
