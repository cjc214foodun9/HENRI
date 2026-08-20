# Phase 8.39 — MMLU wave-rank verdict

Status: **EVALUATED — FALSIFIED (pre-registered kill)**
Branch: `phase839/humaneval-wave-ast` @ `3b00b01` (runner) / verdict `25f9f96`
Date: 2026-08-20

## Result (OBSERVED, remote RTX 5090 CUDA, D=65,536)
- 3,648 / 14,042 correct = **0.2598** accuracy
- Pre-registered gate: chance 0.25 + 0.05 = 0.30 → **margin +0.0098 < 0.05 → KILL**
- wall_clock 1,150.9 s; dataset SHA-256 `15b6785d49e0012602e089558a7a0dfb916baf97e9295aa25b48062f13c6afbb` (local = remote)
- egress_path = STRUCTURED_CODEC_WAVE_RANK; checkpoint_used = False (wave-rank path, not token decode)

## Geometry control (codec_geometry_control)
- mean distinct-option cosine: **0.3588** (vs random baseline 1/sqrt(D) = 0.0039) → codec vectors quasi-collinear, NOT orthogonal
- mean correct-option cosine 0.3430 vs wrong-option 0.3383 → lexical gap **+0.0047**, far below ranking resolution
- Same signature as GPQA Diamond (distinct 0.4529, gap +0.0168, acc 0.2980): **structured char-position codec is the bottleneck**, not the runner or gate.

## Attribution (INFERRED from two independent falsifications)
Character-position structured codec (qFHRR Z_256) encodes shared characters as near-identical superpositions; distinct options share 80%+ of characters at MCQ scale → cosine ~0.36-0.45 at D=65,536. Wave ranking cannot separate options the encoder cannot separate.

## Next fix (pre-registered, bounded)
Compositional text codec: **token-level engrams + position binding** (per run-20 verdict and PDF Evolution I remedy), replacing character-position superposition. Kill criteria: distinct-option cosine < 0.1 at D=65,536 AND correct>wrong gap ≥ 0.01 on held-out 500-item MMLU slice.
