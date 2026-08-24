# System-1 v0.5.3 Grammar-Manifold Expansion — SUPPORT_EXTENDED_NO_DILUTION

**Date:** 2026-08-24 (CUDA 5090, vast-5090, 06:24 UTC, PID 391107)
**Checkpoint:** ckpt_v041/checkpoint.pt sha `11d56121...` (frozen)
**Split:** dev3_v053 (fresh DISPOSABLE, seed 37123, sha `9a17af61...`, n=60: 40 old-family + 20 new-family, disjoint verifier/outcome 4+4)
**Purpose:** directive item 3 — broaden generative support Supp(P_gen) beyond the 7-family DSL. Dev cycle only; NOT a heldout claim.

## OBSERVED (CUDA, budget 64, beam 64, arms matched except grammar size)

| Metric | A token beam | B7 (7 rules) | B13 (13 rules) |
|---|---|---|---|
| outcome pass (disjoint tests) | 0.300 | 0.667 | **0.983** |
| old-family pass (fid<7) | — | **1.0** (40/40) | **1.0** (40/40) |
| new-family pass (fid>=7) | — | 0.0 (no rule) | **0.95** (19/20, CI90 [0.85, 1.0]) |
| mean distinct programs/task | — | 3.63 | **7.17** |
| mean verifier calls | 1.0 | 2.78 | 3.97 |
| paired B13_vs_B7 | — | both 40, B13_only 19, B7_only 0, neither 1 | **McNemar p=3.8e-06** |
| paired B13_vs_A | — | both 18, B13_only 41, A_only 0, neither 1 | **McNemar p=9.1e-13** |

## Pre-registered gates

- G1 SUPPORT_NEW: **TRUE** (B13 new-family 0.95, CI lb 0.85 > 0) — support extended to 6 new DSL families.
- G2 NO_DILUTION: **TRUE** (B13_old 1.0 >= B7_old 1.0 − 0.10) — run14 rank-dilution warning NOT reproduced.
- G3 OVERALL: **TRUE** (0.983 >= 0.667).
- G4 COST: **TRUE** (3.97 <= 2.78 × 1.5 = 4.17) — not explosive, but expansion costs +43% verifier calls (more rules to scan).
- kill = **null** → 13-rule grammar validated on dev.

## Interpretation (all OBSERVED)

1. B7's 0.667 overall = old-only support (40/60): new-family tasks have NO matching rule, so B7 fails all 20. B13's 0.983 = old 1.0 + new 0.95 — clean additive extension.
2. The run14 warning (more grammar shapes displaced simple correct programs) was the failure mode to watch; it did not occur: old-family pass stayed 1.0 → 1.0. The monotonicity contract (C6: B7 pool ⊆ B13 pool) makes old-family regression structurally impossible at budget ≥ 13.
3. Grammar expansion is a capability carrier: B13 vs A is the widest gap yet measured (0.983 vs 0.300, McNemar 9.1e-13).

## Boundary (explicit)

- dev-split result only. The v0.5.2 heldout (5e5f4a00) remains the ONLY consumed heldout; the 13-rule grammar has NO heldout claim yet.
- A heldout promotion claim for the expanded grammar requires a NEW single-use seal + pre-registered verdict — a separate future cycle.

## Artifacts

- eval_v053_expansion.json `c69d6076...`; results `46b9d687...`; log `6aec29eb...`; split `9a17af61...`
- lineage: `e74731f` (impl+contracts) + verdict/telemetry commit; bundle r11
