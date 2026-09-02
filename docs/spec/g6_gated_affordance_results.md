# Carrier G6 Results — Piecewise Gated Subspace Selection & Pure-Support Preservation

**Directive:** `HENRI-DIR-2026-09-V3-CARRIER-G6-PIECEWISE-GATING` (SHA `d5ec31cd…`, 18,268 B, 341 lines), packet sealed `#5de066da` @1,185.
**Prereg:** `docs/spec/g6_gated_affordance_preregistration.md` (SHA `7c371c0f…`, sealed `#52782c82` @1,186).
**Branch:** `feat/carrier-g6-gated-affordance` @ `b7292c0` (engine + tests; remote CUDA 17/17).
**Parent:** G5 verdict `G5_AFFORDANCE_FIT_COLLAPSE` `#eeed5b17` @1,181.
**Verdict:** `G6_AFFORDANCE_FIT_COLLAPSE` — sealed `#e57da0034d61e7e5` (ledger @1,187). **26th sealed falsification / 27th carrier.**

## Gauntlet (OBSERVED, remote RTX 5090, seed 20260929, 12 envs × 150, 0 live steps by design)

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| PG1 global min_auc_subset | ≥ 0.8800 | 0.9412 | ✅ |
| PG1a a0 | ≥ 0.9500 | 1.0 | ✅ |
| PG1a a1 | ≥ 0.9500 | 1.0 | ✅ |
| PG1a a2 | ≥ 0.9500 | **1.0** (G5: 0.9231) | ✅ **FIXED** |
| PG1a a3 | ≥ 0.9500 | **0.9412** | ❌ **KILL** |
| PG1a a4 | ≥ 0.9500 | 1.0 | ✅ (bridge) |
| PG1a a5 | ≥ 0.8800 | 1.0 | ✅ |
| PG1a a6 | ≥ 0.8800 | 1.0 | ✅ (bridge) |
| PG2 norm drift | ≤ 1e-6 | 1.19e-7 | ✅ |
| PG3 CPU==CUDA | exact | passed | ✅ |
| C2 dense α | exactly 0.0 | all 7 α = 0.0 | ✅ |

Full-bank AUC: a0 1.0 · a1 1.0 · a2 0.9863 · a3 0.9775 · a4 1.0 · a5 0.9911 · a6 1.0.
Routes: a0–a3/a5 `topk_pure` (α=0.0), a4/a6 `bridge` (N<20). Moving counts: 56/56/48/65/15/74/10.
Receipt `ddd720143cc0bb3af4719e58453008b643543d7b78740b4cbbd44a23fff57797`; log `4e2ac8709472cc30553b369f1cae510c29451f20528be87caabe4cf30d5dfcad`.

## Mechanism findings (DERIVED)

1. **The α=0 anti-kill CONFIRMED the G5 Ranking-Inversion hypothesis on the G5 kill axis:** a2 subset 0.9231 → 1.0 with pure empirical support. Prior-perturbation of dense support is repaired.
2. **a3's residual miss is NOT support perturbation:** with α=0.0 and full-bank 0.9775 (65 moving rows), the subset draw (17 samples) misclassified exactly 1 row → 16/17 = 0.9412. One sample flips the gate. This is small-sample subset-estimator noise at the calibration boundary, not a selection failure.
3. **Bridge lever held the permanent small-sample fix:** a4/a6 = 1.0 subset AND full.
4. **Mid regime (20–39) never engaged on the real bank** — no action had N_moving in [20,40). The piecewise gate was effectively binary (pure vs bridge). Regime-2 shrinkage remains untested on production data.

## Next levers (NOT authorized — require a new directive)

1. PG1a estimator stability: larger per-action subset (e.g., N=256 or full-bank PG1a with subset as diagnostic) or a bootstrapped CI gate — 17-sample AUC has ±1-sample granularity (0.058 step).
2. Per-action τ/θ calibration refinement at the boundary (a3's 16/17 miss sits at the θ threshold).
3. Any mid-regime (20–39) shrinkage tuning requires a bank/action mix that actually engages regime 2.

## W0 status

**STAYS GATED** — PG1 did not fully clear (PG1a a3). `WavePacketPathSearch` wiring remains a separate approval-gated carrier.

## Failure action executed

`HALT_IMMEDIATELY_SEAL_FALSIFICATION` — genuine kill (receipt complete, PG2/PG3 pass, labels calibrated, fit engaged). No relaunch. No promotion to `main`; branch retained as sealed record.
