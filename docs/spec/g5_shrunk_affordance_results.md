# Carrier G5 Results — Empirical Bayes Shrinkage & Sample-Gated Dual-Subspace Affordance

**Directive:** `HENRI-DIR-2026-09-V3-CARRIER-G5-SHRINKAGE-DIRECTIVE` (SHA `fc2dd03a…`, 17,334 B, 315 lines).
**Prereg:** `docs/spec/g5_shrunk_affordance_preregistration.md` (SHA `8a6fd670…`, sealed `#88e18da9` @1,179).
**Verdict:** `G5_AFFORDANCE_FIT_COLLAPSE` — sealed `#<ledger-hash>` (ledger @1,180). 25th sealed falsification / 26th carrier.
**Branch:** `feat/carrier-g5-shrunk-affordance` @ `56ba8e8` (engine + tests, remote CUDA 13/13).

## Gates (OBSERVED, remote CUDA run, seed 20260928, 12 envs × 150 pre-flight)

| Gate | Metric | Threshold | Measured | Result |
|---|---|---|---|---|
| PG1 (global) | min_action_auc (subset N=128) | ≥ 0.8800 | **0.9231** | ✅ **FIRST PASS in the chain** |
| PG1a per-action | a0–a4 ≥ 0.9500; a5/a6 ≥ 0.8800 | — | a2 = 0.9231, a3 = 0.9231 | ❌ KILL |
| PG2 | norm drift | ≤ 1e-6 | 1.19e-7 | ✅ |
| PG3 | routing + top-k CPU == CUDA | exact | passed | ✅ |

Per-action subset AUC: a0 1.0 · a1 1.0 · a2 0.9231 · a3 0.9231 · a4 1.0 · a5 1.0 · **a6 1.0**.
Per-action full-bank AUC: a0 1.0 · a1 1.0 · a2 0.9863 · a3 0.9775 · a4 1.0 · a5 0.9911 · **a6 1.0**.

Routes: a0–a3, a5 → shrunken top-k; **a4 (15 moving) + a6 (10 moving) → D=64 bridge arm**.
α: 0.098–0.444 as designed (α(a6) = 0.444; α(a0) = 0.125).
Labels match G2/G4 calibration exactly (0.210/0.218/0.197/0.224/0.234/0.205/0.189). 0 live steps (pre-flight kill by design). Receipt SHA `ea6a7308…`; log SHA `7bfcaf55…`.

## Mechanism findings (DERIVED)

1. **The sample-gated bridge lever FIXED the small-sample collapse.** a6 (10 moving) — the axis that killed G2 (0.4512) and G4 (0.4 subset / 0.2 full) — scored **1.0 subset / 1.0 full via the D=64 bridge arm**. a4 (15 moving, bridge-routed) scored 1.0/1.0.
2. **PG1 global passed for the first time in 26 carriers** (G1 0.7768, G2 0.4512, G4 0.4 → G5 0.9231). The global affordance-collapse class is resolved; the remaining failure is a per-action threshold on well-supported actions.
3. **Kill cause: shrinkage prior slightly perturbed support selection on a2/a3.** Full-bank AUC unchanged vs G4 (0.9863/0.9775 ≈ G4 0.9863/0.9773), so the subset drop (1.0 → 0.9231) is the 18-row estimate (≈3–4 moving rows) plus the α = 0.11–0.14 shrinkage blending toward the pooled prior. Not a mechanism collapse — a support-selection + small-subset-estimate interaction.

## Honest limits

- Kill is genuine per the preregistered PG1a targets — no relaunch, no quarantine (run completed; PG2/PG3/labels all verified).
- W0 (G3 `WavePacketPathSearch` wiring) stays **GATED** — C3 requires a full PG1 pass, which includes PG1a. Not unlocked by this carrier.
- 0 live steps; no ARC environment was attempted. No capability claim.

## Next levers (NOT authorized; require a new directive)

1. Shrinkage gated to sparse actions only (α = 0 when N_a ≥ 40) — leaves well-supported top-k untouched while retaining the bridge fallback.
2. Subset estimator: for well-supported actions, report full-bank AUC (or a larger stratified draw) as the binding PG1a metric; reserve N=128 for sparse actions.
3. Tighter per-action support selection under shrinkage (e.g. top-k on unshrunk variance when α < 0.15).
4. W0 remains the gated prize behind a full PG1 pass.
