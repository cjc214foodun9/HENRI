# Carrier G1 — Topological Gauge-Wave Scattering — Results

**Directive:** HENRI-DIR-2026-09-V3-TOPOLOGICAL-GAUGE-WAVE-ORDER
(`818968573ada74e28353af6d2779390ac09a203cecfacd495183ac2f06c6e0b0`, 23,536 B, 352 lines)
**Branch:** `carrier/g1-topological-scattering`
**Engine:** `HENRI V2/experiments/verification/arc_g1_topological_engine.py`
**Prereg seal:** `#b5c6a12257398fdd…` (ledger 1,152)
**Verdict seal:** `#f116b9e2acb6043b…` (ledger 1,153)

## 1. Verdict

```
G1_AFFORDANCE_FIT_COLLAPSE  (PG1 PRE-FLIGHT KILL — no live steps)
min_auc = 0.7768 < 0.8500 (PG1 gate)
```

22nd sealed carrier run. Fail-closed at the pre-flight gate exactly as the
directive prescribes (PG1 Failure Action = PRE-FLIGHT KILL).

## 2. Exact gate values (OBSERVED, receipt `7ab641a4…`)

| Gate | Requirement | Measured | Disposition |
|---|---|---|---|
| PG1 | min per-action moving-vs-blocked AUC ≥ 0.8500 | **0.7768** | ✗ KILL (no run) |
| G1 | mean step latency ≤ 2.0 ms | — | not reached |
| G2 | ≥ 1/12 envs solved | — | not reached |
| G3 | mean live Δν ≥ +0.0150 | — | not reached |
| G4 | mean Δ_Affordance ≤ 0.0500 | — | not reached |

Per-action AUC: a0 0.7787, a1 0.7906, a2 0.8883, a3 0.7768, a4 1.0000,
a5 0.9978, a6 0.9124. min = a3 0.7768.

## 3. Mechanism engagement and kill validity

The kill is REAL, not a harness defect:

- All 7 actions have ≥ 10 bank samples (n: 267, 257, 244, 290, 64, 361, 53).
- All per-action AUCs are present and sensible; `min_auc` is genuine.
- Affordance structure EXISTS in the bank: every AUC ≫ 0.5 (the directive's
  own state-independent reference W = I·(mean−0.5) gives exactly 0.5, control
  C11). The state-dependent bilinear classifier is doing real work.
- The quadratic classifier on the mean-pooled D=64 bridge fails the
  pre-registered 0.85 gate for 4/7 actions.

## 4. Bank statistics (OBSERVED, remote diagnostic)

| Action | n | moving rate | AUC |
|---|---|---|---|
| 0 | 267 | 0.820 | 0.7787 |
| 1 | 257 | 0.840 | 0.7906 |
| 2 | 244 | 0.885 | 0.8883 |
| 3 | 290 | 0.834 | 0.7768 |
| 4 | 64 | 0.234 | 1.0000 |
| 5 | 361 | 0.252 | 0.9978 |
| 6 | 53 | 0.453 | 0.9124 |

## 5. Epiplexity / derived implication

Actions with a rare moving minority (a4 23%, a5 25%) separate almost
perfectly (AUC 1.0 / 0.998); actions dominated by motion (a0–a3, 82–89%
moving) cannot separate the scattered blocked frames at the 0.05
displacement threshold on the mean-pooled bridge. The F22/F23 "stall"
regime (|cos(ψ_next, ψ_t)| ≥ 0.90) is NOT the same as the 0.05 displacement
threshold; the blocked frames that matter for ARC walls are high-cosine
small-displacement events that the mean-pooled D=64 projection erases.

The Homogeneous Manifold Fallacy diagnosis survives, but the G1 carrier
FALSIFIES the specific claim that a quadratic affordance classifier on the
mean-pooled D=64 bridge can separate moving vs blocked at AUC ≥ 0.85 with
the directive's 0.05 displacement threshold.

Next carrier directions (not authorized here, for the next directive):
(a) a finer ingress (per-block feature, not mean-pooled) so small
displacements survive; (b) a threshold-aligned label (stall-cosine regime
|cos| ≥ 0.90 instead of raw ‖ΔΨ‖ > 0.05); (c) a per-action-class prior or
calibrated τ_sharp.

## 6. Software verification

- Local G1 contract suite: 12/12 PASS (final).
- Remote CUDA contract suite at exact SHA `8926776f`: 12/12 PASS.
- Full local regression: not rerun (engine-only carrier; no shared-module
  changes beyond the new engine/test files).
- One harness defect found and fixed: CUDA device placement in
  affordance/identity allocations (local CPU suite could not see it);
  committed `8926776`.

## 7. Governance

- Prereg sealed `#b5c6a12257398fdd…` @1,152.
- Verdict sealed `#f116b9e2acb6043b…` @1,153.
- Receipt SHA `7ab641a4d4c340a9f204207affd51360ae0eda8b09bb6ad54e73393bc4db7cf6`;
  log SHA `3ac33fc8a9dfd2a80b55617553a42977abb70ce9b49efab9f0999b7a24ec9b8b`.
- Remote worktree `/tmp/g1-topological-wt` at `8926776f`, CLEAN.
- No promotion to `main`.
