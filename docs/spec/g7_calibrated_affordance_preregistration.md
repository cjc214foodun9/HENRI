# Carrier G7 Pre-Registration — Calibrated Sample-Density Expansion & Finite-Sample Variance Regularization

**Directive:** `Carrier_G7_Master_Directive___G6_Post-Mortem.md` — `HENRI-DIR-2026-09-V3-CARRIER-G7-CALIBRATION-EXPANSION`
SHA-256 `24b7665d426fc26a9957acbe3a6549accd6755bc83536a77ab8c0502ceeb0800` (18,073 B, 344 lines), packet sealed `#d2cbf9b5…` @1,190.
**Causal parent:** G6 verdict `G6_AFFORDANCE_FIT_COLLAPSE` `#e57da003` @1,187 (PG1 global 0.9412 PASS; PG1a a3 0.9412 < 0.9500 KILL — 16/17 subset draw, full-bank 0.9775; a2 FIXED 0.9231→1.0 via α=0; a4/a6 1.0 via bridge); G6 branch `feat/carrier-g6-gated-affordance` @ `adb441c`.

## 1. Hypothesis (falsifiable)

G6's residual kill was **finite-sample quantization of the PG1a subset estimator**, not support perturbation: a3's measured 0.9412 = 16/17 on a 17-sample draw while the full-bank AUC is 0.9775 (65 moving rows). One misclassified row flips the gate because the empirical Mann-Whitney AUC grid step near the boundary is 1/17 ≈ 0.0588 > (1.0 − 0.9500). **If the stratified evaluation subset is expanded to N=256 (≥ 32 rows/action, grid step ≤ 1/32 = 0.03125), then per-action subset AUC ≥ 0.9500 is reachable with one legitimate outlier (31/32 = 0.9688), and the a3 gate passes (full-bank 0.9775 unchanged).** The per-action median-calibrated temperature τ_a (directive-mandated) is retained for the score surface and cross-action normalization; it is a monotone per-action transform and therefore **rank-invariant per action** — the a3 fix is the subset resolution, not τ_a.

## 2. Mechanism (as audited against live code)

| Regime | N_moving | Route | α |
|---|---|---|---|
| 1 (sparse) | < 20 | D=64 bridge (`_bridge_to_d64_batch`, origin `arc_f21_edmd_engine.py`) | — |
| 2 (mid) | 20–39 | D=65,536 shrunken top-k | ν₀/(ν₀ + N·d), ν₀=64 |
| 3 (dense) | ≥ 40 | D=65,536 **pure empirical top-k** | **α ≡ 0.0 exactly** |

- **Retained from G6 (verified):** piecewise routing (α=0 dense, bridge sparse, shrunk mid), centered per-block displacement variance (`per_block_displacement_variance`), verified `_bridge_to_d64_batch` with fail-closed `ingress is None` guard. Directive code-block `psi_t[:, :8]` bridge slice and uncentered `sample_var` REJECTED (`CONFLICTS_WITH_LIVE_CODE`, same class as G4–G6 corrections).
- **NEW — score surface (directive §2.1):** `S_calibrated(a) = exp(−mean_loss(a) / τ_a)` where `mean_loss(a)` is the C1 mean-quadratic residual over the action's top-k support (fit functional == score functional), and
  `τ_a = clamp(Median({e_m(a) | moving rows of action a}), 0.05, 2.0)`; empty moving set → τ_a = 1.0.
- **NEW — PG1a evaluation subset (directive §1.2):** stratified N=256 draw (36 rows/action + 4 extra on the four largest actions = 256), seed-locked per action, replacing G6's N=128 (18/action). Per-action grid step ≤ 1/32 = 0.03125 ≤ 0.03125 → one-outlier tolerance (31/32 = 0.9688 ≥ 0.9500).
- **Mid regime note:** 20–39 never engaged on the pinned bank (G6 receipt moving counts 56/56/48/65/15/74/10); retained as the directive-specified piecewise rule, documented corpus-bounded.
- Label: flat norm-divided cosine on raw as-captured waves (G2 C13-locked), τ_stall 0.90.
- Bank: pinned `trajectories_production_run_f3v2.npz` (SHA `9e3c01b4…`, same as F22/F23/G1/G2/G4/G5/G6). 12 envs × 150, seed **20260930**, top-k 64, ridge 0.01.
- Gate: `HENRI_G7_CALIBRATED_AFFORDANCE=1` (default-OFF). Verdict: `G7_CALIBRATED_AFFORDANCE_VERIFIED` / `G7_AFFORDANCE_FIT_COLLAPSE`.

## 3. Gates (YAML §3.2 governs; matrix prose superseded by YAML where they differ)

- **PG1 global** min_action_auc_subset ≥ **0.9000** (YAML; tightened from G6's 0.8800).
- **PG1a per-action subset AUC (N=256):** a0–a4 ≥ **0.9500** (a4 bridge-routed); a5, a6 ≥ **0.8800** (a6 bridge-routed).
- **PG2** norm drift ≤ 1e-6.
- **PG3** CPU == CUDA exact identity across the 17-suite regression.

## 4. Contracts

- **C1** functional homology per arm: fit functional == score functional (bridge: `bridge_fit_transitions`/`bridge_mean_quadratic`; full: `fit_aligned_transitions`/`aligned_mean_quadratic`, forward orientation). τ_a scales the score surface only — the underlying residual functional is identical.
- **C2** temperature bounds: τ_a strictly positive, 0.05 ≤ τ_a ≤ 2.0 for every action with a moving set; τ_a = 1.0 when the moving set is empty; τ_a computed from moving-row residuals only.
- **C3** W0 gateway: FULL PG1 clearance (global + all PG1a + PG2 + PG3) automatically authorizes Carrier W0 (G3 `WavePacketPathSearch` planner wiring) as a **separate approval-gated carrier** — NOT wired in G7. Any gate failure leaves W0 gated.

## 5. Failure action

Any gate failure → `HALT_IMMEDIATELY_SEAL_FALSIFICATION`: seal `G7_AFFORDANCE_FIT_COLLAPSE`, no relaunch (unless a harness defect with 0 live steps is proven). W0 stays gated. No promotion to `main`; branch retained as sealed record.
