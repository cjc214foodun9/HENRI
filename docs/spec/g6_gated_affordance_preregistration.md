# Carrier G6 Pre-Registration — Piecewise Gated Subspace Selection & Pure-Support Preservation

**Directive:** `Carrier_G6_Master_Directive___G5_Post-Mortem.md` — `HENRI-DIR-2026-09-V3-CARRIER-G6-PIECEWISE-GATING`
SHA-256 `d5ec31cd38560e4eb07f9d75a779d0f8aec1f03fd358396898dd9d2b49389e83` (18,268 B, 341 lines), packet sealed `#5de066da` @1,185.
**Causal parent:** G5 verdict `G5_AFFORDANCE_FIT_COLLAPSE` `#eeed5b17` @1,181 (PG1 global 0.9231 PASS — first in chain; PG1a a2/a3 0.9231 < 0.9500 KILL; a4/a6 1.0 via bridge); G5 branch `feat/carrier-g5-shrunk-affordance` @ `d7b60e6`.

## 1. Hypothesis (falsifiable)

G5's kill was support perturbation by the global shrinkage prior on well-supported actions (Ranking Inversion Theorem, directive §1.1: α≈0.12 injects shared cross-action modes into a2/a3 support). **If shrinkage is gated piecewise — α≡0.0 for N_moving ≥ 40, shrunk α for 20 ≤ N_moving < 40, D=64 bridge for N_moving < 20 — then dense actions recover their pure empirical support (a2/a3 full-bank AUC ≈ G4's 0.9863/0.9775, subset ≥ 0.9500) while sparse actions keep the bridge-cured AUC (a4/a6 = 1.0).**

## 2. Mechanism (as audited against live code)

| Regime | N_moving | Route | α |
|---|---|---|---|
| 1 (sparse) | < 20 | D=64 bridge (`_bridge_to_d64_batch`, origin `arc_f21_edmd_engine.py`) | — |
| 2 (mid) | 20–39 | D=65,536 shrunken top-k | ν₀/(ν₀ + N·d), ν₀=64 |
| 3 (dense) | ≥ 40 | D=65,536 **pure empirical top-k** | **α ≡ 0.0 exactly** |

- Variance statistic: **centered** per-block displacement variance (`per_block_displacement_variance`, G4/G5-verified). The directive's code-block `mean(Σ_d Δ²)` (uncentered) is REJECTED — same class as the G4 centered-variance correction.
- Bridge arm: `_bridge_to_d64_batch` (PatchIngress, verified origin `arc_f21_edmd_engine.py`) with fail-closed `ingress is None` guard. Directive's `psi_t[:, :8]` slice REJECTED (not the verified bridge).
- Label: flat norm-divided cosine on raw as-captured waves (G2 C13-locked), τ_stall 0.90.
- Bank: pinned `trajectories_production_run_f3v2.npz` (SHA `9e3c01b4…`, same as F22/F23/G1/G2/G4/G5). 12 envs × 150, seed **20260929**, top-k 64, ridge 0.01.
- Gate: `HENRI_G6_GATED_AFFORDANCE=1` (default-OFF). Verdict: `G6_GATED_AFFORDANCE_VERIFIED` / `G6_AFFORDANCE_FIT_COLLAPSE`.

## 3. Gates (YAML §3.2 governs; matrix prose 0.9200 superseded by YAML)

- **PG1 global** min_action_auc_subset ≥ **0.8800**.
- **PG1a per-action subset AUC:** a0–a4 ≥ **0.9500**; a5, a6 ≥ **0.8800**. (Stratified N=128, ≥ 10/action.)
- **PG2** norm drift ≤ 1e-6.
- **PG3** CPU == CUDA exact routing + top-k identity.

## 4. Contracts

- **C1** functional homology per arm: fit functional == score functional (bridge: `bridge_fit_transitions`/`bridge_mean_quadratic`; full: `fit_aligned_transitions`/`aligned_mean_quadratic`, forward orientation).
- **C2** piecewise gating stability: α evaluates to exactly 0.0 for every action with N_moving ≥ 40; α monotone non-increasing in N_moving within regime 2; route('bridge') iff N_moving < 20.
- **C3** W0 gateway: FULL PG1 clearance (global + all PG1a + PG2 + PG3) automatically authorizes Carrier W0 (G3 `WavePacketPathSearch` planner wiring) as a separate approval-gated carrier. Any gate failure leaves W0 gated.

## 5. Failure action

Any gate failure → `HALT_IMMEDIATELY_SEAL_FALSIFICATION`: seal `G6_AFFORDANCE_FIT_COLLAPSE`, no relaunch (unless a harness defect with 0 live steps is proven). W0 stays gated. No promotion to `main`; branch retained as sealed record.
