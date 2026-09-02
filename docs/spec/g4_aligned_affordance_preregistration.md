# Carrier G4 — Functionally Aligned Sparse Affordance: Pre-Registration

**Directive:** `Sprint_Closeout_Synthesis___Carrier_G4_Master_Directive.md`
(`HENRI-DIR-2026-09-V3-SPRINT-CLOSEOUT-G2-G3-G4`, 18,288 B, SHA `bb25dfe247…`, byte-identical `(1).md` copy) +
`Project_HENRI_V3_Carrier_G4_Master_Directive___Functional_Consistency_Synthesis.md`
(`HENRI-DIR-2026-09-V3-CARRIER-G4-FUNCTIONAL-CONSISTENCY`, 4,845 B, SHA `1203d7d8…`).
Parents: G2 sealed `#832c74d1` (23rd falsification), G3 verified `#a73ed272` (24th carrier, sidecar).
Branch: `feat/carrier-g4-aligned-affordance` (directive-mandated). Seed: 20260927.

## Mechanism

Per-action per-block ridge 8×8 transition operators `T_a^(m)` fit on MOVING
samples (stall-cosine label, τ_stall 0.90, G2-verified flat norm-divided cosine)
of the pinned bank (`trajectories_production_run_f3v2.npz`, `9e3c01b4…`).
Action-specific top-k=64 block masks by per-block displacement variance
(`argmax_k Var_t(‖Ψ_t^(m) − Ψ_(t−1)^(m)‖)`, directive PG3 rule).
Aligned score = `exp(−(1/k)Σ_{m∈TopK(a)}‖ψ_next^(m) − T_a^(m)ψ_t^(m)‖²)` — the
exact fit functional (mean quadratic residual) evaluated on the sparse support.
Π_a = `σ((θ_a − mean_residual_a)/τ_a)`; θ_a/τ_a calibrated from the action's
moving-sample residual stats (in-sample calibration, same convention as G2).

**Canonical geometry:** per-block unit norm (HENRI invariant ‖w_m‖₂=1). PG2 =
`max_m |‖w_m‖−1| ≤ 1e-6` and dimension-normalized `|‖Ψ‖/√8192 − 1| ≤ 1e-6`
(reconciles the directive's `abs(norm(Ψ)−1.0)` with the invariant; total norm of
a canonical wave is √8192 ≈ 90.51). Ridge λ = 1e-2 (X entries ~1/√8).

## Audited corrections (disclosed pre-seal)

1. **Score form:** the directive's engine code (`forward_aligned_score`,
   `exp(−mean_loss)`) is authoritative. The exec-flow sigmoid
   `σ((mean − θ)/τ)` has a sign slip that would INVERT affordance (blocked ⇒
   Π→1); implemented as `σ((θ_a − mean)/τ_a)` so Π rises as residual falls
   (matches the code's semantics). Guarded by contract C13.
2. **Live causality:** the residual functional requires the observed next
   state. PG1 (binding gate) measures AUC on the bank where ψ_next exists —
   exactly per the directive. The live loop uses the most recent OBSERVED
   transition pair (one-step lag, causal, zero leakage) as the affordance
   input; live gates (carried from the G1/G2 harness: G1 ≤2.0 ms, G2 ≥1/12,
   G3 Δν ≥ +0.0150, G4 ≤0.05) gate the lagged regime separately.
3. **Sagnac formula** (`|Re|` vs `[0,2]` invariant) is deferred to Carrier W0
   (G3 wiring); not exercised by G4's gates. Discrepancy noted, not resolved
   in this carrier.
4. **N=128 reading:** `action_stratified_trajectory_bank_v2 (N=128)` read as
   128 total rows, action-stratified (18/action + 2 extras to the largest
   actions), deterministic seeded draw. Fit on the full bank; PG1 AUC on the
   subset (binding); full-bank AUC diagnostic.

## Gates (binding, STRICT_FAIL_CLOSED — HALT_IMMEDIATELY_SEAL_FALSIFICATION)

- **PG1:** `min_action_auc ≥ 0.8800` over all 7 actions on the N=128 stratified
  subset (binding; full-bank diagnostic). Local synthetic pre-flight: AUC ≥ 0.85
  on the controlled fixture (contract C2).
- **PG2:** norm drift ≤ 1e-6 on all canonicalized bank rows (per-block + dim-normalized).
- **PG3:** k_support = 64 per action, total_blocks 8192, variance selection rule.
- **C1:** score functional ≡ fit functional (shared code path; gradient-homology test at toy scale).
- **C2:** zero policy leakage — G3 sidecar read-only; production runner and G4 engine do not import G3.
- **C3:** identical pass/fail/skip sets local (CPU) vs remote (RTX 5090 CUDA) at the exact SHA; same-seed determinism.

## Kill criteria

Any PG1–PG3 or C1–C3 failure ⇒ verdict `G4_AFFORDANCE_FIT_COLLAPSE` (or typed
gate verdict), seal, no promotion. All pass ⇒ `G4_ALIGNED_AFFORDANCE_VERIFIED`
then Carrier W0 (G3 wave-packet planner wiring) is the mandated next carrier
with its own prereg + approval.

## Evidence chain

Ledger: G4 packet @ (next index after 1,172); this prereg seal next. Bank
provenance `9e3c01b4…`; 12-env cohort carried (F15 DEFAULT_ENVS); 150 steps/env;
remote `/tmp/henri_g4_aligned/`.
