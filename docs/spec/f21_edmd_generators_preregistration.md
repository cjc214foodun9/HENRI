# Carrier F21 — In-Situ Empirical EDMD & Trajectory-Span Generator Synthesis — Pre-Registration

- Directive: `Project_HENRI_F20_Post-Mortem_Audit___Dynamic_Generator_Synthesis_Directive.md`
- Directive ID: `HENRI-DIR-2026-08-F20-POSTMORTEM-DYNAMIC-GENERATOR-ORDER`
- Directive SHA-256: `9ecabe24ec255591327f5830219581962774f7d3108f774027d7122f993438f4` (21,582 B, 252 lines)
- Evaluated commit: `c3c29643` (branch `carrier/f20-adjoint-conjugation`); ledger 1,129 verified
- Author: henri-arbiter | Date: 2026-09-01 | Seal target: ledger record 1,130+

## 1. Hypothesis

The 17-family falsification chain (F4–F20) closed on the **Lie sub-algebra span deficit**: dim Span{D_1..D_8} = 8 ≪ 2,016 = dim so(64); the goal displacement Ω_goal is ~99.6% orthogonal to any static random generator dictionary, so no coordinate warp, SNR rebalance, or Nyquist clamp can steer the state (directive §1.1). F21 replaces static random skew dictionaries with **data-driven transition Lie generators** compiled in-situ from the verified F3 v2 trajectory bank (directive §1.2, §3).

## 2. Mechanism (verbatim, §1.2 + §3)

1. Action partition: X_a = [ψ_{t1}, ψ_{t2}, …], Y_a = [ψ_{t1+1}, …] per action a from bank rows (bridge: block-mean + PatchIngress — identical to F15/F20 live-state ingress).
2. Koopman solve: K_a = (Y_aᵀ X_a)(X_aᵀ X_a + λI)⁻¹ ∈ ℝ^{d×d} — normal-equation form of the directive's K_a = Y_a X_aᵀ (X_a X_aᵀ + λI)⁻¹, required for the [d,d] matrix log (λ = 1e-3, disclosed; directive leaves λ unvalued).
3. Stiefel retraction: SVD K_a = U S Vᵀ ⇒ W_a = U Vᵀ ∈ SO(D).
4. Lie generator: D_a = ½(Logm(W_a) − Logm(W_a)ᵀ) ∈ so(D) (principal matrix log via torch.linalg.eig).
5. Runtime (zero SVD in the timed loop — resolves F20 G1): ψ̂_{t+1}(a) = exp(D_a) ψ_t; K=8 horizon beam J(a_{1:8}) = |⟨ψ̂_{t+8}, ψ_goal⟩| − β Σ Δ_Sagnac(k), β = 0.025 fixed (F20-ratified regulator); select a* = a_1*; live action mapped by NAME to the bank generator (fallback: legacy modulo).

## 3. Gates (verbatim §4)

- **PG1** (pre-flight kill): in-sample recon cosine ⟨exp(D_a)ψ_t, ψ_{t+1}⟩ — interpretation: per-action mean ≥ 0.7000 (min over actions; stricter than global mean — disclosed).
- **G1** ≤ 5.0 ms/step (1,800 steps, 12 envs, 150 steps/env).
- **G2** ≥ 1 of 12 live envs solved (score > 0.0%).
- **G3** mean per-step directional Δν_goal ≥ +0.0200.
- **G4** single-pass K=8 horizon Sagnac ≤ 0.0500.
- Seed `20260920`; horizon 8; beta-sagnac 0.025; envs = the 12 NAMED F10 whitelist.

## 4. Disclosed Deviations (pre-seal)

1. n_actions = **7** (bank `action_names` (7,)); directive §1.2 writes a∈{0…7} (8) — engine derives n from the bank and reports it.
2. D = 64 verification substrate (directive §1.1); dense 64×64 SVD/Logm offline (production D=65,536 dense Logm infeasible — scope disclosed, same as F15–F20).
3. PG1 = per-action min mean recon (≥0.70) rather than a single global mean.
4. λ ridge = 1e-3 (unvalued in directive).
5. Live action → generator mapping by action NAME; modulo fallback when the env exposes no named action in the bank.
6. Logm = principal matrix log (eig-based; disclosed implementation choice).
7. `--trajectory-jsonl` consumed for env indexing/order (directive requires the flag; bank npz carries psi/next_wave/actions_onehot/action_names).
8. Goal source: F15 protocol — per-env terminal wave from the bank (bridge+ingress), same as F15/F20.
9. Sagnac delta = F10 `sagnac_delta` (normalized 1 − Re⟨·⟩/‖·‖‖·‖, bounded [0,2]); G4 uses the F20 `sagnac_raw_mean` semantics.

## 5. Kill Criteria

- PG1 < 0.70 on any action → pre-flight kill: `F21_EDMD_FIT_COLLAPSE` (zero live steps, no verdict).
- G1/G2/G3/G4 failures → `F21_GATE_GX_FAILED` per gate; any failure = carrier FALSIFIED (no partial promotion).
- Verdict event: `F21_GATES_VERDICT` with receipt SHA-256 + per-gate table + mechanism telemetry (per-action recon, D_a spectral radius, engagement, creeps).

## 6. Artifacts

- Branch `carrier/f21-edmd-generators`; engine `HENRI V2/experiments/verification/arc_f21_edmd_engine.py`; tests `HENRI V2/tests/contract/test_f21_edmd_engine.py`; out `/tmp/henri_f21_edmd/`; receipt `f21_gates_receipt.json`; bank `/root/f3-run/telemetry/f3_bank_capture_v2/` (sha prefix `9e3c01b4…`).
