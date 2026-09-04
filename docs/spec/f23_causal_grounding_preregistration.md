# Carrier F23 — Online Causal Calibration & Semigroup Axiom Grounding — Pre-Registration

**Directive:** HENRI-DIR-2026-08-F22-POSTMORTEM-CAUSAL-GROUNDING
(`9cafa2a3fcc457b0784cc9d9e321d7f02af8a3ae65e6a1696921673a65ec106c`, 20,587 B, 244 lines)
**Branch:** `carrier/f23-causal-grounding`
**Engine:** `HENRI V2/experiments/verification/arc_f23_causal_engine.py`
**Seal target:** ledger record 1,142+ (`F23_PREREG_SEALED`)

## 1. Ratified context (F22 verdict `#0612ff513a2f40d4…`, ledger 1,141)

| Metric | Threshold | F22 measured | Disposition |
|---|---|---|---|
| PG1 recon | ≥ 0.8500 | 0.881334 | PASS — offline fit healthy |
| G1 latency | ≤ 2.00 ms/step | 0.876707 | PASS — sub-millisecond |
| G2 solved | ≥ 1/12 | 0 / 12 | FAIL — sim-to-real split |
| G3 live valence | ≥ +0.0200 | +0.000894 | FAIL — 80× drop vs F21.1 sim (+0.0719) |
| G4 sagnac | ≤ 0.0500 | 0.806277 | FAIL — un-grounded axiom reference |

Root causes (directive §1, §2): (a) static offline EDMD operators `T_a` model
aggregate kinematic drift but not local obstacles/walls → the planner hallucinates
progress; (b) G4 compared forward steps against an arbitrary static reference on
D=64 instead of the transition semigroup's stationary direction.

## 2. F23 mechanism (directive §3.1)

1. **In-situ online calibration:** after each live step,
   `E_t = Ψ_{t+1} − T_a Ψ_t`; `T̃_a = T_a + η_cal E_t Ψ_tᵀ` with `η_cal = 0.05`;
   `T_a ← U Vᵀ` from `SVD(T̃_a)` (Stiefel/orthogonal retraction). Only the
   executed action's operator updates. Expected one-step error contraction:
   `‖T̃_a Ψ_t − Ψ_{t+1}‖ = (1 − η_cal)‖T_a Ψ_t − Ψ_{t+1}‖` (exact pre-retraction).
2. **Semigroup stationary axiom:** `T̄ = (1/7) Σ_a T_a`;
   `Ψ_axiom = LeadingEigenvector(T̄)` (largest real eigenvalue, real part),
   normalized to S^{63}. When all actions share an invariant axis `n`
   (`T_a n = n` ∀a), `Ψ_axiom = n` exactly (test C2 constructs this).
3. **Causal horizon verification:** ring buffer of recent actual transitions
   `(Ψ_t, a, Ψ_{t+1})`. A candidate path for action `a` at state `Ψ ≈ Ψ_t` is
   penalized when history shows the same action at a near-identical state
   produced no movement (stall: `|cos(Ψ_{t+1}, Ψ_t)| ≥ 0.90`). Penalty
   `STALL_PENALTY = 0.05` subtracted from that candidate's score.
4. **G4 grounding:** `Δ_Sagnac = 1 − |⟨T_a Ψ_t, Ψ_axiom⟩|` vs the synthesized
   axiom (not goal distance, not a random vector).
5. **G3 discipline (unchanged from F22):** valence measured on the ACTUAL
   post-action frame `s_{t+1}`; G3 and waypoint advancement never use the
   internally predicted state.

## 3. Data path

```
trajectory bank (f3v2 npz+jsonl, sha 9e3c01b4…, N=1,536)
  → F21.1 compile_generators_capped → {generators, transitions [7,64,64], t_pow, recon}
  → PG1 preflight (kill if min_recon < 0.85)
  → synthesize_semigroup_axiom(T̄)
  → live Arcade loop (12 envs × 150 steps = 1,800), PatchIngress [1,8,8]→[1,64]
      per step: score (horizon-8 rollouts + stall penalty + Langevin escape) →
      step(action) → encode s_{t+1} → Δν vs active waypoint → advance if ≥ 0.60 →
      calibrate executed T_a → record telemetry
  → receipt → verdict seal
```

## 4. Gates (directive §4)

| Gate | Requirement | Kill |
|---|---|---|
| PG1 | min per-action recon ≥ 0.8500 | PRE-FLIGHT KILL (no run) |
| G1 | mean step latency ≤ 2.0 ms (1,800 steps) | K1 |
| G2 | ≥ 1 of 12 envs solved (levels_completed ≥ 1) | K2 |
| G3 | mean live Δν ≥ +0.0150 (relaxed from F22 +0.0200) | K3 |
| G4 | mean Δ_Sagnac vs axiom ≤ 0.0500 | K4 |

Verdict precedence: EDMD collapse → G1 → G2 → G3 → G4 → `F23_PASS` (F22 pattern).
Receipt keys extend F22's set with `axiom_ev`, `calibration_updates`,
`stall_penalties`, `eta_calibration`.

## 5. Parameters

`seed=20260923`, `horizon=8`, `omega_bound=π/32≈0.0982`, `beta_sagnac=0.015`,
`waypoint_advance_thresh=0.60`, `langevin_temp=0.50`, `langevin_steps=3`,
`eta_calibration=0.05`, `stall_cos=0.90`, `stall_memory=32`, `stall_penalty=0.05`.
Bank and envs identical to F22 (12 named env ids). Default-OFF flag
`HENRI_F23_CAUSAL=1`; startup refusal without it.

## 6. Falsification targets

- **F23 PASS** requires G2 ≥ 1/12 AND G3 ≥ +0.0150 AND G4 ≤ 0.0500 together —
  the first task-level resolution in the F4–F22 chain.
- Calibration engaged but G2/G3 fail → in-situ correction is insufficient for
  task-level control (mechanism-level negative result; verdict `F23_GATE_G*_FAILED`).
- Calibration never fires (`calibration_updates == 0`) → `F23_NO_CALIBRATION_ENGAGEMENT`
  (harness defect, not a mechanism verdict).

## 7. Cheapest kill experiment

Unit test C3 (calibration contracts: orthogonality preserved, blend identity,
alignment non-decrease) + C2 (axiom synthesis recovers the shared invariant axis).
If either fails, the mechanism is FALSIFIED before any GPU spend.
