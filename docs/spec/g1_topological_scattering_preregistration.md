# Carrier G1 — Topological Gauge-Wave Scattering (HENRI V3) — Pre-Registration

**Directive:** HENRI-DIR-2026-09-V3-TOPOLOGICAL-GAUGE-WAVE-ORDER
(`818968573ada74e28353af6d2779390ac09a203cecfacd495183ac2f06c6e0b0`, 23,536 B, 352 lines)
**Branch:** `carrier/g1-topological-scattering`
**Engine:** `HENRI V2/experiments/verification/arc_g1_topological_engine.py`
**Seal target:** ledger record 1,151+ (`G1_PREREG_SEALED`)

## 1. Ratified context (F23 verdict `#c2ee09e2…`, ledger 1,147)

| Metric | Threshold | F23 measured | Disposition |
|---|---|---|---|
| PG1 recon | ≥ 0.8500 | 0.8935 | PASS — offline fit healthy |
| G1 latency | ≤ 2.00 ms/step | 2.5398 | FAIL — per-step SVD calibration cost |
| G2 solved | ≥ 1/12 | 0 / 12 | FAIL |
| G3 live valence | ≥ +0.0150 | −0.000384 | FAIL — valence went NEGATIVE |
| G4 sagnac | ≤ 0.0500 | 0.9230 | FAIL — internal axiom, not external physics |

Root cause (directive §Executive Summary): the Homogeneous Manifold Fallacy.
F21/F22/F23 fit ONE global linear operator `T_a` to BOTH free motion AND wall
collisions; the regression computed the statistical mean of moving and hitting
walls (an operator that did neither). F23's online calibration then collapsed
`T̄ → I` (stagnation) because blocked frames dominated.

## 2. G1 mechanism (directive §1.2, §1.3, §4)

1. **Free-motion generators:** `T_free,a = exp(D_free,a) ∈ SO(64)` compiled
   STRICTLY from non-zero-displacement transitions (`‖Ψ_{t+1} − Ψ_t‖₂ > 0.05`)
   via regularized normal equations → SVD Stiefel retraction → skew log →
   spectral cap (`ω_bound = π/32`, F21.1 `_logm_skew`/`spectral_cap` reused).
   Actions with < 5 moving transitions get `T_free,a = I` (no free-motion
   evidence; affordance gate carries all information).
2. **Affordance transmittance (state-dependent):**
   `Π_pass,a(Ψ_t) = σ(Ψ_tᵀ W_contact,a Ψ_t + b_contact,a)`, where
   `W_contact,a = (1/N_a) Σ_i (y_i − ȳ_a) Ψ_i Ψ_iᵀ` (centered quadratic
   correlation; y = 1 moving / 0 blocked) and `b_contact,a` calibrated to the
   prior. This is a REAL state-dependent bilinear classifier. The directive's
   reference `W = I·(mean−0.5)` is state-INDEPENDENT (same score for every
   state → AUC ≈ 0.5 → guaranteed PG1 kill); it is a defect, not the contract.
3. **Symplectic scattering operator:**
   `Ψ̂_{t+1}(a) = Π_pass,a · T_free,a Ψ_t + (1 − Π_pass,a) · Ψ_t`.
   Blocked action → `Ψ̂ = Ψ_t` (predicts zero movement, no hallucination).
4. **Vectorized homotopy beam (K = 8):** `J_a = |⟨Ψ̂_{t+K}(a), Ψ_wp⟩| · Π_pass,a^K`
   — F21.1 `t_pow` unroll for kinematics, affordance product for pruning
   (directive §1.3 formula). No per-step SVD: `T_free`/`t_pow` are static;
   online updates touch only the affordance classifier.
5. **Dual-speed online affordance plasticity (directive §4):**
   `b_a += η·error`, `W_a += η·error·outer(Ψ_t, Ψ_t)`, `η = 0.10`,
   `error = was_moving − Π_pred` on the EXECUTED action after each live step.
   Unexpected collision (predicted pass, observed block) lowers Π;
   unexpected motion raises Π.
6. **G4 redefined honestly (directive §3 gate):** single-pass horizon physical
   consistency `Δ_Affordance = 1 − |⟨Ψ̂_{t+1}(a_exec), Ψ_{t+1}⟩|` — the
   scattered prediction vs the ACTUAL post-action observation. A correct
   affordance gate makes Ψ̂ ≈ Ψ_{t+1} for BOTH regimes (moving: T_free path;
   blocked: identity path). This is external physics, not an internal axiom
   (F23 defect) and not a random reference (F22 defect).
7. **G3 discipline (unchanged from F22/F23):** valence Δν = c_next − c_t on
   the ACTUAL post-action frame vs the active waypoint; never the internally
   predicted state. Reset → 3-step Langevin escape (F22/F23 verified pattern).

## 3. Data path

```
trajectory bank (f3v2 npz+jsonl, sha 9e3c01b4…, N=1,536)
  → PatchIngress D=64 bridge (F22/F23 verified) → psi, nxt, onehot
  → moving mask (‖ΔΨ‖ > 0.05)
  → compile_free_generators_capped (moving-only) → T_free, t_pow [8,7,64,64], recon
  → fit_affordance_classifiers (quadratic correlation + bias)
  → PG1 preflight (per-action min AUC ≥ 0.85; kill → no run)
  → live Arcade loop (12 envs × 150 steps = 1,800; F15 DEFAULT_ENVS)
      per step: ingress → Π_pass → beam J = align·Π^K → argmax →
      step(action) → encode s_{t+1} → Δν vs waypoint → advance if ≥ 0.60 →
      G4 scattered consistency vs actual → online affordance update → telemetry
  → receipt → verdict seal
```

## 4. Gates (directive §3)

| Gate | Requirement | Kill |
|---|---|---|
| PG1 | per-action min in-sample moving-vs-blocked AUC ≥ 0.8500 | PRE-FLIGHT KILL (no run) |
| G1 | mean step latency ≤ 2.0 ms (1,800 steps) | K1 |
| G2 | ≥ 1 of 12 envs solved (levels_completed ≥ 1) | K2 |
| G3 | mean live Δν ≥ +0.0150 | K3 |
| G4 | mean Δ_Affordance ≤ 0.0500 | K4 |

Verdict precedence: PG1 (preflight) → engagement (affordance updates > 0) →
G1 → G2 → G3 → G4 → `G1_PASS` (F22/F23 pattern).

## 5. Parameters

`seed=20260924`, `horizon=8`, `omega_bound=π/32≈0.0982`,
`waypoint_advance_thresh=0.60`, `eta_affordance=0.10`, `moving_thresh=0.05`,
`langevin_temp=0.50`, `langevin_steps=3`. Bank and envs identical to F22/F23
(12 named env ids from F15 `DEFAULT_ENVS`). Default-OFF flag
`HENRI_G1_TOPOLOGICAL=1`; startup refusal without it.

## 6. Falsification targets

- **G1 PASS** requires G2 ≥ 1/12 AND G3 ≥ +0.0150 AND G4 ≤ 0.0500 together —
  the first task-level resolution in the F4–F23 chain.
- Affordance gate engaged but G2/G3 fail → state-dependent affordance is
  insufficient for task-level control (verdict `G1_GATE_G*_FAILED`).
- G4 fails while G2/G3 pass → the scattering prior is physically inconsistent
  with observed kinematics (the directive's central claim is FALSIFIED).
- Affordance never updates (`affordance_updates == 0`, steps > 0) →
  `G1_NO_AFFORDANCE_ENGAGEMENT` (harness defect, not a mechanism verdict).

## 7. Cheapest kill experiment

Unit test C3 (classifier AUC ≥ 0.85 on separable synthetic bank) + C4
(scattering identity: Π→0 ⇒ Ψ̂ = Ψ_t; Π→1 ⇒ Ψ̂ = T_free Ψ_t) + C5 (beam
prunes blocked actions: J ≈ 0). Any failure FALSIFIES the mechanism before
GPU spend.
