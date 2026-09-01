# Carrier F19 — SNR-Rebalanced Geodesic Steering — Pre-Registration

- Directive: `Project_HENRI_F18_Post-Mortem_Audit___Signal-to-Noise_Rebalancing_Directive.md`
- Directive SHA-256: `7b3199296ca3e316f8dc24da9824b09e0371bd40d94683764da69a74fa9642c6` (21,118 B, 253 lines)
- Directive ID: `HENRI-DIR-2026-08-F18-POSTMORTEM-SNR-REBALANCING`
- Branch: `carrier/f19-snr-rebalancing` (base: F18 head `5163c43` — the carrier the directive explicitly builds on)
- Governance: ledger head 1,119 at auth (directive mandates "Record 1,119+"; the seal lands at record 1,120, deviation D8)
- Prior state: F18 sealed `#f4485122` (`F18_GATE_G2_FAILED` @ record 1,117, chain 1,118) **RATIFIED** by the directive (header "Formal Determination: RATIFIED"); F18 delivered the first positive Δν_goal in F4–F18 (+4.61e-05) and isolated the SNR deficit (S_goal ≈ 0.19 vs β·ΣΔ_Sagnac ≈ 0.332; SNR ≈ 0.572)

## 1. Mechanism (directive §3 Tier 1–3, §3.1)

F18's beam was SNR-submerged: with β fixed at 0.05, the accumulated K=8 Sagnac penalty (8 × 0.83 ≈ 0.332) exceeded the goal term (|⟨Ψ̂_{t+8}, Ψ_goal⟩| ≈ 0.19), so J < 0 for all candidates and selection was driven by phase-noise variance. F19 replaces the fixed β with **alignment-gated adaptive attenuation** and amplifies the directional tilt:

1. **Goal extraction**: identical to F15/F16/F17/F18 (trajectory-bank terminal state, `resolve_trajectory_goal`, PG1 ≤ 0.90 pre-flight) — verified ingress retained unchanged.
2. **Tier 1 — skew goal matrix + Killing coefficients** (unchanged from F16–F18): `Ω_goal(t) = Ψ_goal Ψ_t† − Ψ_t Ψ_goal† ∈ so(D)`; `γ_a = −½ Tr(D_a Ω_goal(t))` per action (all 8 in parallel).
3. **Tier 2 — high-gain step-scaled unitary tangent** (THE F19 TILT MECHANISM):
   - `D_a′ = D_a + κ_diff·tanh(γ_a)·Ω_goal`, **`κ_diff = 2.50`** (directive §5; 3.33× over F18's 0.75 — "amplifies the angular velocity differential between aligned and opposed actions");
   - `D̂_a = step_scale · ‖D_a‖_F · D_a′ / ‖D_a′‖_F`, **`step_scale = 1.50`** (directive §5 — "increases the geodesic arc length per step").
   - **Invariant: ‖D̂_a‖_F ≡ 1.50·‖D_a‖_F for all a** (exact up to fp) — the norm lever stays removed; the ONLY steering channels are the direction tilt (κ) and the geodesic step length (step_scale).
4. **Tier 3 — alignment-gated adaptive Sagnac attenuation** (THE F19 SNR MECHANISM):
   - `β_adaptive(t) = β_base · (1.0 − |⟨Ψ_t, Ψ_goal⟩|)`, **`β_base = 0.010`** (directive §5; replaces F18's fixed β = 0.05);
   - `J(a_{1:8}) = |⟨Ψ̂_{t+8}, Ψ_goal⟩| − β_adaptive(t)·Σ_{k=1..8} Δ_Sagnac(k)` (NO Frobenius term — μ_damp ≡ 0.0 locked, carried from F18 C15);
   - `a_t* = a_1*`; `Δν_t = |⟨Ψ_{t+1}, Ψ_goal⟩| − |⟨Ψ_t, Ψ_goal⟩|`.
   - As the wave aligns, the Sagnac penalty relaxes (×5 noise reduction at alignment ≈ 0.2: 0.010·0.8 = 0.008 vs F18's 0.05); directive projects SNR ≈ 8.125 (S_goal ↑ 0.650, P_Sagnac ↓ 0.080) — a projection, not a gate.
5. **Tier 4 — Hebbian creep** (carried from F18): updates strictly on Δν > 0.

## 2. Gates (directive §4, verbatim bounds — identical to F18)

| Gate | Criterion | Failure action |
|---|---|---|
| PG1 | `max_env |⟨Ψ_0, Ψ_goal⟩| ≤ 0.90` | PRE-FLIGHT KILL (degenerate trajectory ingress) |
| G1 | 1,800 steps / 12 envs, latency ≤ 5.0 ms/step | KILL K1 |
| G2 | ≥ 1 of 12 solved (Score > 0.0%) | KILL K2 |
| G3 | mean Δν_goal ≥ +0.0200 | KILL K3 (directional steering failure / SNR masking persists) |
| G4 | single-pass K=8 horizon Δ_Sagnac ≤ 0.0500 | KILL K4 (waveform coherence degradation) |
| E1 (carried from F17, pre-registered) | mean over steps of `std_a(γ_a)` > 1e-6 (population std, nan-safe fail-closed) | `F19_FALSIFIED_NO_ENGAGEMENT` |

## 3. Contract tests (directive §3/§4 + F18 §4 pattern; C1–C16 + flag fail-closed)

1. C1 so(8) skew symmetry of step-scaled normalized generators
2. C2 Killing-form variance `std(γ_a) > 0.10` on non-collinear goal states
3. C3 **step-scaled** tangent norm conservation `‖D̂_a‖_F == 1.50·‖D_a‖_F ± 1e-5` across the tilt range (κ ∈ {0, 0.5, 1.0, 2.5, 4.0}, γ ∈ [−1, 1])
4. C4 positive gradient alignment: ∂/∂γ_a `|⟨exp(D̂_a)Ψ_t, Ψ_goal⟩| > 0` (numerical derivative, non-collinear positive-Killing fixtures, κ = 2.50)
5. C5 Sagnac homodyne coherence: single-pass K=8 Δ ≤ 0.05 with step_scale = 1.50 (fixture θ = 0.11 rad/step → effective 0.165 rad/step × 8 = 1.32 rad → sin = 0.969)
6. C6 vectorized einsum beam == sequential loop within 1e-5 (adaptive β mirrored exactly)
7. C7 Hebbian creep strictly on Δν > 0
8. C8 zero CUDA VRAM leak over 1,000 continuous steps (CPU functional loop)
9. C9 PG1 preflight rejection (fail-closed degenerate bank)
10. C10 trajectory loader integrity (SHA-256 `9e3c01b4…` + array dims of `trajectories_production_run_f3v2.npz`)
11. C11 arcade environment handshake (live API, all 12 games)
12. C12 deterministic seed reproducibility (seed `20260918`, byte-identical trajectories)
13. C13 module constants bound (κ_diff=2.50, step_scale=1.50, beta_base=0.010, μ_damp=0.0 locked, seed=20260918)
14. C14 NaN/Inf guard (degenerate zero tensors → clean fallback; non-finite engagement telemetry → fail-closed verdict)
15. C15 zero-Frobenius-penalty lock (μ_damp ≡ 0.0 cannot be overridden via CLI)
16. C16 clean receipt generation (schema `f19-snr-rebalanced-engine.v1` + verdict mapping)
17. flag fail-closed (`HENRI_F19_SNR_REBALANCED=1` required; absent → RuntimeError)

## 4. Pre-registered deviations (disclosed, not silent)

- **D1 (β formula selection)**: directive §1.2 shows an exponential form `β(t) = β_0·exp(−λ_align|cos|)` with α_damp; §3 Tier 3, §3.1, and the executable §5 command specify the linear alignment-gated form `β_adaptive(t) = 0.010·(1.0 − |cos|)`. **Implemented form = §3/§5 (the executable blueprint)**; the §1.2 exp form is treated as explanatory prose.
- **D2 (C5 fixture)**: with step_scale 1.50, the F18 θ=0.16 fixture would total 1.92 rad (sin = 0.94 < 0.95); θ reduced to 0.11 rad/step so the single-pass K=8 horizon lands at 1.32 rad (sin = 0.969 ≥ 0.95; 1−align = 0.031 ≤ 0.05). Live G4 measures the seeded random generators on the real gauntlet.
- **D3 (C3 semantics)**: F18's ‖D̂‖ ≡ ‖D‖ invariant becomes the step-scaled invariant ‖D̂‖ = step_scale·‖D‖ (directive Tier 2 formula); the relative error check (±1e-5) is unchanged.
- **D4 (C12 determinism)**: byte-identical determinism tested on CPU (CUDA einsum/matrix_exp fp nondeterminism documented); live runs share the identical seed `20260918`.
- **D5 (constants per §5 command)**: κ 2.50, step 1.50, β_base 0.010, seed 20260918, μ 0.0, 150 steps × 12 envs = 1,800; `--beam` omitted by the command block (default 8, carried).
- **D6 (E1 + nan-safety)**: engagement gate carried from F17 amendment 1 (population std `correction=0`; non-finite engagement telemetry → fail-closed `F19_FALSIFIED_NO_ENGAGEMENT`), folded into C14.
- **D7 (flag/schema)**: engine gated by `HENRI_F19_SNR_REBALANCED=1`; receipt schema `f19-snr-rebalanced-engine.v1`; verdict prefix `F19_`.
- **D8 (ledger offset)**: directive mandates prereg seal "Record 1,119+"; ledger head at auth = 1,119 (post-F18 ingest/context events), so the seal lands at record 1,120.
- **D9 (projected SNR)**: the directive's SNR ≈ 8.125 / S_goal ≈ 0.650 / P_Sagnac ≈ 0.080 are **architect projections, not gates**; the falsifiable targets remain G3 ≥ +0.0200 and G2 ≥ 1 (measured live), with the measured SNR reported as DERIVED telemetry.

## 5. Execution order

1. TDD: `tests/contract/test_f19_rebalanced_engine.py` (C1–C16 + flag per above).
2. Implement `experiments/verification/arc_f19_rebalanced_engine.py` (flag fail-closed; canonical imports from `arc_f15_trajectory_engine` / `arc_f10_live_engine` / `arc_f11_plasticity_engine` only; no local shadowing).
3. Local suite + full regression from repo root (isolated Python 3.14).
4. Commit + push `carrier/f19-snr-rebalancing`; remote detached-worktree CUDA verify @ exact SHA.
5. Live gauntlet per directive §5 command: 12 envs × 150 = 1,800 steps, seed `20260918`, K=8, κ 2.50, step 1.50, β_base 0.010, μ 0.0, bank `/root/f3-run/telemetry/f3_bank_capture_v2/` (npz sha `9e3c01b4…` remote-verified).
6. Seal `F19_PREREG_SEALED` (record 1,120) → run → seal `F19_GATES_VERDICT`; deliver scorecard + lessons.

## 6. Kill experiment / cheapest falsification

- Local C2 fail (std(γ_a) ≤ 0.10 on non-collinear pair) or C4 fail (no positive gradient alignment at κ = 2.50) → the high-gain tilt does not preserve the F18 sign-flip on the D=64 substrate → **KILL before launch**.
- Live E1 no engagement (mean std_a(γ_a) ≤ 1e-6) → `F19_FALSIFIED_NO_ENGAGEMENT`.
- Live G3 ≤ 0 with engagement confirmed and G2 = 0 → **SNR rebalancing is falsified against the trajectory goal** — the alignment-gated β + κ 2.50 + step 1.50 combination does not produce positive directional valence at live scale (closes the F18 SNR-deficit hypothesis).
- Live G3 ≥ +0.02 but G2 = 0 and G4 > 0.05 → coherence trade-off regime: the penalty relaxation sacrificed waveform coherence without task resolution (partial falsification, reported as such).

## 7. Falsifiable claim

"HYPOTHESIS: with the Sagnac penalty attenuated alignment-adaptively (β_adaptive(t) = 0.010·(1.0 − |⟨Ψ_t, Ψ_goal⟩|), replacing the fixed 0.05), the Killing tilt amplified (κ_diff = 2.50), and the geodesic step norm expanded (step_scale = 1.50, ‖D̂_a‖_F ≡ 1.50·‖D_a‖_F), the beam objective J(a) = |⟨Ψ̂_{t+8}, Ψ_goal⟩| − β_adaptive(t)·Σ Δ_Sagnac(k) is goal-dominant (projected SNR ≈ 8.1, measured and reported), producing positive directional valence (G3 ≥ +0.02) and ≥ 1 live solve (G2 ≥ 1) on the 12-env live gauntlet under the verified bank ingress, with K=8 horizon coherence G4 ≤ 0.05."
