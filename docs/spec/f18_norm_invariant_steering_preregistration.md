# Carrier F18 — Norm-Invariant Steering — Pre-Registration

- Spec: `Project_HENRI_Carrier_F18_Pre-Registration_Specification.md`
- Spec SHA-256: `9a8b97168564b70b8b46d0171f362fbec9630c844f9c65cce2adeb5c2b6505d9` (13,290 B, 171 lines)
- Spec ID: `HENRI-SPEC-2026-08-F18-NORM-INVARIANT-STEERING`
- Branch: `carrier/f18-norm-invariant-steering` (base: F17 head `b6e45b2` — the carrier the spec explicitly builds on)
- Governance: ledger head 1,115 at auth (spec mandates "Record 1,113+" — a floor; the ledger has advanced past it, so the prereg seal lands at record 1,116; deviation D8)
- Prior state: F17 sealed `#18aac03d` (`F17_GATE_G2_FAILED` @ record 1,112) RATIFIED by the spec (§1.1 quotes F17's diagnosis verbatim: damping penalty ≈ 9.6 vs goal term ≈ 0.05, ~192× imbalance; γ std 0.306; Δν_goal −3.2e-05)

## 1. Mechanism (spec §1.2, §2.1–2.2, pipeline Steps 1–4)

Replaces F17's damping-dominated beam (FALSIFIED: μ‖D̃‖² ≈ 9.6 vs goal term ≈ 0.05; ‖D̃‖² = (1+κ·tanh γ)²‖D‖² anti-correlates with γ ⇒ the damping term actively anti-selected goal-aligned candidates; alignment 0.099→0.074) with **Unitary Tangent Normalization + zero Frobenius damping**:

1. **Goal extraction**: identical to F15/F16/F17 (trajectory-bank terminal state, `resolve_trajectory_goal`, PG1 ≤ 0.90 pre-flight) — verified ingress pathway retained unchanged.
2. **Step 1 — skew goal matrix**: `Ω_goal(t) = Ψ_goal Ψ_t† − Ψ_t Ψ_goal† ∈ so(D)` (unchanged from F16/F17).
3. **Step 2 — Killing alignment coefficients**: `γ_a = −½ Tr(D_a Ω_goal(t))` per action (all 8 in parallel); skew-D algebra ⇒ `γ_a = ⟨Ψ_goal, D_a Ψ_t⟩`.
4. **Step 3 — Unitary tangent normalization** (THE F18 MECHANISM):
   `D_a′ = D_a + κ_diff·tanh(γ_a)·Ω_goal`, `κ_diff = 0.75`;
   `D̂_a = ‖D_a‖_F · D_a′ / ‖D_a′‖_F ∈ so(8)^M`.
   **Invariant: ‖D̂_a‖_F ≡ ‖D_a‖_F for all a** (per-generator norm conservation, exact up to fp) — the norm lever is removed; the ONLY steering channel is the direction tilt of D̂_a toward Ω_goal. This is the direct kill of F17's anti-selection: there is no norm term left for the beam to exploit.
5. **Step 4 — exact unitary exponentiation + horizon beam (K=8)**:
   `Ψ̂_{t+1}(a) = exp(D̂_a) Ψ_t ∈ S^{D−1}`;
   `J(a_{1:8}) = |⟨Ψ̂_{t+8}, Ψ_goal⟩| − β_Sagnac·Σ_{k=1..8} Δ_Sagnac(k)`, `β_Sagnac = 0.05` (NO Frobenius term — μ_damp ≡ 0.0 locked, spec §4 C15);
   `a_t* = a_1*`; `Δν_t = |⟨Ψ_{t+1}, Ψ_goal⟩| − |⟨Ψ_t, Ψ_goal⟩|`.
6. **Tier 4 — Hebbian creep** (spec §4 C7): updates occur **strictly on Δν > 0** (engine guards `delta_nu <= 0.0` → no update; stronger than F17's `== 0.0` guard per C7 wording).

## 2. Gates (spec §3, verbatim bounds)

| Gate | Criterion | Failure action |
|---|---|---|
| PG1 | `max_env |⟨Ψ_0, Ψ_goal⟩| ≤ 0.90` | PRE-FLIGHT KILL (degenerate trajectory ingress) |
| G1 | 1,800 steps / 12 envs, latency ≤ 5.0 ms/step | KILL K1 |
| G2 | ≥ 1 of 12 solved (Score > 0.0%) | KILL K2 |
| G3 | mean Δν_goal ≥ +0.0200 | KILL K3 (directional steering failure / inversion persists) |
| G4 | single-pass K=8 horizon Δ_Sagnac ≤ 0.0500 | KILL K4 (waveform coherence degradation) |
| E1 (carried from F17, pre-registered) | mean over steps of `std_a(γ_a)` > 1e-6 (population std, nan-safe fail-closed) | `F18_FALSIFIED_NO_ENGAGEMENT` |

## 3. Contract tests (spec §4, C1–C16 — implemented verbatim)

1. C1 so(8) skew symmetry of normalized generators
2. C2 Killing-form variance `std(γ_a) > 0.10` on non-collinear goal states
3. C3 tangent norm conservation `‖D̂_a‖_F == ‖D_a‖_F ± 1e-6` across the tilt range (κ ∈ {0, 0.5, 0.75, 1.0, 2.0} covers γ ∈ [−1, 1] and beyond)
4. C4 positive gradient alignment: ∂/∂γ_a `|⟨exp(D̂_a)Ψ_t, Ψ_goal⟩| > 0` (numerical derivative, non-collinear positive-Killing fixtures)
5. C5 Sagnac homodyne coherence: single-pass K=8 Δ ≤ 0.05
6. C6 vectorized einsum beam == sequential loop within 1e-5
7. C7 Hebbian creep strictly on Δν > 0
8. C8 zero CUDA VRAM leak over 1,000 continuous steps
9. C9 PG1 preflight rejection (fail-closed degenerate bank)
10. C10 trajectory loader integrity (SHA-256 `9e3c01b4…` + array dims of `trajectories_production_run_f3v2.npz`)
11. C11 arcade environment handshake (live API, all 12 games)
12. C12 deterministic seed reproducibility (seed `20260917`, byte-identical trajectories)
13. C13 module constants bound (κ_diff=0.75, β_sagnac=0.05, μ_damp=0.0 locked)
14. C14 NaN/Inf guard (degenerate zero tensors → clean fallback, fail-closed verdict)
15. C15 zero-Frobenius-penalty lock (μ_damp ≡ 0.0 cannot be overridden via CLI)
16. C16 clean receipt generation (schema + verdict mapping)

## 4. Pre-registered deviations (disclosed, not silent)

- **D1 (μ lock semantics)**: spec §5 command block passes `--mu-damp 0.0`; §4 C15 forbids overriding. Engine: constructor raises `ValueError` on |μ|>1e-12; CLI `--mu-damp` uses a validating type that rejects any non-zero value with `ArgumentTypeError` (exit 2); module constant `MU_DAMP_LOCKED = 0.0` asserted by C13/C15.
- **D2 (C4 regime)**: C4 validates the derivative in the mechanism's steering regime — candidate generators with γ_a > 0 that are non-collinear with Ω (a collinear generator has no direction left to tilt: derivative identically 0, which is the norm-free mechanism's boundary case). The global claim "∂J/∂γ_a > 0 ∀ a ∈ A" is carried by live G3 (the pre-registered falsification).
- **D3 (C5 fixture)**: C5 uses goal-aligned generators (coherence property of the horizon on aligned candidates); live G4 measures the seeded random generators on the real gauntlet.
- **D4 (C12 determinism)**: byte-identical determinism tested on CPU (CUDA einsum/matrix_exp fp nondeterminism documented); live runs share the identical seed `20260917` for run-level reproducibility.
- **D5 (beam)**: spec §5 command block omits `--beam`; engine keeps `--beam` default 8 (carried from F17 default) for parity; §2 pipeline Step 4 states K=8.
- **D6 (E1 + nan-safety)**: engagement gate carried from F17 amendment 1 (population std `correction=0`; non-finite engagement telemetry → fail-closed `F18_FALSIFIED_NO_ENGAGEMENT`), folded into C14.
- **D7 (flag/schema)**: engine gated by `HENRI_F18_NORM_INVARIANT=1` (mirrors F15/F16/F17); receipt schema `f18-norm-invariant-engine.v1`.
- **D8 (ledger offset)**: spec mandates prereg seal "Record 1,113+"; ledger head at auth = 1,115 (post-F17 context/ingest events), so the seal lands at record 1,116.
- **D9 (seed)**: spec §5 command seed `20260917`; all constants per command block (κ 0.75, β 0.05, μ 0.0, 150 steps × 12 envs = 1,800).

## 5. Execution order

1. TDD: `tests/contract/test_f18_norm_invariant_engine.py` (C1–C16 per spec §4).
2. Implement `experiments/verification/arc_f18_norm_invariant_engine.py` (flag fail-closed; canonical imports from `arc_f15_trajectory_engine` / `arc_f10_live_engine` / `arc_f11_plasticity_engine` only; no local shadowing).
3. Local suite + full regression from repo root (isolated Python 3.14).
4. Commit + push `carrier/f18-norm-invariant-steering`; remote detached-worktree CUDA verify @ exact SHA.
5. Live gauntlet per spec §5 command: 12 envs × 150 = 1,800 steps, seed `20260917`, K=8, κ 0.75, μ 0.0, β 0.05, bank `/root/f3-run/telemetry/f3_bank_capture_v2/` (npz sha `9e3c01b4…` remote-verified).
6. Seal `F18_PREREG_SEALED` (record 1,116) → run → seal `F18_GATES_VERDICT`; deliver scorecard + lessons.

## 6. Kill experiment / cheapest falsification

- Local C2 fail (std(γ_a) ≤ 0.10 on non-collinear pair) or C4 fail (no positive gradient alignment) → the norm-free warp does not break rank invariance / does not tilt toward the goal on the D=64 substrate → **KILL before launch**.
- Live E1 no engagement (mean std_a(γ_a) ≤ 1e-6) → `F18_FALSIFIED_NO_ENGAGEMENT`.
- Live G3 ≤ 0 with engagement confirmed and G2 = 0 → **norm-invariant steering is falsified against the trajectory goal** — the direction tilt alone does not produce positive directional valence at live scale (closes the F17 anti-selection hypothesis).

## 7. Falsifiable claim

"HYPOTHESIS: with the Frobenius damping term eliminated (μ_damp ≡ 0.0, locked) and the warped generators normalized to their per-generator base norms (‖D̂_a‖_F ≡ ‖D_a‖_F), the beam objective J(a) = |⟨Ψ̂_{t+8}, Ψ_goal⟩| − β·Σ Δ_Sagnac(k) is damping-free and goal-aligned candidates are no longer penalized, producing positive directional valence (G3 ≥ +0.02) and ≥ 1 live solve (G2 ≥ 1) on the 12-env live gauntlet under the verified bank ingress, with K=8 horizon coherence G4 ≤ 0.05."
