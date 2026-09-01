# Carrier F20 — Adjoint Symplectic Conjugation & Phase-Wrap Bounded Lie Flow — Pre-Registration

- Directive: `Project_HENRI_F19_Post-Mortem_Audit___Adjoint_Symplectic_Conjugation_Directive.md`
- Directive SHA-256: `97b563148a33069ff3323b9f64263b957efd5d0d57d7911b98e4b54d32e6c14f` (22,431 B, 258 lines)
- Directive ID: `HENRI-DIR-2026-08-F19-POSTMORTEM-ADJOINT-CONJUGATION`
- Branch: `carrier/f20-adjoint-conjugation` (base: F19 head `9e8f837` — the carrier the directive explicitly builds on)
- Governance: ledger head 1,124 at auth (directive cites "Record 1,122" = seal-time depth; seal lands at record 1,125+, deviation D7)
- Prior state: F19 sealed `#84c5aa60` (`F19_GATE_G2_FAILED` @ record 1,121, chain 1,122) **RATIFIED** by the directive; F19 falsified SNR-masking and isolated phase-wrap aliasing (θ_{K=8} ≈ 13.50 rad ≫ 2π, wrap count 2; Sagnac term = Lyapunov barrier, not noise)

## 1. Mechanism (directive §3 Tier 1–4, §3.1)

F19's beam overshot: κ = 2.50 × step_scale 1.50 → ≈ 3.75 rad/step → 13.50 rad over K=8 → SO(8) phase wrap annihilated the directional gradient (alignment 0.233→0.031, Δν −1.45e-04). F20 replaces additive tangent warping with **exact adjoint group action + Nyquist spectral clamping**:

1. **Goal extraction**: identical to F15–F19 (trajectory-bank terminal state, `resolve_trajectory_goal`, PG1 ≤ 0.90 pre-flight) — verified ingress retained unchanged.
2. **Tier 1 — goal rotation element** (directive §3 verbatim):
   - `Ω_goal(t) = Ψ_goal Ψ_t† − Ψ_t Ψ_goal† ∈ so(D)` (same skew tensor as F16–F19);
   - `R_goal(t) = exp(α_rot · Ω_goal / ‖Ω_goal‖_F) ∈ SO(D)`, **`α_rot = 0.35`** (directive §5 command).
3. **Tier 2 — exact adjoint conjugation** (THE F20 TILT MECHANISM, replaces F19's additive κ·tanh(γ)·Ω warp):
   - `D_a^conj = R_goal(t) · D_a · R_goal†(t) ∈ so(8)^M`;
   - **Invariant (directive §1.2): Spec(D_a^conj) ≡ Spec(D_a), ‖D_a^conj‖_F ≡ ‖D_a‖_F** — generator norm/spectrum distortion eliminated by construction (no κ_diff, no step_scale — those parameters are REMOVED).
4. **Tier 3 — symplectic Nyquist spectral-radius clamping** (THE F20 ALIASING MECHANISM):
   - `D̃_a = ClampSpectralRadius(D_a^conj, ω_max)` with per-block scale `min(1, ω_max / σ_max)`;
   - **`ω_max = π/(2K) = π/16 ≈ 0.1963 rad/step`** (directive §3.1/§1.2; CLI `--omega-max 0.1963`); guarantee: K=8 unroll total rotation ≤ π/2 → zero phase-wrap aliasing.
5. **Tier 4 — non-aliased vectorized beam search** (K = 8):
   - `Ψ̂_{t+1}(a) = exp(D̃_a) Ψ_t`; `J(a_{1:8}) = |⟨Ψ̂_{t+8}, Ψ_goal⟩| − β_Sagnac·Σ_{k=1..8} Δ_Sagnac(k)`;
   - **`β_Sagnac = 0.025` FIXED** (directive §3.1: calibrated Lyapunov energy regulator — replaces F19's adaptive β; NO attenuation channel), NO Frobenius term (μ_damp ≡ 0.0 locked, carried from F18 C15);
   - `a_t* = a_1*`; `Δν_t = |⟨Ψ_{t+1}, Ψ_goal⟩| − |⟨Ψ_t, Ψ_goal⟩|`.
6. **Carried mechanisms**: Hebbian goal-valence creep strictly on Δν > 0 (F18 C7 / F19 — directive F20 blueprint is silent; retained for substrate continuity, deviation D5).

**Engagement gate E1 (redefined for F20 — Killing γ no longer exists):** mean over steps of `std_a(‖D_a^conj − D_a‖_F / ‖D_a‖_F)` (population std, nan-safe fail-closed) must exceed 1e-6, else `F20_FALSIFIED_NO_ENGAGEMENT`. Additional diagnostic telemetry (not gates): `clamp_activity_mean` (fraction of candidate generators actually clamped), `spectral_radius_max_mean` (must be ≤ ω_max + 1e-6 — the mechanism-invariant proof), `adjoint_isometry_err_mean` (≈ 1e-7).

## 2. Gates (directive §4, verbatim bounds — identical to F19)

| Gate | Criterion | Failure action |
|---|---|---|
| PG1 | `max_env |⟨Ψ_0, Ψ_goal⟩| ≤ 0.90` | PRE-FLIGHT KILL (degenerate trajectory ingress) |
| G1 | 1,800 steps / 12 envs, latency ≤ 5.0 ms/step | KILL K1 |
| G2 | ≥ 1 of 12 solved (Score > 0.0%) | KILL K2 |
| G3 | mean Δν_goal ≥ +0.0200 | KILL K3 (directional steering failure / aliasing persists) |
| G4 | single-pass K=8 horizon Δ_Sagnac ≤ 0.0500 | KILL K4 (waveform coherence degradation) |
| E1 (F20) | mean over steps of `std_a(‖D_a^conj−D_a‖_F/‖D_a‖_F)` > 1e-6 (population std, nan-safe) | `F20_FALSIFIED_NO_ENGAGEMENT` |

## 3. Contract tests (directive §3/§4 + F19 §4 pattern; C1–C16 + flag fail-closed)

1. C1 so(8) skew symmetry of conjugated + clamped generators (`D̃ᵀ == −D̃`, err < 1e-6)
2. C2 adjoint isometry + spectrum invariance: `‖R D_a R†‖_F == ‖D_a‖_F ± 1e-5`; sorted singular values equal ± 1e-4 (all candidates, α ∈ {0.1, 0.35, 1.0})
3. C3 spectral-radius clamp: `σ_max(D̃_a) ≤ ω_max + 1e-6` for all a; no-op (scale == 1) when σ_max ≤ ω_max; active shrink when σ_max > ω_max (post-clamp σ_max == ω_max ± 1e-4)
4. C4 **aliasing elimination**: θ = 0.5 rad/step generator → raw K=8 unroll wraps (align ≈ 0.757 < 0.99); clamped K=8 unroll lands at exactly π/2 total (align ≥ 0.999); `8·ω_max ≤ π/2 + 1e-9`
5. C5 single-pass K=8 Sagnac coherence: clamped fixture → 1−align ≤ 0.05 (align ≥ 0.95) and `score_action` finite > 0.5
6. C6 vectorized einsum beam == sequential loop within 1e-5 (fixed β mirrored exactly)
7. C7 Hebbian creep strictly on Δν > 0 (carried)
8. C8 zero CUDA VRAM leak over 1,000 continuous steps (CPU functional loop)
9. C9 PG1 preflight rejection: degenerate bank → fail-closed (`F20_PREFLIGHT_DEGENERATE_GOAL`); no bank → `F20_BLOCKED_NO_TRAJECTORY_BANK`
10. C10 trajectory loader integrity (bank schema + dims + sealed SHA prefix `9e3c01b4` of `trajectories_production_run_f3v2.npz`)
11. C11 arcade environment handshake (live API, all 12 games)
12. C12 deterministic seed reproducibility (seed `20260919`, byte-identical trajectories)
13. C13 module constants bound (α_rot 0.35, ω_max = π/16, β_Sagnac 0.025, K 8, beam 8, seed 20260919, μ_damp 0.0 locked, E1 threshold 1e-6, gates 5.0/0.05/0.02/0.90)
14. C13b **fixed-beta differential** (vs F19 adaptive): β == 0.025 regardless of alignment; explicit β override honored (β=0 → higher J)
15. C14 NaN/Inf guard (degenerate zero generators → clean finite fallback; non-finite engagement telemetry → fail-closed verdict)
16. C15 **removed mechanisms + μ lock**: engine constructor REJECTS `kappa_diff` / `step_scale` / `beta_base` (TypeError — F19 parameters removed by directive); `mu_damp ≠ 0` rejected (ValueError + argparse `ArgumentTypeError`)
17. C16 clean receipt generation (schema `f20-adjoint-conjugation-engine.v1` + verdict mapping)
18. flag fail-closed (`HENRI_F20_ADJOINT_CONJUGATION=1` required; absent → RuntimeError)

## 4. Pre-registered deviations (disclosed, not silent)

- **D1 (ω_max value)**: directive §1.2/§3.1 formula is `π/(2K) = π/16 = 0.1963495…`; the §5 CLI prints `--omega-max 0.1963` (4-digit rounding). Engine default = π/16 exact; the launcher passes `0.1963` verbatim per §5. Difference 5e-5; the zero-aliasing guarantee (8·ω_max ≤ π/2) holds for both.
- **D2 (G4 instrument)**: G4 measured with the SAME `SinglePassHorizon` (K=8) + `sagnac_delta` vs goal as F18/F19 (cross-carrier comparability); the beam's J uses per-step exp-unroll Sagnac per directive Tier 4. Both reported.
- **D3 (E1 redefinition)**: F19's engagement was `std_a(γ_a)` (Killing coefficients). F20 removes the Killing warp; E1 is redefined as the conjugation-deviation statistic (see §1). Same threshold 1e-6, same nan-safe fail-closed semantics.
- **D4 (C12 determinism)**: byte-identical determinism tested on CPU (CUDA einsum/matrix_exp fp nondeterminism documented); live runs share the identical seed `20260919`.
- **D5 (creep carried)**: directive F20 blueprint omits the Hebbian creep; retained from F18 C7/F19 (fires only on Δν > 0, cannot oppose the goal direction), telemetry `creeps` reported.
- **D6 (isometry vs clamp)**: directive §1.2 property 1 states ‖D̃‖_F ≡ ‖D_a‖_F (pure conjugation). The Tier 3 clamp intentionally shrinks σ_max above ω_max — the isometry invariant is tested on the CONJUGATED generator (C2); the clamp contract is the spectral bound (C3/C4). Both reported as separate telemetry (`adjoint_isometry_err_mean`, `spectral_radius_max_mean`).
- **D7 (ledger offset)**: directive mandates prereg seal "Record 1,123+"; ledger head at auth = 1,124 (post-F19 ingest/context events), so the seal lands at record 1,125+.
- **D8 (flag/schema)**: engine gated by `HENRI_F20_ADJOINT_CONJUGATION=1`; receipt schema `f20-adjoint-conjugation-engine.v1`; verdict prefix `F20_`.
- **D9 (CLI defaults)**: directive §5 command omits `--envs` (default F10 cohort, 12 envs) and `--beam` (default 8, carried).

## 5. Execution order

1. TDD: `tests/contract/test_f20_adjoint_engine.py` (C1–C16 + flag per above).
2. Implement `experiments/verification/arc_f20_adjoint_engine.py` (flag fail-closed; canonical imports from `arc_f15_trajectory_engine` / `arc_f10_live_engine` / `arc_f11_plasticity_engine` only; no local shadowing).
3. Local suite + full regression from repo root (isolated Python 3.14).
4. Commit + push `carrier/f20-adjoint-conjugation`; remote detached-worktree CUDA verify @ exact SHA.
5. Live gauntlet per directive §5 command: 12 envs × 150 = 1,800 steps, seed `20260919`, K=8, α_rot 0.35, ω_max 0.1963, β_Sagnac 0.025, bank `/root/f3-run/telemetry/f3_bank_capture_v2/` (npz sha `9e3c01b4…` remote-verified).
6. Seal `F20_PREREG_SEALED` → run → seal `F20_GATES_VERDICT`; deliver scorecard + lessons.

## 6. Kill experiment / cheapest falsification

- Local C2 fail (adjoint not isometric) or C3 fail (clamp not binding) or C4 fail (wrap persists under the clamp) → the adjoint/clamp substrate is broken on the D=64 substrate → **KILL before launch**.
- Live E1 no engagement (mean conjugation deviation ≤ 1e-6) → `F20_FALSIFIED_NO_ENGAGEMENT`.
- Live G3 ≤ 0 with engagement confirmed and G2 = 0 → **adjoint conjugation + Nyquist clamping is falsified against the trajectory goal** — the warp DIRECTION (conjugation) is not the fix for the F19 phase-wrap collapse (closes the F19 post-mortem's primary directive).
- Live G3 ≥ +0.02 but G2 = 0 and G4 > 0.05 → coherence trade-off regime: clamp bound insufficient at live scale (partial falsification, reported as such).

## 7. Falsifiable claim

"HYPOTHESIS: replacing additive tangent warping with the exact adjoint group action (D_a^conj = R_goal D_a R_goal†, R_goal = exp(0.35·Ω_goal/‖Ω_goal‖_F)) plus symplectic Nyquist spectral-radius clamping (σ_max ≤ π/16 rad/step, guaranteeing total K=8 rotation ≤ π/2) with a fixed calibrated Sagnac regulator (β = 0.025) produces positive directional valence (G3 ≥ +0.0200) and ≥ 1 live solve (G2 ≥ 1) on the 12-env live gauntlet under the verified bank ingress, with single-pass K=8 horizon coherence G4 ≤ 0.05 — because the beam objective J(a) = |⟨Ψ̂_{t+8}, Ψ_goal⟩| − 0.025·Σ Δ_Sagnac(k) no longer phase-wraps on the compact group."
