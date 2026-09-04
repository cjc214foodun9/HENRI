# Carrier K3 Pre-Registration (Sealed): Empirical Koopman Transition Generator & Action-Outcome Grounding

**Document Identifier:** `HENRI-SPEC-2026-09-V3-CARRIER-K3-SEALED-PREREG`
**Branch:** `feat/carrier-k3-empirical-koopman` (base `2f9bc57`, post-C1 closeout, clean worktree)
**Seal date:** 2026-09-03 (live session, direct OBSERVED reads)

## 1. Supplied artifacts (authenticated, byte-identical)

| Artifact | Path (Drive inbox) | SHA-256 (full) | Bytes | Read |
|---|---|---|---|---|
| Supplied prereg spec | `Carrier_K3_Pre-Registration_Specification.md` | `841ac58159935f8a27cc19f602c357bf4c612dba934ed01dedd462c060543ecc` | 15,459 | FULL (203 lines) |
| Supplied kernel | `Carrier_K3_Fused_Triton_Kernel.py` | `bff0174955e5eea7d22be222c4a8056f2e04d02ad077de272776a7bf8ce66e4e` | 6,205 | FULL (155 lines) |

Staged byte-identical copies: `docs/spec/carrier_k3_supplied_prereg.md`, `HENRI V2/experiments/verification/carrier_k3_supplied_kernel.py` (SHA asserted in contract tests).

## 2. Dispositions of supplied-spec claims vs live code (audit)

| Claim | Disposition | Evidence |
|---|---|---|
| "Fused Triton kernel executes the entire block-wise solve in-kernel, 45 µs" | **CONFLICTS_WITH_SUPPLIED_CODE**: the file's kernel (`_block_covariance_accum_kernel`) accumulates only A/B covariance; the solve is batched `torch.linalg.cholesky_solve` in the class. 45 µs is projection, not measurement. | File lines 12–58 (kernel), 109–127 (class) |
| Spectral contraction | **BOUNDED_IMPLEMENTABLE** with mandatory engagement telemetry: scaling by σ_max is unconditional in the class; if ridge already contracts, the projection never fires. Raw ρ/σ_max + fired-block counts are reported; an unfired projection is NOT a pass. | File lines 122–126 |
| Ridge solve per block | **BOUNDED_IMPLEMENTABLE**; α = 1e-4, SPD via αI, pinv fallback retained with count. | File lines 107–120 |
| Block factorization M=8,192 × d=8 | **ALREADY_CONSISTENT** with the live `[num_blocks, 8]` wave boundary and C1 engine geometry. | Live seam (arc_c1_steering_engine.py) |
| 12-env cohort `ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0`, 1,800 steps | **BLOCKED_MISSING_PREMISE (partial)**: the only 7-action trajectory bank on vast-5090 (`/root/f3-run/telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.npz`, 1,536 rows fp16) has ≥ 30 rows per env for **7 of the 12** (`ar25 bp35 cd82 ft09 g50t ka59 lp85`); **zero rows** for `cn04 dc22 lf52 ls20 m0r0`. Bank-derived full-wave goals (`build_p1_full_goals`) are therefore unavailable for those 5 → the K3 goal-coupling meter cannot measure them with the existing bank. | Remote bank probe (OBSERVED, 2026-09-03) |
| "45 µs solve; τ_step ≤ 2.00 ms (KG5)" | **REQUIRES_MEASUREMENT**: KG5 is bound to the LOCAL K3 mechanism time (score-path + refit CUDA-event mean ≤ 2.00 ms). Remote arcade round trips (~5 ms wall) are reported separately and are NOT the KG5 basis (they flagged C1's LG3 spuriously). | C1 receipt precedent |
| Cold-start 10 exploratory steps/env | **BOUNDED_IMPLEMENTABLE**; detailed per-action fit/eval protocol sealed below (the spec defines no N floors). | This doc §4 |

## 3. Live-input surface (OBSERVED)

- Bank: `/root/f3-run/telemetry/f3_bank_capture_v2/` npz keys `psi|next_wave|actions_onehot|action_names`; fp16 `[N, 65536]`; N=1,536; rows map 1:1 to `trajectories_production_run_f3v2.jsonl`.
- Per-env rows (npz, sidecar-matched): ar25 134, bp35 118, cd82 100, ft09 150 (A6-only), g50t 130, ka59 100, lp85 150 (A6-only); per-action ≥ 19 rows for the env's native actions except A5/A7 (near-zero globally).
- Remote: `/venv/main/bin/python` torch 2.12.0+cu130, triton 3.7.0, CUDA available (OBSERVED). RTX 5090.
- Launcher seam (`arc_g7_calibrated_engine.py`): `use_g8 → use_c1 → use_p1` elif chain; G7 default; modules lazy flag-gated. K3 branch is inserted as `elif use_k3:` after `use_c1` (C1 is sealed FALSIFIED; K3 replaces the rotor dictionary per the spec).
- K3 operators are fit **live** from the run-loop's own transitions (`update_online_affordance(psi_full, idx, psi_full_next)` post-step hook in `G4AlignedEngine.run_gauntlet`), NOT from the bank. The bank supplies only goals for the 7 covered envs.

## 4. Sealed protocol amendments (visible, pre-code)

1. **Per-action fit floor:** an action operator is fitted only when its ring holds ≥ 10 total rows with ≥ 8 fit rows (`KFIT_MIN_N = 8`; held-out window `W = clamp(N/4, 2, 8)` rows excluded from the fit sums). Unfitted actions score `drops = 0` (π^H governs), exactly like P1/C1 missing-transition arms, and are reported per-action.
2. **Ring:** per-action ring cap 256 (fp16 rows, ~458 MB peak on GPU), cleared at every environment boundary (`p1_bind_env_goal` override). Fit sums maintained incrementally O(M·d²) per pushed row; a sliding held-out window of the newest rows is never in the fit sums → KG1 is a genuine held-out one-step error (prediction error measured PRE-update of that row).
3. **Covariance source:** live refits use the incremental torch sums (numerically identical Σ over the same ring). The supplied Triton kernel (fixed `N_transitions: tl.constexpr`, hardcoded M=8,192 stride) is exercised by the CUDA equivalence test at fixed N and by the τ_solve measurement; per-step constexpr-N recompilation would churn the JIT cache (engineering amendment, disclosed).
4. **Engagement counters (KG4 non-vacuity):** every refit reports raw max σ_max, raw max ρ (eigvals), and the count of blocks whose σ_max > 1.0 (scaled = projection fired). KG4's bound is enforced by the scaling; the gate's scientific content is the fired count + raw values.
5. **Verdict precedence (fail-closed, mirroring C1):** engagement → KG5 local latency → KG1 held-out error → KG3 separation → KG4 spectral → **KG2 coupling (seal basis)** → KG6 solved. First-fired symbol ≠ seal basis; KG2 governs W0, exactly as C1's LG1 governed.
6. **Cohort:** dispatch envs = the 12 named in the supplied spec; envs without bank goals (`cn04 dc22 lf52 ls20 m0r0`) run with `goal_unavailable` recorded per env and are **excluded from the KG2/KG6 seal-basis aggregation** (reported per-env, diagnostic only). The seal-basis Δν is computed over the 7 goal-available envs and is labeled CONDITIONAL vs C1's 12-env basis. A fresh capture bank for the 5 envs is a separate pre-requisite if the user wants a full 12-env seal basis.
7. **NaN/Inf fail-closed:** any non-finite value in fit sums, operator, or candidate raises `K3NumericalAbort` after serializing ring state to `<out_dir>/_abort_k3/`; the exception propagates (non-zero exit), per spec §3.4.
8. **Dispatch governance:** remote dispatch of the K3 gauntlet requires a sealed human instrument (`HENRI-AUTH-…-CARRIER-K3-DISPATCH`, APPROVE_REMOTE_RUN), naming envs, steps, seed, commit, machine, and gates — mirroring C1 (`cc485247…`). No dispatch before that instrument exists.

## 5. Verification gates (bound to live receipt fields)

| Gate | Metric (live) | Bound |
|---|---|---|
| KG1 | Held-out one-step relative error, mean over actions with eval rows | ≤ 0.1500 |
| KG2 | `mean_delta_nu_wp` over goal-available envs (seal basis) | ≥ 0.0200 |
| KG3 | Min pairwise operator separation over fitted pairs | ≥ 0.0500 |
| KG4 | Max post-scale ρ (enforced) + raw ρ / fired-block engagement | ≤ 1.000001 (enforced); engagement reported |
| KG5 | Local score-path + refit CUDA-event mean | ≤ 2.00 ms |
| KG6 | Envs solved (goal-available subset) | ≥ 1 |
| — | W0 (`WavePacketPathSearch`) | stays GATED until KG2 passes |

Constants implemented: `K3_FLAG = HENRI_K3_KOOPMAN`, seed 20260930, M=8192, d=8, α₀=1e-4, cond guard 1e5 (α←2α), ring cap 256, `KFIT_MIN_N=8`, eval window W=clamp(N/4,2,8), cold start 10 steps/env.

## 6. Kill / verdict taxonomy

`K3_BLOCKED_NAN` | `K3_NO_ENGAGEMENT` (steps>0, zero K3-fitted scores) | `K3_GATE_KG5_LATENCY_FAILED` | `K3_GATE_KG1_PREDICTION_FAILED` | `K3_GATE_KG3_SEPARATION_FAILED` | `K3_GATE_KG4_SPECTRAL_FAILED` | `K3_FALSIFIED_ACTION_COUPLING` (KG2) | `K3_FALSIFIED_TASK_SOLVE_KG6` | `K3_EMPIRICAL_KOOPMAN_VERIFIED`.

No parameters are tuned post-seal. C1's sealed FALSIFIED record is untouched; K3 is a new mechanism with its own gates.
