# Decision Matrix D1 + D2 — Pre-Registration (feat/low-rank-wave-jepa)

Source: `HENRI V2 Next Decision Matrix.pdf` (Drive inbox, 5 pp, raw SHA-256
`2e2cf71151ed39732563a53d898f516281a6d6e3eb5c7934c1de9526ec03df66`,
LF-normalized text SHA-256 `d8519f8d4f883f2f6a88677f6c01c21d1ba119a72fb9db94da99c052bc836fc0`).

## Premise reconciliation (OBSERVED from live code @ `2218ec4`)

| Matrix claim | Live-code truth | Disposition |
|---|---|---|
| Branch `feat/low-rank-wave-jepa @ 34c1ab2 (Sealed/Unpromoted)` exists | NO such branch (local or remote); `34c1ab2` = Phase 8.4 seal on `phase/8.4-jepa-transition-training` | Branch created fresh from `main` `2218ec4` |
| Production transition = legacy block-diagonal `UnitaryWaveTransition`, zero cross-channel Jacobian, loss floor 0.995 | Production transition IS `LowRankCoupledTransition` (V·W† + R_block, QR retractions), live at `efe_planner.py:70`, constructed `:251`; `UnitaryWaveTransition` = backward-compat alias `:184` | D1 = REUSE-ONLY adapter; no new operator |
| D1 targets `wave_jepa.py` + `multiscale_temporal_coupler.py` | `WaveJEPA` uses `RecursiveDualEDMD`; both files have ZERO production callers | Wire production operator into `WaveJEPA` behind default-OFF flag; coupler untouched (dead code) |
| D1 spec `V_a, W_a ∈ ℂ^{D×64}` complex per-action tensors | Third tensor family — incompatible with real `[num_blocks,8]` UWE waves (rejected Phase 8.4, Wavejepatrain.txt) | REJECTED; reuse real-valued production operator |
| D1 gate: Sagnac loss decrease > 5% within 30 online steps on CUDA | Sealed E1 (2026-08-14): 0.07% descent over 200 steps with `train_transition_step` @ D=65,536 | Gate re-probed as specified; expectation = INERT (pre-falsified at longer horizon) |
| D2: thermostat noise is isotropic, no P_null projector | CONFIRMED: `adaptive_viscoelastic_thermostat.py:214` `torch.randn_like * scale`; no `P_null = I − VV†` anywhere | D2 = genuine gap; implement |

## D1 — Low-rank coupled transition in WaveJEPA (REUSE-ONLY)

Mechanism: `WaveJEPA.predictor` swaps `RecursiveDualEDMD` → production
`LowRankCoupledTransition` (imported from `efe_planner`, same `[num_blocks,8]`
real contract). No new operator code. Default OFF.

- Flag: constructor param `use_lowrank_coupled: bool = False` (env
  `HENRI_WAVEJEPA_LOWRANK_COUPLED=1` also honored).
- Gate probe (fresh, 30 steps): train the LIVE production transition via
  `EFEPlanner.train_transition_step` on real encoder waves from known-transform
  grids (D=65,536), exactly 30 steps, lr=0.05, surprise on, valence=0.
  Metric: `loss_ema` at step 30 vs step 1.
- G1: `loss_decrease_pct > 5.0` → D1 PASS.
- Kill: `loss_decrease_pct <= 5.0` → D1 INERT (consistent with sealed E1).
- G2: when flag ON the predictor is the reuse adapter whose `.transition`
  is the production `LowRankCoupledTransition` (no duplicate operator class).
- G3: default OFF → predictor remains `RecursiveDualEDMD` (byte-identical
  default path).
- G4: `diagnostic_only=true`; no env stepping anywhere in probe sources.
- No score eligibility. No promotion.

## D2 — Anisotropic Langevin injection (P_null projection)

Mechanism (PDF §2.2): when Sagnac veto fires, thermal noise is injected
exclusively into the error-aligned null subspace:

    Ψ_injected = Ψ_t + sqrt(2 T_eff Δt) · P_null ξ,   P_null = I − V V†

Implementation in `AdaptiveViscoelasticThermostat`:
- Constructor params (both default OFF): `use_null_subspace_projection: bool =
  False`, plus a basis setter `set_null_basis(V)`; V real `[d, r]` column-
  orthonormal (the coupled transition's `field_V` is the canonical source).
- When enabled AND basis set: `noise = (ξ − V(V†ξ)) * scale`, applied in
  `step_viscoelastic_creep` to the same fresh white draw; total noise energy
  is reduced by the projection (directional covariance change is the
  mechanism, energy multiplication is NOT).
- Fail-closed: non-finite basis, wrong dtype, wrong leading dim, or rank ≤ 0
  → `ValueError` before any update.
- Default path (flag off or no basis): byte-identical to legacy isotropic.

- A1 (D2 contract): enabled arm noise satisfies
  `||V† noise|| / ||noise|| < 1e-3` while isotropic arm is at baseline
  `E[||V† ξ||/||ξ||] ≈ sqrt(r/d)`; energy ratio `||noise_proj||/||noise_iso||`
  ∈ (0, 1] and reported.
- A2 (D2 gate, PDF): > 40% reduction in phase-space variance drift during
  recovery from invalid AST states — measured via paired recovery harness
  (isotropic vs projected) on the SAME white draws at D=65,536, r=64.
  Dimensional note: P_null removes r/D ≈ 0.1% of isotropic noise energy at
  production scale (energy ratio ≈ sqrt(1 − r/D) ≈ 0.9995), so the PDF's
  >40% gate is expected to FAIL unless the error signal concentrates in the
  V subspace; a D=1,024 / r=256 sanity arm (r/D = 0.25) provides mechanism
  evidence that the projection path changes directional covariance. D2 is a
  noise-shape change, not a recovery accelerator; report separately, do not
  conflate with wavelet gating (Phase 5 P2 ACCEPT).
- A3: fail-closed raises exercised in contract tests.
- No env stepping; `diagnostic_only=true`; no score eligibility.

## Protocol

- Implementation: default-OFF, one bounded change per file.
- Local: contract tests (6) + full suite (base `2218ec4`, no regression).
- Remote: clean worktree @ candidate SHA, GPU-exclusive, detached matrix
  `setsid nohup`, arms = A0 OFF / A1 D1 / A2 D2 (+sanity) / A3 D1+D2 (WaveJEPA adapter
 smoke @ 65,536), aggregated exit codes,
 DONE marker only when all arms rc=0.
- Verdicts: D1 PASS/INERT per G1; D2 PASS/FAIL per A2 with A1/A3 contract
  evidence. Independent verdicts (a combined improvement cannot conceal an
  inert mechanism).
- Seal on branch; NO promotion; `main` untouched `2218ec4`.
