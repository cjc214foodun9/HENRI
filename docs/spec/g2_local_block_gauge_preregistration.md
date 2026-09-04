# Carrier G2 — Local Block-Gauge Affordance: Pre-Registration

**Directive:** user message (2026-09-01, "transitioning to a relativistic attention algorithm… multi-phased sprint") + `Example code.pdf` — HENRI-EVAL-2026-09-V3-G1-FALSIFICATION-AUDIT (SHA-256 `9d36971c46d244023f6437bc9aa0713e27308fabfac81c78292cc3632a80bf0c`, 175,315 B, `%PDF-1.4`). No separate G2 directive file exists in `G:/My Drive/HENRI_Inbox` (OBSERVED 2026-09-01). Supporting refs (authenticated, filed, out of G2 scope): `Project_HENRI_V3_Gravitational_Attention_and_Holographic_Spacetime_Synthesis.md` (`4102e2d1…`, 12,861 B), `Project_HENRI_V3_Universal_Attention___Relativistic_Holography_Specification.md` (`8b4ac64d…`, 15,611 B), `HENRI_V3_Relativistic_Attention_Engine.py` (`123f663c…`, 12,039 B).

**Prior (22nd sealed falsification):** `G1_AFFORDANCE_FIT_COLLAPSE` @ ledger 1,153 (`#f116b9e2…`) — PG1 min_auc **0.7768 < 0.8500**, 0 live steps, per-action AUCs 0.7768–1.0000. G1 + this PDF agree on the root cause: the **mean-pooled D=64 bridge** (16×4096 block-mean → PatchIngress) erases the local block phase that distinguishes "at a wall"; motion-dominated actions a0–a3 (D=64 "moving" rate 0.82–0.89) under-separated.

**Hypothesis (falsifiable):** evaluating the affordance on the FULL `[8192, 8]` wave — per-block quadratic metric + softmax pooling — separates moving-vs-blocked at in-sample AUC ≥ **0.8800** for all 7 actions, where the identical fitting rule on the D=64 mean-pooled bridge yields < 0.8500 (G1 kill reproduction, C2b).

## Mechanism (PDF "Local Block-Gauge Affordance", degeneracy fixed)

1. **Label** (PDF lever 2 — stall-cosine): `y_moving = [cos(ψ_t, ψ_{t+1}) < τ_stall]`, **τ_stall = 0.90**, measured on the FULL-D bank. OBSERVED calibration (remote, bank, 2026-09-01): per-action moving minority at cos<0.90: a0 0.20, a1 0.20, a2 0.20, a3 0.22, a4 0.234, a5 0.205, a6 0.19 — balanced classes, physically honest (wall collision ⇒ cos≈1). G1's D=64 `‖ΔΨ‖>0.05` label at 82–89% "moving" was bridge-noise artifact.
2. **State:** full bank wave `[N, 8192, 8]` (no mean-pooling). Per-block augmented vector `φ_b = [ψ_b; 1] ∈ R^9`.
3. **Per-action metric:** `q_a(ψ)_b = φ_bᵀ W_a φ_b`, `W_a ∈ Sym(9)`, fit closed-form as centered quadratic correlation (G1 fit rule at full-D, block-separable): `W_a = (1/N_a) Σ_i (y_i − ȳ_a) Σ_b φ_i,b φ_i,bᵀ`, symmetrized. **The PDF's skew-Hermitian form `i·W_skew` is IDENTICALLY ZERO for real ψ** (`Re(ψᵀ(iW)ψ) = 0` ⇒ constant affordance ⇒ AUC 0.5) — a degenerate form that would reproduce G1's kill class. Fixed to real symmetric (a metric, not a generator). Control C2a reproduces the PDF form.
4. **Softmax pooling** (PDF lever 1): `pooled_a = Σ_b softmax(β·q_a(ψ)_b) · q_a(ψ)_b`, **β = 10** — isolates the maximum local collision coordinate without averaging dilution.
5. **Per-action temperature** (PDF lever 3): `logit_a = pooled_a / τ_a + b_a`; `b_a = logit(clamp(ȳ_a, 0.05, 0.95))`; `τ_a = τ_base·exp(clamp(log(std_a(pooled)/σ_ref), log 0.1, log 10))` with τ_base=0.05, σ_ref=0.05 — deterministic, closed-form, no label-peeking beyond the fit.
6. **Kinematics (unchanged from G1):** `T_free,a = exp(D_free,a) ∈ SO(64)` fit on D=64 bridge moving-only transitions (`‖ΔΨ‖>0.05`), ω-bound π/32, ridge 1e-4; scattering `Ψ̂ = π·T·Ψ + (1−π)·Ψ`; homotopy beam K=8, `J = align·π^K`; dual-speed online affordance update η=0.10.
7. **Live representation:** full-D wave per step via `FastFullDWaveEncoder` — vectorized chunked phase accumulation, CC-OS parity-preserving (production-equivalent). **Equivalence contract vs `HENRIVisionEncoder`: cos ≥ 0.9999 AND max|Δ| ≤ 1e-3 on 3 synthetic grids (10×10, 20×20, 30×30, seed 20260925), verified on CUDA before the gauntlet.** Motivation (OBSERVED): production encoder = 738 ms/step at 30×30 (per-cell Python loop) — unusable live.

## Gates

| Gate | Bound | Failure |
|---|---|---|
| **PG1** (binding) | in-sample per-action AUC ≥ **0.8800** (≥10 samples/action), all 7 actions | PRE-FLIGHT KILL `G2_AFFORDANCE_FIT_COLLAPSE` |
| G1 (reference) | mean per-step latency ≤ 2.00 ms (1,800 steps) | `G2_GATE_G1_FAILED` |
| G2 (reference) | ≥ 1/12 envs solved | `G2_GATE_G2_FAILED` |
| G3 (reference) | mean post-action waypoint Δν ≥ +0.0150 | `G2_GATE_G3_FAILED` |
| G4 (reference) | mean `1 − |⟨Ψ̂_{t+1}(a_exec), Ψ_{t+1}⟩|` ≤ 0.0500 (actual observation) | `G2_GATE_G4_FAILED` |

PDF promotion language: "Upon achieving AUC ≥ 0.8500 on a0–a3, unlock gates G1 through G4" — i.e. G1–G4 formally bind the NEXT carrier; G2 runs the 12-env gauntlet (PDF step 3) and reports G1–G4 with the G1-directive thresholds as **reference metrics**. Verdict is fail-closed regardless: any reference-gate failure seals `G2_GATE_G*_FAILED` with full telemetry (the measured full-D live path cost is itself a primary finding; the 2.0 ms gate was calibrated for the D=64 loop).

## Controls (pre-registered)

- **C2a — PDF skew form degeneracy:** `q = Re(ψᵀ(i·W_skew)·ψ)` ⇒ constant score ⇒ AUC ≈ 0.5 on the synthetic bank (form-level reproduction).
- **C2b — G1 kill reproduction (remote):** the exact G1 D=64 bridge + G1 fit on the REAL bank ⇒ min per-action AUC < 0.8500 (G1 measured 0.7768). Validates the harness and bank; the representation change is the causal fix.
- **C2c — label-shuffle anti-memorization:** fit on shuffled labels ⇒ AUC ≤ 0.60 on the synthetic bank.
- **C2d — fast-encoder equivalence:** cos ≥ 0.9999, max|Δ| ≤ 1e-3 vs production on 3 grids (CUDA).

## Bounds

- Branch `carrier/g2-local-block-gauge` @ `3a22c02`; engine `HENRI V2/experiments/verification/arc_g2_local_gauge_engine.py`; tests `HENRI V2/tests/contract/test_g2_local_gauge_engine.py`; flag `HENRI_G2_LOCAL_GAUGE=1` (default-OFF fail-closed).
- Bank `trajectories_production_run_f3v2.npz` (SHA `9e3c01b4…`; OBSERVED schema: psi/next_wave `[1536, 65536]`, actions_onehot `[1536, 7]`, action_names `[7]`); jsonl sidecar for env indexing.
- 12 envs × 150 steps = 1,800; seed **20260925**; horizon 8; waypoint threshold 0.60; η=0.10; Langevin T=0.50/3; τ_stall 0.90; β=10; τ_base 0.05; σ_ref 0.05.
- Host vast-5090 (`ssh -p 45864 root@107.206.71.138`), `/venv/main/bin/python`, torch 2.12.0+cu130; GPU free (OBSERVED 2026-09-01: 2 MiB / 32,607 MiB).

## Failure actions

- PG1 kill → seal `G2_AFFORDANCE_FIT_COLLAPSE` (0 live steps), preserve receipt, no quarantine.
- Live gate kill → seal the specific `G2_GATE_G*_FAILED` with full telemetry.
- Harness defect (device placement, bank schema, arcade make) → classify `BLOCKED_INFRASTRUCTURE`/`ERROR_FAIL_CLOSED`, fix, rerun the SAME bounds.
- Refs' fabricated receipts (SHAs/bytes/directive IDs contradicted by the direct probe) — zero accepted; my probe + reads are the evidence.

## Seal

Pre-registration sealed in the governance ledger BEFORE implementation; TDD engine + tests; local contracts; remote CUDA contracts; PG1 + gauntlet; verdict seal; results doc; memory update.
